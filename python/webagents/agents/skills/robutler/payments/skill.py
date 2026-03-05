"""
PaymentSkill - WebAgents V2.0 Platform Integration

Payment processing and billing skill for WebAgents platform.
Validates payment tokens, locks budget, and settles via Robutler's /settle endpoint.
Cost computation is done server-side from raw usage records (MODEL_PRICING).
"""

import os
import inspect
import functools
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook, prompt
from robutler.api import RobutlerClient
from robutler.api.types import ApiResponse
from .exceptions import (
    PaymentError,
    create_token_required_error,
    create_token_invalid_error,
    create_insufficient_balance_error,
    create_charging_error,
    create_platform_unavailable_error
)


@dataclass
class PricingInfo:
    """Pricing information returned by decorated functions for dynamic pricing"""
    credits: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None
    on_success: Optional[Callable] = None
    on_fail: Optional[Callable] = None


def pricing(credits_per_call: Optional[float] = None,
           lock: Optional[float] = None,
           reason: Optional[str] = None,
           on_success: Optional[Callable] = None, 
           on_fail: Optional[Callable] = None):
    """
    Pricing decorator for tool functions - integrates with PaymentSkill
    
    This decorator attaches pricing metadata to tool functions. The actual
    cost calculation and billing is handled by the PaymentSkill when loaded.
    
    Usage patterns:
    1. Fixed pricing: @pricing(credits_per_call=0.05)
    2. Dynamic pricing: @pricing() + return (result, PricingInfo(...))
    3. Callback pricing: @pricing(credits_per_call=0.10, on_success=callback_func)
    
    Args:
        credits_per_call: Fixed credits to charge per call
        lock: Amount to lock before execution (default: credits_per_call)
        reason: Custom reason for usage record  
        on_success: Callback after successful payment
        on_fail: Callback after failed payment
        
    Note: Pricing only applies when PaymentSkill is loaded and billing is enabled.
    Without PaymentSkill, the decorator is inert but doesn't break functionality.
        
    Example usage:
    
    @tool
    @pricing(credits_per_call=0.02, reason="Weather API lookup")
    async def get_weather(location: str) -> str:
        return f"Weather for {location}: Sunny, 72°F"
    
    @tool  
    @pricing()  # Dynamic pricing based on processing complexity
    async def analyze_data(data: str) -> tuple:
        complexity = len(data)
        result = f"Analysis of {complexity} characters"
        # Simple complexity-based pricing: 0.001 credits per character
        credits = max(0.01, complexity * 0.001)  # Minimum 0.01 credits
        pricing_info = PricingInfo(
            credits=credits,
            reason=f"Data analysis of {complexity} chars",
            metadata={"character_count": complexity, "rate_per_char": 0.001}
        )
        return result, pricing_info
    """
    def decorator(func: Callable) -> Callable:
        # Store pricing metadata on function for extraction by PaymentSkill
        func._webagents_pricing = {
            'credits_per_call': credits_per_call,
            'lock': lock,
            'reason': reason or f"Tool '{func.__name__}' execution",
            'on_success': on_success,
            'on_fail': on_fail,
            'supports_dynamic': credits_per_call is None  # If no fixed price, supports dynamic pricing
        }
        
        def _attach_usage_tuple(result):
            # If dynamic pricing returns (result, PricingInfo) -> convert to (result, usage_dict)
            if isinstance(result, tuple) and len(result) == 2:
                res0, res1 = result
                try:
                    if hasattr(res1, '__dict__'):
                        usage = {
                            'pricing': {
                                'credits': getattr(res1, 'credits', None),
                                'reason': getattr(res1, 'reason', None),
                                'metadata': getattr(res1, 'metadata', None),
                            }
                        }
                        return res0, usage
                    if isinstance(res1, dict):
                        return result
                except Exception:
                    return result
                return result
            # If fixed pricing configured and function returned plain result -> add usage tuple
            if credits_per_call is not None:
                usage = {
                    'pricing': {
                        'credits': float(credits_per_call),
                        'reason': reason or f"Tool '{func.__name__}' execution",
                    }
                }
                return result, usage
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            return _attach_usage_tuple(result)
        
        @functools.wraps(func) 
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return _attach_usage_tuple(result)
        
        if inspect.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        # Copy pricing metadata to wrapper
        wrapper._webagents_pricing = func._webagents_pricing
        
        return wrapper
    
    return decorator


@dataclass
class PaymentContext:
    """Payment context for billing.

    Lifecycle: verify → lock → settle
        1. ``payment_token`` is extracted from request headers on connection.
        2. The token is **verified** (POST /api/payments/verify).
        3. A **lock** reserves a budget from the token (POST /api/payments/lock).
        4. On finalization the actual cost is **settled** against the lock
           (POST /api/payments/settle).
    """
    payment_token: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    # Lock state (populated after successful lock)
    lock_id: Optional[str] = None
    locked_amount_dollars: float = 0.0
    # Settlement state (populated after finalize)
    payment_successful: bool = False
    # BYOK: provider names from JWT byok claim
    byok_providers: List[str] = field(default_factory=list)
    byok_user_id: Optional[str] = None


class PaymentSkill(Skill):
    """Payment processing and billing skill for WebAgents platform.

    Uses the Robutler payment token APIs:
        1. **Verify** (POST /api/payments/verify) – validate token + check balance
        2. **Lock**   (POST /api/payments/lock)   – reserve a budget ceiling
        3. **Settle** (POST /api/payments/settle)  – charge actual usage, release remainder

    Cost computation is done server-side: agents forward raw usage records
    (model, prompt_tokens, completion_tokens, cached tokens) to /settle,
    and Robutler computes the dollar cost from MODEL_PRICING.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all", dependencies=['auth'])
        
        self.config = config or {}
        self.enable_billing = self.config.get('enable_billing', True)
        self.minimum_balance = float(self.config.get('minimum_balance', os.getenv('MINIMUM_BALANCE', '0.01')))
        self.per_message_lock = float(self.config.get('per_message_lock', os.getenv('PER_MESSAGE_LOCK', '0.005')))
        self.default_tool_lock = float(self.config.get('default_tool_lock', os.getenv('DEFAULT_TOOL_LOCK', '0.20')))
        
        self.webagents_api_url = (
            self.config.get('webagents_api_url')
            or os.getenv('ROBUTLER_INTERNAL_API_URL')
            or os.getenv('ROBUTLER_API_URL')
            or 'http://localhost:3000'
        )
        self.robutler_api_key = self.config.get('robutler_api_key') or getattr(self.agent, 'api_key', None)
        
        self.client: Optional[RobutlerClient] = None
    
    async def initialize(self, agent) -> None:
        """Initialize PaymentSkill with WebAgents Platform client"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.payments', agent.name)
        
        # Log level is configured globally via setup_logging() - no need for manual configuration
        
        # Resolve WebAgents API key: config -> agent's api_key -> environment -> default
        if not self.robutler_api_key:
            self.logger.error("No agent API key provided - cannot initialize platform client for payments")
            raise ValueError("Agent API key is required for payment processing")
        
        # Initialize WebAgents Platform client
        try:
            # Log init (no secrets)
            self.logger.debug(
                f"🔐 Initializing RobutlerClient | base_url={self.webagents_api_url} using agent API key"
            )
            self.client = RobutlerClient(
                api_key=self.robutler_api_key,
                base_url=self.webagents_api_url
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebAgents Platform client: {e}")
            # Continue without platform integration for testing
            self.client = None
        
        log_skill_event(agent.name, 'payments', 'initialized', {
            'enable_billing': self.enable_billing,
            'minimum_balance': self.minimum_balance,
            'per_message_lock': self.per_message_lock,
            'webagents_api_url': self.webagents_api_url,
            'has_webagents_client': bool(self.client),
        })
    
    # ===== CONNECTION LIFECYCLE HOOKS =====
    @prompt(priority=20, scope="all")
    def payments_general_prompt(self, context: Any = None) -> str:
        return "Agent network: Pay each other for services, help owners earn."

    
    @hook("on_connection", priority=10)
    async def setup_payment_context(self, context) -> Any:
        """Setup payment context: verify payment token → lock budget.

        Flow (all via Robutler payment-token APIs):
            1. Extract payment token from request headers.
            2. Verify token + check its balance (POST /api/payments/verify).
            3. Lock a budget from the token (POST /api/payments/lock).
            4. Store lockId in PaymentContext for later settlement.

        No owner/account balance is checked — only the payment token matters.
        """
        self.logger.debug("🔧 PaymentSkill.setup_payment_context() called")
        self.logger.debug(f"   - enable_billing: {self.enable_billing}")
        self.logger.debug(f"   - minimum_balance: {self.minimum_balance}")

        if not self.enable_billing:
            self.logger.debug("   - Billing disabled, skipping payment verification/lock")
            # Still extract and store the payment token so downstream tools
            # (e.g. NLI) can forward it for agent-to-agent paid calls.
            try:
                passthrough_token = self._extract_payment_token(context)
                if passthrough_token:
                    context.payments = PaymentContext(payment_token=passthrough_token)
                    self.logger.debug(f"   - Passthrough payment token stored for NLI forwarding: {passthrough_token[:20]}...")
            except Exception:
                pass
            return context

        # ── 1. Extract identity & payment token ──
        try:
            payment_token = self._extract_payment_token(context)

            caller_user_id = None
            asserted_agent_id = None
            try:
                auth_ns = getattr(context, 'auth', None) or context.get('auth')
                if auth_ns:
                    caller_user_id = getattr(auth_ns, 'user_id', None)
                    asserted_agent_id = getattr(auth_ns, 'agent_id', None)
            except Exception:
                pass

            self.logger.debug(f"   - payment_token: {'present' if payment_token else 'MISSING'}")
            self.logger.debug(f"   - user_id: {caller_user_id}")
            self.logger.debug(f"   - agent_id (asserted): {asserted_agent_id}")

            payment_context = PaymentContext(
                payment_token=payment_token,
                user_id=caller_user_id,
                agent_id=asserted_agent_id,
            )

            # ── 2. Verify + lock payment token ──
            if payment_token:
                if not self.client:
                    raise create_platform_unavailable_error("token verify+lock")

                # 2a. Verify token balance (POST /api/payments/verify)
                self.logger.debug(f"   - Verifying payment token: {payment_token[:20]}...")
                verification = await self.client.tokens.validate_with_balance(payment_token)
                self.logger.debug(f"   - Verification result: {verification}")

                if not verification.get('valid'):
                    raise create_token_invalid_error(
                        token_prefix=payment_token[:20],
                        reason=verification.get('error', 'Token validation failed'),
                    )

                balance = verification.get('balance', 0.0)
                # Require enough for at least one interaction (~$0.002 typical).
                # Don't gate on minimum_balance — that's the initial funding amount,
                # not the per-request threshold. The lock below handles partial budgets.
                min_usable = 0.001
                if balance < min_usable:
                    raise create_insufficient_balance_error(
                        current_balance=balance,
                        required_balance=min_usable,
                        token_prefix=payment_token[:20],
                    )

                self.logger.info(f"   - ✅ Token verified: {payment_token[:20]}... (balance: ${balance:.4f})")

                # 2b. Read BYOK claim from JWT
                try:
                    import base64, json as _json
                    parts = payment_token.split('.')
                    if len(parts) >= 2:
                        payload = parts[1] + '=' * (4 - len(parts[1]) % 4)
                        claims = _json.loads(base64.urlsafe_b64decode(payload))
                        byok_providers = claims.get('byok', [])
                        byok_user_id = claims.get('sub')
                        if byok_providers:
                            payment_context.byok_providers = byok_providers
                            payment_context.byok_user_id = byok_user_id
                            context.byok_providers = byok_providers
                            context.byok_user_id = byok_user_id
                            self.logger.info(f"   - 🔑 BYOK providers from token: {byok_providers}")
                except Exception as e:
                    self.logger.debug(f"   - Failed to read byok claim: {e}")

                # 2c. Lock budget from the token (POST /api/payments/lock)
                # Lock only a small per-message amount for the agent's own LLM
                # inference (~$0.002–$0.005 typical).  The remaining unlocked
                # balance stays available for NLI payment delegation (creating
                # child tokens for sub-agents).  `minimum_balance` is the gate
                # threshold, NOT the lock amount.
                per_message_lock = getattr(self, 'per_message_lock', 0.005)
                lock_amount = min(balance, per_message_lock)
                self.logger.debug(f"   - Locking ${lock_amount:.4f} from token...")
                try:
                    lock_result = await self.client.tokens.lock(payment_token, lock_amount)
                    payment_context.lock_id = lock_result['lockId']
                    payment_context.locked_amount_dollars = lock_result.get('lockedAmountDollars', lock_amount)
                    self.logger.info(
                        f"   - 🔒 Locked ${payment_context.locked_amount_dollars:.4f} "
                        f"(lockId={payment_context.lock_id})"
                    )
                except Exception as lock_err:
                    status = getattr(lock_err, 'status_code', None)
                    if status == 400:
                        self.logger.warning(
                            f"   - ⚠️ Lock failed (HTTP 400, likely insufficient effective balance). "
                            f"Falling back to zero-amount lock for tracking."
                        )
                        try:
                            lock_result = await self.client.tokens.lock(payment_token, 0)
                            payment_context.lock_id = lock_result['lockId']
                            payment_context.locked_amount_dollars = 0.0
                        except Exception:
                            self.logger.warning("   - ⚠️ Zero-amount lock also failed, proceeding without lock")
                    else:
                        raise

            elif self.enable_billing and self.minimum_balance > 0:
                self.logger.info("   - 💳 Billing enabled but no payment token provided — returning 402")
                agent_name = getattr(self.agent, 'name', None) if hasattr(self, 'agent') else None
                err = create_token_required_error(agent_name=agent_name)
                # Enrich the error with x402 V2 accepts so the platform can auto-create a token
                err.context['x402Version'] = 2
                err.context['accepts'] = [{
                    'scheme': 'token',
                    'network': 'robutler',
                    'amount': str(self.minimum_balance),
                    'asset': 'robutler:credits',
                    'maxTimeoutSeconds': 300,
                    'extra': {
                        'tokenType': 'jwt',
                    },
                }]
                raise err

            # ── 3. Attach to context ──
            context.payments = payment_context
            self.logger.debug(
                f"Payment context ready: token={'✓' if payment_token else '✗'}, "
                f"lock={'✓ ' + str(payment_context.lock_id) if payment_context.lock_id else '✗'}, "
                f"user_id={caller_user_id}"
            )

        except PaymentError as e:
            self.logger.info(f"💳 Payment flow: {e.error_code} — {e.user_message}")
            self.logger.debug(f"   - Payment details: {e.to_dict()}")
            raise e
        except Exception as e:
            self.logger.error(f"🚨 Payment context setup failed: {e}")
            raise

        return context
    
    @hook("on_message", priority=90, scope="all")
    async def prepare_byok_keys(self, context) -> Any:
        """Lazily fetch BYOK keys if byok claim is present (so native LLM skills can use them)."""
        byok_providers = getattr(context, 'byok_providers', [])
        if byok_providers and not getattr(context, 'byok_keys', None):
            await self._fetch_byok_keys(context)
        return context
    
    @hook("after_toolcall", priority=90, scope="all")
    async def handle_tool_completion(self, context) -> Any:
        """Release lock extension if tool failed (error in result)."""
        if not self.enable_billing:
            return context

        tool_result = None
        if hasattr(context, 'get'):
            tool_result = context.get("tool_result")
        else:
            tool_result = getattr(context, 'tool_result', None)

        if tool_result and isinstance(tool_result, str) and "error" in tool_result.lower():
            payment_ctx = getattr(context, 'payments', None)
            if payment_ctx and payment_ctx.lock_id:
                self.logger.info(f"🔓 Tool failed, lock {payment_ctx.lock_id} will be fully settled at finalization")

        return context
    
    @hook("before_toolcall", priority=10, scope="all")
    async def preauth_tool_lock(self, context) -> Any:
        """Extend connection lock before executing @pricing-decorated tools."""
        if not self.enable_billing:
            return context

        tool_call = context.get("tool_call") if hasattr(context, 'get') else getattr(context, 'tool_call', None)
        if not tool_call:
            return context

        pricing_meta = self._get_pricing_for_tool(tool_call)
        if not pricing_meta:
            pricing_meta = {
                'lock': self.default_tool_lock,
                'reason': f"Tool execution (default lock)",
            }

        lock_amount = pricing_meta.get('lock') or pricing_meta.get('credits_per_call') or 0

        if lock_amount <= 0 and pricing_meta.get('supports_dynamic'):
            skill_instance = self._find_skill_for_tool(tool_call)
            if skill_instance and hasattr(skill_instance, 'calculate_lock'):
                try:
                    tool_args = self._extract_tool_args(tool_call)
                    lock_amount = skill_instance.calculate_lock(tool_args)
                except Exception as e:
                    self.logger.warning(f"calculate_lock failed: {e}")
                    lock_amount = float(pricing_meta.get('max_lock', 0.50))

        if lock_amount <= 0:
            return context

        payment_ctx = getattr(context, 'payments', None)

        if not payment_ctx or not payment_ctx.lock_id:
            if payment_ctx and payment_ctx.payment_token and self.client:
                self.logger.info(f"🔒 No lock exists, attempting fresh lock for ${lock_amount:.4f}")
                try:
                    lock_result = await self.client.tokens.lock(payment_ctx.payment_token, lock_amount)
                    payment_ctx.lock_id = lock_result['lockId']
                    payment_ctx.locked_amount_dollars = lock_result.get('lockedAmountDollars', lock_amount)
                    self.logger.info(f"🔒 Created fresh lock: {payment_ctx.lock_id} for ${payment_ctx.locked_amount_dollars:.4f}")
                    return context
                except Exception as lock_err:
                    self.logger.warning(f"🔒 Fresh lock creation failed: {lock_err}")

            tool_name = self._get_tool_name(tool_call) or pricing_meta.get('reason', 'unknown')
            self.logger.warning(f"🚫 Tool '{tool_name}' blocked: billing enabled but no payment lock available")
            if hasattr(context, 'set'):
                context.set("tool_result", f"Tool '{tool_name}' blocked: no payment lock available. "
                            f"Required ${lock_amount:.4f}. The caller must provide a valid payment token with sufficient balance.")
                context.set("tool_skipped", True)
            return context

        # Try to extend the existing lock
        result = await self._extend_lock(payment_ctx.lock_id, lock_amount)

        if not result.get('success'):
            topped_up = await self._request_token_topup(context, lock_amount, pricing_meta)
            if topped_up:
                result = await self._extend_lock(payment_ctx.lock_id, lock_amount)

            if not result.get('success'):
                tool_name = self._get_tool_name(tool_call) or pricing_meta.get('reason', 'unknown')
                if hasattr(context, 'set'):
                    context.set("tool_result", f"Tool '{tool_name}' blocked: spending limit exceeded. "
                                f"Required ${lock_amount:.4f}, insufficient token balance.")
                    context.set("tool_skipped", True)
                return context

        payment_ctx.locked_amount_dollars += lock_amount
        return context

    @hook("finalize_connection", priority=95, scope="all")
    async def finalize_payment(self, context) -> Any:
        """Finalize payment: forward raw usage records to /settle for server-side cost computation."""
        if not self.enable_billing:
            return context

        try:
            payment_context: Optional[PaymentContext] = getattr(context, 'payments', None)
            if not payment_context:
                return context

            usage_records = getattr(context, 'usage', []) or []
            is_byok = getattr(context, 'is_byok', False)
            byok_provider_key_id = getattr(context, 'byok_provider_key_id', None)
            agent_name = getattr(self.agent, 'name', 'unknown')

            # Separate LLM and tool usage records
            llm_records = [r for r in usage_records if isinstance(r, dict) and r.get('type') == 'llm']
            tool_records = [r for r in usage_records if isinstance(r, dict) and r.get('type') == 'tool']

            self.logger.info(
                f"Payment finalization for '{agent_name}': "
                f"{len(llm_records)} LLM records, {len(tool_records)} tool records, BYOK={is_byok}"
            )

            for r in llm_records:
                self.logger.info(
                    f"  LLM: model={r.get('model')}, "
                    f"tokens={r.get('prompt_tokens', 0)}+{r.get('completion_tokens', 0)}"
                    f"{', cached_read=' + str(r['cached_read_tokens']) if r.get('cached_read_tokens') else ''}"
                )

            has_usage = bool(llm_records or tool_records)

            if not has_usage:
                self.logger.debug("  No usage to settle")
                if payment_context.lock_id:
                    try:
                        await self._settle_payment(payment_context.lock_id, 0, release=True)
                    except Exception:
                        pass
                return context

            if not payment_context.lock_id:
                self.logger.error(f"Billing enabled but no lock for settlement")
                raise create_token_required_error(agent_name=agent_name)

            lock_id = payment_context.lock_id

            # BYOK LLM: settle with charge_type='byok_llm' (server charges 0 for LLM, only agent fees)
            if is_byok and llm_records:
                r = await self._settle_payment(
                    lock_id,
                    usage=llm_records,
                    description="BYOK LLM usage",
                    charge_type='byok_llm',
                    provider_key_id=byok_provider_key_id,
                )
                self.logger.info(f"Settled byok_llm: {r.get('chargedDollars', 0)} -> {r.get('success')}")

                if tool_records:
                    tool_amount = sum(
                        float(r.get('pricing', {}).get('credits', 0))
                        for r in tool_records if isinstance(r, dict)
                    )
                    if tool_amount > 0:
                        r = await self._settle_payment(
                            lock_id, amount=tool_amount, description="Tool costs",
                        )
                        self.logger.info(f"Settled tool costs: ${tool_amount:.6f} -> {r.get('success')}")
            else:
                # Platform billing: forward all usage, server computes cost from MODEL_PRICING
                all_usage = llm_records + tool_records
                r = await self._settle_payment(
                    lock_id,
                    usage=all_usage,
                    description="LLM + tool usage",
                )
                self.logger.info(f"Settled usage: {r.get('chargedDollars', 0)} -> {r.get('success')}")

            # Release remaining lock balance
            try:
                await self._settle_payment(lock_id, amount=0, release=True)
            except Exception:
                pass

            payment_context.payment_successful = True

        except Exception as e:
            self.logger.error(f"Payment finalization failed: {e}")

        return context
    
    
    # ===== INTERNAL METHODS =====

    async def _fetch_byok_keys(self, context) -> Dict[str, Any]:
        """Lazily fetch BYOK keys from the internal portal API.
        
        Caches on context.byok_keys for the duration of the connection.
        Returns a dict of { provider: { key, tokenId } }.
        """
        cached = getattr(context, 'byok_keys', None)
        if cached is not None:
            return cached

        byok_user_id = getattr(context, 'byok_user_id', None)
        byok_providers = getattr(context, 'byok_providers', [])
        if not byok_user_id or not byok_providers:
            context.byok_keys = {}
            return {}

        try:
            import httpx
            internal_url = (
                os.getenv('ROBUTLER_INTERNAL_API_URL')
                or os.getenv('ROBUTLER_API_URL')
                or 'http://localhost:3000'
            )
            service_token = os.getenv('INTERNAL_SERVICE_TOKEN', '')
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{internal_url}/api/internal/users/{byok_user_id}/provider-keys",
                    headers={'Authorization': f'Bearer {service_token}'},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    keys = data.get('keys', {})
                    context.byok_keys = keys
                    self.logger.info(f"   - 🔑 Fetched BYOK keys for {len(keys)} providers")
                    return keys
                else:
                    self.logger.warning(f"   - ⚠️ Failed to fetch BYOK keys: HTTP {resp.status_code}")
        except Exception as e:
            self.logger.warning(f"   - ⚠️ BYOK key fetch failed: {e}")

        context.byok_keys = {}
        return {}

    def _extract_payment_token(self, context) -> Optional[str]:
        """Extract payment token: transport-agnostic context.payment_token first, then HTTP headers/query."""
        self.logger.debug("🔍 Extracting payment token from context")

        # 1. Transport-agnostic: set by transport layer before agent runs (UAMP, Realtime, etc.)
        if hasattr(context, "payment_token") and context.payment_token:
            token = context.payment_token
            if token and str(token).strip():
                self.logger.debug(f"   - Found context.payment_token: {str(token)[:20]}...")
                return str(token).strip()

        # 2. Fallback: HTTP headers and query params (backward compat for direct HTTP calls)
        if hasattr(context, "request") and context.request is not None:
            headers = getattr(context.request, "headers", None) or {}
            query_params = getattr(context.request, "query_params", None) or {}
            self.logger.debug(f"   - headers: {list(headers.keys()) if headers else 'NONE'}")
            self.logger.debug(f"   - query_params: {list(query_params.keys()) if query_params else 'NONE'}")

            payment_token = (
                headers.get("X-Payment-Token")
                or headers.get("x-payment-token")
                or headers.get("X-PAYMENT")
                or headers.get("x-payment")
            )
            if payment_token:
                self.logger.debug(f"   - Found payment header: {payment_token[:20]}...")
                return payment_token
            token = query_params.get("payment_token")
            if token:
                self.logger.debug(f"   - Found query param payment_token: {token[:20]}...")
                return token

        self.logger.debug("   - No payment token found in any location")
        return None
    
    def _extract_header(self, context, header_name: str) -> Optional[str]:
        """Extract header value from context"""
        headers = context.get('headers', {})
        return headers.get(header_name) or headers.get(header_name.lower())

    def _get_tool_name(self, tool_call) -> Optional[str]:
        """Extract tool name from a tool_call object."""
        if isinstance(tool_call, dict):
            fn = tool_call.get('function', {})
            return fn.get('name') if isinstance(fn, dict) else None
        return getattr(getattr(tool_call, 'function', None), 'name', None)
    
    # ------------------------------------------------------------------
    # Internal: Robutler payment API wrappers
    # ------------------------------------------------------------------

    async def _settle_payment(
        self, lock_id: str, amount: Optional[float] = None,
        usage: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        charge_type: Optional[str] = None, release: bool = False,
        provider_key_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Settle against a payment lock.

        Supports two modes:
        - ``amount``: pre-computed dollar cost (backward compat, flat-rate tools)
        - ``usage``: raw usage records -- server computes cost from MODEL_PRICING
        """
        try:
            if not self.client:
                raise create_platform_unavailable_error("payment settle")
            kwargs: Dict[str, Any] = dict(
                lock_id=lock_id,
                description=description,
                charge_type=charge_type,
                release=release,
            )
            if amount is not None:
                kwargs['amount'] = amount
            if usage is not None:
                kwargs['usage'] = usage
            if provider_key_id:
                kwargs['provider_key_id'] = provider_key_id
            return await self.client.tokens.settle(**kwargs)
        except Exception as e:
            if isinstance(e, PaymentError):
                raise
            self.logger.error(f"Payment settle failed: {e}")
            raise create_charging_error(
                amount=amount or 0,
                token_prefix=lock_id[:20] if lock_id else None,
                reason=str(e),
            )

    async def _charge_payment_token(self, token: str, amount_usd: float, description: str) -> bool:
        """Legacy fallback: charge payment token directly (no lock). Prefer lock+settle."""
        try:
            if not self.client:
                raise create_platform_unavailable_error("token charging")
            return await self.client.tokens.redeem(token, amount_usd, description=description)
        except Exception as e:
            if isinstance(e, PaymentError):
                raise
            self.logger.error(f"Payment token charge failed: {e}")
            raise create_charging_error(
                amount=amount_usd,
                token_prefix=token[:20] if token else None,
                reason=str(e),
            )

    async def _extend_lock(self, lock_id: str, amount_usd: float) -> Dict[str, Any]:
        """Extend an existing lock via PATCH /api/payments/lock/:id."""
        try:
            if not self.client:
                return {'success': False, 'error': 'No client'}
            return await self.client.tokens.extend_lock(lock_id, amount_usd)
        except Exception as e:
            self.logger.error(f"Lock extension failed: {e}")
            return {'success': False, 'error': str(e)}

    async def _request_token_topup(
        self, context, required_amount: float, pricing_meta: dict
    ) -> bool:
        """Send PaymentRequired over UAMP, await top-up ack."""
        session = getattr(context, 'uamp_session', None)
        if not session:
            return False  # HTTP mode -- cannot do mid-stream top-up

        payment_ctx = getattr(context, 'payments', None)
        if not payment_ctx:
            return False

        try:
            from webagents.uamp.events import PaymentRequiredEvent, PaymentScheme, PaymentRequirements
            await session.send_event(PaymentRequiredEvent(
                requirements=PaymentRequirements(
                    amount=str(required_amount),
                    currency='USD',
                    reason=pricing_meta.get('reason', 'Additional funds required for tool execution'),
                    schemes=[PaymentScheme(scheme='token', network='robutler')],
                ),
            ))

            import asyncio
            event = await asyncio.wait_for(
                session.wait_for_event('payment.submit'),
                timeout=300,
            )
            return event.data.get('success', False) if hasattr(event, 'data') else False
        except Exception as e:
            self.logger.warning(f"Token top-up request failed: {e}")
            return False

    def _get_pricing_for_tool(self, tool_call) -> Optional[Dict[str, Any]]:
        """Extract @pricing metadata for a tool call."""
        if not tool_call:
            return None
        # tool_call may be a dict with 'function' key, or have a name attr
        tool_name = None
        if isinstance(tool_call, dict):
            func = tool_call.get('function', {})
            tool_name = func.get('name') if isinstance(func, dict) else None
        elif hasattr(tool_call, 'function'):
            tool_name = getattr(tool_call.function, 'name', None)

        if not tool_name:
            return None

        # Look up the tool in the agent's registered tools
        agent = getattr(self, 'agent', None)
        if not agent:
            return None

        for t in getattr(agent, '_tools', []):
            func = getattr(t, '_func', t) if hasattr(t, '_func') else t
            if getattr(func, '__name__', '') == tool_name or getattr(t, 'name', '') == tool_name:
                return getattr(func, '_webagents_pricing', None)
        return None

    def _find_skill_for_tool(self, tool_call) -> Optional[Any]:
        """Find the skill instance that owns a given tool call."""
        tool_name = None
        if isinstance(tool_call, dict):
            func = tool_call.get('function', {})
            tool_name = func.get('name') if isinstance(func, dict) else None
        elif hasattr(tool_call, 'function'):
            tool_name = getattr(tool_call.function, 'name', None)

        if not tool_name:
            return None

        agent = getattr(self, 'agent', None)
        if not agent:
            return None

        for skill_name, skill in getattr(agent, 'skills', {}).items():
            for attr_name in dir(skill):
                attr = getattr(skill, attr_name, None)
                if callable(attr) and getattr(attr, '__name__', '') == tool_name:
                    return skill
                if callable(attr) and getattr(attr, '_webagents_pricing', None) is not None:
                    func = getattr(attr, '_func', attr) if hasattr(attr, '_func') else attr
                    if getattr(func, '__name__', '') == tool_name:
                        return skill
        return None

    def _extract_tool_args(self, tool_call) -> dict:
        """Parse the tool call's arguments JSON string into a dict."""
        import json as _json
        args_str = None
        if isinstance(tool_call, dict):
            func = tool_call.get('function', {})
            args_str = func.get('arguments', '{}') if isinstance(func, dict) else '{}'
        elif hasattr(tool_call, 'function'):
            args_str = getattr(tool_call.function, 'arguments', '{}')

        if not args_str:
            return {}
        if isinstance(args_str, dict):
            return args_str
        try:
            return _json.loads(args_str)
        except (ValueError, TypeError):
            return {}
    