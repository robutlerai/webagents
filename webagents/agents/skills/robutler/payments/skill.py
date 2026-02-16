"""
PaymentSkill - WebAgents V2.0 Platform Integration

Payment processing and billing skill for WebAgents platform.
Validates payment tokens, calculates costs using LiteLLM, and charges on connection finalization.
Based on webagents_v1 implementation patterns.
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

# Try to import LiteLLM for cost calculation
try:
    from litellm import completion_cost, cost_per_token
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion_cost = None
    cost_per_token = None


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


class PaymentSkill(Skill):
    """Payment processing and billing skill for WebAgents platform.

    Uses the Roborum payment token APIs:
        1. **Verify** (POST /api/payments/verify) – validate token + check balance
        2. **Lock**   (POST /api/payments/lock)   – reserve a budget ceiling
        3. **Settle** (POST /api/payments/settle)  – charge actual usage, release remainder

    Key Features:
        - Payment token verify + lock on connection
        - LiteLLM-based cost calculation with configurable agent markup
        - Settlement against the lock on connection finalization
        - Optional external amount calculator for revenue-sharing models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all", dependencies=['auth'])
        
        # Configuration
        self.config = config or {}
        self.enable_billing = self.config.get('enable_billing', True)
        # Agent pricing percent (e.g., 100 means 100% markup on LLM cost)
        self.agent_pricing_percent = float(self.config.get('agent_pricing_percent', os.getenv('AGENT_PRICING_PERCENT', '100')))
        self.minimum_balance = float(self.config.get('minimum_balance', os.getenv('MINIMUM_BALANCE', '0.01')))
        # Platform fee percent (fraction of agent markup that goes to platform)
        self.platform_fee_percent = float(self.config.get('platform_fee_percent', os.getenv('PLATFORM_FEE_PERCENT', '20'))) / 100.0
        
        # WebAgents integration
        # Prefer internal portal URL for in-cluster calls, then public URL, then localhost for dev
        self.webagents_api_url = (
            self.config.get('webagents_api_url')
            or os.getenv('ROBUTLER_INTERNAL_API_URL')
            or os.getenv('ROBUTLER_API_URL')
            or 'http://localhost:3000'
        )
        # IMPORTANT: Inside the PaymentSkill, payment token charges must use the agent's key only
        # No fallbacks to service keys here
        self.robutler_api_key = self.config.get('robutler_api_key') or getattr(self.agent, 'api_key', None)
        
        # API client for platform integration
        self.client: Optional[RobutlerClient] = None
        
        # Check LiteLLM availability
        if not LITELLM_AVAILABLE:
            self.logger.warning("LiteLLM not available - cost calculations will use fallback methods")
    
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
            'agent_pricing_percent': self.agent_pricing_percent,
            'minimum_balance': self.minimum_balance,
            'webagents_api_url': self.webagents_api_url,
            'has_webagents_client': bool(self.client),
            'litellm_available': LITELLM_AVAILABLE
        })
    
    # ===== CONNECTION LIFECYCLE HOOKS =====
    @prompt(priority=20, scope="all")
    def payments_general_prompt(self, context: Any = None) -> str:
        return "Agent network: Pay each other for services, help owners earn."

    
    @hook("on_connection", priority=10)
    async def setup_payment_context(self, context) -> Any:
        """Setup payment context: verify payment token → lock budget.

        Flow (all via Roborum payment-token APIs):
            1. Extract payment token from request headers.
            2. Verify token + check its balance (POST /api/payments/verify).
            3. Lock a budget from the token (POST /api/payments/lock).
            4. Store lockId in PaymentContext for later settlement.

        No owner/account balance is checked — only the payment token matters.
        """
        self.logger.debug("🔧 PaymentSkill.setup_payment_context() called")
        self.logger.debug(f"   - enable_billing: {self.enable_billing}")
        self.logger.debug(f"   - agent_pricing_percent: {self.agent_pricing_percent}")
        self.logger.debug(f"   - minimum_balance: {self.minimum_balance}")

        if not self.enable_billing:
            self.logger.debug("   - Billing disabled, skipping payment token flow")
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
                if balance < self.minimum_balance:
                    raise create_insufficient_balance_error(
                        current_balance=balance,
                        required_balance=self.minimum_balance,
                        token_prefix=payment_token[:20],
                    )

                self.logger.info(f"   - ✅ Token verified: {payment_token[:20]}... (balance: ${balance:.4f})")

                # 2b. Lock budget from the token (POST /api/payments/lock)
                # Lock only the minimum per message, not the full token balance.
                # This allows the token to be reused across many messages
                # (actual cost per message is typically ~$0.002).
                lock_amount = min(balance, self.minimum_balance)
                self.logger.debug(f"   - Locking ${lock_amount:.4f} from token...")
                lock_result = await self.client.tokens.lock(payment_token, lock_amount)
                payment_context.lock_id = lock_result['lockId']
                payment_context.locked_amount_dollars = lock_result.get('lockedAmountDollars', lock_amount)
                self.logger.info(
                    f"   - 🔒 Locked ${payment_context.locked_amount_dollars:.4f} "
                    f"(lockId={payment_context.lock_id})"
                )

            elif self.agent_pricing_percent > 0.0 and self.minimum_balance > 0:
                self.logger.error("   - ❌ Billing enabled but no payment token provided")
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
            self.logger.error(f"🚨 Payment validation failed: {e}")
            self.logger.error(f"   - Error details: {e.to_dict()}")
            raise e
        except Exception as e:
            self.logger.error(f"🚨 Payment context setup failed: {e}")
            raise

        return context
    
    @hook("on_message", priority=90, scope="all")
    async def accumulate_llm_costs(self, context) -> Any:
        """No-op: cost is calculated in finalize_connection from context.usage"""
        return context
    
    @hook("after_toolcall", priority=90, scope="all")
    async def accumulate_tool_costs(self, context) -> Any:
        """No-op: BaseAgent appends usage for tools."""
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
            return context

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
            return context

        # Try to extend the existing lock
        result = await self._extend_lock(payment_ctx.lock_id, lock_amount)

        if not result.get('success'):
            # Extension failed -- request token top-up via UAMP
            topped_up = await self._request_token_topup(context, lock_amount, pricing_meta)
            if topped_up:
                # Retry extension after top-up
                result = await self._extend_lock(payment_ctx.lock_id, lock_amount)

            if not result.get('success'):
                # Still failed -- return spend-limit error to LLM, don't crash
                tool_name = pricing_meta.get('reason') or 'unknown'
                if hasattr(context, 'set'):
                    context.set("tool_result", f"Tool '{tool_name}' blocked: spending limit exceeded. "
                                f"Required ${lock_amount:.4f}, insufficient token balance.")
                    context.set("tool_skipped", True)
                return context

        payment_ctx.locked_amount_dollars += lock_amount
        return context

    @hook("finalize_connection", priority=95, scope="all")
    async def finalize_payment(self, context) -> Any:
        """Finalize payment: calculate actual cost, settle platform fee first, then LLM cost, then agent markup."""
        if not self.enable_billing:
            return context

        try:
            payment_context: Optional[PaymentContext] = getattr(context, 'payments', None)
            if not payment_context:
                return context

            # ── 1. Sum LLM + tool costs from context.usage ──
            usage_records = getattr(context, 'usage', []) or []
            llm_cost_usd = 0.0
            tool_cost_usd = 0.0
            llm_breakdown: List[Dict[str, Any]] = []
            tool_breakdown: List[Dict[str, Any]] = []

            for record in usage_records:
                if not isinstance(record, dict):
                    continue
                record_type = record.get('type')
                if record_type == 'llm':
                    model = record.get('model')
                    prompt_tokens = int(record.get('prompt_tokens') or 0)
                    completion_tokens = int(record.get('completion_tokens') or 0)
                    self.logger.info(f"💰 Processing LLM usage: model={model}, tokens={prompt_tokens}+{completion_tokens}")
                    try:
                        if LITELLM_AVAILABLE and cost_per_token and model:
                            p_cost, c_cost = cost_per_token(
                                model=model,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )
                            record_cost = float((p_cost or 0.0) + (c_cost or 0.0))
                            llm_cost_usd += record_cost
                            self.logger.info(f"💰 Cost ${record_cost:.6f} for {model}")
                            llm_breakdown.append({
                                'model': model, 'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens, 'cost_usd': record_cost,
                            })
                    except Exception as e:
                        self.logger.warning(f"💰 cost_per_token failed for {model}: {e}")
                        continue
                elif record_type == 'tool':
                    pricing = record.get('pricing') or {}
                    credits = pricing.get('credits')
                    if credits is not None:
                        try:
                            record_cost = float(credits)
                            tool_cost_usd += record_cost
                            tool_breakdown.append({
                                'tool_name': record.get('tool_name', 'unknown'),
                                'reason': pricing.get('reason', 'Tool usage'),
                                'credits': record_cost,
                                'metadata': pricing.get('metadata', {}),
                            })
                        except Exception:
                            continue

            # ── 2. Calculate costs ──
            subtotal = llm_cost_usd + tool_cost_usd
            agent_markup = subtotal * (self.agent_pricing_percent / 100.0)
            platform_fee = agent_markup * self.platform_fee_percent
            agent_share = agent_markup - platform_fee
            is_byok = getattr(context, 'is_byok', False)

            # ── 3. Log breakdown ──
            agent_name = getattr(self.agent, 'name', 'unknown')
            self.logger.info(f"💰 Payment Breakdown for '{agent_name}':")
            self.logger.info(f"   📊 LLM: ${llm_cost_usd:.6f}")
            for r in llm_breakdown:
                self.logger.info(f"      - {r['model']}: {r['prompt_tokens']}+{r['completion_tokens']} = ${r['cost_usd']:.6f}")
            self.logger.info(f"   🛠️  Tools: ${tool_cost_usd:.6f}")
            for r in tool_breakdown:
                self.logger.info(f"      - {r['tool_name']}: ${r['credits']:.6f} ({r['reason']})")
            self.logger.info(
                f"   💵 Subtotal=${subtotal:.6f} | Agent markup ({self.agent_pricing_percent:.0f}%)=${agent_markup:.6f} | "
                f"Platform fee ({self.platform_fee_percent*100:.0f}%)=${platform_fee:.6f} | BYOK={is_byok}"
            )

            if subtotal <= 0 and agent_markup <= 0:
                self.logger.debug("   - Nothing to charge (cost=0)")
                # Release the lock if it exists
                if payment_context.lock_id:
                    try:
                        await self._settle_payment(payment_context.lock_id, 0, release=True)
                    except Exception:
                        pass
                return context

            # ── 4. Settle against lock: platform_fee first, then LLM cost, then agent markup ──
            if not payment_context.lock_id:
                self.logger.error(f"💳 Billing enabled but no lock for ${subtotal + agent_markup:.6f}")
                raise create_token_required_error(agent_name=agent_name)

            lock_id = payment_context.lock_id

            # 4a. Platform fee FIRST
            if platform_fee > 0:
                r = await self._settle_payment(
                    lock_id, platform_fee,
                    description=f"Platform fee ({self.platform_fee_percent*100:.0f}%)",
                    charge_type='platform_fee',
                )
                self.logger.info(f"💳 Settled platform_fee: ${platform_fee:.6f} -> {r.get('success')}")

            # 4b. LLM + tool cost
            if subtotal > 0:
                charge_type = 'agent_fee' if is_byok else 'platform_llm'
                r = await self._settle_payment(
                    lock_id, subtotal,
                    description=f"LLM + tool costs",
                    charge_type=charge_type,
                )
                self.logger.info(f"💳 Settled {charge_type}: ${subtotal:.6f} -> {r.get('success')}")

            # 4c. Agent's share of markup
            if agent_share > 0:
                r = await self._settle_payment(
                    lock_id, agent_share,
                    description=f"Agent {agent_name} fee ({self.agent_pricing_percent:.0f}%)",
                    charge_type='agent_fee',
                )
                self.logger.info(f"💳 Settled agent_fee: ${agent_share:.6f} -> {r.get('success')}")

            # 4d. Release remaining lock balance
            try:
                await self._settle_payment(lock_id, 0, release=True)
            except Exception:
                pass

            payment_context.payment_successful = True

        except Exception as e:
            self.logger.error(f"Payment finalization failed: {e}")

        return context
    
    
    # ===== INTERNAL METHODS =====
    
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
                headers.get("X-PAYMENT")
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
    
    # ------------------------------------------------------------------
    # Internal: Roborum payment API wrappers
    # ------------------------------------------------------------------

    async def _settle_payment(
        self, lock_id: str, amount_usd: float, description: str = "",
        charge_type: Optional[str] = None, release: bool = False,
    ) -> Dict[str, Any]:
        """Settle actual usage against a payment lock (POST /api/payments/settle)."""
        try:
            if not self.client:
                raise create_platform_unavailable_error("payment settle")
            return await self.client.tokens.settle(
                lock_id=lock_id,
                amount=amount_usd,
                description=description,
                charge_type=charge_type,
                release=release,
            )
        except Exception as e:
            if isinstance(e, PaymentError):
                raise
            self.logger.error(f"Payment settle failed: {e}")
            raise create_charging_error(
                amount=amount_usd,
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
                timeout=120,
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
    