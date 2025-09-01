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
    """Payment context for billing"""
    payment_token: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None


class PaymentSkill(Skill):
    """
    Payment processing and billing skill for WebAgents platform
    
    Key Features:
    - Payment token validation on connection
    - Origin/peer identity context management
    - LiteLLM cost calculation with markup
    - Connection finalization charging
    - Transaction creation via Portal API
    
    Based on webagents_v1 implementation patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all", dependencies=['auth'])
        
        # Configuration
        self.config = config or {}
        self.enable_billing = self.config.get('enable_billing', True)
        self.min_balance_agent = float(self.config.get('min_balance_agent', os.getenv('MIN_BALANCE_AGENT', '0.11')))
        # Agent pricing percent as percent (e.g., 20 means 20%)
        self.agent_pricing_percent = float(self.config.get('agent_pricing_percent', os.getenv('AGENT_PRICING_PERCENT', '20')))
        self.minimum_balance = float(self.config.get('minimum_balance', os.getenv('MINIMUM_BALANCE', '0.1')))
        # Optional external amount calculator: (llm_cost_usd, tool_cost_usd, agent_pricing_percent_percent) -> amount_to_charge
        self.amount_calculator: Optional[Callable[[float, float, float], float]] = self.config.get('amount_calculator')
        
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
            'min_balance_agent': self.min_balance_agent,
            'webagents_api_url': self.webagents_api_url,
            'has_webagents_client': bool(self.client),
            'litellm_available': LITELLM_AVAILABLE
        })
    
    # ===== CONNECTION LIFECYCLE HOOKS =====
    @prompt(priority=20, scope="all")
    def payments_general_prompt(self, context: Any = None) -> str:
        return "You are part of a network of AI agents working for their owners. You pay each other for services and help the owners make money."

    
    @hook("on_connection", priority=10)
    async def setup_payment_context(self, context) -> Any:
        """Setup payment context and validate payment token on connection"""
        self.logger.debug("🔧 PaymentSkill.setup_payment_context() called")
        self.logger.debug(f"   - enable_billing: {self.enable_billing}")
        self.logger.debug(f"   - agent_pricing_percent: {self.agent_pricing_percent}")
        self.logger.debug(f"   - minimum_balance: {self.minimum_balance}")
        
        if not self.enable_billing:
            self.logger.debug("   - Billing disabled, validating agent owner's min balance")
            try:
                if not self.client:
                    raise create_platform_unavailable_error("owner balance check")
                # With agent key, /user returns the agent owner's profile
                # Prefer /user/credits (availableCredits) if exposed; fallback to /user
                try:
                    credits = await self.client.user.credits()
                    available = float(credits)
                except Exception:
                    user_profile = await self.client.user.get()
                    available = float(getattr(user_profile, 'available_credits', 0))
                self.logger.debug(f"   - Owner available credits: ${available:.6f} (required: ${self.min_balance_agent:.2f})")
                if available < self.min_balance_agent:
                    raise create_insufficient_balance_error(
                        current_balance=available,
                        required_balance=self.min_balance_agent,
                        token_prefix=None,
                    )
            except Exception as e:
                if hasattr(e, 'status_code'):
                    raise
                self.logger.error(f"   - Owner min balance check failed: {e}")
                raise
            return context
        
        try:
            # Extract payment token and identity headers
            payment_token = self._extract_payment_token(context)
            
            # Get harmonized identity from auth skill context
            caller_user_id = None
            asserted_agent_id = None
            try:
                auth_ns = getattr(context, 'auth', None) or context.get('auth')
                if auth_ns:
                    caller_user_id = getattr(auth_ns, 'user_id', None)
                    asserted_agent_id = getattr(auth_ns, 'agent_id', None)
            except Exception:
                caller_user_id = None
                asserted_agent_id = None
            
            self.logger.debug(f"   - payment_token: {'present' if payment_token else 'MISSING'}")
            self.logger.debug(f"   - user_id: {caller_user_id}")
            self.logger.debug(f"   - agent_id (asserted): {asserted_agent_id}")
            
            # Create payment context
            payment_context = PaymentContext(
                payment_token=payment_token,
                user_id=caller_user_id,
                agent_id=asserted_agent_id,
            )
            
            # If agent_pricing_percent < 100, ensure owner's min balance first
            if self.agent_pricing_percent < 100.0:
                try:
                    if not self.client:
                        raise create_platform_unavailable_error("owner balance check")
                    try:
                        owner_available = float(await self.client.user.credits())
                    except Exception:
                        owner_profile = await self.client.user.get()
                        owner_available = float(getattr(owner_profile, 'available_credits', 0))
                    self.logger.debug(f"   - Owner available credits: ${owner_available:.6f} (required: ${self.min_balance_agent:.2f})")
                    if owner_available < self.min_balance_agent:
                        raise create_insufficient_balance_error(
                            current_balance=owner_available,
                            required_balance=self.min_balance_agent,
                            token_prefix=None,
                        )
                except Exception as e:
                    if hasattr(e, 'status_code'):
                        raise
                    self.logger.error(f"   - Owner min balance check failed: {e}")
                    raise

            # Validate payment token if provided (and agent_pricing_percent > 0 requires token)
            if payment_token:
                self.logger.debug(f"   - Validating payment token: {payment_token[:20]}...")
                validation_result = await self._validate_payment_token_with_balance(payment_token)
                self.logger.debug(f"   - Validation result: {validation_result}")
                
                if not validation_result['valid']:
                    self.logger.error(f"   - ❌ Payment token validation failed: {validation_result['error']}")
                    raise create_token_invalid_error(
                        token_prefix=payment_token[:20] if payment_token else None,
                        reason=validation_result.get('error', 'Token validation failed')
                    )
                
                # Check if balance meets minimum requirement
                balance = validation_result['balance']
                self.logger.debug(f"   - Balance check: ${balance:.2f} >= ${self.minimum_balance:.2f}")
                
                if balance < self.minimum_balance:
                    self.logger.error(f"   - ❌ Insufficient balance: ${balance:.2f} < ${self.minimum_balance:.2f} required")
                    raise create_insufficient_balance_error(
                        current_balance=balance,
                        required_balance=self.minimum_balance,
                        token_prefix=payment_token[:20] if payment_token else None
                    )
                
                self.logger.info(f"   - ✅ Payment token validated: {payment_token[:20]}... (balance: ${balance:.2f})")
            elif self.enable_billing and self.agent_pricing_percent > 0.0:
                # If billing is enabled but no payment token provided, require one (unless minimum_balance is 0)
                if self.minimum_balance > 0:
                    self.logger.error("   - ❌ Billing enabled but no payment token provided")
                    agent_name = getattr(self.agent, 'name', None) if hasattr(self, 'agent') else None
                    raise create_token_required_error(agent_name=agent_name)
            
            # Set payment context in payments namespace
            context.payments = payment_context
            
            self.logger.debug(
                f"Payment context setup: token={'✓' if payment_token else '✗'}, user_id={caller_user_id}, agent_id={asserted_agent_id}"
            )
            
        except PaymentError as e:
            # These are payment-specific errors that should return 402
            self.logger.error(f"🚨 Payment validation failed: {e}")
            self.logger.error(f"   - Error details: {e.to_dict()}")
            # Re-raise the specific payment error (it already has status_code=402)
            raise e
        except Exception as e:
            self.logger.error(f"🚨 Payment context setup failed: {e}")
            self.logger.error(f"   - enable_billing: {self.enable_billing}, payment_token: {'present' if payment_token else 'missing'}")
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
    
    @hook("finalize_connection", priority=95, scope="all")
    async def finalize_payment(self, context) -> Any:
        """Finalize payment by calculating total from context.usage and charging the token"""
        if not self.enable_billing:
            return context

        try:
            payment_context = getattr(context, 'payments', None)
            if not payment_context:
                return context

            # Sum LLM and tool costs from context.usage
            usage_records = getattr(context, 'usage', []) or []
            llm_cost_usd = 0.0
            tool_cost_usd = 0.0

            for record in usage_records:
                if not isinstance(record, dict):
                    continue
                record_type = record.get('type')
                if record_type == 'llm':
                    model = record.get('model')
                    prompt_tokens = int(record.get('prompt_tokens') or 0)
                    completion_tokens = int(record.get('completion_tokens') or 0)
                    try:
                        if LITELLM_AVAILABLE and cost_per_token and model:
                            p_cost, c_cost = cost_per_token(
                                model=model,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens
                            )
                            llm_cost_usd += float((p_cost or 0.0) + (c_cost or 0.0))
                    except Exception as e:
                        self.logger.debug(f"LLM cost_per_token failed for model {model}: {e}")
                        continue
                elif record_type == 'tool':
                    pricing = record.get('pricing') or {}
                    credits = pricing.get('credits')
                    if credits is not None:
                        try:
                            tool_cost_usd += float(credits)
                        except Exception:
                            continue

            # Calculate total to charge using external calculator if provided, else default formula
            if callable(self.amount_calculator):
                try:
                    # Log inputs and client auth fingerprint used for downstream calls
                    self.logger.debug(
                        f"🧮 Amount calculator input | llm_cost_usd={llm_cost_usd:.6f} "
                        f"tool_cost_usd={tool_cost_usd:.6f} agent_pricing_percent={self.agent_pricing_percent:.2f}% "
                        f"client_key_src={getattr(self.client, '_api_key_source', 'unknown')} "
                        f"client_key_fp={getattr(self.client, '_api_key_fingerprint', 'na')}"
                    )
                    result = self.amount_calculator(llm_cost_usd, tool_cost_usd, self.agent_pricing_percent)
                    if inspect.isawaitable(result):
                        to_charge = float(await result)
                    else:
                        to_charge = float(result)
                    self.logger.debug(f"🧮 Amount calculator output | to_charge={to_charge:.6f}")
                except Exception as e:
                    self.logger.error(f"Amount calculator failed: {e}; falling back to default formula")
                    to_charge = (llm_cost_usd + tool_cost_usd) * (1.0 + (self.agent_pricing_percent / 100.0))
            else:
                to_charge = (llm_cost_usd + tool_cost_usd) * (1.0 + (self.agent_pricing_percent / 100.0))

            if to_charge <= 0:
                return context

            # Charge payment token directly
            if payment_context.payment_token:
                success = await self._charge_payment_token(
                    payment_context.payment_token,
                    to_charge,
                    f"Agent {getattr(self.agent, 'name', 'unknown')} usage (margin {self.agent_pricing_percent:.2f}%)"
                )
                if success:
                    self.logger.info(
                        f"💳 Payment finalized: ${to_charge:.6f} charged to token {payment_context.payment_token[:20]}..."
                    )
                else:
                    self.logger.error(
                        f"💳 Payment failed: ${to_charge:.6f} could not charge token {payment_context.payment_token[:20]}..."
                    )
            else:
                # Enforce billing policy: if there are any costs and no token, raise 402
                self.logger.error(f"💳 Billing enabled but no payment token available for ${to_charge:.6f}")
                raise create_token_required_error(agent_name=getattr(self.agent, 'name', None))

        except Exception as e:
            self.logger.error(f"Payment finalization failed: {e}")

        return context
    
    
    # ===== INTERNAL METHODS =====
    
    def _extract_payment_token(self, context) -> Optional[str]:
        """Extract payment token from context headers"""
        headers = context.request.headers
        query_params = context.request.query_params
        
        self.logger.debug(f"🔍 Extracting payment token from context")
        self.logger.debug(f"   - headers: {list(headers.keys()) if headers else 'NONE'}")
        self.logger.debug(f"   - query_params: {list(query_params.keys()) if query_params else 'NONE'}")
        
        # Try X-Payment-Token header
        payment_token = headers.get('X-Payment-Token') or headers.get('x-payment-token')
        
        if payment_token:
            self.logger.debug(f"   - Found X-Payment-Token: {payment_token[:20]}...")
            return payment_token
        
        
        # Try query parameters
        token = query_params.get('payment_token')
        if token:
            self.logger.debug(f"   - Found query param payment_token: {token[:20]}...")
            return token
            
        self.logger.debug(f"   - No payment token found in any location")
        return None
    
    def _extract_header(self, context, header_name: str) -> Optional[str]:
        """Extract header value from context"""
        headers = context.get('headers', {})
        return headers.get(header_name) or headers.get(header_name.lower())
    
    async def _validate_payment_token(self, token: str) -> bool:
        """Validate payment token with WebAgents Platform"""
        try:
            if not self.client:
                self.logger.warning("Cannot validate payment token - no platform client")
                raise create_platform_unavailable_error("token validation")
            
            # Use the object-oriented token validation method
            return await self.client.tokens.validate(token)
            
        except Exception as e:
            self.logger.error(f"Payment token validation error: {e}")
            # If it's already a PaymentError, re-raise it
            if isinstance(e, PaymentError):
                raise e
            # Otherwise, create a validation error
            token_prefix = token[:20] if token else None
            raise create_token_invalid_error(
                token_prefix=token_prefix,
                reason=str(e)
            )
    
    async def _validate_payment_token_with_balance(self, token: str) -> Dict[str, Any]:
        """Validate payment token and check balance with WebAgents Platform"""
        try:
            if not self.client:
                raise create_platform_unavailable_error("token balance check")
            
            # Use the object-oriented token validation with balance method
            return await self.client.tokens.validate_with_balance(token)
            
        except PaymentError as e:
            # If it's already a PaymentError, convert to dict format expected by caller
            return {'valid': False, 'error': str(e), 'balance': 0.0}
        except Exception as e:
            self.logger.error(f"Payment token balance check failed: {e}")
            return {'valid': False, 'error': str(e), 'balance': 0.0}
    
    
    async def _charge_payment_token(self, token: str, amount_usd: float, description: str) -> bool:
        """Charge payment token for the specified amount"""
        try:
            if not self.client:
                raise create_platform_unavailable_error("token charging")
            
            # Convert amount to credits (using current conversion rate)
            credits = amount_usd
            
            # Require full token format id:secret for redemption
            try:
                has_secret = isinstance(token, str) and (":" in token) and len(token.split(":", 1)[1]) > 0
            except Exception:
                has_secret = False
            if not has_secret:
                token_prefix = token[:20] if token else None
                raise create_token_invalid_error(
                    token_prefix=token_prefix,
                    reason="Token must include secret in 'id:secret' format for redemption"
                )

            # Use the object-oriented token redeem method
            return await self.client.tokens.redeem(token, credits)
            
        except Exception as e:
            self.logger.error(f"Payment token charge failed: {e}")
            # If it's already a PaymentError, re-raise it
            if isinstance(e, PaymentError):
                raise e
            # Otherwise, create a charging error
            token_prefix = token[:20] if token else None
            raise create_charging_error(
                amount=amount_usd,
                token_prefix=token_prefix,
                reason=str(e)
            )
    