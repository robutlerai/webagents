"""
PaymentSkillX402 - x402 Payment Protocol Support

Enhanced payment skill with full x402 protocol support.
Extends PaymentSkill with multi-scheme payments and automatic handling.
Supports JWT payment tokens with local JWKS verification and /api/payments/* endpoints.
"""

import os
import logging
from typing import Dict, Any, List, Optional

import jwt
from webagents.agents.skills.robutler.payments.skill import PaymentSkill
from webagents.agents.tools.decorators import hook
from .exceptions import (
    PaymentRequired402,
    X402UnsupportedScheme,
    X402VerificationFailed,
    X402SettlementFailed,
    X402ExchangeFailed
)
from .schemes import (
    encode_robutler_payment,
    decode_payment_header,
    extract_token_from_payment,
    create_x402_requirements,
    create_x402_response
)


class PaymentSkillX402(PaymentSkill):
    """
    Enhanced payment skill with full x402 protocol support.
    
    Superset of PaymentSkill:
    - All PaymentSkill functionality (token validation, cost calculation, etc.)
    - Full x402 protocol support (verify, settle, multiple schemes)
    - Agent B: Return 402 responses with x402 payment requirements
    - Agent A: Automatic payment via hooks (no explicit tools needed)
    - Exchange: Crypto-to-credits conversion
    
    Payment scheme: 'token' for network 'robutler' (not 'robutler-credits').
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Initialize parent PaymentSkill
        super().__init__(config)
        
        config = config or {}
        
        # x402-specific configuration (default: /api/payments for lock, verify, settle)
        self.facilitator_url = (
            config.get("facilitator_url")
            or os.getenv("X402_FACILITATOR_URL")
            or f"{self.webagents_api_url}/api/payments"
        )
        # Optional JWKS manager for local JWT verification (payment tokens)
        self._jwks_manager = config.get("jwks_manager")
        if self._jwks_manager is None:
            try:
                from webagents.crypto.jwks import JWKSManager
                self._jwks_manager = JWKSManager(config={"jwks_cache_ttl": 3600})
            except ImportError:
                self._jwks_manager = None
        
        # Payment schemes to accept (for Agent B)
        # Default: accept robutler token scheme
        self.accepted_schemes = config.get('accepted_schemes', [
            {'scheme': 'token', 'network': 'robutler'}
        ])
        
        # Payment schemes to use (for Agent A)
        # Priority order: robutler token, then blockchain if wallet configured
        self.payment_schemes = config.get('payment_schemes', [
            'token'  # Robutler token scheme
        ])
        
        # Optional: wallet for blockchain payments
        self.wallet_private_key = (
            config.get('wallet_private_key')
            or os.getenv('X402_WALLET_PRIVATE_KEY')
        )
        if self.wallet_private_key:
            self.payment_schemes.append('exact')  # Add blockchain support
        
        # Auto-exchange: convert crypto to credits when available
        self.auto_exchange = config.get('auto_exchange', True)
        
        # Max payment amount (safety limit)
        self.max_payment = float(
            config.get('max_payment')
            or os.getenv('X402_MAX_PAYMENT', '10.0')
        )
        
        self.logger = logging.getLogger(__name__)
    
    # =========================================================================
    # Agent B: HTTP Endpoint Payment Requirements
    # =========================================================================
    
    @hook("before_http_call", priority=10, scope="all")
    async def check_http_endpoint_payment(self, context) -> Any:
        """
        Intercept HTTP endpoint calls requiring payment.
        Return 402 with x402 PaymentRequirements if no valid payment.
        
        This hook is called before HTTP endpoints execute, allowing us to
        check for payment requirements and verify payments.
        """
        # Check if endpoint requires payment
        endpoint_func = getattr(context, 'endpoint_func', None)
        if not endpoint_func:
            return context
        
        requires_payment = getattr(endpoint_func, '_http_requires_payment', False)
        if not requires_payment:
            return context
        
        # Check for X-PAYMENT header
        payment_header = None
        if hasattr(context, 'request'):
            payment_header = context.request.headers.get('X-PAYMENT') or context.request.headers.get('x-payment')
        
        if not payment_header:
            # No payment provided - return 402 with x402 requirements
            pricing_info = endpoint_func._webagents_pricing
            payment_requirements = self._create_x402_requirements(
                endpoint_func, pricing_info, context
            )
            raise PaymentRequired402(payment_requirements)
        
        # Payment provided - verify and settle via x402 facilitator
        await self._process_x402_payment(payment_header, context, endpoint_func)
        return context
    
    def _create_x402_requirements(
        self,
        func,
        pricing,
        context
    ) -> Dict[str, Any]:
        """Create x402 PaymentRequirements response"""
        agent_id = getattr(self.agent, 'id', 'unknown')
        amount = pricing.get('credits_per_call', 0.0)
        
        # Get resource path
        resource = "/"
        if hasattr(context, 'request'):
            resource = context.request.url.path
        
        # Build list of accepted payment schemes
        accepts = []
        for scheme_config in self.accepted_schemes:
            accepts.append(create_x402_requirements(
                scheme=scheme_config['scheme'],
                network=scheme_config.get('network', 'robutler'),
                amount=amount,
                resource=resource,
                pay_to=agent_id,
                description=pricing.get('reason', 'API call'),
                mime_type='application/json'
            ))
        
        return create_x402_response(accepts)
    
    async def _verify_payment_token(
        self, token: str, expected_audience: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Verify payment token locally via JWKS when possible (JWT).
        Returns dict with isValid and balance, or None to fall back to API.
        When expected_audience is provided, JWT aud claim must be present and match.
        """
        if not self._jwks_manager:
            return None
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
            issuer = (unverified.get("iss") or "").strip()
            if not issuer:
                return None
            kid = jwt.get_unverified_header(token).get("kid")
            if not kid:
                return None
            jwks_uri = f"{issuer.rstrip('/')}/.well-known/jwks.json"
            public_key = await self._jwks_manager.get_public_key_from_jwks(jwks_uri, kid)
            if not public_key:
                return None
            decode_kw: Dict[str, Any] = {
                "algorithms": ["RS256"],
                "options": {"require": ["exp", "payment"]},
            }
            if expected_audience:
                decode_kw["audience"] = expected_audience
            verified = jwt.decode(token, public_key, **decode_kw)
            payment = verified.get("payment") or {}
            balance = payment.get("balance")
            if balance is None:
                return None
            return {"isValid": True, "balance": float(balance)}
        except Exception:
            return None

    async def _process_x402_payment(
        self,
        payment_header: str,
        context,
        endpoint_func
    ) -> None:
        """
        Verify and settle x402 payment. Uses local JWKS verification for JWT
        tokens when available, otherwise facilitator verify; always settles via facilitator.
        """
        pricing_info = endpoint_func._webagents_pricing
        amount = pricing_info.get("credits_per_call", 0.0)

        try:
            payment_data = decode_payment_header(payment_header)
        except ValueError as e:
            raise X402VerificationFailed(f"Invalid payment header: {e}") from e

        scheme = payment_data.get("scheme")
        network = payment_data.get("network")
        token_for_api = extract_token_from_payment(payment_data)

        resource = "/"
        if hasattr(context, "request"):
            resource = context.request.url.path

        requirements = {
            "scheme": scheme,
            "network": network,
            "maxAmountRequired": str(amount),
            "payTo": getattr(self.agent, "id", "unknown"),
            "resource": resource,
            "description": pricing_info.get("reason", "API call"),
        }

        # Try local JWT verification first
        verify_result = None
        if payment_data.get("_is_jwt") and self._jwks_manager:
            verify_result = await self._verify_payment_token(token_for_api)

        if verify_result is None:
            verify_result = await self.client.facilitator.verify(
                payment_header, requirements
            )

        if not verify_result.get("isValid"):
            reason = verify_result.get("invalidReason", "Payment verification failed")
            raise X402VerificationFailed(reason)

        if float(verify_result.get("balance", 0)) < amount:
            raise X402VerificationFailed("Insufficient token balance")

        settle_result = await self.client.facilitator.settle(
            payment_header, requirements
        )

        if not settle_result.get("success"):
            error = settle_result.get("error", "Settlement failed")
            raise X402SettlementFailed(error)

        self.logger.info(
            f"x402 payment settled: {scheme}:{network} {amount} credits",
            extra={"txHash": settle_result.get("transactionHash")},
        )
    
    # =========================================================================
    # Agent A: Automatic Payment Handling
    # =========================================================================
    
    # Note: Full Agent A HTTP client integration would require hooks into
    # the HTTP client library being used. The plan specifies no tools initially,
    # so we provide helper methods that can be used by the agent's HTTP client
    # or added as tools later.
    
    async def _get_available_token(self, context) -> Optional[str]:
        """
        Get available robutler payment token.
        
        Checks:
        1. Context payment token (from PaymentSkill)
        2. Agent's token list via API
        
        No KV dependency.
        """
        # Check context first (from PaymentSkill)
        payment_context = getattr(context, 'payments', None)
        if payment_context:
            token = getattr(payment_context, 'payment_token', None)
            if token:
                # Verify token still valid
                try:
                    result = await self.client.tokens.validate_with_balance(token)
                    if result.get('valid') and result.get('balance', 0) > 0:
                        return token
                except Exception as e:
                    self.logger.warning(f"Token validation failed: {e}")
        
        # Check agent's token list via API
        # This includes virtual tokens from blockchain payments
        agent_id = getattr(self.agent, 'id', None)
        if agent_id and self.client:
            try:
                # Use existing agent tokens endpoint
                response = await self.client._make_request(
                    'GET',
                    f'/agents/{agent_id}/payment-tokens',
                    params={'status': 'active'}
                )
                if response.success:
                    tokens = response.data.get('tokens', [])
                    # Find first valid token with balance
                    for token_data in tokens:
                        token = token_data.get('token')
                        balance = token_data.get('balance', 0)
                        if token and balance > 0:
                            return token
            except Exception as e:
                self.logger.warning(f"Failed to fetch agent tokens: {e}")
        
        return None
    
    async def _create_payment(
        self,
        accepts: List[Dict],
        context
    ) -> tuple[str, str, float]:
        """
        Create payment for one of the accepted schemes.
        
        Priority:
        1. scheme='token', network='robutler' with existing token
        2. scheme='token' via exchange (if auto_exchange)
        3. Direct blockchain payment
        
        Returns:
            (payment_header, scheme_description, cost)
        """
        # Try robutler token first
        for req in accepts:
            if req.get('scheme') == 'token' and req.get('network') == 'robutler':
                # Check for existing token
                token = await self._get_available_token(context)
                
                if token:
                    # Use existing token
                    payment_header = encode_robutler_payment(
                        token, req['maxAmountRequired']
                    )
                    return (
                        payment_header,
                        'token:robutler',
                        float(req['maxAmountRequired'])
                    )
                
                # No token - try exchange
                if self.auto_exchange and self.wallet_private_key:
                    amount = float(req['maxAmountRequired'])
                    try:
                        new_token = await self._exchange_for_credits(amount, context)
                        payment_header = encode_robutler_payment(
                            new_token, req['maxAmountRequired']
                        )
                        return (
                            payment_header,
                            'token:robutler-via-exchange',
                            float(req['maxAmountRequired'])
                        )
                    except Exception as e:
                        self.logger.warning(f"Exchange failed: {e}")
        
        # Try blockchain schemes
        if self.wallet_private_key:
            for req in accepts:
                if req.get('scheme') == 'exact':
                    try:
                        payment_header = await self._create_blockchain_payment(
                            float(req['maxAmountRequired']),
                            req['scheme'],
                            req['network']
                        )
                        return (
                            payment_header,
                            f"{req['scheme']}:{req['network']}",
                            float(req['maxAmountRequired'])
                        )
                    except NotImplementedError:
                        # Blockchain payment not yet implemented
                        pass
        
        raise X402UnsupportedScheme(
            "No compatible payment method available"
        )
    
    async def _exchange_for_credits(
        self,
        amount: float,
        context
    ) -> str:
        """
        Exchange cryptocurrency for robutler token.
        
        Returns:
            Token string
        """
        # Check exchange rates
        rates = await self.client.facilitator.exchange_rates()
        
        # Find suitable exchange rate (prefer USDC on Base)
        rate_key = "exact:base-mainnet:USDC"
        rate_info = rates.get('exchangeRates', {}).get(rate_key)
        
        if not rate_info:
            raise X402ExchangeFailed(
                f"Exchange not supported for {rate_key}"
            )
        
        # Calculate output amount after fees
        fee = float(rate_info.get('fee', 0.02))
        rate = float(rate_info.get('rate', 1.0))
        output_amount = (amount * rate) * (1 - fee)
        
        # Create blockchain payment
        payment_header = await self._create_blockchain_payment(
            amount, "exact", "base-mainnet"
        )
        
        # Call exchange endpoint
        result = await self.client.facilitator.exchange(
            payment_header=payment_header,
            payment_requirements={
                'scheme': "exact",
                'network': "base-mainnet",
                'maxAmountRequired': str(amount),
                'payTo': 'exchange',
                'resource': '/exchange'
            },
            requested_output={
                'scheme': 'token',
                'network': 'robutler',
                'amount': str(output_amount)
            }
        )
        
        if not result.get('success'):
            raise X402ExchangeFailed(
                result.get('error', 'Exchange failed')
            )
        
        return result['token']
    
    async def _create_blockchain_payment(
        self,
        amount: float,
        scheme: str,
        network: str
    ) -> str:
        """
        Create blockchain payment using wallet.
        
        Returns:
            Base64 encoded x402 payment header
            
        Note: This is a placeholder. Full blockchain payment implementation
        requires web3.py for EVM chains, solana-py for Solana, etc.
        """
        if not self.wallet_private_key:
            raise X402ExchangeFailed("Wallet not configured for blockchain payments")
        
        # TODO: Implement blockchain payment creation
        # - Sign transaction with wallet
        # - Include proof in payment header
        # - Return encoded header
        
        raise NotImplementedError(
            "Blockchain payment creation not yet implemented. "
            "This will require web3.py or solana-py depending on the network."
        )

