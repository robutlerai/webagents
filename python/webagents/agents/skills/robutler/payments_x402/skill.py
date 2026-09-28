"""
PaymentSkillX402: the payment skill that also SELLS over standard x402 and
MPP on the agent's priced `@http` endpoints (webagents gap-closure plan 2.6,
2026-09-26).

WHAT CHANGED. This skill used to answer a priced endpoint itself, from a
`before_http_call` hook no server ever fired, in a private 402 shape
(`scheme: 'token'` on `network: 'robutler'`, `payTo:` the agent), and to
verify and settle through `self.client.facilitator`, an attribute the
platform client never had, so every paid request raised. All of that is
gone. The server (`server/core/app.py`) now asks this skill's `paywall`
(`..payments.paywall`) for any handler carrying `@pricing`, and the paywall
answers a standard x402 402 (v2 header, v1 body) with the credits scheme
and, when configured, a chain scheme and MPP challenges beside it, verifies
a payment before the handler runs and settles once after it answered.

WHAT STAYED. Everything `PaymentSkill` does for a chat run (verify, lock,
settle at finalize), and `_verify_payment_token`: local verification of a
Robutler payment token against the PLATFORM's key set (S-135: only a token
whose unverified `iss` is the configured platform issuer is checked locally,
so a priced endpoint's header can never make this host fetch a key set of
the caller's choosing). The paywall's credits scheme tries it first and
falls back to `POST /api/payments/verify`.

CONFIGURATION, under `config["x402"]` (the environment is the fallback):

    credits:      take Robutler credits (default True)
    nonce_secret: the HMAC secret behind the credits nonces (X402_NONCE_SECRET)
    chain:        {pay_to, network, asset, decimals, units_per_credit?, schemes?,
                   max_timeout_seconds?, extra?, facilitator: client | {url, headers, cdp}}
                  X402_PAY_TO switches it on from the environment, with
                  X402_NETWORK, X402_ASSET, X402_ASSET_DECIMALS, X402_ASSET_NAME,
                  X402_ASSET_VERSION, X402_FACILITATOR_URL, CDP_API_KEY_ID/SECRET
    resource:     {service_name, description, mime_type, tags, icon_url}
    mpp:          {realm, secret, stripe: {profile_id, client, ...}, credits}

WHO RECEIVES A CHAIN PAYMENT is `chain["pay_to"]`: on a self-hosted agent the
developer's own address, in the developer's own software. The platform's
dispatcher never reads this and keeps its own chain path off until counsel
clears it. Credits are named as credits throughout.
"""

import os
import logging
from typing import Dict, Any, Optional

import jwt
from webagents.agents.skills.robutler.payments.skill import PaymentSkill


class PaymentSkillX402(PaymentSkill):
    """`PaymentSkill` plus the paywall for priced endpoints (file comment)."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        config = config or {}
        # Optional JWKS manager for local JWT verification (payment tokens)
        self._jwks_manager = config.get("jwks_manager")
        if self._jwks_manager is None:
            try:
                from webagents.crypto.jwks import JWKSManager
                self._jwks_manager = JWKSManager(config={"jwks_cache_ttl": 3600})
            except ImportError:
                self._jwks_manager = None
        self.x402_config: Dict[str, Any] = dict(config.get("x402") or {})
        self._paywall = None
        self.logger = logging.getLogger(__name__)

    # =========================================================================
    # The paywall the server asks for a priced endpoint
    # =========================================================================

    @property
    def paywall(self):
        """The paywall (`..payments.paywall.Paywall`), built once on first use so
        `initialize()` has had its chance to find the agent's key. None only
        when this skill has nothing to sell with: credits switched off and no
        chain seller named."""
        if self._paywall is not None:
            return self._paywall
        from ..payments.mpp_seller import MppSeller
        from ..payments.paywall import Paywall
        from ..payments.x402_credits import CreditsScheme, PlatformCreditsClient

        credits = None
        if self.x402_config.get("credits", True):
            client = PlatformCreditsClient(
                platform_url=self.webagents_api_url,
                api_key=self.robutler_api_key,
                verify_locally=self._verify_payment_token if self._jwks_manager else None,
            )
            credits = CreditsScheme(
                client,
                nonce_secret=self.x402_config.get("nonce_secret") or os.getenv("X402_NONCE_SECRET"),
                platform_url=self.webagents_api_url,
            )
        chain = self._chain_seller_config()
        if credits is None and chain is None:
            return None
        mpp = None
        mpp_config = self.x402_config.get("mpp")
        if mpp_config:
            mpp = MppSeller(
                realm=mpp_config["realm"],
                secret=mpp_config["secret"],
                stripe=mpp_config.get("stripe"),
                credits=mpp_config.get("credits", True),
                challenge_ttl_seconds=mpp_config.get("challenge_ttl_seconds"),
            )
        self._paywall = Paywall(credits=credits, chain=chain, mpp=mpp, resource=dict(self.x402_config.get("resource") or {}))
        return self._paywall

    def _chain_seller_config(self) -> Optional[Dict[str, Any]]:
        """The chain seller from config, else from the environment (`X402_PAY_TO` switches it on)."""
        chain = dict(self.x402_config.get("chain") or {})
        pay_to = chain.get("pay_to") or os.getenv("X402_PAY_TO")
        if not pay_to:
            return None
        from ..payments.x402_facilitator import facilitator_from_config

        facilitator = chain.get("facilitator")
        if facilitator is None or isinstance(facilitator, dict):
            facilitator = facilitator_from_config(facilitator)
        try:
            decimals = int(chain.get("decimals", os.getenv("X402_ASSET_DECIMALS", "6")))
        except (TypeError, ValueError):
            decimals = 6
        return {
            "pay_to": pay_to,
            "network": chain.get("network") or os.getenv("X402_NETWORK") or "eip155:84532",
            "asset": chain.get("asset") or os.getenv("X402_ASSET") or "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
            "decimals": decimals,
            "units_per_credit": chain.get("units_per_credit", 1),
            "schemes": chain.get("schemes"),
            "max_timeout_seconds": chain.get("max_timeout_seconds"),
            "extra": chain.get("extra") or {"name": os.getenv("X402_ASSET_NAME", "USDC"), "version": os.getenv("X402_ASSET_VERSION", "2")},
            "facilitator": facilitator,
        }

    # =========================================================================
    # Local verification of a platform payment token
    # =========================================================================

    def _platform_issuer(self) -> Optional[str]:
        """The `iss` a platform-minted payment token must carry: `platform_issuer`
        in config, else ROBUTLER_PLATFORM_ISSUER, else ROBUTLER_API_URL (the
        public base URL the portal stamps), else the payments API base URL.
        The same ladder the auth skill uses for its `platform_issuer`."""
        value = (
            self.config.get("platform_issuer")
            or os.getenv("ROBUTLER_PLATFORM_ISSUER")
            or os.getenv("ROBUTLER_API_URL")
            or self.webagents_api_url
            or ""
        )
        return str(value).strip().rstrip("/") or None

    def _platform_jwks_url(self) -> str:
        """The platform's key set at the URL this agent can actually reach
        (`webagents_api_url` is ROBUTLER_INTERNAL_API_URL in-cluster, so plain
        http to a service name is the normal shape). `platform_jwks_url` in
        config, or OWNER_ASSERTION_JWKS_URL, overrides it exactly as it does for
        the auth skill's `_platform_jwks_url`."""
        return (
            self.config.get("platform_jwks_url")
            or os.getenv("OWNER_ASSERTION_JWKS_URL")
            or f"{str(self.webagents_api_url).rstrip('/')}/.well-known/jwks.json"
        )

    async def _verify_payment_token(self, token: str, expected_audience: Any = None) -> Optional[Dict[str, Any]]:
        """
        Verify payment token locally via JWKS when possible (JWT).
        Returns dict with isValid and balance, or None to fall back to API.
        When expected_audience is provided, JWT aud claim must be present and match.

        S-135 twin (2026-09-17): the key set is the PLATFORM's, at the configured
        platform URL, and only a token whose unverified `iss` equals the
        configured platform issuer gets that far. This method used to build
        `${iss}/.well-known/jwks.json` from the token and fetch it before any
        claim was checked, which let anyone with a priced endpoint's X-PAYMENT
        header make this host GET an address of their choosing. An unexpected
        issuer is None with no request, and the platform's verify API (the
        paywall's credits scheme) remains the fallback for it.
        """
        if not self._jwks_manager:
            return None
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
            issuer = str(unverified.get("iss") or "").strip().rstrip("/")
            if not issuer:
                return None
            kid = jwt.get_unverified_header(token).get("kid")
            if not kid:
                return None
            expected_issuer = self._platform_issuer()
            if not expected_issuer or issuer != expected_issuer:
                return None
            jwks_uri = self._platform_jwks_url()
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
