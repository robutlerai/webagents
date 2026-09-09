"""
Secrets Skill

Named credentials in the operating system's own keystore (macOS Keychain,
Linux Secret Service, Windows Credential Manager), with a fallback that
refuses to be quiet about being a fallback. The backends live in ``store.py``;
this file is the agent-facing wrapper.

THE ONE DESIGN DECISION WORTH ARGUING ABOUT: ``secrets_get`` does NOT return
the secret. It returns whether one exists.

The reason is that the primary consumer of a stored credential is CODE, not a
model. ``store.get(name)`` in an HTTP call is the normal path; the model needs
to know a credential is present so it can stop asking for one, and almost
never needs the bytes. Handing a bearer to the model instead puts it in the
transcript, in whatever the transcript is persisted to, and in the inference
provider's records, which are three copies of a credential in places nobody
will think to rotate. ``reveal=True`` exists for the genuine exception and is
refused unless the developer set ``allow_reveal`` in the config, so revealing
is a decision made in code rather than one the model can make for itself.

That matters more here than it would for an API key, because the credential
this skill was built for is the platform bearer from ``register_with_platform``:
seven days, ``agents:own``, and no ``jti``, so nothing can revoke it (platform
security log S-037). Rotating the key on the agent card does not touch it.

Every tool is ``owner`` scope. A secret store readable by a counterparty agent
is not a secret store.
"""

from typing import Any, Dict, Optional

from ...base import Skill
from webagents.agents.tools.decorators import tool, prompt

from .store import SecretStore, open_secret_store


class SecretsSkill(Skill):
    """Named credentials in the OS keystore, with an honest plaintext fallback"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config, scope="owner")
        cfg = config or {}
        self._namespace = cfg.get("namespace")
        self._require_keystore = cfg.get("require_keystore")
        self._secrets_dir = cfg.get("secrets_dir")
        self._backend = cfg.get("backend")
        self._quiet = bool(cfg.get("quiet", False))
        # Off by default. See the module docstring: a revealed secret is a
        # secret copied into a transcript.
        self._allow_reveal = bool(cfg.get("allow_reveal", False))
        self._store: Optional[SecretStore] = None

    async def initialize(self, agent) -> None:
        """Open the store eagerly so the fallback warning is logged at boot
        rather than at the first tool call, which may be hours later or never.

        The namespace defaults to the agent's name, which is what keeps two
        agents on one machine from reading each other's credentials out of a
        machine-wide keystore.
        """
        await super().initialize(agent)
        if self._namespace is None:
            self._namespace = getattr(agent, "name", None)
        self.get_store()

    def get_store(self) -> SecretStore:
        """The store this skill wraps, for code that wants the credential
        rather than a tool call. This is the handle to pass to
        ``register_with_platform``."""
        if self._store is None:
            self._store = open_secret_store(
                namespace=self._namespace,
                require_keystore=self._require_keystore,
                secrets_dir=self._secrets_dir,
                quiet=self._quiet,
                backend=self._backend,
            )
        return self._store

    def _envelope(self) -> Dict[str, Any]:
        """Backend fields carried on every result, so it is never implicit.

        The ``warning`` reaches the MODEL, which is the point: process logs
        are easy to miss and a tool result is not.
        """
        status = self.get_store().status()
        if status["keystore"]:
            return {"backend": "keystore", "keystore": True}
        return {
            "backend": "file",
            "keystore": False,
            "warning": status["warning"],
        }

    @prompt(priority=45, scope="owner")
    def secrets_guide(self, context: Any = None) -> str:
        return (
            "## Secrets skill\n"
            "\n"
            "Named credentials in the operating system keystore, or in an owner-only "
            "file when the machine has no keystore.\n"
            "\n"
            "### Rules\n"
            "- `secrets_get` tells you whether a secret EXISTS. It does not give you the "
            "value, and that is deliberate: the code that uses a credential reads it "
            "directly, and a value returned here would be copied into this conversation.\n"
            "- Never paste a credential into a normal reply, a file, or a commit. "
            "`secrets_set` is the only place one belongs.\n"
            "- Read `backend` on every result. When it is `file` the secret is stored as "
            "plaintext on disk and the `warning` field says where. Tell the owner rather "
            "than treating it as normal.\n"
            "- Deleting is cheap and reversible only by the owner re-entering the "
            "credential. Confirm before `secrets_delete`.\n"
            "\n"
            "### Names\n"
            "Use 1 to 128 characters from A-Z a-z 0-9 . _ and -. Names are scoped to this "
            "agent, so `platform_token` here does not collide with another agent's "
            "`platform_token` on the same machine."
        )

    @tool(name="secrets_set", scope="owner")
    async def secrets_set(self, name: str, value: str) -> Dict[str, Any]:
        """Store a named secret in the OS keystore, or an owner-only file if the machine has no keystore.

        Returns which backend it landed in. Prefer having the owner enter the
        value here over keeping it in a file or an environment variable.

        Args:
            name: Secret name, 1-128 characters from A-Z a-z 0-9 . _ and -
            value: The secret value. Never echo this back.

        Returns:
            The secret name, plus the backend it was stored in.
        """
        store = self.get_store()
        store.set(name, value)
        store.note_index(name, True)
        return {"name": name, "stored": True, **self._envelope()}

    @tool(name="secrets_get", scope="owner")
    async def secrets_get(self, name: str, reveal: bool = False) -> Dict[str, Any]:
        """Report whether a named secret exists.

        Does NOT return the value: code reads the credential directly, and
        returning it here would copy it into this conversation.

        Args:
            name: Secret name.
            reveal: Return the value as well. Refused unless the developer
                enabled it when constructing the skill.

        Returns:
            Whether the secret exists, plus the backend holding it.
        """
        store = self.get_store()
        value = store.get(name)
        result: Dict[str, Any] = {
            "name": name,
            "exists": value is not None,
            **self._envelope(),
        }
        if not reveal:
            return result
        if not self._allow_reveal:
            result["note"] = (
                "reveal refused: this skill was constructed without allow_reveal. "
                "The value is readable from code via the skill store, which keeps it "
                "out of this conversation."
            )
            return result
        if value is not None:
            result["value"] = value
        return result

    @tool(name="secrets_list", scope="owner")
    async def secrets_list(self) -> Dict[str, Any]:
        """List the names of stored secrets. Never returns values.

        On a keystore backend the list is what this agent has written, which
        may be short of what the keystore actually holds.

        Returns:
            The known secret names, and whether that list is complete.
        """
        names, complete = self.get_store().list()
        result: Dict[str, Any] = {"names": names, "complete": complete}
        if not complete:
            result["note"] = (
                "the OS keystore cannot be enumerated portably, so this lists only the "
                "names this agent recorded. A secret set by something else is still "
                "readable by name."
            )
        result.update(self._envelope())
        return result

    @tool(name="secrets_delete", scope="owner")
    async def secrets_delete(self, name: str) -> Dict[str, Any]:
        """Remove a named secret from every backend.

        Includes a plaintext copy left behind from before a keystore was
        available.

        Args:
            name: Secret name.

        Returns:
            Whether anything was removed.
        """
        store = self.get_store()
        removed = store.delete(name)
        store.note_index(name, False)
        return {"name": name, "removed": removed, **self._envelope()}

    @tool(name="secrets_status", scope="owner")
    async def secrets_status(self) -> Dict[str, Any]:
        """Report where secrets are being stored and why.

        Use this before telling the owner their credential is safe.

        Returns:
            The active backend, and on the fallback the reason and the path.
        """
        return self.get_store().status()
