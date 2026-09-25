"""
Daemon Client

Client for communicating with webagentsd over HTTP.
"""

import httpx
from typing import Optional, Dict, Any, List
from pathlib import Path


class DaemonClient:
    """Client for communicating with webagentsd
    
    The client uses a configurable agents_prefix to construct URLs.
    By default, agent routes are at /agents/{name}/...
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        agents_prefix: str = "/agents",
        working_dir: Optional[str] = None,
    ):
        """Initialize daemon client.

        Args:
            base_url: Base URL of the daemon server (e.g., "http://127.0.0.1:8765").
                Omitted, it is the configured address: `daemon.host`/`daemon.port`
                from the config store, read for `working_dir`, else 127.0.0.1:8765.
                This defaulted to a hard-coded `http://localhost:8765`, so every
                bare `DaemonClient()` (the chat's among them) ignored the config
                (2026-09-24, `cli/daemon_address.py`).
            agents_prefix: URL prefix for agent routes (default: "/agents")
            working_dir: Working directory to use for agent operations (sent via header)

        Raises:
            DaemonAddressError: when `base_url` is omitted and the configured
                address cannot be used.
        """
        if base_url is None:
            from ..daemon_address import resolve_daemon_address

            base_url = resolve_daemon_address(cwd=Path(working_dir) if working_dir else None).base_url
        self.base_url = base_url.rstrip("/")
        self.agents_prefix = agents_prefix.rstrip("/") if agents_prefix else ""
        self.working_dir = working_dir or str(Path.cwd())
        
        # A CREDENTIAL ON EVERY REQUEST (2026-09-24). The daemon's model routes
        # sit behind the credential floor (`server/core/credential_floor.py`),
        # which refuses a request that carries none, and this client sent only
        # `X-Working-Dir`. So since the floor reached the daemon, every chat from
        # `webagents connect` and the TUI answered 401, for everyone, logged in
        # or not. Found walking the CLI quickstart as a first-time developer;
        # the e2e run had only exercised `run -p`, which never uses the daemon.
        #
        # A fixed local marker rather than the platform token: the floor checks
        # PRESENCE (by design, and documented), the daemon is this user's own
        # loopback process, and nothing it does needs the user's bearer, so
        # there is no reason to hand it over. When the daemon learns to require
        # a real per-daemon key (the fix S-224 calls for), this is where it goes.
        headers = {
            "X-Working-Dir": self.working_dir,
            "Authorization": "Bearer webagents-cli-local",
        }
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)
    
    def _agents_url(self, path: str = "") -> str:
        """Build URL for agent endpoints.
        
        Args:
            path: Path after the agents prefix (e.g., "/{name}/command")
            
        Returns:
            Full URL like "http://localhost:8765/agents/{name}/command"
        """
        if path and not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{self.agents_prefix}{path}"

    @staticmethod
    def _chat_payload(
        messages: List[Dict], *, stream: bool, model: Optional[str]
    ) -> Dict[str, Any]:
        """Build a chat/completions body, omitting `model` when unset.

        OMITTING IS THE POINT. Both call sites used to hardcode
        `"model": "gpt-4o-mini"` (2026-09-23), and because the daemon path is
        the only live one for the REPL and the TUI, that silently overrode the
        `model:` every AGENT.md declares. An agent configured for one provider
        answered from another, with nothing in the UI saying so.

        The daemon already resolves the agent's own model when the field is
        absent, so absent is the correct default and an explicit value is only
        ever a deliberate override (a `/model` slash command, a `--model` flag).
        """
        payload: Dict[str, Any] = {"messages": messages, "stream": stream}
        if model:
            payload["model"] = model
        return payload
    
    async def is_running(self) -> bool:
        """Check if daemon is running"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    async def health(self) -> Dict[str, Any]:
        """Get daemon health status
        
        Returns:
            Health status dict with 'status' key
            
        Raises:
            httpx.HTTPError if daemon is not reachable
        """
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def list_agents(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """List or search registered agents
        
        Args:
            query: Optional search query (name pattern)
        
        Returns:
            List of agent metadata
        """
        params = {"query": query} if query else {}
        response = await self.client.get(self._agents_url("/"), params=params)
        response.raise_for_status()
        return response.json()["agents"]
    
    async def get_agent(self, name: str) -> Dict[str, Any]:
        """Get agent details
        
        Args:
            name: Agent name
            
        Returns:
            Agent metadata dict
        """
        response = await self.client.get(self._agents_url(f"/{name}"))
        response.raise_for_status()
        return response.json()
    
    async def register_agent(self, path: Path) -> Dict[str, Any]:
        """Register an agent from file
        
        Args:
            path: Path to agent file (AGENT.md)
            
        Returns:
            Registered agent metadata
        """
        response = await self.client.post(
            self._agents_url("/"),
            json={"path": str(path)}
        )
        response.raise_for_status()
        return response.json()
    
    async def unregister_agent(self, name: str) -> Dict[str, Any]:
        """Unregister an agent
        
        Args:
            name: Agent name
            
        Returns:
            Unregistration result
        """
        response = await self.client.delete(self._agents_url(f"/{name}"))
        response.raise_for_status()
        return response.json()
    
    async def chat(
        self, agent_name: str, message: str, history: List[Dict], model: Optional[str] = None
    ) -> Dict:
        """Send chat message to agent
        
        Args:
            agent_name: Name of the agent
            message: User message
            history: Conversation history
            model: Model override. When None the DAEMON decides, which means
                the model the agent declared in its AGENT.md is used.
            
        Returns:
            Chat completion response
        """
        messages = history + [{"role": "user", "content": message}]
        
        response = await self.client.post(
            self._agents_url(f"/{agent_name}/chat/completions"),
            json=self._chat_payload(messages, stream=False, model=model)
        )
        response.raise_for_status()
        return response.json()

    async def chat_stream(
        self, agent_name: str, message: str, history: List[Dict], model: Optional[str] = None
    ):
        """Stream chat responses from agent
        
        Args:
            agent_name: Name of the agent
            message: User message
            history: Conversation history
            model: Model override. When None the DAEMON decides, which means
                the model the agent declared in its AGENT.md is used.
            
        Yields:
            SSE data chunks (without "data: " prefix)
        """
        import logging
        logger = logging.getLogger(__name__)
        import time
        start_time = time.time()
        logger.debug(f"STREAM START: {start_time}")
        
        messages = history + [{"role": "user", "content": message}]
        
        async with self.client.stream(
            "POST",
            self._agents_url(f"/{agent_name}/chat/completions"),
            json=self._chat_payload(messages, stream=True, model=model)
        ) as response:
            if response.is_error:
                # Read the body before raising: it says why (the daemon answers
                # a run that failed before its first chunk, a model it could not
                # reach, with a 500 whose body is the reason), and a streamed
                # response has not read it yet. See `daemon_error_detail`.
                await response.aread()
            response.raise_for_status()
            logger.debug(f"STREAM HEADERS: {time.time() - start_time:.3f}s")
            
            async for line in response.aiter_lines():
                if line.strip():
                    logger.debug(f"STREAM CHUNK: {time.time() - start_time:.3f}s | {line[:50]}...")
                
                if line.startswith("data: "):
                    yield line[6:]  # Strip "data: " prefix
    
    async def list_commands(self, agent_name: str) -> List[Dict[str, Any]]:
        """List available commands for an agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of command info dicts
        """
        import logging
        logger = logging.getLogger(__name__)
        
        url = self._agents_url(f"/{agent_name}/command")
        logger.debug(f"Fetching commands from: {url}")
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        commands = data.get("commands", [])
        logger.debug(f"Received {len(commands)} commands for '{agent_name}'")
        return commands
    
    async def execute_command(self, agent_name: str, path: str, data: Dict[str, Any] = None) -> Any:
        """Execute a command on an agent
        
        Args:
            agent_name: Name of the agent
            path: Command path (e.g., "/checkpoint/create")
            data: Command arguments
            
        Returns:
            Command result
        """
        url_path = path.lstrip("/")
        response = await self.client.post(
            self._agents_url(f"/{agent_name}/command/{url_path}"),
            json=data or {}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_command_docs(self, agent_name: str, path: str) -> Dict[str, Any]:
        """Get documentation for a specific command
        
        Args:
            agent_name: Name of the agent
            path: Command path (e.g., "/checkpoint/create")
            
        Returns:
            Command documentation dict
        """
        url_path = path.lstrip("/")
        response = await self.client.get(
            self._agents_url(f"/{agent_name}/command/{url_path}")
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()


def daemon_error_detail(error: BaseException) -> str:
    """What to tell the person when a request to the daemon failed.

    For an error ANSWER, what the daemon said (`detail`, `error` or the body),
    not httpx's "Server error '500 Internal Server Error' for url ...", which
    is what the chat printed for a model it could not reach (2026-09-24). For
    no answer at all, the daemon's address and the reason.
    """
    import httpx

    from webagents.utils.errors import describe_exception

    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        try:
            text = response.text.strip()
        except httpx.ResponseNotRead:
            text = ""
        detail = text
        try:
            body = response.json()
            if isinstance(body, dict):
                found = body.get("detail") or body.get("error") or body.get("message")
                if isinstance(found, dict):
                    found = found.get("message") or found
                if found:
                    detail = str(found)
        except ValueError:
            pass
        return detail or f"The daemon answered {response.status_code}."
    return describe_exception(error)

