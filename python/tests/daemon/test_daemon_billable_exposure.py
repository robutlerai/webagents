"""THE FIFTH DOOR: ``webagentsd``.

The floor exercise found four doors to the billable endpoint across three
rounds, and the fix for the fourth was to stop writing per-route checks and put
one chokepoint in front of routing. Walking what else in this repo constructs an
HTTP app turned up a fifth: ``WebAgentsDaemon`` builds its OWN ``FastAPI``
instance (``webagents/cli/daemon/server.py``), serves
``POST {url_prefix}/{name}/chat/completions`` straight into ``agent.run`` /
``agent.run_streaming``, and used to bind ``0.0.0.0`` — every interface — with
no credential check of any kind. Not a served-agent route, so no served-agent
floor; the same open billable endpoint, on the LAN, from a tool people leave
running while they work.

It cannot simply get the floor: ``DaemonClient`` (``webagents chat``) sends no
credential and there is none to send locally, so a blanket floor would break the
CLI the daemon exists for. So the two cases are separated, and this file pins
both halves:

* loopback — the caller is already on the machine; no floor, CLI unchanged;
* anything else — you are publishing a billable model endpoint, so the SAME
  ``install_credential_floor`` a served agent gets goes on.

The default moved to loopback. That is the part that actually closes the door.
"""

import pytest
from fastapi.testclient import TestClient

from webagents.cli.daemon.server import WebAgentsDaemon


ANON_BODY = {"messages": [{"role": "user", "content": "Hello"}], "stream": False}
COMPLETIONS = "/agents/some-agent/chat/completions"


class TestTheDaemonDoesNotPublishAnAnonymousModelEndpoint:
    def test_it_binds_loopback_by_default(self):
        """The door itself. Before this, the default was ``0.0.0.0``."""
        daemon = WebAgentsDaemon(port=0)
        assert daemon.host == "127.0.0.1"
        assert daemon.is_loopback is True

    def test_the_bind_host_is_what_uvicorn_is_actually_given(self):
        """Guard the guard: a `host` attribute nobody passes to uvicorn would
        make the assertion above decorative."""
        import inspect

        source = inspect.getsource(WebAgentsDaemon.start)
        assert "host=self.host" in source
        assert '"0.0.0.0"' not in source

    def test_an_exposed_daemon_refuses_an_anonymous_completions_request(self):
        """Bound anywhere but loopback, the daemon is a published billable
        endpoint and gets the same floor a served agent gets — exactly 401, and
        without ever loading the agent."""
        daemon = WebAgentsDaemon(port=0, host="0.0.0.0")
        assert daemon.is_loopback is False
        client = TestClient(daemon.app, raise_server_exceptions=False)
        response = client.post(COMPLETIONS, json=ANON_BODY)
        assert response.status_code == 401, response.text
        assert "Authentication required" in response.text

    def test_an_exposed_daemon_lets_a_credentialed_request_through_the_floor(self):
        """The 401 above is the floor, not a missing route: with a credential
        the request gets past it and fails for its own reasons (no such agent),
        which is a 404 rather than a refusal."""
        daemon = WebAgentsDaemon(port=0, host="0.0.0.0")
        client = TestClient(daemon.app, raise_server_exceptions=False)
        response = client.post(
            COMPLETIONS, json=ANON_BODY, headers={"Authorization": "Bearer local-token"}
        )
        assert response.status_code != 401, response.text

    def test_a_loopback_daemon_does_not_gate_the_local_cli(self):
        """The other half. `DaemonClient` sends no credential, so a floor on a
        loopback daemon would break `webagents chat` — the reason the floor is
        conditional rather than unconditional."""
        daemon = WebAgentsDaemon(port=0)
        client = TestClient(daemon.app, raise_server_exceptions=False)
        response = client.post(COMPLETIONS, json=ANON_BODY)
        assert response.status_code != 401, response.text

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
    def test_loopback_is_recognised_in_every_spelling(self, host):
        assert WebAgentsDaemon(port=0, host=host).is_loopback is True

    def test_the_env_override_still_works_and_still_carries_the_floor(self, monkeypatch):
        """Someone who really wants the daemon on the LAN can still have it —
        they just do not get it anonymously."""
        monkeypatch.setenv("WEBAGENTS_DAEMON_HOST", "0.0.0.0")
        daemon = WebAgentsDaemon(port=0)
        assert daemon.host == "0.0.0.0"
        client = TestClient(daemon.app, raise_server_exceptions=False)
        assert client.post(COMPLETIONS, json=ANON_BODY).status_code == 401
