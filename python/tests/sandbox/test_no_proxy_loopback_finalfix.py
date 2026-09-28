"""
A `network:` host on loopback or a private range is reachable from a confined
command (2026-09-27, the final e2e re-run, `g1b-netdebug`). srt exports
`NO_PROXY=localhost,127.0.0.1,...` inside the sandbox, ordinary clients
therefore bypassed its proxy, and their direct socket was refused, so a
listed loopback host answered nothing. The command now starts with a NO_PROXY
from which every entry covering a `network:` host is removed, pinned by
`no_proxy` in the shared fixture `tests/fixtures/sandbox/srt.json` (the
TypeScript suite reads the same cases), and proved against real srt: a plain
`curl`, no `--noproxy`, to a listed `127.0.0.1:<port>` and to a listed
`localhost:<port>` answers, while an unlisted loopback name still does not.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
from pathlib import Path

import pytest

from webagents.sandbox import backend_status, policy_from_metadata, run_sandboxed, sandbox_available
from webagents.sandbox import srt as engine

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "sandbox" / "srt.json").read_text())["no_proxy"]

requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")


@pytest.fixture
def site():
    """A local HTTP server on 127.0.0.1, the one host a test can allow-list."""

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"HELLO-FROM-SITE\n")

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


class TestTheFixtureIsTheContract:
    def test_srts_list_is_pinned(self):
        assert list(engine.SRT_NO_PROXY) == FIXTURE["srt_entries"]

    @pytest.mark.parametrize("case", FIXTURE["cases"], ids=[",".join(c["network"]) or "empty" for c in FIXTURE["cases"]])
    def test_each_case(self, case):
        kept = case["kept"]
        assert engine.no_proxy_for(case["network"]) == (FIXTURE["srt_entries"] if kept is None else kept)
        wrapped = engine.wrapped_command("true", "/usr/bin", case["network"])
        if kept is None:
            assert wrapped == "export PATH=/usr/bin\nexport NODE_USE_ENV_PROXY=1\nexport npm_config_cache=\"${TMPDIR:-/tmp}/npm-cache\"\ntrue"
        elif kept:
            joined = ",".join(kept)
            assert wrapped == f"export PATH=/usr/bin\nexport NODE_USE_ENV_PROXY=1\nexport npm_config_cache=\"${{TMPDIR:-/tmp}}/npm-cache\"\nexport NO_PROXY={joined} no_proxy={joined}\ntrue"
        else:
            assert wrapped == "export PATH=/usr/bin\nexport NODE_USE_ENV_PROXY=1\nexport npm_config_cache=\"${TMPDIR:-/tmp}/npm-cache\"\nunset NO_PROXY no_proxy\ntrue"

    def test_the_wrapped_command_is_spelled_as_the_fixture_says(self):
        w = FIXTURE["wrapped"]
        assert engine.wrapped_command(w["command"], w["path"], w["network"]) == w["text"]
        assert engine.wrapped_command(w["command"], w["path"]) == w["unchanged_text"]
        assert engine.wrapped_command(w["command"], w["path"], ["github.com"]) == w["unchanged_text"]
        assert engine.wrapped_command(w["command"], w["path"], FIXTURE["cases"][-1]["network"]) == w["all_removed_text"]

    def test_the_host_of_an_entry(self):
        assert engine.network_host("*.Example.com") == "example.com"
        assert engine.network_host("127.0.0.1:8080") == "127.0.0.1"
        assert engine.network_host("[::1]:9000") == "::1"
        assert engine.network_host("[::1]") == "::1"
        assert engine.network_host("localhost.") == "localhost"

    def test_covers_by_suffix_equality_and_membership(self):
        covers = engine.no_proxy_entry_covers
        assert covers("localhost", "api.localhost") is True
        assert covers("localhost", "notlocalhost.example") is False
        assert covers("localhost", "127.0.0.1") is False
        assert covers("127.0.0.1", "localhost") is False
        assert covers("10.0.0.0/8", "ten.example") is False
        assert covers("10.0.0.0/8", "10.255.255.255") is True
        assert covers("10.0.0.0/8", "11.0.0.1") is False
        assert covers("::1", "[::1]:80") is True


@requires_backend
class TestAgainstRealSrt:
    def test_a_listed_address_answers_a_plain_client(self, tmp_path, site):
        work = tmp_path / "work"
        work.mkdir()
        allowed = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)], "network": [f"127.0.0.1:{site}"]}, cwd=str(work))
        inside = run_sandboxed('echo "np=[$NO_PROXY] lc=[$no_proxy]"', allowed, timeout=30)
        kept = ",".join(engine.no_proxy_for([f"127.0.0.1:{site}"]))
        assert inside.stdout.strip() == f"np=[{kept}] lc=[{kept}]", inside.stderr
        served = run_sandboxed(f"curl -sf -m 4 http://127.0.0.1:{site}/", allowed, timeout=30)
        assert served.stdout.strip() == "HELLO-FROM-SITE", served.stderr

    def test_a_listed_name_answers_and_an_unlisted_loopback_name_does_not(self, tmp_path, site):
        work = tmp_path / "work"
        work.mkdir()
        by_name = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)], "network": [f"localhost:{site}"]}, cwd=str(work))
        served = run_sandboxed(f"curl -sf -m 4 http://localhost:{site}/", by_name, timeout=30)
        assert served.stdout.strip() == "HELLO-FROM-SITE", served.stderr
        # Only the address is listed: the name stays in NO_PROXY, goes direct, and is refused.
        by_address = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)], "network": [f"127.0.0.1:{site}"]}, cwd=str(work))
        refused = run_sandboxed(f"curl -sf -m 4 http://localhost:{site}/ && echo NET", by_address, timeout=30)
        assert "NET" not in refused.stdout

    def test_nothing_listed_stays_unreachable(self, tmp_path, site):
        work = tmp_path / "work"
        work.mkdir()
        denied = policy_from_metadata({"preset": "development", "allowed_folders": [str(work)]}, cwd=str(work))
        direct = run_sandboxed(f"curl -sf -m 4 http://127.0.0.1:{site}/ && echo NET", denied, timeout=30)
        assert "NET" not in direct.stdout
        via_proxy = run_sandboxed(f"curl -sf -m 4 --noproxy '' http://127.0.0.1:{site}/ && echo NET", denied, timeout=30)
        assert "NET" not in via_proxy.stdout
