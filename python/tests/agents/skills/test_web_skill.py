"""
S-245 (2026-09-25): the `web` skill fetches only public addresses, checks every
redirect hop, and connects to the address it checked. It used to fetch any URL
the model named, following redirects, so a caller could make the agent's host
read its own loopback, private network or cloud metadata and hand back the
answer. The TypeScript twin's cases are `tests/unit/skills/web.test.ts`.
"""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webagents.agents.skills.local.web.skill import WebSkill, page_text


class _Site:
    """A local server: GET /page is a small HTML page, /hop redirects to `redirect_to`."""

    def __init__(self, redirect_to=None):
        self.hits = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                site.hits.append(self.path)
                if self.path == "/hop" and site.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", site.redirect_to)
                    self.end_headers()
                    return
                body = (
                    b"<html><head><style>p{}</style><script>var secret = 1;</script></head>"
                    b"<body><h1>Hello</h1><p>from the page</p></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.redirect_to = redirect_to
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def site():
    s = _Site()
    yield s
    s.close()


def fetch(skill, prompt):
    return asyncio.run(skill.web_fetch(prompt))


def test_a_loopback_address_is_refused_without_a_connection(site):
    out = fetch(WebSkill(), f"Read http://127.0.0.1:{site.port}/page")
    assert "127.0.0.1 is not a public address, so it is not called." in out
    assert site.hits == []


def test_a_name_that_resolves_to_loopback_is_refused(site):
    out = fetch(WebSkill(), f"Read http://localhost:{site.port}/page")
    assert "localhost resolves to" in out and "which is not a public address" in out
    assert site.hits == []


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.1/",
    "http://[::1]/",
])
def test_private_and_metadata_addresses_are_refused(url):
    out = fetch(WebSkill(), f"Read {url}")
    assert "is not a public address, so it is not called." in out


def test_an_allowed_private_address_is_fetched_as_text(site):
    # The loader's call shape: the entry's config is the first argument.
    skill = WebSkill({"allow_private": [f"127.0.0.1:{site.port}"]})
    out = fetch(skill, f"Read http://127.0.0.1:{site.port}/page")
    assert "Hello from the page" in out
    assert "secret" not in out
    assert site.hits == ["/page"]


def test_each_redirect_hop_is_checked():
    target = _Site()
    try:
        first = _Site(redirect_to=f"http://127.0.0.1:{target.port}/page")
        try:
            # Only the first server is allowed; its redirect points elsewhere.
            skill = WebSkill({"allow_private": [f"127.0.0.1:{first.port}"]})
            out = fetch(skill, f"Read http://127.0.0.1:{first.port}/hop")
            assert "is not a public address, so it is not called." in out
            assert first.hits == ["/hop"]
            assert target.hits == []
        finally:
            first.close()
    finally:
        target.close()


def test_a_malformed_allow_list_stops_the_load():
    with pytest.raises(ValueError, match="allow_private"):
        WebSkill({"allow_private": ["not-an-address"]})


def test_page_text_drops_scripts_styles_and_tags():
    body = b"<p>a</p><script>b</script><style>c</style><div>d</div>"
    assert page_text(body, "text/html") == "a d"
