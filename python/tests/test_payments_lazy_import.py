"""
Running an agent does not import the platform stack (2026-09-24).

The transports import `payments.exceptions`, which is only exception classes,
but that ran `payments/__init__.py`, which imported `.skill`, which imports
`robutler.api`, and `robutler` imports `litellm`, whose module import calls
`load_dotenv()`. So a plain `webagents run` loaded an ancestor `.env` into the
agent's environment (the S-216 chain, reopened; see the portal's
SECURITY_ISSUES_LOG.md), cost about 1.2 s, and installed robutler's global
logger class. Checked in a subprocess, because this test process may already
have imported robutler for other tests.
"""

import json
import subprocess
import sys

PROBE = r"""
import json, sys
import webagents.agents.skills.core.transport.a2a.skill
import webagents.agents.skills.core.transport.uamp.skill
from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError
print(json.dumps({"robutler": "robutler" in sys.modules, "litellm": "litellm" in sys.modules}))
"""


def test_the_transports_do_not_import_robutler_or_litellm():
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120, check=True
    )
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded == {"robutler": False, "litellm": False}


def test_every_existing_spelling_still_resolves():
    from webagents.agents.skills.robutler import payments
    from webagents.agents.skills.robutler.payments import PaymentSkill, PricingInfo, pricing
    from webagents.agents.skills.robutler.payments.skill import PaymentSkill as FromSkill

    assert PaymentSkill is FromSkill
    assert callable(pricing) and PricingInfo is not None
    assert {"PaymentSkill", "pricing"} <= set(dir(payments))
