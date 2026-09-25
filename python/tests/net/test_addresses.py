"""
Which addresses the REST tool and the key-directory fetch may connect to
(ADR-0045), from the table both SDKs run
(`tests/fixtures/net/addresses.json`; TypeScript tests/unit/net/addresses.test.ts).
"""

import ipaddress
import json
from pathlib import Path

import pytest

from webagents.net.addresses import (
    AllowListError,
    address_allowed,
    ip_text,
    is_always_blocked,
    is_public_address,
    parse_allow_list,
    parse_ip,
    sort_addresses,
)

TABLE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "net" / "addresses.json").read_text())


@pytest.mark.parametrize("case", TABLE["addresses"], ids=lambda c: c["address"])
def test_classification(case):
    addr = ipaddress.ip_address(case["address"])
    assert is_public_address(addr) is case["public"]
    assert is_always_blocked(addr) is case["always_blocked"]


@pytest.mark.parametrize("case", TABLE["allow_lists"], ids=lambda c: f"{c['entries']}->{c['address']}:{c['port']}")
def test_allow_lists(case):
    allow = parse_allow_list(case["entries"])
    assert address_allowed(ipaddress.ip_address(case["address"]), case["port"], allow) is case["allowed"]


@pytest.mark.parametrize("entry", TABLE["bad_allow_entries"])
def test_bad_allow_entries(entry):
    with pytest.raises(AllowListError):
        parse_allow_list([entry])


def test_shorthand_ipv4_is_an_address():
    assert parse_ip("127.1") == ipaddress.ip_address("127.0.0.1")
    assert parse_ip("0x7f.0.0.1") == ipaddress.ip_address("127.0.0.1")
    assert parse_ip("2130706433") == ipaddress.ip_address("127.0.0.1")
    assert parse_ip("example.com") is None


def test_text_is_the_url_spelling():
    assert ip_text(ipaddress.ip_address("::ffff:127.0.0.1")) == "::ffff:7f00:1"
    assert ip_text(ipaddress.ip_address("2001:db8:0:1:1:1:1:1")) == "2001:db8:0:1:1:1:1:1"
    assert ip_text(ipaddress.ip_address("fe80::1")) == "fe80::1"
    assert ip_text(ipaddress.ip_address("::")) == "::"


def test_v4_first_then_by_value():
    ordered = sort_addresses([ipaddress.ip_address(a) for a in ("::1", "127.0.0.2", "127.0.0.1", "::1")])
    assert [str(a) for a in ordered] == ["127.0.0.1", "127.0.0.2", "::1"]
