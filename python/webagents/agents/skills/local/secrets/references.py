"""
``${secret:NAME}`` and ``${env:NAME}`` in an MCP server's ``env``, ``headers``
and ``url`` (S-292, 2026-09-26): the one grammar both SDKs resolve, the
sentences they say when they cannot, what a report shows instead of a value,
and the patterns that make a literal look like a secret.

WHY. Until this, a self-hosted agent's MCP credentials could only be literal
text in ``AGENT.md`` or ``mcp.json``: a plain file that is easy to commit, and
that the agent's own file and shell tools can read into a conversation. The
SDK already had an owner-only store for exactly this kind of value (the OS
keystore behind ``webagents secrets set``, ``store.py``), and the MCP config
could not point at it. Now ``${secret:NAME}`` names an entry in that store and
``${env:NAME}`` names a process variable; both are replaced at connect time,
in memory, and the value they resolve to is never written back, never logged,
and masked wherever a server's configuration is shown.

THE GRAMMAR, the same in ``typescript/src/skills/secrets/references.ts`` and
pinned by ``secret_refs`` in the shared fixture
``tests/fixtures/mcp_tool/config_shapes.json``:

  * ``${secret:NAME}`` reads the keystore ``webagents secrets set NAME``
    writes, for the active profile; ``${env:NAME}`` reads the process
    environment.
  * NAME is an environment-variable-shaped name: letters, digits and
    underscores, not starting with a digit, at most 128 characters, which is
    also what ``webagents secrets set`` accepts, so every reference that
    parses is one the CLI can satisfy.
  * ``$${`` is a literal ``${``. Anything else that opens with ``${`` is
    refused, because a ``${vault:X}`` or a bare ``${NAME}`` passed through as
    text is a typo nobody sees until a server answers 401.
  * References are resolved in ``env`` values, ``headers`` values and the
    address (``url``, ``httpUrl``, ``mcpUrlTemplate``). A reference in
    ``command`` or ``args`` is REFUSED by the loader: a command line is
    readable by every local account through the process list, so a value
    there is not a secret whatever store it came from.

MASKING. A configuration shown in a report (``server_report()``, ``doctor``,
an error) shows a value as written when it is a well-formed reference, since
``${secret:GH}`` gives nothing away, and ``****`` otherwise: a literal in the
file is a secret too, and that is the case the loader warns about.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

#: What a report shows instead of a resolved or literal value.
SECRET_MASK = "****"

#: NAME in ``${secret:NAME}`` and ``${env:NAME}``: env-var shaped, at most 128 characters.
REFERENCE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

#: The sentences, as templates (``{text}`` is the reference as written). The
#: fixture holds the same table; a sentence cannot change in one SDK only.
REFERENCE_SENTENCES: Dict[str, str] = {
    "notClosed": "{text} is not closed: a reference is ${secret:NAME} or ${env:NAME}, and $${ is a literal ${",
    "unknown": "{text} is not a reference this SDK knows: write ${secret:NAME} or ${env:NAME}, or $${ for a literal ${",
    "badName": "{text} does not name a secret or a variable: use letters, digits and underscores, not starting with a digit",
    "secretNotSet": "{text} is not set: store it with `{hint}`",
    "envNotSet": "{text} is not set in the environment",
    "commandLine": "puts a reference in {field}, which other local accounts can read from the process list: pass it through env instead",
    "literal": "has what looks like a secret written into {field} {key}: keep it out of the file with ${secret:{suggested}} and `{hint}`",
    "atConnect": '{where} of MCP server "{server}": {sentence}',
}

#: A literal that looks like a credential: the key prefixes people paste most
#: (OpenAI and Anthropic ``sk-``, GitHub ``ghp_`` and ``github_pat_``, Slack
#: ``xox``, AWS ``AKIA``) and a bearer of twenty or more characters. Strings,
#: so the fixture can hold the same list for TypeScript.
SECRET_LOOKING_PATTERNS: Tuple[str, ...] = ("^sk-", "^ghp_", "^github_pat_", "^xox", "^AKIA", r"^Bearer\s+\S{20,}")


def _fill(template: str, **values: str) -> str:
    """``{hole}`` filled from ``values`` in ONE pass, as the TypeScript twin
    does, so a filled value is never scanned for holes again. A hole with no
    value stays as written; ``${secret:NAME}`` in a template is not a hole
    (a colon is not a word character)."""
    return re.sub(r"\{(\w+)\}", lambda match: values.get(match.group(1), match.group(0)), template)


class SecretReference:
    """One ``${scheme:NAME}`` as written."""

    __slots__ = ("scheme", "name", "text")

    def __init__(self, scheme: str, name: str, text: str) -> None:
        self.scheme = scheme
        self.name = name
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SecretReference({self.text!r})"


class SecretReferenceError(ValueError):
    """A reference that cannot be resolved, or is not one. ``missing_secret``
    names a ``secret:`` that is not stored; ``missing_env`` an ``env:``
    variable that is not set (2026-09-26, so `doctor` can name it in its fix)."""

    def __init__(self, message: str, reference: str, missing_secret: Optional[str] = None, missing_env: Optional[str] = None) -> None:
        super().__init__(message)
        self.reference = reference
        self.missing_secret = missing_secret
        self.missing_env = missing_env


def secrets_set_hint(name: str) -> str:
    """``webagents secrets set NAME``, with ``--profile <name>`` while a profile
    is active (``cli/config_store.py`` ``cli_command``). A hint without the
    profile would store the secret in the DEFAULT profile, and the reference,
    read from the active one, would stay unresolved (the S-219 shape)."""
    from webagents.cli.config_store import cli_command

    return cli_command(f"secrets set {name}")


def tokenize_references(value: str) -> List[Union[str, SecretReference]]:
    """The literal runs and references of ``value``, in order. Raises
    :class:`SecretReferenceError` for a ``${`` that does not close, a scheme
    this SDK does not know, or a name outside the grammar."""
    out: List[Union[str, SecretReference]] = []
    literal = ""
    i = 0
    length = len(value)
    while i < length:
        if value.startswith("$${", i):
            literal += "${"
            i += 3
            continue
        if value.startswith("${", i):
            close = value.find("}", i + 2)
            if close == -1:
                # The sentence names the reference as far as it parses, never
                # what follows it (S-303, 2026-09-26): the tail of the value
                # may be a literal key, and an error is logged.
                shown = unclosed_reference_text(value[i:])
                raise SecretReferenceError(_fill(REFERENCE_SENTENCES["notClosed"], text=shown), shown)
            text = value[i : close + 1]
            body = value[i + 2 : close]
            scheme, colon, name = body.partition(":")
            if not colon:
                scheme, name = "", ""
            if scheme not in ("secret", "env"):
                raise SecretReferenceError(_fill(REFERENCE_SENTENCES["unknown"], text=text), text)
            if not REFERENCE_NAME.match(name):
                raise SecretReferenceError(_fill(REFERENCE_SENTENCES["badName"], text=text), text)
            if literal:
                out.append(literal)
                literal = ""
            out.append(SecretReference(scheme, name, text))
            i = close + 1
            continue
        literal += value[i]
        i += 1
    if literal:
        out.append(literal)
    return out


_UNCLOSED_RE = re.compile(r"^\$\{[A-Za-z]{0,16}:?[A-Za-z0-9_]{0,32}")


def unclosed_reference_text(rest: str) -> str:
    """What an unclosed ``${...`` is named as: ``${``, the scheme, the colon and the name as far as they parse, and nothing after."""
    match = _UNCLOSED_RE.match(rest)
    return match.group(0) if match else "${"


def has_reference(value: str) -> bool:
    """Whether ``value`` holds at least one well-formed reference (and nothing malformed)."""
    try:
        return any(not isinstance(token, str) for token in tokenize_references(value))
    except SecretReferenceError:
        return False


def mentions_reference(value: str) -> bool:
    """Whether ``value`` contains ``${`` at all: what the command-line refusal looks for."""
    return "${" in value


def expand_references(
    value: str,
    secret: Callable[[str], Optional[str]],
    env: Mapping[str, str],
) -> Tuple[str, List[str]]:
    """``value`` with its references replaced, plus every resolved value (for
    masking a report or an error that might repeat one). Raises
    :class:`SecretReferenceError`, naming the reference and never a value,
    when one cannot be resolved."""
    text = ""
    values: List[str] = []
    for token in tokenize_references(value):
        if isinstance(token, str):
            text += token
            continue
        if token.scheme == "env":
            found = env.get(token.name)
            if found is None or found == "":
                raise SecretReferenceError(_fill(REFERENCE_SENTENCES["envNotSet"], text=token.text), token.text, missing_env=token.name)
            values.append(found)
            text += found
            continue
        found = secret(token.name)
        if found is None or found == "":
            raise SecretReferenceError(
                _fill(REFERENCE_SENTENCES["secretNotSet"], text=token.text, hint=secrets_set_hint(token.name)),
                token.text,
                token.name,
            )
        values.append(found)
        text += found
    return text, values


def command_line_refusal(field: str) -> str:
    """The loader's refusal for a reference in ``command`` or ``args``."""
    return _fill(REFERENCE_SENTENCES["commandLine"], field=field)


def looks_like_secret(value: str) -> bool:
    """Whether a literal value (one with no reference) looks like a credential."""
    if mentions_reference(value):
        return False
    return any(re.search(pattern, value) for pattern in SECRET_LOOKING_PATTERNS)


def suggested_secret_name(server: str, key: str) -> str:
    """The secret name a warning suggests for a literal: ``<SERVER>_<KEY>``,
    upper case, anything outside the grammar folded to ``_``, so two servers
    with an ``Authorization`` header do not get the same suggestion."""
    name = re.sub(r"[^A-Z0-9_]", "_", f"{server}_{key}".upper())
    if re.match(r"^[0-9]", name):
        name = "_" + name
    return name[:128]


def literal_warning(server: str, field: str, key: str) -> str:
    """The loader's warning for a secret-looking literal in ``env`` or ``headers``."""
    suggested = suggested_secret_name(server, key)
    return _fill(REFERENCE_SENTENCES["literal"], field=field, key=key, suggested=suggested, hint=secrets_set_hint(suggested))


def at_connect_sentence(server: str, field: str, key: Optional[str], sentence: str) -> str:
    """A resolution failure as the connect-time error says it: where it sits, which server, and the sentence."""
    where = f"{field} {key}" if key else field
    return _fill(REFERENCE_SENTENCES["atConnect"], where=where, server=server, sentence=sentence)


def _mask_literal_run(run: str) -> str:
    """A literal run beside a reference, as a report shows it (S-303, 2026-09-26):
    a value that holds a reference used to be shown as written, so a literal
    key sitting beside one was printed. Every whitespace-delimited token of
    the run that looks like a secret, or is 16 or more characters long, is
    masked; a scheme word (``Bearer``), a short path (``/x``) and the spaces
    around them stay, so the report still says how the value is shaped."""
    return re.sub(r"\S+", lambda m: SECRET_MASK if looks_like_secret(m.group(0)) or len(m.group(0)) >= 16 else m.group(0), run)


def mask_value(value: str) -> str:
    """A value as a report shows it: each reference as written, every literal beside one masked as ``_mask_literal_run`` says, else the mask."""
    if not has_reference(value):
        return SECRET_MASK
    return "".join(_mask_literal_run(token) if isinstance(token, str) else token.text for token in tokenize_references(value))


def mask_url(url: str) -> str:
    """An address as a report shows it: without any ``user:password@`` (kept
    when it is references), and with every query value masked unless the
    value is a reference, the host and path kept so the report still says
    which server. Plain string work rather than ``urllib``, so both SDKs print
    the same bytes. A reference anywhere in the address used to show the
    whole address as written, literal query values and password included
    (S-303)."""

    def _user_info(match: "re.Match[str]") -> str:
        return match.group(0) if has_reference(match.group(2)) else match.group(1)

    no_user = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)([^/?#@]*)@", _user_info, url)
    q = no_user.find("?")
    if q == -1:
        return no_user
    hash_at = no_user.find("#", q)
    query = no_user[q + 1 :] if hash_at == -1 else no_user[q + 1 : hash_at]
    parts = []
    for part in query.split("&"):
        eq = part.find("=")
        if eq == -1 or has_reference(part[eq + 1 :]):
            parts.append(part)
        else:
            parts.append(f"{part[:eq]}={SECRET_MASK}")
    return no_user[: q + 1] + "&".join(parts) + ("" if hash_at == -1 else no_user[hash_at:])


def mask_map(values: Optional[Mapping[str, Any]]) -> Optional[Dict[str, str]]:
    """A map of values as a report shows it."""
    if values is None:
        return None
    return {key: mask_value(str(value)) for key, value in values.items()}


def mask_text(text: str, values: List[str]) -> str:
    """``text`` with every resolved value replaced by the mask, longest first, so an error can be logged."""
    out = text
    for value in sorted({v for v in values if v}, key=len, reverse=True):
        out = out.replace(value, SECRET_MASK)
    return out
