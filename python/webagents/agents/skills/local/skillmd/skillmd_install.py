"""
Installing SKILL.md skills from outside: `webagents skills add <source>`
(gap-closure plan item 1.4, 2026-09-26). The TypeScript twin is
`typescript/src/skills/skillmd/skillmd-install.ts`; both run
`tests/fixtures/skillmd/skillmd.json` (`install`).

WHAT A SOURCE IS (`parse_source`): a git URL (`https://...`, `git@host:...`,
`file://...`), `owner/repo` on GitHub, a `.../tree/<ref>/<path>` or
`.../blob/<ref>/<path>/SKILL.md` page, or a local folder (`./x`, `../x`,
`/x`, `~/x`). A bare word (`shell`, `pdf`) is a NAME, never a source: names
belong to the existing editor of the agent file's `skills:` list.

WHY THE VETTING. ClawHavoc (341 malicious ClawHub skills) is the reason
nothing here installs silently: the repository is fetched at ONE commit, its
skills are located with the same search rules skills.sh uses, every file is
listed with its size, scripts and binaries are flagged, the person confirms
(or passes `--yes`, which is required when there is no terminal), and only
then is the skill copied into `<agent folder>/.agents/skills/<name>` and
recorded in `<agent folder>/.webagents/skills.lock` with its source, the
commit, a digest of its files and the file list. Limits: 10 MiB fetched,
25 MiB and 1,000 files installed, no symbolic links. What is installed is
owner-only to the agent until `access: tools:` opens it, and its scripts run
only in the sandbox (`skillmd_skill.py`).

THE LOCK is the record of what was installed and from where: the skills the
CLI may replace or remove are the ones it wrote. A folder under
`.agents/skills` that the lock does not know is the person's, and `add` and
`remove` leave it alone and say so.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .skillmd_loader import SKILL_FILE_NAMES, SKIPPED_DIRS, SkillMd, SkippedSkill, load_skill_dir, skill_file_in

#: The limits skills.sh applies, applied here too.
DOWNLOAD_LIMIT = 10 * 1024 * 1024
EXTRACTED_LIMIT = 25 * 1024 * 1024
FILE_LIMIT = 1000

#: Where skills live in a repository (skills.sh's rules): the root itself,
#: then these folders, walked `SEARCH_DEPTH` levels deep, plus any
#: `.<agent>/skills` folder at the root and the paths a Claude Code plugin
#: manifest names. A shallower SKILL.md shadows a deeper one of the same name.
SEARCH_ROOTS = (
    ".", "skills", "skills/.curated", "skills/.experimental", "skills/.system",
    ".agents/skills", ".claude/skills", ".codex/skills", ".opencode/skills", ".hermes/skills",
    ".cursor/skills", ".github/skills",
)
SEARCH_DEPTH = 3
PLUGIN_MANIFESTS = (".claude-plugin/marketplace.json", ".claude-plugin/plugin.json")

#: What is flagged in the file list before the person confirms.
SCRIPT_DIRS = ("scripts",)
SCRIPT_EXTENSIONS = (".py", ".sh", ".js", ".mjs", ".cjs", ".ts", ".rb", ".pl", ".ps1", ".bat", ".cmd")
BINARY_EXTENSIONS = (".so", ".dylib", ".dll", ".exe", ".wasm", ".bin", ".pyc", ".o", ".a", ".jar", ".class", ".node")

LOCK_FILE = os.path.join(".webagents", "skills.lock")
LOCK_VERSION = 1
LOCK_ENTRY_KEYS = ("source", "url", "ref", "subpath", "commit", "tree", "files", "installed_at")

#: Hosts whose HTTPS repository pages are normalised to the `.git` clone URL.
_FORGE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_PAGE_RE = re.compile(r"^(https?://([^/]+)/[^/]+/[^/]+?)(?:\.git)?(?:/(tree|blob)/([^/]+)(?:/(.*?))?)?/?$")


class InstallError(Exception):
    """A refusal or a failure, worded for the person at the terminal."""


@dataclass
class Source:
    kind: str  # git | local | name
    text: str
    url: Optional[str] = None
    ref: Optional[str] = None
    subpath: Optional[str] = None
    path: Optional[str] = None


def parse_source(text: str) -> Source:
    """What `skills add <text>` means (module docstring; pinned by the fixture's `install.sources`)."""
    s = text.strip()
    if s in (".", "..") or s.startswith(("./", "../", "/", "~")):
        return Source("local", text, path=s)
    if "://" in s:
        match = _PAGE_RE.match(s)
        if match and match.group(2).lower() in _FORGE_HOSTS:
            base, _host, kind, ref, subpath = match.group(1), match.group(2), match.group(3), match.group(4), match.group(5)
            if kind == "blob" and subpath:
                for name in SKILL_FILE_NAMES:
                    if subpath.endswith("/" + name) or subpath == name:
                        subpath = subpath[: -len(name)].rstrip("/")
                        break
            return Source("git", text, url=base + ".git", ref=ref or None, subpath=(subpath or None) if kind else None)
        return Source("git", text, url=s)
    if s.startswith("git@") and ":" in s:
        return Source("git", text, url=s)
    if _OWNER_REPO_RE.match(s) and not os.path.exists(s):
        return Source("git", text, url=f"https://github.com/{s}.git")
    if "/" in s and os.path.exists(os.path.expanduser(s)):
        return Source("local", text, path=s)
    return Source("name", text)


def repo_name(source: Source) -> str:
    """The repository's own name, for a skill at its root."""
    base = (source.url or source.path or source.text).rstrip("/")
    base = base.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return base[:-4] if base.endswith(".git") else base


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _git(args: Sequence[str], cwd: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_ASKPASS", "echo")
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env, timeout=600)


def _folder_bytes(root: str) -> int:
    total = 0
    for current, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(current, name)).st_size
            except OSError:
                pass
    return total


def _detail(result: subprocess.CompletedProcess) -> str:
    lines = [line.strip() for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else f"git exited with {result.returncode}"


def fetch_git(source: Source, workdir: str) -> Tuple[str, str, int]:
    """Clone `source` at one commit into `workdir`: `(checkout, commit, bytes fetched)`.
    A branch or tag name goes to `--branch`; a full commit SHA is fetched by
    itself. Refuses when more than `DOWNLOAD_LIMIT` came down."""
    if shutil.which("git") is None:
        raise InstallError(f"Could not fetch {source.text}: git is not installed")
    checkout = os.path.join(workdir, "repo")
    url = str(source.url)
    if source.ref and _SHA_RE.match(source.ref):
        os.makedirs(checkout)
        for args in (["init", "-q"], ["remote", "add", "origin", url], ["fetch", "-q", "--depth", "1", "origin", source.ref], ["checkout", "-q", "FETCH_HEAD"]):
            result = _git(args, checkout)
            if result.returncode != 0:
                raise InstallError(f"Could not fetch {source.text}: {_detail(result)}")
    else:
        args = ["clone", "-q", "--depth", "1", "--single-branch"]
        if source.ref:
            args += ["--branch", source.ref]
        args += [url, checkout]
        result = _git(args, workdir)
        if result.returncode != 0:
            raise InstallError(f"Could not fetch {source.text}: {_detail(result)}")
    head = _git(["rev-parse", "HEAD"], checkout)
    if head.returncode != 0:
        raise InstallError(f"Could not fetch {source.text}: {_detail(head)}")
    fetched = _folder_bytes(os.path.join(checkout, ".git"))
    if fetched > DOWNLOAD_LIMIT:
        raise InstallError(f"Refused: fetching {source.text} took {human_size(fetched)}, more than the 10 MiB limit.")
    return checkout, head.stdout.strip(), fetched


# ---------------------------------------------------------------------------
# Locating skills in a checkout
# ---------------------------------------------------------------------------


def _skill_dirs_below(base: str, depth: int) -> List[str]:
    """Folders holding a SKILL.md under `base`, breadth first up to `depth`
    levels, a folder with one not descended into."""
    found: List[str] = []
    level = [base]
    for _ in range(depth):
        next_level: List[str] = []
        for directory in level:
            try:
                names = sorted(os.listdir(directory))
            except OSError:
                continue
            for name in names:
                if name in SKIPPED_DIRS or name.startswith("."):
                    continue
                child = os.path.join(directory, name)
                if os.path.islink(child) or not os.path.isdir(child):
                    continue
                if skill_file_in(child):
                    found.append(child)
                else:
                    next_level.append(child)
        level = next_level
    return found


def _manifest_paths(root: str) -> List[str]:
    """The skill folders a Claude Code plugin manifest names, relative to the root."""
    paths: List[str] = []
    for relative in PLUGIN_MANIFESTS:
        file = os.path.join(root, relative)
        if not os.path.isfile(file):
            continue
        try:
            with open(file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        entries: List[Any] = []
        if isinstance(data, dict):
            plugins = data.get("plugins")
            if isinstance(plugins, list):
                for plugin in plugins:
                    if isinstance(plugin, dict):
                        skills = plugin.get("skills")
                        entries += skills if isinstance(skills, list) else [skills] if isinstance(skills, str) else []
            skills = data.get("skills")
            entries += skills if isinstance(skills, list) else [skills] if isinstance(skills, str) else []
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                paths.append(entry.strip())
    return paths


def locate_skills(root: str, subpath: Optional[str], name_for_root: str) -> Tuple[List[Tuple[str, str]], List[SkippedSkill]]:
    """`([(name, folder)], skipped)`: the skill folders in a checkout, by
    skills.sh's rules (`SEARCH_ROOTS`), or under `subpath` when a page URL
    named one. Folders whose SKILL.md cannot load are in `skipped`."""
    root = os.path.realpath(root)
    candidates: List[Tuple[str, str]] = []
    seen = set()

    # THE INSTALLED FOLDER IS NAMED AFTER THE SKILL'S OWN VALIDATED `name:`
    # (2026-09-26, the e2e run): the folder took the repository's name (or the
    # checkout folder's), so a skill declared `greeter` in a repository
    # `greeter-skill` landed as `greeter-skill` and warned about its own name
    # on every load. A declared name that is not a skill name (upper-case, a
    # space) is not used; the folder name, or `name_for_root` for a skill at
    # the repository root, stands in as before.
    def add(directory: str, name: Optional[str] = None) -> None:
        real = os.path.realpath(directory)
        if not real.startswith(root + os.sep) and real != root:
            return
        if not skill_file_in(real):
            return
        loaded = load_skill_dir(real)
        declared = loaded.declared_name if isinstance(loaded, SkillMd) else None
        label = declared or name or os.path.basename(real)
        if label in seen:
            return
        seen.add(label)
        candidates.append((label, real))

    if subpath:
        base = os.path.realpath(os.path.join(root, subpath))
        if not base.startswith(root + os.sep) or not os.path.isdir(base):
            return [], []
        if skill_file_in(base):
            add(base)
        else:
            for directory in _skill_dirs_below(base, SEARCH_DEPTH):
                add(directory)
    else:
        # A skill at the repository root: its validated `name:` (`add`
        # prefers it), else the repository's own name, never the checkout
        # folder's.
        if skill_file_in(root):
            add(root, name_for_root)
        for relative in SEARCH_ROOTS[1:]:
            directory = os.path.join(root, relative)
            if os.path.isdir(directory):
                for found in _skill_dirs_below(directory, SEARCH_DEPTH):
                    add(found)
        try:
            for entry in sorted(os.listdir(root)):
                if entry.startswith(".") and entry not in SKIPPED_DIRS and os.path.isdir(os.path.join(root, entry, "skills")):
                    for found in _skill_dirs_below(os.path.join(root, entry, "skills"), SEARCH_DEPTH):
                        add(found)
        except OSError:
            pass
        for relative in _manifest_paths(root):
            add(os.path.join(root, relative))

    skipped: List[SkippedSkill] = []
    located: List[Tuple[str, str]] = []
    for label, directory in candidates:
        loaded = load_skill_dir(directory)
        if isinstance(loaded, SkippedSkill):
            skipped.append(SkippedSkill(location=loaded.location, name=label, problem=loaded.problem, reason=loaded.reason))
        else:
            located.append((label, directory))
    return located, skipped


# ---------------------------------------------------------------------------
# Files, flags, digests
# ---------------------------------------------------------------------------


@dataclass
class SkillFile:
    path: str
    size: int
    script: bool = False
    binary: bool = False


def _is_binary_file(full: str) -> bool:
    try:
        with open(full, "rb") as handle:
            return b"\x00" in handle.read(8192)
    except OSError:
        return False


def list_files(directory: str) -> Tuple[List[SkillFile], List[str]]:
    """Every regular file under a skill folder (`.git` and the like left out),
    sorted, with its size and flags; and the symbolic links found, which
    refuse the install."""
    real = os.path.realpath(directory)
    files: List[SkillFile] = []
    links: List[str] = []
    for current, dirs, names in os.walk(real):
        dirs[:] = sorted(d for d in dirs if d not in SKIPPED_DIRS)
        for name in dirs:
            if os.path.islink(os.path.join(current, name)):
                links.append(os.path.relpath(os.path.join(current, name), real).replace(os.sep, "/"))
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(current, d))]
        for name in sorted(names):
            full = os.path.join(current, name)
            relative = os.path.relpath(full, real).replace(os.sep, "/")
            if os.path.islink(full):
                links.append(relative)
                continue
            if not os.path.isfile(full):
                continue
            size = os.path.getsize(full)
            extension = os.path.splitext(name)[1].lower()
            in_scripts = relative.split("/")[0] in SCRIPT_DIRS and "/" in relative
            executable = bool(os.stat(full).st_mode & stat.S_IXUSR)
            binary = extension in BINARY_EXTENSIONS or _is_binary_file(full)
            script = not binary and (in_scripts or extension in SCRIPT_EXTENSIONS or executable)
            files.append(SkillFile(path=relative, size=size, script=script, binary=binary))
    files.sort(key=lambda f: f.path)
    return files, sorted(links)


def tree_digest(directory: str, files: Optional[Sequence[str]] = None) -> str:
    """`sha256:<hex>` over the sorted file list: each path, then the SHA-256 of
    its content, one per line. The same bytes give the same digest in both
    SDKs (fixture `install.lock.digest_vector`)."""
    real = os.path.realpath(directory)
    names = sorted(files) if files is not None else [f.path for f in list_files(real)[0]]
    digest = hashlib.sha256()
    for relative in names:
        with open(os.path.join(real, *relative.split("/")), "rb") as handle:
            content = hashlib.sha256(handle.read()).hexdigest()
        digest.update(f"{relative}\n{content}\n".encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


def lock_path(folder: str) -> str:
    return os.path.join(folder, LOCK_FILE)


def read_lock(folder: str) -> Dict[str, Any]:
    path = lock_path(folder)
    if not os.path.isfile(path):
        return {"version": LOCK_VERSION, "skills": {}}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {"version": LOCK_VERSION, "skills": {}}
    if not isinstance(data, dict) or not isinstance(data.get("skills"), dict):
        return {"version": LOCK_VERSION, "skills": {}}
    data["version"] = LOCK_VERSION
    return data


def write_lock(folder: str, data: Dict[str, Any]) -> None:
    path = lock_path(folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    directory: str
    description: str
    files: List[SkillFile] = field(default_factory=list)
    links: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


@dataclass
class InstallReport:
    installed: List[str] = field(default_factory=list)
    skipped: List[SkippedSkill] = field(default_factory=list)


Say = Callable[[str], None]
Confirm = Callable[[str], bool]


def _short(description: str, limit: int = 80) -> str:
    description = " ".join(description.split())
    return description if len(description) <= limit else description[: limit - 3].rstrip() + "..."


def install_from_source(
    source: Source,
    folder: str,
    *,
    skill: Optional[str] = None,
    yes: bool = False,
    tty: bool = False,
    confirm: Optional[Confirm] = None,
    out: Say = print,
    err: Say = print,
) -> int:
    """`skills add <source>`: fetch, locate, show, confirm, install, record.
    Returns the exit code; every refusal leaves the folder untouched."""
    folder = os.path.realpath(folder)
    workdir = tempfile.mkdtemp(prefix="webagents-skillmd-")
    try:
        try:
            if source.kind == "git":
                root, commit, _fetched = fetch_git(source, workdir)
                root = os.path.realpath(root)
            else:
                root = os.path.realpath(os.path.expanduser(str(source.path)))
                if not os.path.isdir(root):
                    err(f"{source.text} is not a folder.")
                    return 1
                commit = None
        except InstallError as error:
            err(str(error))
            return 1

        located, skipped = locate_skills(root, source.subpath, repo_name(source))
        for item in skipped:
            err(f"Skipped {item.name}: {item.reason}")
        if not located:
            err(f"No SKILL.md skills found in {source.text}.")
            return 1
        if skill is not None:
            chosen = [(name, directory) for name, directory in located if name == skill]
            if not chosen:
                err(f'No skill called "{skill}" in {source.text}. Skills there: {", ".join(name for name, _ in located)}.')
                return 1
        else:
            chosen = located

        candidates: List[Candidate] = []
        for name, directory in chosen:
            loaded = load_skill_dir(directory)
            description = loaded.description if isinstance(loaded, SkillMd) else ""
            files, links = list_files(directory)
            candidates.append(Candidate(name=name, directory=directory, description=description, files=files, links=links))
        names = ", ".join(c.name for c in candidates)

        # The limits, before anything is shown as installable.
        for candidate in candidates:
            if candidate.links:
                err(f"Refused: {candidate.name} contains a symbolic link ({candidate.links[0]}), which could point outside the skill folder.")
                return 1
        total_files = sum(len(c.files) for c in candidates)
        total_size = sum(c.size for c in candidates)
        if total_files > FILE_LIMIT:
            err(f"Refused: {names} would install {total_files} files, more than the limit of {FILE_LIMIT}.")
            return 1
        if total_size > EXTRACTED_LIMIT:
            err(f"Refused: {names} would install {human_size(total_size)}, more than the 25 MiB limit.")
            return 1
        lock = read_lock(folder)
        for candidate in candidates:
            target = os.path.join(folder, ".agents", "skills", candidate.name)
            if os.path.lexists(target) and candidate.name not in lock["skills"]:
                err(f"{candidate.name} already exists in .agents/skills and was not installed by webagents; remove it first.")
                return 1

        # The vetting: what would be installed, file by file.
        out(f"Found {len(candidates)} in {source.text}: {names}")
        for candidate in candidates:
            out(f"{candidate.name}: {_short(candidate.description)}")
            for item in candidate.files:
                flag = " [binary]" if item.binary else " [script]" if item.script else ""
                out(f"  {item.path} ({item.size} B){flag}")
        if not yes:
            if not tty:
                err("Pass --yes to install without a prompt.")
                return 1
            asked = confirm or (lambda question: input(question).strip().lower() in ("y", "yes"))
            if not asked(f"Install {names} into .agents/skills? [y/N] "):
                out("Nothing installed.")
                return 0

        # The install, then the record.
        for candidate in candidates:
            target = os.path.join(folder, ".agents", "skills", candidate.name)
            if os.path.lexists(target):
                shutil.rmtree(target)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copytree(candidate.directory, target, symlinks=False, ignore=shutil.ignore_patterns(*SKIPPED_DIRS))
            paths = [f.path for f in candidate.files]
            lock["skills"][candidate.name] = {
                "source": source.text,
                "url": source.url,
                "ref": source.ref,
                "subpath": source.subpath if source.subpath else os.path.relpath(candidate.directory, root).replace(os.sep, "/"),
                "commit": commit,
                "tree": tree_digest(target, paths),
                "files": paths,
                "installed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }
            write_lock(folder, lock)
            if commit:
                out(f"Installed {candidate.name} into .agents/skills/{candidate.name} ({len(paths)} files) at {commit[:7]}.")
            else:
                out(f"Installed {candidate.name} into .agents/skills/{candidate.name} ({len(paths)} files).")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def installed_skill_names(folder: str) -> List[str]:
    """The skills the lock records, that are still there."""
    folder = os.path.realpath(folder)
    return sorted(name for name in read_lock(folder)["skills"] if os.path.isdir(os.path.join(folder, ".agents", "skills", name)))


def remove_installed(name: str, folder: str, *, out: Say = print, err: Say = print) -> Optional[int]:
    """`skills remove <name>` for an installed SKILL.md skill: the folder and
    its lock entry go. None when `name` is not one the lock knows (the
    caller tries the agent file's `skills:` list next); else the exit code."""
    folder = os.path.realpath(folder)
    lock = read_lock(folder)
    if name not in lock["skills"]:
        return None
    target = os.path.join(folder, ".agents", "skills", name)
    if os.path.lexists(target):
        shutil.rmtree(target)
    del lock["skills"][name]
    write_lock(folder, lock)
    out(f"Removed {name} from .agents/skills.")
    return 0
