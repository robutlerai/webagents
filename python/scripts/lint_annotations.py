#!/usr/bin/env python3
"""Static lint for the F-040 annotation idiom. No install required.

Flags annotations that are evaluated at class-creation time (Python <= 3.13)
and reference a name that a `try: import ... except ImportError:` guard
either binds to None or leaves unbound — the failure that raises
AttributeError / NameError at `import webagents` and that an
`except ImportError` clause can never catch. On Python 3.14 the import
SUCCEEDS (PEP 649 defers annotations) and the same error relocates to the
first `typing.get_type_hints()` call, which is why this must be a static
check and not just an import smoke test.

Rules (deliberately narrow, so the lint is green on day one and stays
green):

- FLAG an attribute-rooted annotation (`types.Tool`) whose root name is
  bound to None in an import-guard except handler: evaluating it is
  `None.Tool` -> AttributeError.
- FLAG any annotation referencing a name imported in a guarded try block
  and NOT bound at all in the handler: NameError.
- SKIP quoted (string) annotations — they are not evaluated at class
  creation, and stand-in bindings keep get_type_hints() working.
- SKIP names imported under `if TYPE_CHECKING:` (their annotations are
  quoted by convention; unquoted ones are caught by the unbound rule only
  when they ALSO appear in an import guard).
- SKIP annotations inside function bodies (evaluated at call time at
  worst, not at import; a bare name bound to None evaluates fine).

Usage: python3 scripts/lint_annotations.py [package_dir]
Exit 1 when any finding exists.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _guard_bindings(tree: ast.Module) -> Tuple[Set[str], Set[str]]:
    """(none_bound, unbound_if_missing) names from module-level import guards."""
    none_bound: Set[str] = set()
    unbound: Set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Try):
            continue
        guards_import = any(
            isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "Exception", "ModuleNotFoundError")
            for h in node.handlers
            if h.type is not None
        )
        if not guards_import:
            continue
        imported: Set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported.add(alias.asname or alias.name)
        handler_bound_none: Set[str] = set()
        handler_bound_other: Set[str] = set()
        for h in node.handlers:
            for stmt in ast.walk(ast.Module(body=h.body, type_ignores=[])):
                if isinstance(stmt, ast.Assign):
                    is_none = isinstance(stmt.value, ast.Constant) and stmt.value.value is None
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            (handler_bound_none if is_none else handler_bound_other).add(tgt.id)
                elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    handler_bound_other.add(stmt.name)
        none_bound |= imported & handler_bound_none
        # Names the handler binds to a real stand-in are safe; names it never
        # binds at all are NameErrors when referenced.
        unbound |= imported - handler_bound_none - handler_bound_other
    return none_bound, unbound


def _type_checking_names(tree: ast.Module) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_tc:
                for stmt in node.body:
                    if isinstance(stmt, ast.ImportFrom):
                        for alias in stmt.names:
                            names.add(alias.asname or alias.name)
                    elif isinstance(stmt, ast.Import):
                        for alias in stmt.names:
                            names.add((alias.asname or alias.name).split(".")[0])
    return names


def _annotation_findings(
    ann: ast.expr,
    none_bound: Set[str],
    unbound: Set[str],
    tc_names: Set[str],
) -> List[str]:
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        return []  # quoted: not evaluated at class creation
    findings: List[str] = []
    attr_roots: Set[int] = set()
    for node in ast.walk(ann):
        if isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                attr_roots.add(id(root))
                if root.id in none_bound:
                    findings.append(
                        f"attribute annotation on None-bound '{root.id}' "
                        f"(AttributeError at class creation on <=3.13)"
                    )
                elif root.id in unbound and root.id not in tc_names:
                    findings.append(
                        f"annotation on unbound '{root.id}' (NameError at class creation on <=3.13)"
                    )
    for node in ast.walk(ann):
        if isinstance(node, ast.Name) and id(node) not in attr_roots:
            if node.id in unbound and node.id not in tc_names:
                findings.append(
                    f"annotation on unbound '{node.id}' (NameError at class creation on <=3.13)"
                )
    return findings


def _class_creation_annotations(tree: ast.Module):
    """Yield (lineno, annotation) evaluated at import time: signatures and
    AnnAssigns at module level or directly inside class bodies."""

    def iter_scope(body, in_class: bool):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for a in (
                    list(args.posonlyargs)
                    + list(args.args)
                    + list(args.kwonlyargs)
                    + ([args.vararg] if args.vararg else [])
                    + ([args.kwarg] if args.kwarg else [])
                ):
                    if a.annotation is not None:
                        yield node.lineno, a.annotation
                if node.returns is not None:
                    yield node.lineno, node.returns
                # do NOT descend into the function body
            elif isinstance(node, ast.ClassDef):
                yield from iter_scope(node.body, True)
            elif isinstance(node, ast.AnnAssign):
                yield node.lineno, node.annotation
            elif isinstance(node, (ast.If, ast.Try)):
                yield from iter_scope(node.body, in_class)
                for h in getattr(node, "handlers", []):
                    yield from iter_scope(h.body, in_class)
                yield from iter_scope(getattr(node, "orelse", []), in_class)
                yield from iter_scope(getattr(node, "finalbody", []), in_class)

    yield from iter_scope(tree.body, False)


def lint_file(path: Path) -> List[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: syntax error: {e.msg}"]
    none_bound, unbound = _guard_bindings(tree)
    if not none_bound and not unbound:
        return []
    tc_names = _type_checking_names(tree)
    out: List[str] = []
    for lineno, ann in _class_creation_annotations(tree):
        for finding in _annotation_findings(ann, none_bound, unbound, tc_names):
            out.append(f"{path}:{lineno}: {finding}")
    return out


def main(root: str = "webagents") -> int:
    findings: List[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        findings.extend(lint_file(path))
    for f in findings:
        print(f)
    if findings:
        print(f"\n{len(findings)} F-040-class annotation finding(s).")
        return 1
    print("annotation lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "webagents"))
