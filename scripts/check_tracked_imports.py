#!/usr/bin/env python3
"""Fail if a tracked file references a codebase_rag module or script that isn't itself tracked.

make verify runs against the working tree, so a file present on disk but never committed
is invisible to every test written inside that tree. This is the gap that let
provider_factory.py and openai_compat_client.py sit untracked for three commits while
cli.py imported the former: everything passed here, and failed on a clean clone.

Two checks:
  1. Every codebase_rag import in a tracked src/, evals/, scripts/, or tests/ file resolves
     to a tracked file (covers both `from codebase_rag.llm.x import y` and the submodule
     form `from codebase_rag.llm import x` / `from .llm import x`).
  2. Every .py path the Makefile invokes directly (`$(PYTHON) scripts/foo.py`) is tracked.
     Without this, a Makefile hunk referencing an untracked script (this script, the first
     time it was added) passes here and dies on a clean clone with a missing-file error,
     the exact failure class this script exists to catch, one level up.
"""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "codebase_rag"
PYTHON_TREES = ("src", "evals", "scripts", "tests")


def _git(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git not found on PATH")
    return subprocess.run(  # noqa: S603 - fixed argv, git resolved via shutil.which above
        [git, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def all_tracked_py_files() -> set[Path]:
    # `git ls-files "scripts/**/*.py"` silently misses scripts/ingest.py: `**` there requires
    # at least one intervening directory, so a file directly inside the tree root doesn't
    # match. List every tracked .py file rather than trusting a per-tree glob.
    output = _git("ls-files", "--", "*.py")
    return {REPO_ROOT / line for line in output.splitlines() if line}


def tracked_files() -> set[Path]:
    """Tracked .py files under the trees whose imports get checked. A narrower view than
    `all_tracked_py_files`, used for `check_imports`; `check_makefile_scripts` needs the
    unfiltered set, since a script the Makefile invokes need not live in one of these trees.
    """
    tree_roots = tuple(REPO_ROOT / tree for tree in PYTHON_TREES)
    return {p for p in all_tracked_py_files() if any(root == p.parent or root in p.parents for root in tree_roots)}


def module_to_path(module: str) -> Path | None:
    """Resolve a dotted codebase_rag module name to the file that defines it."""
    if not (module == PACKAGE or module.startswith(f"{PACKAGE}.")):
        return None
    src_root = REPO_ROOT / "src"
    parts = module.split(".")
    base = src_root.joinpath(*parts)
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return module_file
    return None


def resolve_relative(file_path: Path, level: int, module: str | None) -> list[str]:
    """Resolve `from .foo import bar` / `from ..foo import bar` to dotted module name(s).

    Only meaningful for files that actually live inside the codebase_rag package
    (under src/codebase_rag/); relative imports elsewhere (tests/, scripts/) don't
    address this package and are skipped.
    """
    src_root = REPO_ROOT / "src"
    try:
        rel = file_path.relative_to(src_root)
    except ValueError:
        return []
    package_parts = rel.with_suffix("").parts[:-1]
    ascend = level - 1
    base_parts = package_parts[: len(package_parts) - ascend] if ascend else package_parts
    if not base_parts:
        return []
    if module:
        return [".".join([*base_parts, *module.split(".")])]
    return [".".join(base_parts)]


def imported_modules(file_path: Path) -> set[str]:
    """Parse a file's imports.

    Returns an empty set (rather than raising) if the file is gone (deleted with `rm`
    instead of `git rm`, still listed by `git ls-files`) or mid-edit and currently
    unparseable: this script's job is to catch untracked-file references, not to
    re-litigate file existence or syntax, which ruff/mypy already do later in the same
    `make verify` run. Crashing here instead would take the whole gate down before those
    tools get a chance to give the actual, more specific error.
    """
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError, ValueError):
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # `from x import y` names the package/module in `module` and `y` as an alias,
            # which is ambiguous: `y` may be a name defined in `module`, or `module.y` may
            # itself be a submodule (`from codebase_rag.llm import provider_factory`, or the
            # relative form `from ..llm import provider_factory`). Add both forms; module_to_path
            # only resolves whichever one is actually a real file, so the ambiguity is harmless.
            if node.level and node.level > 0:
                bases = resolve_relative(file_path, node.level, node.module)
            elif node.module:
                bases = [node.module]
            else:
                bases = []
            modules.update(bases)
            for base in bases:
                for alias in node.names:
                    modules.add(f"{base}.{alias.name}")
    return modules


def check_imports(tracked: set[Path]) -> list[str]:
    problems = []
    for file_path in sorted(tracked):
        for module in imported_modules(file_path):
            target = module_to_path(module)
            if target is None or target in tracked:
                continue
            rel_source = file_path.relative_to(REPO_ROOT)
            rel_target = target.relative_to(REPO_ROOT)
            problems.append(f"{rel_source} imports '{module}' -> {rel_target}, which is not tracked by git")
    return problems


def check_makefile_scripts() -> list[str]:
    makefile = REPO_ROOT / "Makefile"
    if not makefile.is_file():
        return []
    tracked_makefile_paths = {REPO_ROOT / line for line in _git("ls-files", "Makefile").splitlines() if line}
    if makefile not in tracked_makefile_paths:
        return []

    # The full tracked set, not the tree-filtered one `check_imports` uses: a script the
    # Makefile invokes doesn't have to live under src/evals/scripts/tests to be legitimately
    # tracked (e.g. a top-level tools/ directory), and testing membership against the
    # narrower set would misreport a correctly committed script as untracked.
    tracked = all_tracked_py_files()
    problems = []
    for match in re.finditer(r"\b([\w./-]+\.py)\b", makefile.read_text()):
        rel_path = match.group(1)
        target = (REPO_ROOT / rel_path).resolve()
        if not target.is_relative_to(REPO_ROOT):
            continue
        if target.suffix == ".py" and target not in tracked and target.is_file():
            problems.append(f"Makefile references {rel_path}, which exists on disk but is not tracked by git")
    return problems


def main() -> int:
    tracked = tracked_files()
    if not tracked:
        print("check_tracked_imports: no tracked Python files found, skipping")
        return 0

    problems = check_imports(tracked) + check_makefile_scripts()

    if problems:
        print("check_tracked_imports: found references to untracked files:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nThese files exist on this disk but not in git. A fresh clone or CI checkout")
        print("cannot see them. Run `git add` on the listed target files.")
        return 1

    print(f"check_tracked_imports: {len(tracked)} tracked Python files, no untracked references found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
