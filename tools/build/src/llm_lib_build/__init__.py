"""`uv run build`: llm-lib as a tarball (sdist) and a wheel in dist/, checked before and after.

    uv run build              # lint + tests, build dist/llm_lib-<version>.tar.gz and .whl, then install
                              # the tarball into a throwaway venv and import it
    uv run build --no-check   # skip lint and tests
    uv run build --sdist      # the tarball only

Another app then installs it with `uv add /path/to/LLM-lib/dist/llm_lib-<version>.tar.gz` (add
`[anthropic]` via `"llm-lib[anthropic] @ file:///.../llm_lib-<version>.tar.gz"`).

A dev-only command (its own package in tools/build, pulled in by llm-lib's dev group) so that apps
depending on llm-lib don't get a `build` on their PATH that would shadow their own.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

# tools/build/src/llm_lib_build/__init__.py -> project root
ROOT = Path(__file__).resolve().parents[4]
DIST = ROOT / "dist"
# What the tarball must hold to be installable and usable
REQUIRED = ("pyproject.toml", "README.md", "LICENSE", "src/llm_lib/__init__.py", "src/llm_lib/py.typed")


def _run(cmd: list[str], what: str) -> None:
    print(f"==> {what}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"build: {what} failed (exit {result.returncode})")


def _project() -> tuple[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return data["name"], data["version"]


def _check_tarball(path: Path) -> None:
    with tarfile.open(path) as tar:
        names = tar.getnames()
    top = names[0].split("/")[0]
    missing = [f for f in REQUIRED if f"{top}/{f}" not in names]
    if missing:
        sys.exit(f"build: {path.name} is missing {', '.join(missing)}")
    modules = sum(n.endswith(".py") and "/src/llm_lib/" in n for n in names)
    print(f"==> {path.name}: {len(names)} entries, {modules} modules, {path.stat().st_size / 1024:.0f} KiB")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="build", description="Build llm-lib's sdist tarball and wheel into dist/.")
    parser.add_argument("--no-check", action="store_true", help="skip ruff and pytest before building")
    parser.add_argument("--sdist", action="store_true", help="build only the .tar.gz")
    parser.add_argument("--no-verify", action="store_true", help="skip installing the tarball into a fresh venv")
    args = parser.parse_args(argv)

    uv = shutil.which("uv")
    if not uv:
        sys.exit("build: uv not found on PATH")
    name, version = _project()
    dist_name = name.replace("-", "_")

    if not args.no_check:
        _run([uv, "run", "ruff", "check", "src", "tests", "tools"], "lint")
        _run([uv, "run", "pytest", "-q"], "tests")

    # Only this version's files are replaced; older releases in dist/ are kept
    DIST.mkdir(exist_ok=True)
    for old in DIST.glob(f"{dist_name}-{version}*"):
        old.unlink()
    _run([uv, "build", "--out-dir", str(DIST), *(["--sdist"] if args.sdist else [])], "build")

    tarball = DIST / f"{dist_name}-{version}.tar.gz"
    if not tarball.exists():
        sys.exit(f"build: expected {tarball} but uv build didn't write it")
    _check_tarball(tarball)

    if not args.no_verify:
        # A fresh venv with nothing but the tarball (and its dependencies): proves it installs and imports
        _run(
            [uv, "run", "--isolated", "--no-project", "--with", str(tarball), "python", "-c",
             "import llm_lib, llm_lib.servers.colibri, llm_lib.servers.lmstudio; "
             "print('    imports fine:', len(llm_lib.PROVIDERS), 'providers')"],
            "verify",
        )

    print("\nBuilt:")
    for path in sorted(DIST.glob(f"{dist_name}-{version}*")):
        print(f"  {path.relative_to(ROOT)}")
    print(f"\nInstall in another app:  uv add {tarball}")


if __name__ == "__main__":
    main()
