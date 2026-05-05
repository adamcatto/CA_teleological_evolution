"""
Zip a repository tree for run reproducibility, excluding ``outs/`` and cache junk.
Embeds ``git_info.txt`` (HEAD + ``git status --short``) when ``.git`` is not zipped in full.
"""
from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Set

_SKIP_DIR_NAMES: Set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    ".venv",
    "venv",
    ".eggs",
    "eggs",
    "dist",
    "build",
}


def _git_info_text(repo_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        h = (r.stdout or "").strip() or "?"
    except (OSError, subprocess.TimeoutExpired):
        h = "?"
    try:
        r2 = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        st = (r2.stdout or "(no output)\n").rstrip() + "\n"
    except (OSError, subprocess.TimeoutExpired):
        st = "(git status failed)\n"
    return f"commit {h}\n\n--- status --short ---\n{st}"


def _skip_file(rel: Path) -> bool:
    name = rel.name
    if name == ".DS_Store":
        return True
    if name.endswith((".pyc", ".pyo", ".pyd", ".pkl")):
        return True
    if "__pycache__" in rel.parts:
        return True
    return False


def write_code_snapshot_zip(
    repo_root: str | Path,
    dest_zip: str | Path,
    *,
    include_dot_git: bool = False,
) -> str:
    """
    Zip *repo_root* for reproducibility. Skips only the top-level ``outs/`` directory (not
    ``other/outs``). Skips common cache/noise dirs, ``__pycache__``, and ``.pyc``.

    If ``include_dot_git`` is False (default), embeds a ``git_info.txt`` in the zip root; if
    True, the ``.git`` directory is included and ``git_info.txt`` is still added for
    convenience.

    Returns the absolute path to the zip.
    """
    root = Path(repo_root).resolve()
    out = Path(dest_zip).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(str(root), topdown=True, followlinks=False):
            dpath = Path(dirpath)
            if dpath == root:
                if "outs" in dirnames:
                    dirnames.remove("outs")
                if not include_dot_git and ".git" in dirnames:
                    dirnames.remove(".git")
            for b in _SKIP_DIR_NAMES:
                if b in dirnames:
                    dirnames.remove(b)
            reld = dpath.relative_to(root)
            if reld.parts and reld.parts[0] == "outs":
                continue
            for fn in filenames:
                p = dpath / fn
                rel = p.relative_to(root)
                if rel.parts and rel.parts[0] == "outs":
                    continue
                if _skip_file(rel):
                    continue
                zf.write(p, arcname=rel.as_posix())
        zf.writestr("git_info.txt", _git_info_text(root))
    return str(out)
