"""
FastAPI server for the CA training visualizer.

  pip install fastapi uvicorn
  uvicorn viewer_server:app --reload --port 8765

Environment:
  VIEWER_ROOTS  pathsep-separated list of roots to discover sessions (default: .)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.viewer_log import vocabulary_meta

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static" / "viewer"


def _allowed_roots() -> List[Path]:
    raw = os.environ.get("VIEWER_ROOTS", ".")
    sep = ";" if os.name == "nt" else os.pathsep
    roots = [Path(p.strip()).resolve() for p in raw.split(sep) if p.strip()]
    return roots or [Path.cwd()]


def _under_root(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _session_dir(session: str) -> Path:
    s = (session or "").strip().replace("\\", "/")
    if s == "" or s == ".":
        p = Path.cwd()
    elif s.startswith("/") or (len(s) > 1 and s[1] == ":"):
        p = Path(s).resolve()
    else:
        p = (Path.cwd() / s).resolve()
    p = p.resolve()
    # If this looks like a training out_dir with a produced viewer, allow any path (local dev).
    if (p / "viewer" / "stream.json").is_file():
        return p
    for r in _allowed_roots():
        if _under_root(p, r) or p == r:
            return p
    if _under_root(p, Path.cwd()):
        return p
    raise HTTPException(status_code=404, detail="session not found (no viewer/stream.json at this path)")


def _viewer_dir(session: str) -> Path:
    p = _session_dir(session)
    v = p / "viewer"
    if not v.is_dir():
        raise HTTPException(status_code=404, detail="no viewer/ for this session")
    return v


def discover_sessions() -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for root in _allowed_roots():
        if not root.is_dir():
            continue
        s_self = root / "viewer" / "stream.json"
        if s_self.is_file():
            try:
                rel = str(root.resolve().relative_to(Path.cwd()))
            except ValueError:
                rel = str(root)
            found.append(
                {
                    "session": rel.replace(os.sep, "/"),
                    "path": str(root.resolve()),
                    "mtime": s_self.stat().st_mtime,
                }
            )
        try:
            for child in root.iterdir():
                if not child.is_dir() or child.name == "viewer":
                    continue
                stream = child / "viewer" / "stream.json"
                if stream.is_file():
                    try:
                        rel = str(child.resolve().relative_to(Path.cwd()))
                    except ValueError:
                        rel = str(child.resolve())
                    found.append(
                        {
                            "session": rel.replace(os.sep, "/"),
                            "path": str(child.resolve()),
                            "mtime": stream.stat().st_mtime,
                        }
                    )
        except OSError:
            continue
    seen: set = set()
    out = []
    for f in found:
        key = f["path"]
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    out.sort(key=lambda x: -x["mtime"])
    return out


app = FastAPI(title="CA Teleological Viewer", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/vocab")
def api_vocab():
    return {"tokens": vocabulary_meta()}


@app.get("/api/sessions")
def api_sessions():
    return {"sessions": discover_sessions()}


def _find_metrics_jsonl(session: str) -> "Path":
    """
    Resolve ``<run_dir>/metrics.jsonl`` without requiring ``viewer/stream.json`` (so
    metrics work even if session path is only the run folder, or stream is missing).
    Never raises HTTPException.
    """
    s = (session or "").strip().replace("\\", "/")
    if not s:
        return Path()
    base = Path(s)
    if not base.is_absolute():
        base = (Path.cwd() / s).resolve()
    else:
        base = base.resolve()
    direct = base / "metrics.jsonl"
    if direct.is_file():
        return direct
    try:
        alt = _session_dir(s) / "metrics.jsonl"
        if alt.is_file():
            return alt
    except HTTPException:
        pass
    return direct


def _metrics_points_from_file(path: Path, max_points: int) -> Tuple[List[Dict[str, Any]], int, str]:
    """
    Read ``metrics.jsonl``; return (points list possibly truncated from the end, total row
    count before truncation, resolved path string).
    """
    if not path.is_file():
        return [], 0, str(path.resolve()) if path.parts else ""
    points: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            points.append(
                {
                    "global_step": d.get("global_step"),
                    "step_in_phase": d.get("step_in_phase"),
                    "phase": d.get("phase"),
                    "reward": d.get("reward"),
                    "loss": d.get("loss"),
                    "gumbel_tau": d.get("gumbel_tau"),
                }
            )
    n = len(points)
    if n > max_points:
        points = points[-max_points:]
    return points, n, str(path.resolve())


@app.get("/api/stream")
def api_stream(
    session: str = Query(..., description="Path to run dir (e.g. outs/myrun)"),
    stream_metrics: int = Query(
        1,
        ge=0,
        le=1,
        description="If 1, attach last N rows from metrics.jsonl as ``metrics_points`` (for charts without /api/metrics).",
    ),
    stream_metrics_max: int = Query(20_000, ge=0, le=200_000, description="Max metrics rows to attach"),
):
    p = _viewer_dir(session) / "stream.json"
    if not p.is_file():
        raise HTTPException(404, "stream.json missing")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if stream_metrics and stream_metrics_max > 0:
        mpath = _find_metrics_jsonl(session)
        pts, total_rows, mresolved = _metrics_points_from_file(mpath, stream_metrics_max)
        data["metrics_points"] = pts
        data["metrics_total_rows"] = total_rows
        if len(pts) < total_rows:
            data["metrics_truncated"] = True
        data["metrics_path"] = mresolved
        if not mpath.is_file():
            data["metrics_error"] = "metrics.jsonl not found"
    return data


@app.get("/api/metrics")
def api_metrics(
    session: str = Query(..., description="Path to run dir (contains metrics.jsonl)"),
    max_points: int = Query(200_000, ge=1, le=2_000_000, description="Max rows from end of file"),
):
    """
    Scalar training log: one JSON object per line in ``<session>/metrics.jsonl``.
    """
    p = _find_metrics_jsonl(session)
    if not p.is_file():
        return {
            "points": [],
            "n": 0,
            "total_rows": 0,
            "path": str(p.resolve()) if p.parts else "",
            "error": "metrics.jsonl not found",
        }
    points, total_rows, resolved = _metrics_points_from_file(p, max_points)
    return {"points": points, "n": len(points), "total_rows": total_rows, "path": resolved}


@app.get("/api/meta")
def api_meta(session: str = Query(...)):
    p = _viewer_dir(session) / "meta.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/episodes")
def api_episodes(session: str = Query(...)):
    p = _viewer_dir(session) / "episodes_manifest.json"
    if not p.is_file():
        return {"items": []}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/episode")
def api_episode(
    session: str = Query(...),
    file: Optional[str] = Query(None, description="e.g. 000042.json"),
    episode_id: Optional[int] = Query(None, description="look up in manifest if file not set"),
):
    vdir = _viewer_dir(session)
    epdir = vdir / "episodes"
    if file is None and episode_id is not None:
        man_path = vdir / "episodes_manifest.json"
        if man_path.is_file():
            with open(man_path, encoding="utf-8") as f:
                m = json.load(f)
            for it in reversed(m.get("items") or []):
                if it.get("id") == episode_id:
                    file = it.get("file")
                    break
        if file is None:
            raise HTTPException(404, "episode id not found in manifest")
    if not file or not str(file).endswith(".json"):
        raise HTTPException(400, "need file= or episode_id=")
    name = Path(file).name
    p = (epdir / name).resolve()
    if not str(p).startswith(str(epdir.resolve())):
        raise HTTPException(400, "bad path")
    if not p.is_file():
        raise HTTPException(404, "episode file not found (may be rotated out)")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def index():
    index_html = STATIC_DIR / "index.html"
    if not index_html.is_file():
        return JSONResponse(
            {"error": f"Missing static bundle at {index_html} (create static/viewer/)"},
            status_code=500,
        )
    return FileResponse(str(index_html))
