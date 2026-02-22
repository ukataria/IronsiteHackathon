"""PreCheck FastAPI backend — connects the Python pipeline to the Lovable frontend."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.utils import load_json, setup_logger

logger = setup_logger("api")

app = FastAPI(title="PreCheck API", version="1.0.0")

# ---------------------------------------------------------------------------
# CORS — allow Lovable (*.lovable.app) and localhost dev
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

DATA_DIR        = Path("data")
FRAMES_DIR      = DATA_DIR / "frames"
DEPTH_DIR       = DATA_DIR / "depth"
DETECTIONS_DIR  = DATA_DIR / "detections"
MEASUREMENTS_DIR = DATA_DIR / "measurements"
RESULTS_DIR     = DATA_DIR / "results"

# { job_id: { status, stage, progress, frames, error } }
_jobs: dict[str, dict[str, Any]] = {}

# WebSocket connections waiting on a job: { job_id: [WebSocket, ...] }
_ws_clients: dict[str, list[WebSocket]] = {}


def _job(job_id: str) -> dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


async def _broadcast(job_id: str, msg: dict) -> None:
    """Send a JSON message to all WebSocket clients watching this job."""
    for ws in list(_ws_clients.get(job_id, [])):
        try:
            await ws.send_json(msg)
        except Exception:
            _ws_clients[job_id].remove(ws)


def _run_pipeline_sync(job_id: str, file_path: str, file_type: str, vlm: str) -> None:
    """Run in a background thread; updates _jobs[job_id] and broadcasts via asyncio."""

    loop = asyncio.new_event_loop()

    def emit(stage: str, progress: int) -> None:
        _jobs[job_id].update({"stage": stage, "progress": progress})
        asyncio.run_coroutine_threadsafe(
            _broadcast(job_id, {"type": "stage", "stage": stage, "progress": progress}),
            loop,
        )

    loop.run_until_complete(_broadcast_noop())  # warm up loop reference

    try:
        _jobs[job_id]["status"] = "running"
        frame_paths: list[str] = []

        if file_type == "video":
            emit("Extracting frames", 5)
            from scripts.extract_frames import extract_frames
            frame_paths = extract_frames(file_path, str(FRAMES_DIR))
            emit(f"Extracted {len(frame_paths)} frames", 15)
        else:
            # Single image — copy into frames dir
            dest = FRAMES_DIR / Path(file_path).name
            FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            frame_paths = [str(dest)]

        frame_ids = [Path(p).stem for p in frame_paths]
        _jobs[job_id]["frames"] = frame_ids

        from pipeline import run_pipeline

        total = len(frame_paths)
        for i, fp in enumerate(frame_paths):
            fid = Path(fp).stem
            base_progress = 20 + int((i / total) * 75)
            emit(f"Processing {fid} ({i+1}/{total})", base_progress)
            run_pipeline(fp, vlm=vlm, skip_vlm=False)

        emit("Complete", 100)
        _jobs[job_id]["status"] = "complete"
        asyncio.run_coroutine_threadsafe(
            _broadcast(job_id, {"type": "complete", "frame_ids": frame_ids}),
            loop,
        )

    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {e}")
        _jobs[job_id].update({"status": "error", "error": str(e)})
        asyncio.run_coroutine_threadsafe(
            _broadcast(job_id, {"type": "error", "message": str(e)}),
            loop,
        )
    finally:
        loop.close()


async def _broadcast_noop() -> None:
    pass


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    vlm: str = "claude",
) -> dict:
    """Accept a video or image, kick off the pipeline in a background thread."""
    suffix = Path(file.filename or "upload").suffix.lower()
    file_type = "video" if suffix in (".mp4", ".mov", ".avi") else "image"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "stage": "Queued",
        "progress": 0,
        "frames": [],
        "error": None,
        "file_type": file_type,
        "filename": file.filename,
    }

    thread = threading.Thread(
        target=_run_pipeline_sync,
        args=(job_id, tmp_path, file_type, vlm),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "file_type": file_type, "filename": file.filename}


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@app.get("/api/job/{job_id}/status")
async def job_status(job_id: str) -> dict:
    j = _job(job_id)
    return {
        "status": j["status"],
        "stage": j["stage"],
        "progress": j["progress"],
        "frames_extracted": len(j.get("frames", [])),
        "error": j.get("error"),
    }


@app.get("/api/job/{job_id}/frames")
async def job_frames(job_id: str) -> dict:
    j = _job(job_id)
    frames = []
    for fid in j.get("frames", []):
        compliance = _frame_compliance(fid)
        frames.append({
            "id": fid,
            "thumbnail_url": f"/api/frames/{fid}/image?type=raw",
            "compliance": compliance,
        })
    return {"frames": frames}


# ---------------------------------------------------------------------------
# Frame data
# ---------------------------------------------------------------------------

@app.get("/api/frames")
async def list_frames() -> dict:
    """List all frames that have been processed (have a measurements file)."""
    if not FRAMES_DIR.exists():
        return {"frames": []}
    frames = []
    for p in sorted(FRAMES_DIR.glob("*.jpg")) + sorted(FRAMES_DIR.glob("*.png")):
        fid = p.stem
        frames.append({
            "id": fid,
            "thumbnail_url": f"/api/frames/{fid}/image?type=raw",
            "compliance": _frame_compliance(fid),
        })
    return {"frames": frames}


@app.get("/api/frames/{frame_id}/image")
async def get_frame_image(frame_id: str, type: str = "raw") -> FileResponse:
    """Serve raw, annotated, or measured image for a frame."""
    if type == "raw":
        for ext in (".jpg", ".jpeg", ".png"):
            p = FRAMES_DIR / f"{frame_id}{ext}"
            if p.exists():
                return FileResponse(str(p))
    elif type == "annotated":
        p = DETECTIONS_DIR / f"{frame_id}_annotated.png"
        if p.exists():
            return FileResponse(str(p))
        # Fall back to raw
        for ext in (".jpg", ".jpeg", ".png"):
            p2 = FRAMES_DIR / f"{frame_id}{ext}"
            if p2.exists():
                return FileResponse(str(p2))
    elif type == "measured":
        p = MEASUREMENTS_DIR / f"{frame_id}_measured.png"
        if p.exists():
            return FileResponse(str(p))
        # Fall back to annotated
        p2 = DETECTIONS_DIR / f"{frame_id}_annotated.png"
        if p2.exists():
            return FileResponse(str(p2))
    elif type == "depth":
        p = DEPTH_DIR / f"{frame_id}_depth.png"
        if p.exists():
            return FileResponse(str(p))

    raise HTTPException(status_code=404, detail=f"Image not found for {frame_id} (type={type})")


@app.get("/api/frames/{frame_id}/measurements")
async def get_measurements(frame_id: str) -> dict:
    p = MEASUREMENTS_DIR / f"{frame_id}_measurements.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Measurements not found — run pipeline first")
    return load_json(str(p))


@app.get("/api/frames/{frame_id}/anchors")
async def get_anchors(frame_id: str) -> dict:
    p = DETECTIONS_DIR / f"{frame_id}_anchors.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Anchors not found")
    return load_json(str(p))


# ---------------------------------------------------------------------------
# Ask a question
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    condition: str = "anchor_calibrated"  # baseline | depth | anchor_calibrated | all
    vlm: str = "claude"                   # claude | gpt4o | ollama


@app.post("/api/frames/{frame_id}/ask")
async def ask_question(frame_id: str, body: AskRequest) -> dict:
    """Run a custom question against one or all conditions for a frame."""
    frame_path = _find_frame_path(frame_id)
    if not frame_path:
        raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found")

    measurements_path = str(MEASUREMENTS_DIR / f"{frame_id}_measurements.json")
    depth_png_path = str(DEPTH_DIR / f"{frame_id}_depth.png")

    from src.vlm.clients import run_inspection

    conditions = (
        ["baseline", "depth", "anchor_calibrated"]
        if body.condition == "all"
        else [body.condition]
    )

    answers = []
    for cond in conditions:
        result = run_inspection(
            image_path=frame_path,
            measurements_json_path=measurements_path,
            depth_png_path=depth_png_path if Path(depth_png_path).exists() else None,
            condition=cond,
            vlm=body.vlm,
            output_dir=str(RESULTS_DIR),
            question=body.question,
        )
        answers.append({
            "condition": cond,
            "response": result["response"],
            "verdict": _infer_verdict(result["response"]),
        })

    return {"question": body.question, "frame_id": frame_id, "answers": answers}


# ---------------------------------------------------------------------------
# WebSocket — real-time pipeline progress
# ---------------------------------------------------------------------------

@app.websocket("/ws/job/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    _ws_clients.setdefault(job_id, []).append(websocket)

    # Send current state immediately
    if job_id in _jobs:
        j = _jobs[job_id]
        await websocket.send_json({
            "type": "stage",
            "stage": j["stage"],
            "progress": j["progress"],
        })
        if j["status"] == "complete":
            await websocket.send_json({"type": "complete", "frame_ids": j.get("frames", [])})
        elif j["status"] == "error":
            await websocket.send_json({"type": "error", "message": j.get("error", "")})

    try:
        while True:
            await websocket.receive_text()  # keep alive; client can send pings
    except WebSocketDisconnect:
        if job_id in _ws_clients and websocket in _ws_clients[job_id]:
            _ws_clients[job_id].remove(websocket)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_frame_path(frame_id: str) -> str | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = FRAMES_DIR / f"{frame_id}{ext}"
        if p.exists():
            return str(p)
    return None


def _frame_compliance(frame_id: str) -> str:
    p = MEASUREMENTS_DIR / f"{frame_id}_measurements.json"
    if not p.exists():
        return "unknown"
    try:
        m = load_json(str(p))
    except Exception:
        return "unknown"
    items = (
        [s.get("compliant", True) for s in m.get("stud_spacings", [])]
        + [s.get("compliant", True) for s in m.get("rebar_spacings", [])]
        + [h.get("compliant", True) for h in m.get("electrical_box_heights", [])]
    )
    if not items:
        return "unknown"
    return "pass" if all(items) else "fail"


def _infer_verdict(text: str) -> str:
    import re
    t = text.lower()
    if re.search(r"overall[:\s]+pass|recommendation[:\s]+pass", t):
        return "PASS"
    if re.search(r"overall[:\s]+fail|recommendation[:\s]+fail", t):
        return "FAIL"
    fails = len(re.findall(r"\bfail\b|\bdeficien|\bviolation\b|\bnon.compliant\b", t))
    passes = len(re.findall(r"\bpass\b|\bcompliant\b|\bwithin tolerance\b", t))
    if fails > passes:
        return "FAIL"
    if passes > fails and passes > 0:
        return "PASS"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "PreCheck API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
