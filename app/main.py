import os, uuid, threading, inspect, traceback
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- SDPA compat shim: ignore enable_gqa if Torch < 2.5
try:
    import torch.nn.functional as F
    if "enable_gqa" not in inspect.signature(F.scaled_dot_product_attention).parameters:
        _orig_sdp = F.scaled_dot_product_attention
        def _sdp_compat(*args, enable_gqa=False, **kwargs):
            return _orig_sdp(*args, **kwargs)
        F.scaled_dot_product_attention = _sdp_compat
except Exception:
    pass

MODELS_DIR = os.getenv("MODELS_DIR", "/models")
OUT_DIR = os.getenv("OUT_DIR", "/data/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()

pipe = None
pipe_lock = threading.Lock()

app = FastAPI(title="WAN 2.2 T2V-A14B API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- safe defaults (multiples of 16; fit A100 80GB)
class GenerateRequest(BaseModel):
    prompt: str
    width: int = 960
    height: int = 544           # /16
    num_frames: int = 25        # will be normalized below
    steps: int = 28
    guidance_scale: float = 4.0
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None

def _load_pipeline():
    import torch
    from diffusers import DiffusionPipeline

    model_id = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    _pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, cache_dir=MODELS_DIR
    )
    _pipe.to("cuda")

    # memory-friendly toggles
    for fn in ("enable_attention_slicing", "enable_vae_slicing"):
        if hasattr(_pipe, fn):
            try:
                getattr(_pipe, fn)("max") if fn == "enable_attention_slicing" else getattr(_pipe, fn)()
            except Exception:
                pass
    try:
        if hasattr(_pipe, "vae") and hasattr(_pipe.vae, "enable_tiling"):
            _pipe.vae.enable_tiling()
    except Exception:
        pass

    # speed bump (A100-safe)
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.set_grad_enabled(False)
    except Exception:
        pass
    return _pipe

def _ensure_pipe():
    global pipe
    if pipe is None:
        with pipe_lock:
            if pipe is None:
                pipe = _load_pipeline()

# ---- Warmup: load only (no tiny render to avoid shape errors)
WARMUP = os.getenv("WARMUP", "1") == "1"

@app.on_event("startup")
def _startup():
    if not WARMUP:
        return
    def _bg():
        try:
            _ensure_pipe()
            print("Warmup load done.")
        except Exception as e:
            print("Warmup load error:", e)
    threading.Thread(target=_bg, daemon=True).start()

def _normalize_hw(width: int, height: int):
    """snap to multiples of 16 and keep >=256"""
    def floor16(x: int) -> int: return max(256, (int(x) // 16) * 16)
    return floor16(width), floor16(height)

def _normalize_frames(nf: int):
    """WAN requires (nf - 1) % 4 == 0; snap to nearest valid >= 9."""
    nf = max(9, int(nf))
    # round to nearest (k*4 + 1)
    k = round((nf - 1) / 4)
    return int(k * 4 + 1)

def _safe_infer(pipeline, **kw):
    import torch
    try:
        return pipeline(**kw)
    except torch.cuda.OutOfMemoryError:
        # last-resort: enable CPU offload + smaller recipe
        torch.cuda.empty_cache()
        try:
            if hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
        except Exception:
            pass
        kw["width"], kw["height"] = 960, 544
        kw["num_frames"] = min(25, kw.get("num_frames", 25))
        kw["num_inference_steps"] = min(28, kw.get("num_inference_steps", 28))
        return pipeline(**kw)

def _run_job(job_id: str, req: GenerateRequest):
    import torch
    from diffusers.utils import export_to_video
    try:
        _ensure_pipe()

        w16, h16 = _normalize_hw(req.width, req.height)
        nf_ok = _normalize_frames(req.num_frames)
        steps = max(10, int(req.steps))  # keep sane minimum

        gen = None
        if req.seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(req.seed))

        kw = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=h16, width=w16,
            num_frames=nf_ok,
            guidance_scale=req.guidance_scale,
            num_inference_steps=steps,
        )
        if gen is not None:
            kw["generator"] = gen

        out = _safe_infer(pipe, **kw)
        frames = out.frames[0]

        out_path = os.path.join(OUT_DIR, f"{job_id}.mp4")
        export_to_video(frames, out_path, fps=24)

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["file"] = out_path

    except Exception as e:
        # always provide a readable error
        msg = f"{e.__class__.__name__}: {e}"
        tb = traceback.format_exc(limit=2)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = msg + "\n" + tb

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/generate")
def generate(req: GenerateRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "file": None, "error": None}
    bg.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "running"}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    resp = {"job_id": job_id, **job}
    if job.get("file"):
        resp["result_url"] = f"/result/{job_id}"
    return JSONResponse(resp)

@app.get("/result/{job_id}")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("file"):
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(job["file"], media_type="video/mp4", filename=f"{job_id}.mp4")
