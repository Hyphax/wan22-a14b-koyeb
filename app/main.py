import os, uuid, threading, inspect
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- SDPA compat shim: ignore enable_gqa if Torch is older than 2.5
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

# ---------- safer defaults (fit A100 80GB reliably)
class GenerateRequest(BaseModel):
    prompt: str
    width: int = 960
    height: int = 540
    num_frames: int = 25
    steps: int = 28
    guidance_scale: float = 4.0
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None

def _load_pipeline():
    import torch
    from diffusers import WanPipeline, AutoencoderKLWan

    model_id = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    # VAE on CPU float32; main model bf16 on GPU
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=MODELS_DIR
    )
    _pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir=MODELS_DIR
    ).to("cuda")

    # Memory-friendly toggles (defensive)
    try: _pipe.enable_attention_slicing("max")
    except Exception: pass
    try: _pipe.enable_vae_slicing()
    except Exception: pass
    try:
        if hasattr(_pipe, "vae") and hasattr(_pipe.vae, "enable_tiling"):
            _pipe.vae.enable_tiling()
    except Exception: pass

    # Speed bump (safe on A100)
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

# Warmup so first real job is fast(er)
WARMUP = os.getenv("WARMUP", "1") == "1"
@app.on_event("startup")
def _warmup():
    if not WARMUP:
        return
    try:
        _ensure_pipe()
        _ = pipe(prompt="warmup", height=192, width=320, num_frames=8,
                 num_inference_steps=4, guidance_scale=3.5)
    except Exception as e:
        print("Warmup error:", e)

def _run_job(job_id: str, req: GenerateRequest):
    import torch
    from diffusers.utils import export_to_video
    try:
        _ensure_pipe()
        gen = None
        if req.seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(req.seed))

        def do_infer(w, h, nf, st):
            kw = dict(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                height=h, width=w,
                num_frames=nf,
                guidance_scale=req.guidance_scale,
                num_inference_steps=st,
            )
            if gen is not None:
                kw["generator"] = gen
            return pipe(**kw)

        # Try user params first; on OOM, auto-fallback to safe profile
        try:
            out = do_infer(req.width, req.height, req.num_frames, req.steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = do_infer(960, 540, min(25, req.num_frames), min(28, req.steps))

        frames = out.frames[0]
        out_path = os.path.join(OUT_DIR, f"{job_id}.mp4")
        export_to_video(frames, out_path, fps=24)

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["file"] = out_path
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)

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
