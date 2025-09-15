import os, uuid, threading
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ==== Storage paths
MODELS_DIR = os.getenv("MODELS_DIR", "/models")
OUT_DIR = os.getenv("OUT_DIR", "/data/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ==== Simple in-memory job store (single replica)
JOBS = {}  # id -> dict(status,file,error)
JOBS_LOCK = threading.Lock()

# ==== Lazy-loaded pipeline
pipe = None
pipe_lock = threading.Lock()

app = FastAPI(title="WAN 2.2 T2V-A14B API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1280     # 720p
    height: int = 720
    num_frames: int = 81  # ~3.4s @24fps (keeps inference fast)
    steps: int = 50
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
    dtype = torch.bfloat16
    _pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=dtype, cache_dir=MODELS_DIR
    )
    _pipe.to("cuda")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    torch.set_grad_enabled(False)
    return _pipe

def _ensure_pipe():
    global pipe
    if pipe is None:
        with pipe_lock:
            if pipe is None:
                pipe = _load_pipeline()

def _run_job(job_id: str, req: GenerateRequest):
    from diffusers.utils import export_to_video
    import torch
    try:
        _ensure_pipe()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"

        gen = None
        if req.seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(req.seed))

        kwargs = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.steps,
        )
        if gen is not None:
            kwargs["generator"] = gen

        out = pipe(**kwargs)
        frames = out.frames[0]  # list of PIL images

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
        JOBS[job_id] = {"status": "queued", "file": None, "error": None}
    bg.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}

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
