# WAN 2.2 Text-to-Video API for Koyeb

This project provides a FastAPI service for the WAN 2.2 text-to-video model, optimized for deployment on Koyeb.

## ⚠️ Important Limitations on Koyeb

### Critical Issues Identified:

1. **No GPU Support**: Koyeb only provides CPU instances, but the WAN 2.2 model requires CUDA GPU
2. **Memory Constraints**: The 14B parameter model requires 40-80GB VRAM, but Koyeb offers max 8GB RAM
3. **Model Size**: Multi-gigabyte model downloads may exceed Koyeb's startup timeouts
4. **Performance**: CPU-only inference will be extremely slow for video generation

### Implemented Fixes:

✅ **CPU Fallback Mode**: Automatically detects environment and uses CPU-only PyTorch
✅ **Lightweight Base Image**: Replaced NVIDIA CUDA image with Python slim
✅ **Memory Optimizations**: Reduced default parameters for limited resources
✅ **Fallback Models**: Attempts smaller models if main model fails
✅ **Error Handling**: Graceful degradation with dummy responses
✅ **Resource Monitoring**: Health endpoint shows system resources
✅ **Environment Detection**: Automatic Koyeb deployment detection

## Environment Variables

- `KOYEB_DEPLOYMENT=1`: Forces CPU mode (automatically set in Dockerfile)
- `WAN_MODEL_ID`: Model to use (default: Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- `WARMUP=1`: Pre-load model on startup (default: enabled)
- `MODELS_DIR`: Model cache directory (default: /models)
- `OUT_DIR`: Output directory (default: /data/outputs)

## API Endpoints

- `GET /`: Service information and warnings
- `GET /healthz`: Health check with resource monitoring
- `POST /generate`: Generate video from text prompt
- `GET /jobs/{job_id}`: Check job status
- `GET /result/{job_id}`: Download generated video

## Deployment Options

### Option 1: Deploy with Limitations (Not Recommended)
The current setup will deploy on Koyeb but with severe limitations:
- Extremely slow generation (CPU-only)
- May timeout on large models
- Limited to small video resolutions
- May fall back to dummy/test responses

### Option 2: Use GPU Cloud Service (Recommended)
For production use, consider:
- **RunPod**: GPU instances with CUDA support
- **Vast.ai**: Affordable GPU rentals
- **Google Cloud Platform**: GPU-enabled containers
- **AWS**: EC2 instances with GPU
- **Lambda Labs**: GPU cloud with ML focus

### Option 3: Hybrid Approach
- Deploy a lightweight proxy on Koyeb
- Route heavy inference to GPU backend service
- Use Koyeb for API management and job queuing

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (will use GPU if available)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Force CPU mode for testing
KOYEB_DEPLOYMENT=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Performance Expectations

| Environment | Performance | Feasibility |
|-------------|-------------|-------------|
| Koyeb (CPU) | Very Slow (minutes per frame) | ❌ Not Practical |
| GPU Cloud (A100) | Fast (seconds per video) | ✅ Recommended |
| Local GPU (RTX 4090) | Good (30s-2min per video) | ✅ Viable |

## Recommendations

1. **Do not deploy on Koyeb for production** - CPU-only performance is impractical
2. **Use Koyeb as API gateway** - Route to GPU backend for actual inference
3. **Consider alternative models** - Smaller text-to-video models that can run on CPU
4. **Implement job queuing** - Queue requests and process on GPU instances

## Alternative Solutions

If you must use Koyeb, consider these alternatives:
- **Stable Video Diffusion**: Smaller image-to-video model
- **Text-to-Image + Video Interpolation**: Generate image first, then animate
- **Pre-generated Content**: Serve pre-made videos based on prompts
- **Proxy to External API**: Route to OpenAI, Runway, or other video APIs