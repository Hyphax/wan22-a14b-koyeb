# Koyeb Deployment Guide

## Pre-Deployment Checklist

⚠️ **Read This First**: This deployment will work but with severe limitations. See ANALYSIS.md for details.

### Requirements Met:
- ✅ Koyeb-compatible base image (Python slim)
- ✅ CPU-only PyTorch dependencies
- ✅ Automatic environment detection
- ✅ Fallback error handling
- ✅ Resource monitoring

### Limitations:
- ❌ Extremely slow performance (CPU-only)
- ❌ Large model may not load (memory constraints)
- ❌ Generation quality will be poor
- ❌ High timeout risk

## Deployment Steps

### 1. Koyeb Configuration

```yaml
# koyeb.yaml (if using Koyeb CLI)
services:
  - name: wan22-api
    git:
      url: https://github.com/your-username/wan22-a14b-koyeb
      branch: main
    instance_type: small  # or medium for more memory
    ports:
      - port: 8000
        protocol: http
    env:
      - key: KOYEB_DEPLOYMENT
        value: "1"
      - key: WARMUP
        value: "0"  # Disable warmup to reduce startup time
      - key: WAN_MODEL_ID
        value: "stabilityai/stable-video-diffusion-img2vid"  # Smaller fallback
    health_check:
      http:
        port: 8000
        path: /healthz
```

### 2. Environment Variables

Set these in Koyeb dashboard:

| Variable | Value | Purpose |
|----------|-------|---------|
| `KOYEB_DEPLOYMENT` | `1` | Enable CPU mode |
| `WARMUP` | `0` | Skip model preloading |
| `MODELS_DIR` | `/tmp/models` | Model cache (ephemeral) |
| `OUT_DIR` | `/tmp/outputs` | Output directory |

### 3. Expected Behavior

**Startup (2-5 minutes):**
- Container builds and starts
- Dependencies install
- Health check becomes ready
- Model loading skipped (WARMUP=0)

**First Request (5-15 minutes):**
- Model download begins
- May timeout or fail due to size
- Falls back to smaller model or dummy response

**Subsequent Requests:**
- Very slow generation (10+ minutes per video)
- Frequent timeouts
- Limited quality

## Testing the Deployment

1. **Check Health**:
   ```bash
   curl https://your-app.koyeb.app/healthz
   ```

2. **Test API**:
   ```bash
   python3 test_api.py https://your-app.koyeb.app
   ```

3. **Monitor Resources**:
   ```bash
   curl https://your-app.koyeb.app/healthz | jq '.resources'
   ```

## Troubleshooting

### Common Issues:

1. **Startup Timeout**:
   - Set `WARMUP=0`
   - Use smaller model in `WAN_MODEL_ID`
   - Increase instance size

2. **Memory Errors**:
   - Models too large for available RAM
   - Will fallback to dummy responses
   - Consider alternative models

3. **Generation Timeout**:
   - CPU inference is extremely slow
   - Reduce frame count and resolution
   - Use caching where possible

4. **Model Download Fails**:
   - Network timeout during download
   - Will attempt fallback models
   - Check health endpoint for status

## Alternative Deployment Options

If Koyeb deployment fails or performs poorly:

1. **GPU Cloud Services**:
   - RunPod: `runpod.io`
   - Vast.ai: `vast.ai`
   - Lambda Labs: `lambdalabs.com`

2. **Hybrid Setup**:
   - Keep Koyeb as API gateway
   - Route inference to GPU backend
   - Use job queuing system

3. **Local Development**:
   ```bash
   # Run locally with GPU
   docker build -t wan22 .
   docker run --gpus all -p 8000:8000 wan22
   ```

## Monitoring & Maintenance

- Monitor memory usage via health endpoint
- Set up alerts for high error rates
- Consider scaling down during low usage
- Regular health checks recommended

## Support

For issues with this deployment:
1. Check ANALYSIS.md for technical details
2. Review health endpoint output
3. Consider GPU cloud alternatives
4. Contact for deployment assistance