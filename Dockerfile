FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODELS_DIR=/models \
    OUT_DIR=/data/outputs \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
    HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git ca-certificates libgl1 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    mkdir -p ${MODELS_DIR} ${OUT_DIR}

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade huggingface_hub==0.25.2
RUN pip install --no-cache-dir "git+https://github.com/huggingface/diffusers"

COPY app /app/app
COPY start.sh /app/start.sh
EXPOSE 8000
CMD ["/bin/bash", "/app/start.sh"]
