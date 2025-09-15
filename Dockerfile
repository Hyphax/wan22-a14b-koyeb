FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODELS_DIR=/tmp/models \
    OUT_DIR=/tmp/outputs \
    HF_HOME=/tmp/models \
    HUGGINGFACE_HUB_CACHE=/tmp/models \
    KOYEB_DEPLOYMENT=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    mkdir -p ${MODELS_DIR} ${OUT_DIR} && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade huggingface_hub==0.25.2
RUN pip install --no-cache-dir "git+https://github.com/huggingface/diffusers"

COPY app /app/app
COPY start.sh /app/start.sh
EXPOSE 8000
CMD ["/bin/bash", "/app/start.sh"]
