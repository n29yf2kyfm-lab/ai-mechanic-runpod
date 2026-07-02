# AI MECHANIC ULTIMATE v10.0
# Docker Hub: alamk123/ai-mechanic-runpod:latest
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/cache
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 ffmpeg libsndfile1 imagemagick \
    && rm -rf /var/lib/apt/lists/*

# vLLM FIRST — brings its own compatible torch, avoids version conflict
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    vllm==0.5.5 sentencepiece protobuf

# Remaining ML packages (torchvision/torchaudio auto-match vLLM's torch)
RUN pip install --no-cache-dir \
    torchvision torchaudio \
    transformers==4.44.0 accelerate bitsandbytes \
    scipy numpy pillow opencv-python \
    soundfile librosa \
    trimesh einops timm \
    kokoro onnxruntime-gpu \
    requests beautifulsoup4 \
    runpod

# NOTE: TripoSR is NOT installed here.
# It requires compiling CUDA extensions (torchmcubes), which needs an actual
# GPU — not available on GitHub Actions build runners. It is installed lazily
# at runtime on the RunPod GPU worker instead, inside handler.py's
# get_tripo_sr() function, the first time image_to_3d/generate_3d is called.

# Copy handler files
COPY handler.py /workspace/handler.py
COPY obd_database.py /workspace/obd_database.py

# RunPod serverless entry point
CMD ["python3", "-u", "handler.py"]
