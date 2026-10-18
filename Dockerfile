# syntax=docker/dockerfile:1
FROM python:3.11-slim AS cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src MUJOCO_GL=egl \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    XDG_CACHE_HOME=/cache WARP_CACHE_PATH=/cache/warp
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglib2.0-0 libosmesa6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/cube
COPY pyproject.toml ./
COPY src/cube/__init__.py src/cube/__init__.py
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir '.[dev]'
WORKDIR /workspace
CMD ["python", "-m", "cube", "--help"]

# Optional CUDA PyTorch build; Warp CUDA also works in the CPU image with GPU access.
FROM cpu AS cuda
RUN pip install --no-cache-dir --force-reinstall --no-deps torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
