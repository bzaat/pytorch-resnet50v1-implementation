# =============================================================================
# ResNet-50 on Oxford-IIIT Pet: training image (dependencies managed with uv)
#
# The image holds only code and dependencies. The dataset, checkpoints and
# TensorBoard logs stay on the host and are mounted at run time.
# Dependencies come from pyproject.toml and uv.lock, so commit both.
#
# Build (default PyPI PyTorch wheels, usable with an NVIDIA GPU):
#   docker build -t resnet50-pet .
#
# Build (CPU-only PyTorch, much smaller image):
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t resnet50-pet:cpu .
#
# Create the host folders first, otherwise Docker creates them owned by root:
#   mkdir -p data trained_model tensorboard
#
# Train from scratch. Arguments after the image name go to oxiii_pet.model.train.
# Drop --gpus all to train on the CPU. --shm-size is required because the
# DataLoader uses 4 worker processes (Docker's default 64 MB is too small):
#   docker run --rm --gpus all --shm-size=2g \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/trained_model:/app/trained_model" \
#     -v "$(pwd)/tensorboard:/app/tensorboard" \
#     resnet50-pet --epochs 50 --batch_size 32
#
# Transfer learning (pretrained ResNet-50). The ImageNet weights are downloaded
# on first use, so the container needs internet access:
#   docker run --rm --gpus all --shm-size=2g \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/trained_model:/app/trained_model" \
#     --entrypoint python resnet50-pet \
#     -m oxiii_pet.transfer_learning.train -n 37 --epoch 10 --batch_size 32
#
# TensorBoard on http://localhost:6006 :
#   docker run --rm -p 6006:6006 -v "$(pwd)/tensorboard:/app/tensorboard" \
#     --entrypoint tensorboard resnet50-pet --logdir /app/tensorboard --host 0.0.0.0
# =============================================================================

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

LABEL org.opencontainers.image.title="resnet50-pet" \
      org.opencontainers.image.source="https://github.com/bzaat/pytorch-resnet50v1-implementation" \
      org.opencontainers.image.licenses="MIT"

# The virtual environment lives outside /app so that bind-mounting ./src or
# ./data over /app never hides it. Putting it first on PATH means plain
# `python` and `tensorboard` use it, and uv is only needed at build time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MPLCONFIGDIR=/tmp/matplotlib \
    DATA_DIR=/app/data

# Shared libraries that OpenCV needs when the dependencies include
# opencv-python (not -headless). Remove this layer if you switch to -headless.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# uv itself, installed from PyPI (pin a version for reproducible builds).
ARG UV_VERSION=0.11.32
RUN pip install "uv==${UV_VERSION}"

# Run as a non-root user. Match your host user so files written to the mounted
# folders are owned by you: --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} appuser \
 && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home appuser

WORKDIR /app

# 1) Dependencies first, so this layer is cached until pyproject.toml or
#    uv.lock changes.
#    Default: install exactly what uv.lock pins (the PyPI PyTorch build).
#    If TORCH_INDEX_URL is set (for example the CPU-only index), PyTorch comes
#    from that index and everything else is still taken from uv.lock; the CUDA
#    libraries that uv.lock lists for the PyPI build are left out.
ARG TORCH_INDEX_URL=
COPY pyproject.toml uv.lock ./
RUN if [ -n "${TORCH_INDEX_URL}" ]; then \
        uv venv "${UV_PROJECT_ENVIRONMENT}" \
     && uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
            torch torchvision --index "${TORCH_INDEX_URL}" \
     && uv export --frozen --no-hashes --no-annotate --no-emit-project \
            --no-emit-package torch --no-emit-package torchvision \
        | grep -vE '^(nvidia-|cuda-|triton)' > /tmp/requirements.txt \
     && uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
            -r /tmp/requirements.txt \
     && rm /tmp/requirements.txt; \
    else \
        uv sync --locked --no-install-project; \
    fi

# 2) The project itself, installed in editable mode, so
#    `python -m oxiii_pet.model.train` works from any directory and a
#    bind-mounted ./src is picked up without a rebuild.
#    README.md is copied because pyproject.toml uses it as the package readme.
COPY README.md ./
COPY src ./src
COPY test_images ./test_images
RUN if [ -n "${TORCH_INDEX_URL}" ]; then \
        uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --no-deps -e .; \
    else \
        uv sync --locked; \
    fi

# Folders the training scripts write to (relative to /app) and the dataset mount point.
RUN mkdir -p /app/data /app/trained_model /app/tensorboard \
 && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m", "oxiii_pet.model.train"]
CMD ["--epochs", "50", "--batch_size", "32"]
