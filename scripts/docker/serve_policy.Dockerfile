# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Needed because LeRobot uses git-lfs.
RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
# sam-2 is a viz-tooling dep that doesn't compile inside this CUDA-runtime image
# (no nvcc) and isn't used by serve_policy.py. Strip it from pyproject.toml +
# uv.lock for the in-container sync, then run unfrozen so deps re-resolve.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/tmp/uv.lock.src,rw=false \
    --mount=type=bind,source=pyproject.toml,target=/tmp/pyproject.src.toml,rw=false \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    cp /tmp/pyproject.src.toml /app/pyproject.toml && \
    cp /tmp/uv.lock.src /app/uv.lock && \
    sed -i '/^[[:space:]]*"sam-2",/d' /app/pyproject.toml && \
    sed -i '/^sam-2 = { path = "third_party\/sam2" }/d' /app/pyproject.toml && \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --no-install-project --no-dev && \
    rm -f /app/pyproject.toml /app/uv.lock

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
