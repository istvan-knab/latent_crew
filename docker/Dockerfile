FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

# System libraries required by OpenCV and Open3D
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglib2.0-0 libgomp1 libgfortran5 libice6 libsm6 libidn2-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The venv lives outside /app so the compose volume mounts do not hide it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_FROZEN=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock .python-version ./
# The uv cache is mounted only during the build, so it does not end up in the image
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Make the project venv the default python inside the container
ENV PATH="/opt/venv/bin:$PATH"

CMD ["bash"]
