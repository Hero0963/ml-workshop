# .devcontainer/Dockerfile

# --- Stage 1: Build the Svelte Frontend ---
# Use the same base image as the original frontend Dockerfile for consistency
FROM node:lts-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY src/custom_components/puzzle_editor/frontend/package.json ./
COPY src/custom_components/puzzle_editor/frontend/package-lock.json ./
RUN npm install

# Copy the rest of the frontend source code
COPY src/custom_components/puzzle_editor/frontend/ ./

# Build the frontend
# The vite.config.ts is expected to already have the correct 'base' path.
RUN npm run build

# --- Stage 2: Final Python Application ---
# Plain Python, not a CUDA image: this container never uses the GPU (the vision model runs
# in the ollama container and RL inference is device="cpu"). The previous base,
# pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel, carried ~17GB of conda and CUDA toolkit that
# nothing here read, on top of the torch `uv sync` installs anyway. The Debian release is
# in the tag so that an OS upgrade is an edit, not a side effect of rebuilding.
FROM python:3.11-slim-trixie

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency description files
COPY pyproject.toml uv.lock pytest.ini ./

# Install all dependencies
RUN uv sync

# Copy the application source code
COPY src ./src

# Copy the built frontend from the builder stage
COPY --from=frontend-builder /app/frontend/dist ./src/custom_components/puzzle_editor/frontend/dist

# Set default environment variable for the port
ENV APP_PORT=7440

# Expose the container port
EXPOSE 7440

# Without this the image builds and the container does nothing: the base image's default
# command is a shell, so `docker compose up` used to report a healthy-looking container
# with no server in it. Shell form on purpose -- APP_PORT has to be expanded at runtime,
# and compose passes it through `env_file`.
CMD ["sh", "-c", "uv run uvicorn src.app.main:app --host 0.0.0.0 --port ${APP_PORT:-7440}"]
