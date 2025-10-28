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
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

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
