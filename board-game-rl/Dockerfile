FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Install prerequisites for uv and general utilities
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install uv using the standalone installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure uv is on PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./


# Copy the rest of the application
COPY src/ ./src/
# Install dependencies using uv. uv will automatically download the required Python 3.13!
RUN uv sync --frozen --no-dev

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 7860

CMD ["./start.sh"]
