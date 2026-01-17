FROM python:3.11-slim

# Install system dependencies and Node.js (for npx)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv (for uvx)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /bin/uvx

# Set working directory
WORKDIR /workspace

# Default command
CMD ["sleep", "infinity"]
