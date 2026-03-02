# Use the NVIDIA CUDA runtime as the base
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Set non-interactive frontend to skip timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    tzdata \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libpq-dev \
    gcc \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 (since distutils is gone, we use the bootstrap script)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create symlink so 'python' and 'python3' point to 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the rest of the code
COPY . .

# Default command
CMD ["uv", "run", "python", "arxiv_client.py"]