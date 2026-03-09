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

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create symlinks
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml ./
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create venv
RUN uv venv /opt/venv

# Step 1: Install Airflow with its official constraints (resolves structlog + all pins)
RUN uv pip install --python /opt/venv/bin/python \
    "apache-airflow==3.1.7" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.7/constraints-3.12.txt"

# Step 2: Install the rest of your app dependencies on top
RUN uv pip install --python /opt/venv/bin/python \
    "arxiv>=2.4.0" \
    "docling>=2.75.0" \
    "httpx>=0.28.1" \
    "langchain>=1.2.10" \
    "langchain-community>=0.4.1" \
    "langchain-ollama>=1.0.1" \
    "opensearch-py>=3.1.0" \
    "psycopg2-binary>=2.9.11" \
    "requests>=2.32.5" \
    "uvicorn>=0.41.0" \
    "fastapi>=0.116.0,<0.118.0" \
    "grpcio!=1.78.1" \
    "asyncpg>=0.29.0"

# Copy the rest of the code
COPY . .

# Default command
CMD ["uv", "run", "python", "arxiv_client.py"]