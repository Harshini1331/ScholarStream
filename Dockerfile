# Use a stable Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for PostgreSQL
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv to manage our packages
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy our project files
COPY pyproject.toml uv.lock ./

# Install dependencies into the container
RUN uv sync --frozen

# Copy the rest of the code
COPY . .

# Command to run our test
CMD ["uv", "run", "heart.py"]