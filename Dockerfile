FROM pytorch/pytorch:2.12.0-cuda12.4-cudnn9-runtime

LABEL maintainer="Sreenath Kyathanahally"
LABEL description="Reproducible environment for corrected cross-vendor qMRI domain-shift benchmark"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (exact pins)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default: run verification (no GPU needed)
CMD ["python3", "scripts/verify_paper.py"]
