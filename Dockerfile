FROM python:3.11-slim

# System dependencies for scipy, pandas, rapidfuzz C extensions
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker layer caching — only rebuilds on requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY . .

# HuggingFace Spaces listens on 7860
EXPOSE 7860

# Health check — HF uses this to know when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Single worker: HF free tier is 2vCPU/8GB — pandas DataFrames in-memory
# async endpoints handle concurrent requests without multiple workers
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]