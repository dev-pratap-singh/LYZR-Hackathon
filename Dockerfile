# Backend Dockerfile for Railway Deployment
# This Dockerfile is placed in the root directory but builds the backend service

FROM python:3.12-slim

WORKDIR /app

# Install uv for faster dependency installation
RUN pip install --no-cache-dir uv

# Copy backend requirements
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the entire backend directory
COPY backend/ ./

# Copy memory submodule (if needed)
COPY memory/ ../memory/

# Expose port (Railway will set $PORT dynamically)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Start command - Railway provides $PORT environment variable
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
