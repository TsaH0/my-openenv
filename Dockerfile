# SQL Query Learning Environment - Dockerfile
# Compatible with Hugging Face Spaces (port 7860, non-root user)

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
COPY pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY client.py .
COPY openenv.yaml .
COPY inference.py .
COPY server/ ./server/

# Ensure server package is importable
RUN touch server/__init__.py

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables (overridden at runtime)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
ENV PYTHONPATH="/app"

# Expose Hugging Face Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
