FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY app/ ./app/

# Config (MODEL_NAME, MODEL_PATH) is injected at runtime via Cloud Run env vars.
# For local development, create a .env file from .env.example.

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
