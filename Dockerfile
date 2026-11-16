
FROM python:3.11-slim

WORKDIR /app

# Install Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

ENV PYTHONUNBUFFERED=1

# Start Flask through Gunicorn
CMD ["sh", "-c", "gunicorn --workers 1 --threads 2 --timeout 180 --graceful-timeout 30 --bind 0.0.0.0:${PORT:-10000} app:app"]
