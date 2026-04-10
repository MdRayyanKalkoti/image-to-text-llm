# Use official Python 3.11 slim image
FROM python:3.11-slim

# Install tesseract and system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer cache — only re-installs if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . .

# Create uploads folder
RUN mkdir -p uploads

# Expose port
EXPOSE 10000

# Start gunicorn
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1