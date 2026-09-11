FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (OpenCV needs libgl)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV INTERSTELLAR_API_HOST="0.0.0.0"

EXPOSE 8000 8501

# Note: The entrypoint will be defined in docker-compose
