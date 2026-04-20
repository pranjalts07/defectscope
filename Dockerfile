FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights must exist before building — run training scripts first.
# In CI/CD, pull from S3: aws s3 cp s3://your-bucket/models/ models/ --recursive
# (see docs/deployment.md)

EXPOSE 8000

# Use 1 worker for GPU inference — multiple workers don't share GPU memory cleanly.
# For CPU deployments, increase workers for throughput.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
