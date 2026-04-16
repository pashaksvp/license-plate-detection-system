FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install numpy==2.2.0 torch torchvision ultralytics opencv-python-headless wandb pyyaml

COPY src/ ./src/
COPY train/ ./train/

RUN mkdir -p /app/data

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "license_plate_detector.cli"]
CMD ["--help"]

