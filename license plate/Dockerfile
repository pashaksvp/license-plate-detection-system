# =========================================================
# Dockerfile для License Plate Detector
# =========================================================
#
# Что такое Docker и зачем он нужен:
# Docker позволяет упаковать приложение со ВСЕМИ зависимостями
# в изолированный контейнер. Контейнер запустится одинаково
# на любой машине — Windows, Mac, Linux, облачный сервер.
#
# Сборка образа:
#   docker build -t license-plate-detector .
#
# Запуск (режим VIDEO):
#   docker run -v $(pwd)/data:/app/data license-plate-detector \
#     video --input data/video.mp4 --output data/result.mp4
#
# Запуск (режим CAMERA — нужен X11 или VNC):
#   docker run --device=/dev/video0 license-plate-detector camera
# =========================================================

# Базовый образ: Python 3.11 на лёгкой сборке Debian
FROM python:3.11-slim

# Метаданные образа
LABEL maintainer="Your Name <your@email.com>"
LABEL description="License Plate Detector based on YOLOv8"
LABEL version="1.0.0"

# ---- Переменные окружения ----
ENV PYTHONDONTWRITEBYTECODE=1  \
    # не писать .pyc файлы (не нужны в контейнере)
    PYTHONUNBUFFERED=1         \
    # вывод Python сразу (не буферизовать)
    PIP_NO_CACHE_DIR=1

# ---- Установка системных зависимостей ----
# OpenCV требует libGL и libGLib для работы с видео
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
    # Очищаем кэш apt — уменьшаем размер образа

# ---- Рабочая директория ----
WORKDIR /app

# ---- Устанавливаем Poetry ----
RUN pip install poetry==1.7.1

# ---- Копируем файлы зависимостей (отдельным слоем для кэширования) ----
# Docker кэширует каждый слой. Если pyproject.toml не изменился,
# зависимости не будут переустанавливаться при повторной сборке.
COPY pyproject.toml poetry.lock* ./

# ---- Устанавливаем зависимости через Poetry ----
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev
    # virtualenvs.create false — устанавливаем прямо в системный Python
    # --without dev — не устанавливаем dev-зависимости (pytest, black)

# ---- Копируем исходный код ----
COPY src/ ./src/
COPY train/ ./train/

# ---- Создаём директорию для логов ----
# Эта директория будет доступна как volume для просмотра логов
RUN mkdir -p /app/data

# ---- Копируем веса модели (если есть) ----
# Если файл весов существует — копируем его в образ
# Если нет — модель скачается при первом запуске
COPY runs/ ./runs/ 2>/dev/null || true

# ---- Точка входа ----
# ENTRYPOINT — команда, которая всегда выполняется при запуске контейнера
# CMD — аргументы по умолчанию (можно переопределить при docker run)
ENTRYPOINT ["python", "-m", "license_plate_detector.cli"]
CMD ["--help"]
