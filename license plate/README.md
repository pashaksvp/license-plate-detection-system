# license plate detection system

> Система автоматического обнаружения номерных знаков на базе **YOLOv8**

**Студент:** Фомин Павел Андреевич 
**Группа:** 972402

---

## 📋 Содержание

- [Описание проекта](#описание)
- [Структура репозитория](#структура)
- [Установка](#установка)
- [Создание датасета](#датасет)
- [Обучение модели](#обучение)
- [Оценка качества](#оценка)
- [Запуск детекции](#запуск)
- [Docker](#docker)
- [Логирование](#логирование)
- [Ресурсы](#ресурсы)

---

## 📖 Описание

Проект реализует систему **ALPR (Automatic License Plate Recognition)** —
автоматического распознавания номерных знаков. Система основана на архитектуре
**YOLOv8** (You Only Look Once), дообученной (fine-tuned) на собственном датасете.

### Возможности

- 🎬 Обработка видеофайлов с сохранением аннотированного результата
- 📷 Детекция в реальном времени через веб-камеру
- 🐳 Развёртывание через Docker без настройки окружения
- 📊 Логирование всех событий в файл
- 🔬 Мониторинг обучения через Weights & Biases

---

## 🗂 Структура репозитория

```
license-plate-detector/
│
├── src/
│   └── license_plate_detector/
│       ├── __init__.py         # публичный API пакета
│       ├── model_impl.py       # главный класс My_LicensePlate_Model
│       ├── cli.py              # CLI-приложение (video / camera)
│       └── logger.py           # Singleton-логгер
│
├── train/
│   └── train.py                # скрипт обучения с Wandb
│
├── tests/
│   └── test_model.py           # юнит-тесты (pytest)
│
├── data/
│   ├── dataset.yaml            # конфиг датасета
│   └── log_file.log            # файл логов (создаётся автоматически)
│
├── runs/                       # результаты обучения (создаётся автоматически)
│   └── detect/train/weights/
│       ├── best.pt             # лучшие веса модели
│       └── last.pt             # веса последней эпохи
│
├── Dockerfile                  # образ Docker
├── docker-compose.yaml         # оркестрация контейнеров
├── pyproject.toml              # Poetry — зависимости и настройки
└── .gitignore
```

---

## ⚙️ Установка

### Вариант 1: Poetry (рекомендуется)

```bash
# 1. Клонируем репозиторий
git clone https://github.com/YOUR_USERNAME/license-plate-detector.git
cd license-plate-detector

# 2. Устанавливаем Poetry (если нет)
pip install poetry

# 3. Устанавливаем зависимости
poetry install

# 4. Активируем виртуальное окружение
poetry shell
```

### Вариант 2: pip + .whl

```bash
# Устанавливаем пакет из .whl файла
pip install license_plate_detector-1.0.0-py3-none-any.whl
```

### Вариант 3: Google Colab (если нет GPU)

```python
# В ячейке Colab:
!git clone https://github.com/YOUR_USERNAME/license-plate-detector.git
%cd license-plate-detector
!pip install ultralytics opencv-python wandb
```

### Установка PyTorch с CUDA (для GPU)

```bash
# CUDA 12.8 (последняя версия)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Проверка GPU:
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📊 Создание датасета

> 💡 **Собственный датасет даёт +15 баллов по заданию**

### Шаг 1: Сбор видео

Запишите видео с дороги или используйте публичные источники:
- Собственная запись на телефон (15-30 минут)
- YouTube (публичные записи dashcam)
- Kaggle datasets

### Шаг 2: Извлечение кадров

```bash
# Извлекаем 2 кадра в секунду из видео
ffmpeg -i your_video.mp4 -vf fps=2 data/images/raw/frame_%04d.jpg

# Разбиваем на train (80%) и val (20%)
python -c "
import os, shutil, random
from pathlib import Path

frames = list(Path('data/images/raw').glob('*.jpg'))
random.shuffle(frames)

split = int(len(frames) * 0.8)
train_frames = frames[:split]
val_frames = frames[split:]

for f in train_frames:
    shutil.copy(f, 'data/images/train/')
for f in val_frames:
    shutil.copy(f, 'data/images/val/')

print(f'Train: {len(train_frames)}, Val: {len(val_frames)}')
"
```

### Шаг 3: Разметка в Roboflow

1. Зайдите на [roboflow.com](https://roboflow.com) → Create Project → Object Detection
2. Загрузите кадры из `data/images/`
3. Для каждого кадра обведите номерные знаки рамкой
4. Укажите класс: `license_plate`
5. Экспортируйте: **Format → YOLOv8** → Download ZIP
6. Распакуйте в папку `data/`

### Шаг 4: Проверка структуры

```
data/
├── dataset.yaml
├── images/
│   ├── train/   # ~800+ изображений
│   └── val/     # ~200+ изображений
└── labels/
    ├── train/   # .txt файлы с разметкой
    └── val/
```

---

## 🧠 Обучение модели

### Базовый запуск

```bash
python train/train.py \
  --data data/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --device cuda
```

### С логированием в Wandb

```bash
# Сначала авторизуйтесь
wandb login

# Затем запустите обучение с флагом --wandb
python train/train.py \
  --data data/dataset.yaml \
  --epochs 100 \
  --batch 32 \
  --device cuda \
  --wandb
```

### Выбор модели по размеру

| Модель | Параметры | Скорость | Точность | Рекомендация |
|--------|-----------|----------|----------|--------------|
| yolov8n.pt | 3.2M | Очень быстро | Базовая | Google Colab / CPU |
| yolov8s.pt | 11.2M | Быстро | Хорошая | GPU 4GB+ |
| yolov8m.pt | 25.9M | Средне | Высокая | GPU 8GB+ |
| yolov8l.pt | 43.7M | Медленно | Очень высокая | GPU 16GB+ |

### Параметры обучения

```bash
python train/train.py --help
```

```
  --data      Путь к dataset.yaml
  --model     Базовая модель (yolov8n.pt по умолчанию)
  --epochs    Количество эпох (50 по умолчанию)
  --batch     Размер батча (16 по умолчанию, уменьшите при OOM)
  --device    cpu / cuda / 0 (номер GPU)
  --wandb     Включить логирование в Wandb
  --patience  Early stopping (20 по умолчанию)
```

После обучения веса сохранятся в:
```
runs/detect/train/weights/best.pt   ← лучшая модель
runs/detect/train/weights/last.pt   ← последняя эпоха
```

---

## 📈 Оценка качества модели

```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
metrics = model.val(data='data/dataset.yaml')
print(f'mAP@0.5:      {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision:    {metrics.box.mp:.4f}')
print(f'Recall:       {metrics.box.mr:.4f}')
"
```

### Критерии оценки (из задания)

| mAP@0.5 | Баллы | Уровень |
|---------|-------|---------|
| > 0.8 | 15 | 🏆 S-Tier |
| 0.6 – 0.8 | 15 | ✅ Good |
| 0.4 – 0.6 | 10 | 📊 Baseline |
| < 0.4 | 0 | ❌ Poor |

---

## 🚀 Запуск детекции

### Режим VIDEO

```bash
# Обработка видеофайла
detect video --input road_video.mp4 --output result.mp4

# С отображением в реальном времени
detect video --input road_video.mp4 --output result.mp4 --show

# С указанием весов и порога уверенности
detect video \
  --input road_video.mp4 \
  --output result.mp4 \
  --weights runs/detect/train/weights/best.pt \
  --conf 0.5
```

### Режим CAMERA (веб-камера)

```bash
# Стандартная веб-камера (ID=0)
detect camera

# Вторая камера + сохранение
detect camera --camera-id 1 --save camera_result.mp4

# С кастомными весами
detect camera --weights runs/detect/train/weights/best.pt --conf 0.4
```

### Использование как Python-библиотека

```python
import cv2
import numpy as np
from license_plate_detector import My_LicensePlate_Model

# Инициализация модели
model = My_LicensePlate_Model(
    weights_path="runs/detect/train/weights/best.pt",
    confidence_threshold=0.4,
    device="cpu",  # или "cuda"
)

# Детекция на изображении
frame = cv2.imread("car.jpg")
detections = model.detect_plates(frame)

# Результат — список словарей:
for plate in detections:
    print(plate)
    # {'x1': 245, 'y1': 380, 'x2': 410, 'y2': 420,
    #  'confidence': 0.934, 'class_name': 'license_plate'}

# Визуализация
annotated = model.draw_detections(frame, detections)
cv2.imshow("Result", annotated)
cv2.waitKey(0)
```

---

## 🐳 Docker

### Сборка образа

```bash
docker build -t license-plate-detector .
```

### Режим VIDEO через docker-compose

```bash
# Положите видео в ./data/input.mp4
docker-compose run detector-video

# Результат появится в ./data/output.mp4
# Логи — в ./data/log_file.log
```

### Режим CAMERA через docker-compose

```bash
# На Linux (нужна камера /dev/video0)
xhost +local:docker
docker-compose run detector-camera
```

### Ручной запуск через docker run

```bash
# VIDEO
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  license-plate-detector \
  video --input /app/data/input.mp4 --output /app/data/output.mp4

# Просмотр логов внутри контейнера
docker run -v $(pwd)/data:/app/data license-plate-detector \
  sh -c "cat /app/data/log_file.log"
```

---

## 📝 Логирование

Все события логируются в `./data/log_file.log`:

```
[2026-04-07 12:00:01] [INFO]  ============================================================
[2026-04-07 12:00:01] [INFO]  Запуск License Plate Detector
[2026-04-07 12:00:01] [INFO]  ============================================================
[2026-04-07 12:00:01] [INFO]  Режим работы: VIDEO
[2026-04-07 12:00:01] [INFO]  Инициализация модели. Веса: best.pt, устройство: cpu
[2026-04-07 12:00:02] [INFO]  Модель успешно загружена.
[2026-04-07 12:00:02] [INFO]  Открываем видеофайл: road.mp4
[2026-04-07 12:00:02] [INFO]  Параметры видео: 1920x1080, 30 FPS, 900 кадров
[2026-04-07 12:00:05] [DEBUG] Обнаружено объектов: 2
[2026-04-07 12:00:35] [INFO]  Готово! Обработано 900 кадров, обнаружено 1247 номерных знаков.
```

Логгер реализован по паттерну **Singleton** — существует только один экземпляр
на всё приложение, что исключает дублирование записей.

---

## 🧪 Тесты

```bash
# Запустить все тесты
pytest tests/ -v

# С отчётом о покрытии
pytest tests/ -v --cov=license_plate_detector
```

---

## 📦 Сборка пакета

```bash
# Создать .whl файл для распространения
poetry build

# Файл появится в dist/
ls dist/
# license_plate_detector-1.0.0-py3-none-any.whl
```

---

## 🌿 Git Workflow

```
main   ← стабильная версия (только merge из dev)
dev    ← активная разработка

# Работа с ветками:
git checkout -b dev           # создать dev ветку
git add .
git commit -m "feat: add video processing mode"
git push origin dev

# После проверки — merge в main:
git checkout main
git merge dev
git push origin main
```

---

## 📚 Ресурсы

### Документация и статьи

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/) — официальная документация
- [YOLO: нейросеть для обнаружения объектов (Habr)](https://habr.com/ru/articles/821971/)
- [Как работает YOLO (Timeweb Blog)](https://timeweb.cloud/blog/yolo-neyroset-obnaruzhenie-obektov)
- [Обзор YOLO (Data Secrets)](https://datasecrets.ru/articles/20)

### Инструменты разметки

- [Roboflow](https://roboflow.com) — онлайн-разметка, экспорт в YOLOv8
- [Label Studio](https://labelstud.io) — self-hosted разметка
- [CVAT](https://cvat.ai) — профессиональный инструмент разметки

### Датасеты (готовые)

- [License Plate Detection — Roboflow Universe](https://universe.roboflow.com/search?q=license+plate)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)

### Мониторинг обучения

- [Weights & Biases](https://wandb.ai) — логирование метрик, графики, артефакты
- [ClearML](https://clear.ml) — альтернатива Wandb

### Предобученные веса

- YOLOv8n: скачивается автоматически через `YOLO("yolov8n.pt")`
- Дообученные веса проекта: `runs/detect/train/weights/best.pt`
