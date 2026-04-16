# License Plate Detector

**Студент:** Фомин Павел Андреевич
**Группа:** 972402  

Проект по домашнему заданию №2 — детекция номерных знаков с помощью YOLOv8.

---

## Что сделано

Написал систему которая находит номерные знаки на видео и в потоке с камеры. За основу взял архитектуру YOLOv8 от Ultralytics, дообучил на датасете с номерами.

Два режима работы:
- **video** — обрабатывает видеофайл и сохраняет результат
- **camera** — работает с веб-камерой в реальном времени

---

## Результаты обучения

Обучал на датасете License Plate Recognition (7058 изображений для train, 2048 для val).

| Метрика | Значение |
|---------|----------|
| mAP@0.5 | 0.952 |
| mAP@0.5:0.95 | 0.607 |
| Precision | 0.959 |
| Recall | 0.923 |

Уже после первой эпохи mAP вышел на 0.95 — датасет оказался качественным.

---

## Структура проекта

```
license plate/
├── src/
│   └── license_plate_detector/
│       ├── __init__.py
│       ├── model_impl.py     # класс My_LicensePlate_Model
│       ├── cli.py            # CLI приложение
│       └── logger.py         # логгер (Singleton)
├── train/
│   └── train.py              # скрипт обучения
├── tests/
│   └── test_model.py         # тесты (11 штук)
├── data/
│   └── log_file.log          # логи
├── dist/
│   └── license_plate_detector-1.0.0-py3-none-any.whl
├── Dockerfile
├── docker-compose.yaml
└── pyproject.toml
```

---

## Установка

Нужен Python 3.10+ и Poetry.

```bash
# Клонировать репозиторий
git clone https://github.com/pashaksvp/license-plate-detector.git
cd license-plate-detector

# Установить зависимости
poetry install

# Активировать окружение
eval $(poetry env activate)
```

Или установить через whl:

```bash
pip install dist/license_plate_detector-1.0.0-py3-none-any.whl
```

---

## Как запустить обучение

Сначала нужен датасет в формате YOLOv8. Я брал с Roboflow Universe.

```bash
python train/train.py \
  --data data/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --device cuda
```

На CPU работает но медленно, лучше использовать GPU или Google Colab.

После обучения веса сохранятся в `runs/train/weights/best.pt`.

---

## Как запустить детекцию

### Режим video

```bash
PYTHONPATH=src python -m license_plate_detector.cli \
  --weights runs/train/weights/best.pt \
  --conf 0.4 \
  video \
  --input video.mp4 \
  --output result.mp4 \
  --show
```

## Как запустить тесты

```bash
pytest tests/ -v
```

Все 11 тестов проходят. Проверяют формат вывода, типы данных, работу логгера и устойчивость к неверным входным данным.

---

## Использование как библиотека

```python
import cv2
from license_plate_detector import My_LicensePlate_Model

model = My_LicensePlate_Model(
    weights_path="runs/train/weights/best.pt",
    confidence_threshold=0.4,
)

frame = cv2.imread("car.jpg")
detections = model.detect_plates(frame)

for plate in detections:
    print(plate)
    # {'x1': 245, 'y1': 380, 'x2': 410, 'y2': 420,
    #  'confidence': 0.934, 'class_name': 'license_plate'}
```

---

## Логирование

Все события пишутся в `./data/log_file.log`:

```
[2026-04-13 16:49:04] [INFO] Запуск License Plate Detector
[2026-04-13 16:49:04] [INFO] Режим работы: VIDEO
[2026-04-13 16:49:04] [INFO] Инициализация модели. Веса: best.pt, устройство: cpu
[2026-04-13 16:49:04] [INFO] Модель успешно загружена.
[2026-04-13 16:49:04] [INFO] Открываем видеофайл: video.mov
[2026-04-13 16:49:04] [INFO] Параметры видео: 1920x1080, 29 FPS, 1252 кадров
[2026-04-13 16:55:03] [INFO] Готово! Обработано 1252 кадров, обнаружено 3476 номерных знаков.
```

Логгер реализован по паттерну Singleton — один экземпляр на всё приложение.

---

## Ресурсы

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLO на Habr](https://habr.com/ru/articles/821971/)
- [Статья про YOLO на Timeweb](https://timeweb.cloud/blog/yolo-neyroset-obnaruzhenie-obektov)
- [Датасет License Plate Recognition](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
- [Roboflow](https://roboflow.com) — разметка и управление датасетом
- [Weights & Biases](https://wandb.ai) — мониторинг обучения
