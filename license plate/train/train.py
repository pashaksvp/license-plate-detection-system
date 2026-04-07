"""
Скрипт обучения (fine-tuning) модели YOLOv8 на датасете номерных знаков.

Что делает этот скрипт:
1. Загружает предобученную модель YOLOv8 (nano или small)
2. Дообучает (fine-tune) её на вашем датасете
3. Логирует весь процесс в Wandb (метрики, графики, примеры детекций)
4. Сохраняет лучшие веса в runs/detect/train/weights/best.pt

Запуск:
    python train/train.py --data data/dataset.yaml --epochs 50 --model yolov8n.pt

Структура датасета (YOLO-формат):
    data/
    ├── dataset.yaml          ← конфиг датасета
    ├── images/
    │   ├── train/            ← обучающие изображения
    │   └── val/              ← валидационные изображения
    └── labels/
        ├── train/            ← разметка для train (файлы .txt)
        └── val/              ← разметка для val
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO
from license_plate_detector.logger import SingletonLogger


def parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для скрипта обучения.

    Returns:
        argparse.Namespace: аргументы обучения
    """
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 на датасете номерных знаков")

    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.yaml",
        help="Путь к файлу конфигурации датасета (dataset.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Базовая модель: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох обучения (по умолчанию: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Размер входного изображения (по умолчанию: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Размер батча (по умолчанию: 16). Уменьшите при нехватке VRAM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство: 'cpu' или 'cuda' или '0' (GPU ID)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Директория для сохранения результатов",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Название эксперимента",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Включить логирование в Weights & Biases",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping: остановить если mAP не улучшается N эпох",
    )

    return parser.parse_args()


def setup_wandb(project_name: str = "license-plate-detector") -> None:
    """
    Инициализирует интеграцию с Weights & Biases.

    Wandb — это платформа для отслеживания экспериментов ML.
    Позволяет видеть графики потерь, метрики mAP, примеры предсказаний
    прямо в браузере в реальном времени.

    Ultralytics YOLO автоматически интегрируется с wandb —
    достаточно его импортировать до начала обучения.

    Args:
        project_name: название проекта в Wandb (отображается на дашборде)
    """
    try:
        import wandb
        wandb.init(
            project=project_name,
            tags=["yolov8", "license-plate", "detection"],
            notes="Fine-tuning YOLOv8 для обнаружения номерных знаков",
        )
        print("[Wandb] Логирование активировано.")
    except ImportError:
        print("[Wandb] Библиотека wandb не установлена. Запустите: pip install wandb")
    except Exception as e:
        print(f"[Wandb] Не удалось инициализировать: {e}")


def create_dataset_yaml(output_path: str = "data/dataset.yaml") -> None:
    """
    Создаёт шаблонный файл конфигурации датасета.

    Файл dataset.yaml описывает датасет для YOLO:
    - где находятся изображения
    - сколько классов
    - как называются классы

    Формат разметки (файлы .txt):
        Каждая строка = один объект:
        <class_id> <x_center> <y_center> <width> <height>
        Все координаты нормированы (от 0 до 1 относительно размера изображения)

    Args:
        output_path: куда сохранить yaml-файл
    """
    yaml_content = """# Конфигурация датасета для обнаружения номерных знаков
# Формат: YOLOv8

# Пути к данным (относительно корня проекта)
path: data              # корневая директория датасета
train: images/train     # папка с обучающими изображениями
val: images/val         # папка с валидационными изображениями

# Количество классов
nc: 1

# Названия классов
names:
  0: license_plate

# --- Как создать собственный датасет ---
# 1. Запишите видео с дороги (или скачайте готовое)
# 2. Извлеките кадры: ffmpeg -i video.mp4 -vf fps=2 data/images/train/frame_%04d.jpg
# 3. Разметьте в Roboflow (roboflow.com) или Label Studio (labelstud.io)
# 4. Экспортируйте в формате "YOLOv8"
# 5. Замените содержимое папок images/ и labels/
"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text(yaml_content)
        print(f"Создан шаблон dataset.yaml: {path}")
    else:
        print(f"dataset.yaml уже существует: {path}")


def train(args: argparse.Namespace) -> None:
    """
    Основная функция обучения модели.

    Процесс fine-tuning:
    - Берём предобученную модель (yolov8n.pt), обученную на COCO (80 классов)
    - Заменяем последний слой под наш 1 класс (license_plate)
    - Дообучаем на нашем датасете с маленьким learning rate
    - YOLO сам управляет lr_scheduler, augmentation, early stopping

    Результат сохраняется в:
        runs/detect/train/
        ├── weights/
        │   ├── best.pt    ← лучшие веса (по mAP50)
        │   └── last.pt    ← веса последней эпохи
        ├── results.csv    ← метрики по эпохам
        └── confusion_matrix.png

    Args:
        args: аргументы обучения
    """
    logger = SingletonLogger.get_logger()

    # Проверяем наличие dataset.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        logger.warning(f"Файл датасета не найден: {data_path}")
        logger.info("Создаём шаблон dataset.yaml...")
        create_dataset_yaml(str(data_path))
        logger.info(
            "Заполните dataset.yaml и добавьте изображения в data/images/, "
            "разметку в data/labels/, затем запустите обучение снова."
        )
        return

    logger.info(f"Начинаем обучение модели: {args.model}")
    logger.info(f"Датасет: {args.data}")
    logger.info(f"Эпох: {args.epochs}, батч: {args.batch}, imgsz: {args.imgsz}")

    # --- Инициализируем Wandb (если запрошен) ---
    if args.wandb:
        setup_wandb()

    try:
        # --- Загружаем базовую предобученную модель ---
        model = YOLO(args.model)

        logger.info("Запуск обучения...")

        # --- Запускаем fine-tuning ---
        results = model.train(
            data=str(args.data),           # путь к dataset.yaml
            epochs=args.epochs,            # количество эпох
            imgsz=args.imgsz,              # размер изображений
            batch=args.batch,              # размер батча
            device=args.device,            # GPU/CPU
            project=args.project,          # куда сохранять
            name=args.name,                # название эксперимента
            patience=args.patience,        # early stopping

            # Гиперпараметры оптимизатора
            lr0=0.01,                      # начальный learning rate
            lrf=0.01,                      # финальный lr (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,

            # Аугментации данных (расширение датасета)
            hsv_h=0.015,                   # вариация оттенка
            hsv_s=0.7,                     # вариация насыщенности
            hsv_v=0.4,                     # вариация яркости
            flipud=0.0,                    # вертикальный флип (не нужен для авто)
            fliplr=0.5,                    # горизонтальный флип
            mosaic=1.0,                    # мозаичная аугментация

            # Логирование
            plots=True,                    # сохранять графики метрик
            save=True,                     # сохранять чекпоинты
            verbose=True,
        )

        logger.info("Обучение завершено!")

        # --- Оцениваем финальную модель на валидации ---
        logger.info("Запускаем валидацию финальной модели...")

        best_weights = Path(args.project) / args.name / "weights" / "best.pt"
        if best_weights.exists():
            best_model = YOLO(str(best_weights))
            metrics = best_model.val(data=str(args.data), device=args.device)

            map50 = metrics.box.map50
            map50_95 = metrics.box.map

            logger.info(f"mAP@0.5:    {map50:.4f}  ({map50 * 100:.1f}%)")
            logger.info(f"mAP@0.5:0.95: {map50_95:.4f}  ({map50_95 * 100:.1f}%)")

            # Оцениваем по критериям задания
            if map50 >= 0.8:
                logger.info("🏆 Результат: S-Tier (mAP > 0.8) — 15 баллов")
            elif map50 >= 0.6:
                logger.info("✅ Результат: Good (mAP 0.6–0.8) — 15 баллов")
            elif map50 >= 0.4:
                logger.info("📊 Результат: Baseline (mAP 0.4–0.6) — 10 баллов")
            else:
                logger.info("❌ Результат: Poor (mAP < 0.4) — 0 баллов")

            logger.info(f"Лучшие веса сохранены: {best_weights}")
        else:
            logger.warning("Файл best.pt не найден — возможно обучение было прервано")

    except Exception as e:
        logger.error(f"Ошибка во время обучения: {e}")
        raise


if __name__ == "__main__":
    args = parse_args()
    train(args)
