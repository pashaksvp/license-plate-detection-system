"""
CLI-приложение для обнаружения номерных знаков.

Поддерживает два режима работы:
  1. video  — обработка существующего видеофайла
  2. camera — обработка потока с веб-камеры в реальном времени

Примеры запуска:
  python -m license_plate_detector.cli video --input road.mp4 --output result.mp4
  python -m license_plate_detector.cli camera --camera-id 0
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

from .model_impl import My_LicensePlate_Model
from .logger import SingletonLogger


def parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.

    Определяет два подкоманды: 'video' и 'camera',
    каждая со своими параметрами.

    Returns:
        argparse.Namespace: распарсенные аргументы
    """
    parser = argparse.ArgumentParser(
        description="🚗 Детектор номерных знаков на базе YOLOv8",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Общие аргументы (для обоих режимов)
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Путь к файлу весов модели (.pt)\n(по умолчанию: runs/detect/train/weights/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Порог уверенности (0.0–1.0, по умолчанию: 0.4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство: 'cpu' или 'cuda' (по умолчанию: cpu)",
    )

    # Подкоманды: video или camera
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- Режим VIDEO ---
    video_parser = subparsers.add_parser(
        "video",
        help="Обработка видеофайла",
    )
    video_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Путь к входному видеофайлу",
    )
    video_parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.mp4",
        help="Путь для сохранения результата (по умолчанию: output.mp4)",
    )
    video_parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать видео в окне во время обработки",
    )

    # --- Режим CAMERA ---
    camera_parser = subparsers.add_parser(
        "camera",
        help="Обработка потока с веб-камеры",
    )
    camera_parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="ID камеры (0 — стандартная веб-камера, по умолчанию: 0)",
    )
    camera_parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Если указан путь — сохранять поток в файл",
    )

    return parser.parse_args()


def process_video(model: My_LicensePlate_Model, args: argparse.Namespace) -> None:
    """
    Режим VIDEO: обрабатывает видеофайл покадрово.

    Алгоритм:
    1. Открываем входной файл через OpenCV
    2. Создаём VideoWriter для записи результата
    3. Для каждого кадра: detect_plates → draw_detections → запись
    4. Выводим итоговую статистику

    Args:
        model: экземпляр My_LicensePlate_Model
        args: аргументы командной строки
    """
    logger = SingletonLogger.get_logger()
    input_path = Path(args.input)

    # --- Проверяем, что файл существует ---
    if not input_path.exists():
        logger.error(f"Входной файл не найден: {input_path}")
        sys.exit(1)

    logger.info(f"Открываем видеофайл: {input_path}")

    # --- Открываем видео ---
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видеофайл: {input_path}")
        sys.exit(1)

    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Параметры видео: {width}x{height}, {fps} FPS, {total_frames} кадров")

    # --- Создаём VideoWriter для записи ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # кодек для .mp4
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0

    logger.info("Начинаем обработку видео...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # видео закончилось

            frame_count += 1

            # --- Детекция номерных знаков на кадре ---
            detections = model.detect_plates(frame)
            total_detections += len(detections)

            # --- Рисуем результаты на кадре ---
            annotated_frame = model.draw_detections(frame, detections)

            # Добавляем счётчик кадров на изображение
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames} | Plates: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # --- Записываем кадр в выходной файл ---
            writer.write(annotated_frame)

            # --- Показываем кадр (если указан флаг --show) ---
            if args.show:
                cv2.imshow("License Plate Detector", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Остановлено пользователем (нажата клавиша Q)")
                    break

            # Логируем прогресс каждые 100 кадров
            if frame_count % 100 == 0:
                logger.info(f"Обработано кадров: {frame_count}/{total_frames}")

    except KeyboardInterrupt:
        logger.info("Обработка прервана пользователем (Ctrl+C)")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    logger.info(
        f"Готово! Обработано {frame_count} кадров, "
        f"обнаружено {total_detections} номерных знаков. "
        f"Результат сохранён: {output_path}"
    )


def process_camera(model: My_LicensePlate_Model, args: argparse.Namespace) -> None:
    """
    Режим CAMERA: обрабатывает поток с веб-камеры в реальном времени.

    Алгоритм:
    1. Открываем камеру через OpenCV
    2. В бесконечном цикле: читаем кадр → detect_plates → показываем
    3. Опционально — сохраняем в файл
    4. Выход по нажатию клавиши Q

    Args:
        model: экземпляр My_LicensePlate_Model
        args: аргументы командной строки
    """
    logger = SingletonLogger.get_logger()
    camera_id = args.camera_id

    logger.info(f"Открываем камеру с ID: {camera_id}")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть камеру с ID {camera_id}")
        sys.exit(1)

    # Параметры камеры
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    logger.info(f"Камера: {width}x{height}, {fps} FPS")

    # --- Опционально: запись потока в файл ---
    writer = None
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        logger.info(f"Запись потока в файл: {save_path}")

    logger.info("Запуск детекции в реальном времени. Нажмите Q для выхода.")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Не удалось прочитать кадр с камеры")
                continue

            frame_count += 1

            # --- Детекция ---
            detections = model.detect_plates(frame)

            # --- Визуализация ---
            annotated_frame = model.draw_detections(frame, detections)

            # Показываем количество найденных номеров
            cv2.putText(
                annotated_frame,
                f"Plates detected: {len(detections)} | Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Live License Plate Detector", annotated_frame)

            if writer:
                writer.write(annotated_frame)

            # Q — выход
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Остановлено пользователем (нажата клавиша Q)")
                break

    except KeyboardInterrupt:
        logger.info("Остановлено пользователем (Ctrl+C)")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    logger.info(f"Сессия завершена. Обработано кадров: {frame_count}")


def main() -> None:
    """
    Точка входа в CLI-приложение.

    Разбирает аргументы, инициализирует модель,
    запускает нужный режим (video или camera).
    """
    logger = SingletonLogger.get_logger()
    logger.info("=" * 60)
    logger.info("Запуск License Plate Detector")
    logger.info("=" * 60)

    args = parse_args()

    logger.info(f"Режим работы: {args.mode.upper()}")
    logger.info(f"Веса модели: {args.weights}")
    logger.info(f"Порог уверенности: {args.conf}")
    logger.info(f"Устройство: {args.device}")

    # --- Инициализируем модель ---
    try:
        model = My_LicensePlate_Model(
            weights_path=args.weights,
            confidence_threshold=args.conf,
            device=args.device,
        )
    except RuntimeError as e:
        logger.error(f"Не удалось инициализировать модель: {e}")
        sys.exit(1)

    # --- Запускаем нужный режим ---
    if args.mode == "video":
        process_video(model, args)
    elif args.mode == "camera":
        process_camera(model, args)
    else:
        logger.error(f"Неизвестный режим: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
