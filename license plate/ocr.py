"""
Скрипт для распознавания текста на номерных знаках.
Использует My_LicensePlate_Model для детекции + EasyOCR для чтения текста.

Запуск:
    python ocr.py --input video.mov --output result_ocr.mp4 --show
"""

import cv2
import argparse
import sys
import easyocr
import numpy as np
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from license_plate_detector.model_impl import My_LicensePlate_Model
from license_plate_detector.logger import SingletonLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Детекция и распознавание номерных знаков")
    parser.add_argument("--input", "-i", required=True, help="Путь к видеофайлу")
    parser.add_argument("--output", "-o", default="result_ocr.mp4", help="Путь к результату")
    parser.add_argument("--weights", default="runs/train/weights/best.pt", help="Веса модели")
    parser.add_argument("--conf", type=float, default=0.4, help="Порог уверенности")
    parser.add_argument("--show", action="store_true", help="Показывать в реальном времени")
    return parser.parse_args()


def main():
    logger = SingletonLogger.get_logger()
    args = parse_args()

    # --- Инициализация модели детекции ---
    logger.info("Загружаем модель детекции...")
    model = My_LicensePlate_Model(
        weights_path=args.weights,
        confidence_threshold=args.conf,
        device="cpu",
    )

    # --- Инициализация EasyOCR ---
    logger.info("Загружаем EasyOCR...")
    # en — английские символы (цифры и буквы латиницы)
    # gpu=False — без GPU, работает на CPU
    reader = easyocr.Reader(['en'], gpu=False)
    logger.info("EasyOCR готов!")

    # --- Открываем видео ---
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видео: {args.input}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Видео: {width}x{height}, {fps} FPS, {total} кадров")

    # --- VideoWriter ---
    out_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_count = 0
    recognized_plates = {}  # номер_кадра: [тексты]

    logger.info("Начинаем обработку...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- Детекция номерных знаков ---
        detections = model.detect_plates(frame)

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["confidence"]

            # --- Вырезаем область номера ---
            plate_crop = frame[y1:y2, x1:x2]

            # Пропускаем слишком маленькие области
            if plate_crop.size == 0 or plate_crop.shape[0] < 10 or plate_crop.shape[1] < 20:
                continue

            # --- Улучшаем изображение для OCR ---
            # Увеличиваем в 2 раза для лучшего распознавания
            scale = 2
            plate_resized = cv2.resize(
                plate_crop,
                (plate_crop.shape[1] * scale, plate_crop.shape[0] * scale),
                interpolation=cv2.INTER_CUBIC
            )

            # Переводим в оттенки серого
            gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)

            # Увеличиваем контрастность
            gray = cv2.equalizeHist(gray)

            # --- Распознаём текст ---
            ocr_results = reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            # Собираем распознанный текст
            plate_text = ""
            plate_conf = 0.0
            for (_, text, ocr_conf) in ocr_results:
                if ocr_conf > 0.3:  # минимальная уверенность OCR
                    plate_text += text.upper().strip()
                    plate_conf = max(plate_conf, ocr_conf)

            # --- Рисуем результаты на кадре ---
            # Зелёная рамка вокруг номера
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Текст с номером и уверенностью
            if plate_text:
                label = f"{plate_text} ({plate_conf:.2f})"
                color = (0, 255, 0)
                logger.debug(f"Кадр {frame_count}: распознан номер '{plate_text}'")
            else:
                label = f"? ({conf:.2f})"
                color = (0, 165, 255)  # оранжевый если не распознан

            # Фон для текста
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 5, y1),
                (0, 0, 0),
                -1,
            )

            # Текст
            cv2.putText(
                frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2,
            )

        # Счётчик кадров
        cv2.putText(
            frame,
            f"Frame: {frame_count}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2,
        )

        writer.write(frame)

        if args.show:
            cv2.imshow("License Plate OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Остановлено пользователем")
                break

        if frame_count % 50 == 0:
            logger.info(f"Обработано кадров: {frame_count}/{total}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    logger.info(f"Готово! Обработано {frame_count} кадров. Результат: {out_path}")


if __name__ == "__main__":
    main()
