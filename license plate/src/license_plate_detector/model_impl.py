"""
Главный модуль модели обнаружения номерных знаков.

Содержит класс My_LicensePlate_Model — основной программный артефакт,
требуемый заданием. Использует предобученную модель YOLO (Ultralytics)
и дообучается на датасете с номерными знаками.
"""

import numpy as np
from pathlib import Path
from typing import Optional

# Импортируем YOLO из библиотеки ultralytics
# pip install ultralytics
from ultralytics import YOLO

from .logger import SingletonLogger


class My_LicensePlate_Model:
    """
    Класс-обёртка над YOLO для обнаружения номерных знаков.

    Инкапсулирует:
    - загрузку весов модели
    - предобработку входного кадра
    - вывод (inference) модели
    - постобработку результатов в удобный формат

    Пример использования:
        model = My_LicensePlate_Model(weights_path="runs/detect/train/weights/best.pt")
        results = model.detect_plates(frame)
        for plate in results:
            print(plate)
        # {'x1': 100, 'y1': 200, 'x2': 300, 'y2': 250, 'confidence': 0.92, 'class_name': 'license_plate'}
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
    ):
        """
        Инициализирует модель, загружает веса.

        Args:
            weights_path: путь к файлу весов (.pt).
                          Если файл не найден — загружается базовая YOLOv8n.
            confidence_threshold: минимальный порог уверенности (0.0 — 1.0).
                                  Детекции ниже порога отбрасываются.
            device: устройство для вычислений: 'cpu', 'cuda', 'cuda:0' и т.д.
        """
        self.logger = SingletonLogger.get_logger()
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.logger.info(f"Инициализация модели. Веса: {weights_path}, устройство: {device}")

        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path: str) -> YOLO:
        """
        Загружает модель YOLO из файла весов.

        Если файл не существует — загружает базовую предобученную
        YOLOv8n (nano) с серверов Ultralytics. Это полезно при
        первом запуске или при отсутствии дообученных весов.

        Args:
            weights_path: путь к файлу весов

        Returns:
            YOLO: загруженная модель

        Raises:
            RuntimeError: если модель не удалось загрузить
        """
        try:
            path = Path(weights_path)

            if path.exists():
                self.logger.info(f"Загружаем веса из файла: {path}")
                model = YOLO(str(path))
            else:
                # Файла нет — скачиваем базовую модель
                self.logger.warning(
                    f"Файл весов '{weights_path}' не найден. "
                    f"Загружаем базовую YOLOv8n..."
                )
                model = YOLO("yolov8n.pt")

            model.to(self.device)
            self.logger.info("Модель успешно загружена.")
            return model

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            raise RuntimeError(f"Не удалось загрузить модель: {e}") from e

    def detect_plates(self, frame: np.ndarray) -> list[dict]:
        """
        Обнаруживает номерные знаки на изображении.

        Это основной метод, требуемый заданием. Принимает кадр в формате
        numpy-массива (BGR или RGB — YOLO обрабатывает оба формата),
        возвращает список словарей с координатами и уверенностью.

        Args:
            frame: изображение в формате np.ndarray.
                   Ожидаемая форма: (H, W, 3) — высота, ширина, каналы.

        Returns:
            list[dict]: список обнаруженных номерных знаков.
            Каждый элемент — словарь вида:
            {
                'x1': int,           # левый край bounding box
                'y1': int,           # верхний край
                'x2': int,           # правый край
                'y2': int,           # нижний край
                'confidence': float, # уверенность модели (0.0 — 1.0)
                'class_name': str,   # название класса ('license_plate')
            }

        Raises:
            ValueError: если frame имеет неверный формат
        """
        # --- Валидация входных данных ---
        if not isinstance(frame, np.ndarray):
            self.logger.error("detect_plates: frame должен быть np.ndarray")
            raise ValueError("frame должен быть np.ndarray")

        if frame.ndim != 3 or frame.shape[2] != 3:
            self.logger.error(f"detect_plates: неверная форма frame: {frame.shape}")
            raise ValueError(f"Ожидается форма (H, W, 3), получено: {frame.shape}")

        self.logger.debug(f"Обработка кадра размером {frame.shape[1]}x{frame.shape[0]}")

        try:
            # --- Запускаем инференс YOLO ---
            # verbose=False отключает вывод YOLO в консоль (у нас свой логгер)
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
            )

            # --- Извлекаем и форматируем результаты ---
            detections = self._parse_results(results)

            self.logger.debug(f"Обнаружено объектов: {len(detections)}")
            return detections

        except Exception as e:
            self.logger.error(f"Ошибка во время инференса: {e}")
            return []  # возвращаем пустой список, не падаем

    def _parse_results(self, results) -> list[dict]:
        """
        Преобразует сырой вывод YOLO в список словарей.

        YOLO возвращает объект Results с тензорами координат.
        Эта функция распаковывает тензоры и формирует читаемый формат.

        Args:
            results: объект Results от Ultralytics YOLO

        Returns:
            list[dict]: список детекций в стандартном формате
        """
        detections = []

        for result in results:
            # result.boxes содержит все bounding boxes для одного изображения
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            # Получаем названия классов из модели
            class_names = result.names  # dict: {0: 'license_plate', ...}

            for box in boxes:
                # xyxy — координаты [x1, y1, x2, y2] в пикселях
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # conf — уверенность модели
                confidence = float(box.conf[0])

                # cls — индекс класса
                class_id = int(box.cls[0])
                class_name = class_names.get(class_id, f"class_{class_id}")

                detections.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": round(confidence, 4),
                    "class_name": class_name,
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Рисует bounding boxes на кадре (вспомогательный метод).

        Полезен для визуализации результатов при сохранении видео
        или отображении в реальном времени.

        Args:
            frame: исходный кадр np.ndarray
            detections: список словарей от detect_plates()

        Returns:
            np.ndarray: кадр с нарисованными рамками и подписями
        """
        import cv2

        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            confidence = det["confidence"]
            class_name = det["class_name"]

            # Зелёная рамка вокруг номерного знака
            cv2.rectangle(output, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Подпись с уверенностью
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Фон для текста (чёрный прямоугольник)
            cv2.rectangle(
                output,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 0, 0),
                -1,
            )

            # Белый текст поверх чёрного фона
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return output
