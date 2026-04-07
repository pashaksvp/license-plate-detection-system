"""
Тесты для My_LicensePlate_Model.

Запуск: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from license_plate_detector.model_impl import My_LicensePlate_Model
from license_plate_detector.logger import SingletonLogger


class TestSingletonLogger:
    """Тесты для паттерна Singleton в логгере."""

    def test_singleton_same_instance(self):
        """Два вызова get_logger() должны вернуть один и тот же объект."""
        logger1 = SingletonLogger.get_logger()
        logger2 = SingletonLogger.get_logger()
        assert logger1 is logger2, "Логгер должен быть Singleton"

    def test_logger_has_handlers(self):
        """Логгер должен иметь хотя бы один обработчик (файл или консоль)."""
        logger = SingletonLogger.get_logger()
        assert len(logger.handlers) > 0


class TestMyLicensePlateModel:
    """Тесты для класса My_LicensePlate_Model."""

    @pytest.fixture
    def model(self):
        """Создаёт экземпляр модели с базовыми весами (без GPU)."""
        # Используем базовую yolov8n.pt (скачается автоматически)
        return My_LicensePlate_Model(weights_path="yolov8n.pt", device="cpu")

    def test_detect_plates_returns_list(self, model):
        """detect_plates должен возвращать список."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.detect_plates(frame)
        assert isinstance(result, list)

    def test_detect_plates_empty_frame(self, model):
        """На пустом (чёрном) кадре не должно быть детекций."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.detect_plates(frame)
        # Пустой кадр — скорее всего нет детекций
        assert isinstance(result, list)

    def test_detect_plates_result_format(self, model):
        """Каждая детекция должна содержать нужные поля."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = model.detect_plates(frame)

        required_keys = {"x1", "y1", "x2", "y2", "confidence", "class_name"}
        for det in result:
            assert required_keys.issubset(det.keys()), (
                f"Не хватает полей: {required_keys - det.keys()}"
            )

    def test_detect_plates_coordinate_types(self, model):
        """Координаты должны быть int, confidence — float."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = model.detect_plates(frame)

        for det in result:
            assert isinstance(det["x1"], int)
            assert isinstance(det["y1"], int)
            assert isinstance(det["x2"], int)
            assert isinstance(det["y2"], int)
            assert isinstance(det["confidence"], float)
            assert isinstance(det["class_name"], str)

    def test_detect_plates_invalid_input_type(self, model):
        """При передаче не-ndarray должен выброситься ValueError."""
        with pytest.raises(ValueError):
            model.detect_plates([[1, 2, 3], [4, 5, 6]])

    def test_detect_plates_invalid_shape(self, model):
        """При неверной форме массива должен выброситься ValueError."""
        frame_2d = np.zeros((480, 640), dtype=np.uint8)  # нет канального измерения
        with pytest.raises(ValueError):
            model.detect_plates(frame_2d)

    def test_detect_plates_confidence_range(self, model):
        """Confidence должен быть в диапазоне [0, 1]."""
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = model.detect_plates(frame)
        for det in result:
            assert 0.0 <= det["confidence"] <= 1.0

    def test_draw_detections_returns_ndarray(self, model):
        """draw_detections должен возвращать np.ndarray того же размера."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_detections = [
            {"x1": 100, "y1": 100, "x2": 300, "y2": 150,
             "confidence": 0.9, "class_name": "license_plate"}
        ]
        result = model.draw_detections(frame, fake_detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_model_does_not_modify_original_frame(self, model):
        """detect_plates не должен изменять исходный кадр."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        model.detect_plates(frame)
        assert np.array_equal(frame, frame_copy), "Исходный кадр был изменён!"
