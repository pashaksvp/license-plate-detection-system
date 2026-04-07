"""
license_plate_detector — пакет для обнаружения номерных знаков.

Основные компоненты:
- My_LicensePlate_Model: главный класс для детекции (model_impl.py)
- SingletonLogger: логгер-синглтон (logger.py)
- CLI: точка входа для командной строки (cli.py)
"""

from .model_impl import My_LicensePlate_Model
from .logger import SingletonLogger

__version__ = "1.0.0"
__all__ = ["My_LicensePlate_Model", "SingletonLogger"]
