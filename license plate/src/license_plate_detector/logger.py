"""
Модуль логирования.
Реализует паттерн Singleton — один экземпляр логгера на всё приложение.
"""

import logging
import os
from pathlib import Path


class SingletonLogger:
    """
    Singleton-логгер для всего приложения.

    Паттерн Singleton гарантирует, что существует только ОДИН
    экземпляр логгера, независимо от того, сколько раз его запрашивают.
    Это важно, чтобы не создавать дублирующихся обработчиков и
    писать все логи в один файл согласованно.
    """

    _instance: logging.Logger | None = None  # хранилище единственного экземпляра

    @classmethod
    def get_logger(cls, log_path: str = "./data/log_file.log") -> logging.Logger:
        """
        Возвращает единственный экземпляр логгера.

        Если логгер ещё не создан — создаёт его и настраивает.
        При повторных вызовах возвращает уже существующий.

        Args:
            log_path: путь к файлу лога (по умолчанию ./data/log_file.log)

        Returns:
            logging.Logger: настроенный экземпляр логгера
        """
        if cls._instance is None:
            cls._instance = cls._create_logger(log_path)
        return cls._instance

    @classmethod
    def _create_logger(cls, log_path: str) -> logging.Logger:
        """
        Создаёт и настраивает логгер с двумя обработчиками:
        - FileHandler  → пишет в файл ./data/log_file.log
        - StreamHandler → выводит в консоль (удобно при разработке)

        Args:
            log_path: путь к файлу лога

        Returns:
            logging.Logger: готовый логгер
        """
        # Убедимся, что директория для лога существует
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("LicensePlateDetector")
        logger.setLevel(logging.DEBUG)  # пишем всё: DEBUG, INFO, WARNING, ERROR

        # Формат: [2026-04-07 12:00:00] [INFO] Сообщение
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # --- Обработчик 1: запись в файл ---
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # --- Обработчик 2: вывод в консоль ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # в консоль только INFO и выше
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
