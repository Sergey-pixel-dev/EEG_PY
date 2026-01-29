import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from data_recorder import DataRecorder


class PlaybackWorker(QThread):
    """Воспроизведение записанных ЭЭГ данных.

    Эмитит те же сигналы что и SerialWorker, позволяя
    использовать существующий pipeline визуализации.
    """

    data_received = pyqtSignal(np.ndarray)
    playback_finished = pyqtSignal()
    playback_progress = pyqtSignal(float, float)  # (current_sec, total_sec)

    BATCH_INTERVAL_MS = 10  # 10мс между батчами

    def __init__(self):
        super().__init__()
        self.data: Optional[np.ndarray] = None
        self.metadata: Optional[dict] = None
        self.sample_freq = 2000
        self.running = False
        self.paused = False
        self.current_index = 0
        self._seek_requested = False
        self._seek_position = 0.0

    def load(self, filepath: str):
        """Загрузить файл записи"""
        self.data, self.metadata = DataRecorder.load(filepath)
        self.sample_freq = self.metadata.get('sample_freq', 2000)
        self.current_index = 0

    def get_duration(self) -> float:
        """Общая длительность в секундах"""
        if self.data is None:
            return 0.0
        return len(self.data) / self.sample_freq

    def get_metadata(self) -> Optional[dict]:
        """Получить метаданные загруженного файла"""
        return self.metadata

    def run(self):
        """Основной цикл воспроизведения"""
        if self.data is None:
            return

        self.running = True
        total_samples = len(self.data)
        total_duration = total_samples / self.sample_freq

        # Количество сэмплов в батче для достижения нужной частоты
        batch_size = self.sample_freq * self.BATCH_INTERVAL_MS // 1000  # 20 сэмплов при 2000Гц
        batch_interval = self.BATCH_INTERVAL_MS / 1000.0  # 0.01 сек

        last_batch_time = time.perf_counter()

        while self.running and self.current_index < total_samples:
            # Обработка запроса на перемотку
            if self._seek_requested:
                self.current_index = int(total_samples * self._seek_position)
                self._seek_requested = False
                last_batch_time = time.perf_counter()
                continue

            if self.paused:
                time.sleep(0.01)
                last_batch_time = time.perf_counter()
                continue

            current_time = time.perf_counter()
            elapsed = current_time - last_batch_time

            if elapsed >= batch_interval:
                # Эмитим батч сэмплов
                end_index = min(self.current_index + batch_size, total_samples)

                for i in range(self.current_index, end_index):
                    self.data_received.emit(self.data[i])

                self.current_index = end_index
                last_batch_time = current_time

                # Эмитим прогресс
                current_sec = self.current_index / self.sample_freq
                self.playback_progress.emit(current_sec, total_duration)
            else:
                # Спим оставшееся время (с запасом)
                sleep_time = batch_interval - elapsed - 0.001
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.running = False
        if self.current_index >= total_samples:
            self.playback_finished.emit()

    def pause(self):
        """Приостановить воспроизведение"""
        self.paused = True

    def resume(self):
        """Возобновить воспроизведение"""
        self.paused = False

    def stop(self):
        """Остановить воспроизведение"""
        self.running = False

    def seek(self, position: float):
        """Перемотка к позиции (0.0-1.0)"""
        self._seek_position = max(0.0, min(1.0, position))
        self._seek_requested = True

    def reset(self):
        """Сброс позиции на начало"""
        self.current_index = 0
        self.paused = False

    def is_loaded(self) -> bool:
        """Проверка загружен ли файл"""
        return self.data is not None
