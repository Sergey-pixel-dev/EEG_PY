import json
import struct
from datetime import datetime
from typing import Optional

import numpy as np


class DataRecorder:
    """Запись ЭЭГ данных в файл .eeg формата.

    Формат файла:
    1. 4 байта — длина JSON заголовка (little-endian uint32)
    2. JSON заголовок (UTF-8) с метаданными
    3. Binary data — сырые данные uint16, interleaved (ch1, ch2, ch1, ch2, ...)
    """

    def __init__(self, sample_freq: int = 2000, channels: int = 2):
        self.sample_freq = sample_freq
        self.channels = channels
        self.is_recording = False
        self.data_buffer: list[np.ndarray] = []
        self.start_time: Optional[datetime] = None

    def start_recording(self):
        """Начать новую запись"""
        self.data_buffer = []
        self.start_time = datetime.now()
        self.is_recording = True

    def stop_recording(self) -> str:
        """Остановить запись и вернуть предложенное имя файла"""
        self.is_recording = False
        if self.start_time:
            return f"recording_{self.start_time.strftime('%Y%m%d_%H%M%S')}.eeg"
        return "recording.eeg"

    def add_sample(self, data: np.ndarray):
        """Добавить сэмпл (вызывается для каждого пакета данных)"""
        if self.is_recording:
            self.data_buffer.append(data.copy())

    def get_duration(self) -> float:
        """Текущая длительность записи в секундах"""
        return len(self.data_buffer) / self.sample_freq

    def get_samples_count(self) -> int:
        """Количество записанных сэмплов"""
        return len(self.data_buffer)

    def save(self, filepath: str, notes: str = ''):
        """Сохранить запись в файл .eeg

        Args:
            filepath: Путь к файлу
            notes: Описание записи
        """
        if not self.data_buffer:
            raise ValueError("Нет данных для сохранения")

        # Конвертируем данные в numpy массив (float32, значения в мкВ)
        data_array = np.array(self.data_buffer, dtype=np.float32)

        # Создаём метаданные
        metadata = {
            'version': 3,
            'data_dtype': 'float32',
            'sample_freq': self.sample_freq,
            'channels': self.channels,
            'created_at': self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
            'duration_seconds': self.get_duration(),
            'samples_count': len(self.data_buffer),
            'device_info': 'EEG ADC',
            'notes': notes
        }

        # Сериализуем JSON
        json_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
        json_len = len(json_bytes)

        # Записываем в файл
        with open(filepath, 'wb') as f:
            # 4 байта — длина JSON
            f.write(struct.pack('<I', json_len))
            # JSON заголовок
            f.write(json_bytes)
            # Binary data — interleaved uint16
            f.write(data_array.tobytes())

    @staticmethod
    def load(filepath: str) -> tuple[np.ndarray, dict]:
        """Загрузить запись из файла .eeg

        Returns:
            tuple: (data, metadata)
                data: np.ndarray shape (N, channels), dtype=uint16
                metadata: dict с метаданными записи
        """
        with open(filepath, 'rb') as f:
            # Читаем длину JSON
            json_len_bytes = f.read(4)
            if len(json_len_bytes) < 4:
                raise ValueError("Некорректный файл: не удалось прочитать длину заголовка")
            json_len = struct.unpack('<I', json_len_bytes)[0]

            # Читаем JSON
            json_bytes = f.read(json_len)
            if len(json_bytes) < json_len:
                raise ValueError("Некорректный файл: не удалось прочитать заголовок")
            metadata = json.loads(json_bytes.decode('utf-8'))

            # Читаем данные
            data_bytes = f.read()
            channels = metadata.get('channels', 2)
            samples_count = metadata.get('samples_count', len(data_bytes) // (2 * channels))

            # Конвертируем в numpy (поддержка старых файлов uint16 и новых float32)
            dtype = metadata.get('data_dtype', 'uint16')
            data = np.frombuffer(data_bytes, dtype=np.dtype(dtype))
            data = data.reshape((samples_count, channels))

        return data, metadata
