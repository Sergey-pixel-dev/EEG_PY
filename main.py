import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QTabWidget, QToolBar, QComboBox, QPushButton, QLabel)
from serial.serialutil import SerialException

from eeg_channel_widget import EEGChannelWidget
from fourier_analysis_widget import FourierAnalysisWidget

PACKET_START = b'\xAA\x55'
PACKET_SIZE = 8

MAX_BUFFER_SIZE = 8192


class SerialWorker(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self, sport):
        super().__init__()
        self.sport = sport
        self.running = False
        self.rxBuf = bytearray(MAX_BUFFER_SIZE)  # ИЗМЕНЕНО на bytearray для скорости
        self.buf_pos = 0

        # Профилирование
        self.packet_count = 0
        self.last_time = time.perf_counter()
        self.last_print_time = self.last_time
        self.parse_times = []

    def run(self):
        self.running = True
        if not self.sport.is_open:
            return

        print(f"[{time.strftime('%H:%M:%S')}] SerialWorker запущен")

        while self.running:
            try:
                bytes_waiting = self.sport.in_waiting
                if bytes_waiting > 0:
                    raw_data = self.sport.read(min(bytes_waiting, 512))

                    bytes_to_add = len(raw_data)
                    if self.buf_pos + bytes_to_add >= MAX_BUFFER_SIZE:
                        valid_data = self.rxBuf[self.buf_pos:]
                        self.rxBuf[:len(valid_data)] = valid_data
                        self.buf_pos = len(valid_data)

                    self.rxBuf[self.buf_pos:self.buf_pos + bytes_to_add] = raw_data
                    self.buf_pos += bytes_to_add

                    while True:
                        start_idx = self.rxBuf.find(PACKET_START, 0, self.buf_pos)
                        if start_idx == -1:
                            break
                        if self.buf_pos - start_idx < PACKET_SIZE:
                            break
                        end_idx = start_idx + PACKET_SIZE - 1
                        if not (self.rxBuf[end_idx - 1] == 0x55 and self.rxBuf[end_idx] == 0xAA):
                            self.rxBuf[start_idx:self.buf_pos - 1] = self.rxBuf[start_idx + 1:self.buf_pos]
                            self.buf_pos -= 1
                            continue
                        parse_start = time.perf_counter()
                        packet = bytes(self.rxBuf[start_idx:start_idx + PACKET_SIZE])

                        data = np.zeros(PACKET_SIZE // 2 - 2, dtype=np.uint16)
                        for i in range(PACKET_SIZE // 2 - 2):
                            data[i] = packet[2 * i + 2] | (packet[2 * i + 3] << 8)

                        parse_time = (time.perf_counter() - parse_start) * 1000
                        self.parse_times.append(parse_time)

                        self.packet_count += 1
                        current_time = time.perf_counter()

                        # Вывод статистики каждую 1 секунду
                        if current_time - self.last_print_time >= 1.0:
                            elapsed = current_time - self.last_time
                            packet_rate = self.packet_count / elapsed
                            avg_parse_time = np.mean(self.parse_times[-100:]) if self.parse_times else 0
                            max_parse_time = np.max(self.parse_times[-100:]) if self.parse_times else 0

                            print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] "
                                  f"Пакетов: {self.packet_count} | "
                                  f"Скорость: {packet_rate:.1f} пак/сек | "
                                  f"Мин.вреемя парсинга: {np.min(self.parse_times[-100:]):.3f}мс | "
                                  f"Макс.время парсинга: {max_parse_time:.3f}мс | "
                                  f"Среднее: {avg_parse_time:.3f}мс")

                            self.last_print_time = current_time
                        self.rxBuf[0:self.buf_pos - PACKET_SIZE] = self.rxBuf[PACKET_SIZE:self.buf_pos]
                        self.buf_pos -= PACKET_SIZE
                        self.data_received.emit(data)
                else:
                    # Нет данных - немного спим чтобы не грузить CPU
                    time.sleep(0.0001)  # 0.1 мс

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Ошибка: {e}")
                self.running = False

    def stop(self):
        print(f"[{time.strftime('%H:%M:%S')}] SerialWorker останавливается. "
              f"Всего пакетов получено: {self.packet_count}")
        self.running = False
        self.wait()


class EEGPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЭЭГ")
        self.setGeometry(100, 100, 1400, 800)

        self.BAUD_RATE = 1500000
        self.sport = serial.Serial(None, self.BAUD_RATE)

        self.X_AXIS_RANGE = 5  # в сек.
        self.SAMPLE_FREQ = 1000  # Гц

        self.serial_worker = None

        self._setup_ui()

        # Создаем 2 канала
        self.channel1 = EEGChannelWidget("Канал 1", self.X_AXIS_RANGE, (0, 3400), self.SAMPLE_FREQ)
        self.channel2 = EEGChannelWidget("Канал 2", self.X_AXIS_RANGE, (0, 3400), self.SAMPLE_FREQ)

        self._init_test_data()

        self.graph_layout.addWidget(self.channel1)
        self.graph_layout.addWidget(self.channel2)

        self.fourier_widget = FourierAnalysisWidget(
            channels=[self.channel1, self.channel2],
            sample_freq=self.SAMPLE_FREQ
        )
        self.tab_widget.addTab(self.fourier_widget, "Фурье-анализ")
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_all_channels)
        self.plot_timer.start(10)

    def _get_available_ports(self):
        """Получение списка доступных COM-портов"""
        ports = serial.tools.list_ports.comports()
        available_ports = [port.device for port in ports]
        return available_ports if available_ports else ["Нет доступных портов"]

    def _setup_ui(self):
        """Настройка интерфейса"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Порт:"))
        self.port_combo = QComboBox()
        self.port_combo.addItems(self._get_available_ports())
        toolbar.addWidget(self.port_combo)

        refresh_btn = QPushButton("Обновить")
        refresh_btn.setToolTip("Обновить список портов")
        refresh_btn.clicked.connect(self._refresh_ports)
        refresh_btn.setMaximumWidth(40)
        toolbar.addWidget(refresh_btn)

        toolbar.addSeparator()

        self.connect_btn = QPushButton("Подключиться")
        self.connect_btn.clicked.connect(self.connect_sport)
        toolbar.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Отключиться")
        self.disconnect_btn.clicked.connect(self.disconnect_sport)
        self.disconnect_btn.setEnabled(False)
        toolbar.addWidget(self.disconnect_btn)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.tab_graph = QWidget()
        self.graph_layout = QVBoxLayout()
        self.tab_graph.setLayout(self.graph_layout)
        self.tab_widget.addTab(self.tab_graph, "Графики")

        self.tab_settings = QWidget()
        settings_layout = QVBoxLayout()
        settings_layout.addWidget(QLabel("Настройки (в разработке)"))
        settings_layout.addStretch()
        self.tab_settings.setLayout(settings_layout)
        self.tab_widget.addTab(self.tab_settings, "Настройки")

    def _refresh_ports(self):
        """Обновление списка доступных портов"""
        current_port = self.port_combo.currentText()
        self.port_combo.clear()
        available_ports = self._get_available_ports()
        self.port_combo.addItems(available_ports)

        # Пытаемся восстановить предыдущий выбор
        index = self.port_combo.findText(current_port)
        if index >= 0:
            self.port_combo.setCurrentIndex(index)

    def _init_test_data(self):
        """Инициализация тестовыми синусоидами"""
        x = np.linspace(0, self.X_AXIS_RANGE, self.X_AXIS_RANGE * self.SAMPLE_FREQ)
        for val in x:
            self.channel1.append_data(np.sin(val) * 1000 + 1700)
        for val in x:
            self.channel2.append_data(np.sin(val * 2) * 800 + 1700)

    def connect_sport(self):
        """Подключение к serial порту"""
        port = self.port_combo.currentText()
        try:
            self.sport = serial.Serial(port, self.BAUD_RATE)
            self.serial_worker = SerialWorker(self.sport)
            self.serial_worker.data_received.connect(self.update_data)
            self.serial_worker.start()
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.port_combo.setEnabled(False)
            print(f"Подключено к {port}")
        except SerialException as e:
            print(f"Ошибка подключения: {e}")

    def disconnect_sport(self):
        """Отключение от serial порта"""
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker = None

        if self.sport.is_open:
            self.sport.close()

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.port_combo.setEnabled(True)

        print("Отключено")

    def update_data(self, data):
        """Обновление данных по событию от воркера"""
        if data is not None:
            self.channel1.append_data(data[0])
            self.channel2.append_data(data[1])

    def _update_all_channels(self):
        """Обновление всех графиков по таймеру"""
        self.channel1.update_display()
        self.channel2.update_display()

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.serial_worker:
            self.serial_worker.stop()
        if self.sport.is_open:
            self.sport.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(
        antialias=False,
        useOpenGL=False
    )

    window = EEGPlotter()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
