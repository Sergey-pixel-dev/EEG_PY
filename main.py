import sys
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

MAX_BUFFER_SIZE = 4096


class SerialWorker(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self, sport):
        super().__init__()
        self.sport = sport
        self.running = False
        self.rxBuf = deque(maxlen=MAX_BUFFER_SIZE)

    def run(self):
        self.running = True
        if not self.sport.is_open:
            return

        while self.running:
            try:
                if self.sport.in_waiting > 0:
                    raw_data = self.sport.read(self.sport.in_waiting)
                    self.rxBuf.extend(raw_data)
                    # выделяем ацп пакет
                    buffer_bytes = bytes(self.rxBuf)
                    start_index = buffer_bytes.find(PACKET_START)
                    if start_index == -1:
                        continue
                    if len(buffer_bytes) - start_index < PACKET_SIZE:
                        continue
                    if not (buffer_bytes[start_index + PACKET_SIZE - 2] == 0x55 and buffer_bytes[
                        start_index + PACKET_SIZE - 1] == 0xAA):
                        continue
                    packet = buffer_bytes[start_index:start_index + PACKET_SIZE]
                    for _ in range(start_index + PACKET_SIZE):
                        self.rxBuf.popleft()
                    data = np.zeros(PACKET_SIZE // 2 - 2, dtype=np.uint16)
                    for i in range(0, PACKET_SIZE // 2 - 2):
                        data[i] = packet[2 * i + 2] | (packet[2 * i + 1 + 2] << 8)
                    self.data_received.emit(data)
            except Exception as e:
                print(f"Ошибка чтения: {e}")
                self.running = False

    def stop(self):
        self.running = False
        self.wait()


class EEGPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЭЭГ")
        self.setGeometry(100, 100, 1400, 800)

        self.BAUD_RATE = 115200
        self.sport = serial.Serial(None, self.BAUD_RATE)

        self.X_AXIS_RANGE = 15  # в сек.
        self.SAMPLE_RATE = 143  # Гц

        self.serial_worker = None

        self._setup_ui()

        # Создаем 2 канала
        self.channel1 = EEGChannelWidget("Канал 1", self.X_AXIS_RANGE, (0, 3400), self.SAMPLE_RATE)
        self.channel2 = EEGChannelWidget("Канал 2", self.X_AXIS_RANGE, (0, 3400), self.SAMPLE_RATE)

        self._init_test_data()

        self.graph_layout.addWidget(self.channel1)
        self.graph_layout.addWidget(self.channel2)

        self.fourier_widget = FourierAnalysisWidget(
            channels=[self.channel1, self.channel2],
            sample_rate=self.SAMPLE_RATE
        )
        self.tab_widget.addTab(self.fourier_widget, "Фурье-анализ")
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_all_channels)
        self.plot_timer.start(25)

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
        x = np.linspace(0, self.X_AXIS_RANGE, self.X_AXIS_RANGE * self.SAMPLE_RATE)
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
