import sys
import time
from enum import IntEnum
from typing import Optional

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QToolBar, QComboBox, QPushButton, QLabel,
                             QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, QFormLayout,
                             QFileDialog, QSlider, QDialog, QTextEdit, QDialogButtonBox,
                             QScrollArea)
from PyQt6.QtCore import Qt
from serial.serialutil import SerialException

from eeg_channel_widget import EEGChannelWidget
from fourier_analysis_widget import FourierAnalysisWidget
from data_recorder import DataRecorder
from playback_worker import PlaybackWorker

NUM_CHANNELS = 8  # <-- меняй здесь количество каналов
BYTES_PER_CHANNEL = 3  # 24-битное знаковое целое в мкВ

# Типы пакетов SerProt
TYPE_COMMAND  = 0xCC  # Master → Slave (зарезервировано)
TYPE_RESPONSE = 0xDD  # Slave → Master (зарезервировано)
TYPE_ERROR    = 0xEE  # Slave → Master (зарезервировано)
TYPE_PUSH     = 0xFF  # Slave → Master, стриминговые данные

_VALID_TYPES = {TYPE_COMMAND, TYPE_RESPONSE, TYPE_ERROR, TYPE_PUSH}


def crc16_modbus(data: bytes) -> int:
    """CRC-16/Modbus: полином 0x8005 (reflected 0xA001), init=0xFFFF, покрывает [type][seq][payload]"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


class _S(IntEnum):
    S0     = 0  # ожидание 0xAA (старт)
    S0_ESC = 7  # в S0 получен 0xBB — следующий байт литерал, не маркер
    S1     = 1  # получен 0xAA, ждём type
    S2     = 2  # получен type, ждём seq
    S3     = 3  # получен seq, ждём 1-й escaped-байт
    S4     = 4  # получен 1-й байт, ждём 2-й (гарантия ≥ 2 байт для CRC)
    S5     = 5  # накапливаем payload+CRC до end-маркера 0xAA
    SESC   = 6  # получен 0xBB, следующий байт — литерал


class SerProtParser:
    """Конечный автомат разбора пакетов SerProt (по Machine_state_parser.csv)."""

    def __init__(self, num_channels: int, bytes_per_channel: int):
        self.num_channels = num_channels
        self.bytes_per_channel = bytes_per_channel
        self._state = _S.S0
        self._type = 0
        self._seq = 0
        self._esc_return: _S = _S.S3
        self._buf = bytearray()
        self._last_push_seq: Optional[int] = None

    def process_byte(self, b: int) -> Optional[np.ndarray]:
        s = self._state

        if s == _S.S0:
            if b == 0xAA:
                self._state = _S.S1
            elif b == 0xBB:
                self._state = _S.S0_ESC  # escape-префикс: следующий байт — литерал

        elif s == _S.S0_ESC:
            self._state = _S.S0  # пропускаем литерал (0xAA или 0xBB), ищем дальше

        elif s == _S.S1:
            if b == 0xAA:
                pass  # повторный 0xAA = новый старт-маркер, остаёмся в S1
            elif b in _VALID_TYPES:
                self._type = b
                self._state = _S.S2
            else:
                self._state = _S.S0

        elif s == _S.S2:
            self._seq = b
            self._buf = bytearray()
            self._state = _S.S3

        elif s == _S.S3:
            if b == 0xAA:
                self._state = _S.S0          # слишком короткий
            elif b == 0xBB:
                self._esc_return = _S.S4
                self._state = _S.SESC
            else:
                self._buf.append(b)
                self._state = _S.S4

        elif s == _S.S4:
            if b == 0xAA:
                self._state = _S.S0          # нужны ≥ 2 байта (CRC_L + CRC_H)
            elif b == 0xBB:
                self._esc_return = _S.S5
                self._state = _S.SESC
            else:
                self._buf.append(b)
                self._state = _S.S5

        elif s == _S.S5:
            if b == 0xAA:
                self._state = _S.S0
                return self._finalize()
            elif b == 0xBB:
                self._esc_return = _S.S5
                self._state = _S.SESC
            else:
                self._buf.append(b)

        elif s == _S.SESC:
            if b == 0xAA or b == 0xBB:
                self._buf.append(b)
                self._state = self._esc_return
            else:
                self._state = _S.S0          # недопустимый байт после escape

        return None

    def _finalize(self) -> Optional[np.ndarray]:
        buf = self._buf
        if len(buf) < 2:
            return None

        crc_received = buf[-2] | (buf[-1] << 8)
        payload = bytes(buf[:-2])

        crc_check = bytes([self._type, self._seq]) + payload
        if crc16_modbus(crc_check) != crc_received:
            print(f"[SerProt] CRC mismatch (type=0x{self._type:02X} seq={self._seq})")
            return None

        if self._type != TYPE_PUSH:
            return None  # Response/Error/Command — обработка в будущем

        expected_payload = self.num_channels * self.bytes_per_channel
        if len(payload) != expected_payload:
            print(f"[SerProt] Push payload size {len(payload)} != {expected_payload}")
            return None

        # Детектирование потерь пакетов по seq
        if self._last_push_seq is not None:
            expected_seq = (self._last_push_seq + 1) & 0xFF
            if self._seq != expected_seq:
                missed = (self._seq - expected_seq) & 0xFF
                print(f"[WARN] Push seq loss: missed {missed} (got {self._seq}, expected {expected_seq})")
        self._last_push_seq = self._seq

        # Декодирование 8 каналов × 3 байта LE signed → мкВ
        data = np.empty(self.num_channels, dtype=np.float32)
        for i in range(self.num_channels):
            off = i * self.bytes_per_channel
            raw = payload[off] | (payload[off + 1] << 8) | (payload[off + 2] << 16)
            if raw >= 0x800000:
                raw -= 0x1000000
            data[i] = float(raw)
        return data


class SerialWorker(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self, sport):
        super().__init__()
        self.sport = sport
        self.running = False
        self.parser = SerProtParser(NUM_CHANNELS, BYTES_PER_CHANNEL)

        # Профилирование
        self.packet_count = 0
        self.last_time = time.perf_counter()
        self.last_print_time = self.last_time

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
                    for b in raw_data:
                        result = self.parser.process_byte(b)
                        if result is not None:
                            self.packet_count += 1
                            self.data_received.emit(result)

                    current_time = time.perf_counter()
                    if current_time - self.last_print_time >= 1.0:
                        elapsed = current_time - self.last_time
                        print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] "
                              f"Пакетов: {self.packet_count} | "
                              f"Скорость: {self.packet_count / elapsed:.1f} пак/сек")
                        self.last_print_time = current_time
                else:
                    time.sleep(0.0001)

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

        self.BAUD_RATE = 2000000  # MCU: APB1=45MHz, BRR=23 → ~1.957 Мбод
        self.sport = serial.Serial(None, self.BAUD_RATE)

        self.X_AXIS_RANGE = 8  # в сек.
        self.SAMPLE_FREQ = 2048  # Гц

        self.serial_worker = None

        # Запись
        self.data_recorder = DataRecorder(self.SAMPLE_FREQ, NUM_CHANNELS)
        self.is_recording = False

        # Воспроизведение
        self.playback_worker = None
        self.playback_mode = False
        self._slider_dragging = False

        self._setup_ui()

        # Создаем каналы (количество задаётся константой NUM_CHANNELS вверху файла)
        self.channels = [
            EEGChannelWidget(f"Канал {i + 1}", self.X_AXIS_RANGE, (-500, 500), self.SAMPLE_FREQ)
            for i in range(NUM_CHANNELS)
        ]

        self._init_test_data()

        self.channel_checkboxes = []
        for i, ch in enumerate(self.channels):
            ch.setMinimumHeight(250)
            ch.setMaximumHeight(500)
            self.graph_layout.addWidget(ch)

            cb = QCheckBox(f"Канал {i + 1}")
            cb.setChecked(i == 0)
            ch.setVisible(i == 0)
            cb.toggled.connect(lambda checked, c=ch: c.setVisible(checked))
            self.channel_checks_layout.insertWidget(self.channel_checks_layout.count() - 1, cb)
            self.channel_checkboxes.append(cb)

        self.fourier_widget = FourierAnalysisWidget(
            channels=self.channels,
            sample_freq=self.SAMPLE_FREQ
        )
        self.tab_widget.addTab(self.fourier_widget, "Фурье-анализ")
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_all_channels)
        self.plot_timer.start(33)

    def _get_available_ports(self):
        """Получение списка доступных USB COM-портов"""
        ports = serial.tools.list_ports.comports()
        usb_ports = [
            f"{port.device} - {port.description}"
            for port in ports
            if port.vid is not None
        ]
        return usb_ports if usb_ports else ["Нет доступных портов"]

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

        toolbar.addSeparator()

        # Кнопка записи
        self.record_btn = QPushButton("Rec")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_btn.setEnabled(False)
        self.record_btn.setStyleSheet("QPushButton:checked { background-color: #ff4444; color: white; }")
        toolbar.addWidget(self.record_btn)

        toolbar.addSeparator()

        # Выбор единиц отображения мкВ / мВ
        toolbar.addWidget(QLabel("Единицы:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["мкВ", "мВ"])
        self.unit_combo.setToolTip("Единицы отображения амплитуды")
        self.unit_combo.currentTextChanged.connect(self._toggle_display_unit)
        toolbar.addWidget(self.unit_combo)

        toolbar.addSeparator()

        # Кнопка открытия файла
        self.open_btn = QPushButton("Открыть")
        self.open_btn.clicked.connect(self._open_recording)
        toolbar.addWidget(self.open_btn)

        # Кнопка воспроизведения/паузы
        self.play_btn = QPushButton("Воспр.")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        toolbar.addWidget(self.play_btn)

        # Кнопка остановки воспроизведения
        self.stop_playback_btn = QPushButton("Стоп")
        self.stop_playback_btn.clicked.connect(self._stop_playback)
        self.stop_playback_btn.setEnabled(False)
        toolbar.addWidget(self.stop_playback_btn)

        # Слайдер позиции воспроизведения
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setRange(0, 1000)
        self.playback_slider.setValue(0)
        self.playback_slider.setFixedWidth(150)
        self.playback_slider.setEnabled(False)
        self.playback_slider.sliderPressed.connect(self._on_slider_pressed)
        self.playback_slider.sliderReleased.connect(self._on_slider_released)
        self.playback_slider.valueChanged.connect(self._on_slider_value_changed)
        toolbar.addWidget(self.playback_slider)

        # Метка времени воспроизведения
        self.time_label = QLabel("00:00 / 00:00")
        toolbar.addWidget(self.time_label)

        # Кнопка информации о файле (скрыта до загрузки)
        self.file_info_btn = QPushButton("")
        self.file_info_btn.setFlat(True)
        self.file_info_btn.clicked.connect(self._show_file_info)
        self.file_info_action = toolbar.addWidget(self.file_info_btn)
        self.file_info_action.setVisible(False)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.tab_graph = QWidget()
        tab_graph_layout = QVBoxLayout()

        # Панель чекбоксов каналов
        self.channel_checks_layout = QHBoxLayout()
        self.channel_checks_layout.addWidget(QLabel("Каналы:"))
        self.channel_checks_layout.addStretch()
        tab_graph_layout.addLayout(self.channel_checks_layout)

        # Прокручиваемая область для графиков
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.graph_layout = QVBoxLayout()
        scroll_content.setLayout(self.graph_layout)
        scroll_area.setWidget(scroll_content)
        tab_graph_layout.addWidget(scroll_area)

        self.tab_graph.setLayout(tab_graph_layout)
        self.tab_widget.addTab(self.tab_graph, "Графики")

        self.tab_settings = QWidget()
        settings_layout = QVBoxLayout()

        # Пресеты фильтров
        preset_group = QGroupBox("Пресеты фильтров")
        preset_layout = QFormLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["— Выбрать пресет —", "ЭЭГ", "ЭКГ", "ЭМГ"])
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        preset_layout.addRow("Пресет:", self.preset_combo)
        preset_group.setLayout(preset_layout)
        settings_layout.addWidget(preset_group)

        # Группа Notch-фильтра
        notch_group = QGroupBox("Notch-фильтр (режекторный)")
        notch_layout = QFormLayout()

        self.notch_enabled_cb = QCheckBox("Включен")
        self.notch_enabled_cb.setChecked(True)
        notch_layout.addRow(self.notch_enabled_cb)

        self.notch_freq_spin = QDoubleSpinBox()
        self.notch_freq_spin.setRange(30.0, 70.0)
        self.notch_freq_spin.setValue(50.0)
        self.notch_freq_spin.setSuffix(" Гц")
        notch_layout.addRow("Частота:", self.notch_freq_spin)

        self.notch_q_spin = QDoubleSpinBox()
        self.notch_q_spin.setRange(5.0, 50.0)
        self.notch_q_spin.setValue(20.0)
        notch_layout.addRow("Добротность Q:", self.notch_q_spin)

        notch_group.setLayout(notch_layout)
        settings_layout.addWidget(notch_group)

        # Группа ФНЧ
        lowpass_group = QGroupBox("Низкочастотный фильтр (ФНЧ)")
        lowpass_layout = QFormLayout()

        self.lowpass_enabled_cb = QCheckBox("Включен")
        self.lowpass_enabled_cb.setChecked(False)
        lowpass_layout.addRow(self.lowpass_enabled_cb)

        self.lowpass_freq_spin = QDoubleSpinBox()
        self.lowpass_freq_spin.setRange(10.0, 500.0)
        self.lowpass_freq_spin.setValue(40.0)
        self.lowpass_freq_spin.setSuffix(" Гц")
        lowpass_layout.addRow("Частота среза:", self.lowpass_freq_spin)

        self.lowpass_order_spin = QSpinBox()
        self.lowpass_order_spin.setRange(2, 8)
        self.lowpass_order_spin.setValue(4)
        lowpass_layout.addRow("Порядок:", self.lowpass_order_spin)

        lowpass_group.setLayout(lowpass_layout)
        settings_layout.addWidget(lowpass_group)

        # Группа ФВЧ
        highpass_group = QGroupBox("Высокочастотный фильтр (ФВЧ)")
        highpass_layout = QFormLayout()

        self.highpass_enabled_cb = QCheckBox("Включен")
        self.highpass_enabled_cb.setChecked(False)
        highpass_layout.addRow(self.highpass_enabled_cb)

        self.highpass_freq_spin = QDoubleSpinBox()
        self.highpass_freq_spin.setRange(0.1, 10.0)
        self.highpass_freq_spin.setValue(0.5)
        self.highpass_freq_spin.setSuffix(" Гц")
        self.highpass_freq_spin.setDecimals(1)
        highpass_layout.addRow("Частота среза:", self.highpass_freq_spin)

        highpass_group.setLayout(highpass_layout)
        settings_layout.addWidget(highpass_group)

        # Группа антиалиасингового фильтра
        aa_group = QGroupBox("Антиалиасинговый фильтр (для дисплея)")
        aa_layout = QFormLayout()

        self.aa_enabled_cb = QCheckBox("Включен")
        self.aa_enabled_cb.setChecked(True)
        aa_layout.addRow(self.aa_enabled_cb)

        self.aa_freq_spin = QDoubleSpinBox()
        self.aa_freq_spin.setRange(20.0, 61.5)
        self.aa_freq_spin.setValue(55.0)
        self.aa_freq_spin.setSuffix(" Гц")
        aa_layout.addRow("Частота среза:", self.aa_freq_spin)

        self.aa_order_spin = QSpinBox()
        self.aa_order_spin.setRange(2, 8)
        self.aa_order_spin.setValue(4)
        aa_layout.addRow("Порядок:", self.aa_order_spin)

        aa_group.setLayout(aa_layout)
        settings_layout.addWidget(aa_group)

        # Кнопка применения настроек
        apply_btn = QPushButton("Применить настройки")
        apply_btn.clicked.connect(self._apply_filter_settings)
        settings_layout.addWidget(apply_btn)

        settings_layout.addStretch()
        self.tab_settings.setLayout(settings_layout)
        self.tab_widget.addTab(self.tab_settings, "Настройки")

    def _toggle_display_unit(self, unit: str):
        """Переключить единицы отображения всех каналов: мкВ / мВ"""
        for ch in self.channels:
            ch.set_display_unit(unit)
        self.fourier_widget.set_display_unit(unit)

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
        """Инициализация тестовыми синусоидами (в мкВ)"""
        x = np.linspace(0, self.X_AXIS_RANGE, self.X_AXIS_RANGE * self.SAMPLE_FREQ)
        for i, ch in enumerate(self.channels):
            freq = i + 1
            amplitude = 200 - i * 20  # от 200 до 60 мкВ
            for val in x:
                ch.append_data(np.sin(2 * np.pi * val * freq) * amplitude)

    def _apply_filter_settings(self):
        """Применение настроек фильтров к обоим каналам"""
        # Notch-фильтр
        notch_enabled = self.notch_enabled_cb.isChecked()
        notch_freq = self.notch_freq_spin.value()
        notch_q = self.notch_q_spin.value()

        # ФНЧ
        lowpass_enabled = self.lowpass_enabled_cb.isChecked()
        lowpass_freq = self.lowpass_freq_spin.value()
        lowpass_order = self.lowpass_order_spin.value()

        # ФВЧ
        highpass_enabled = self.highpass_enabled_cb.isChecked()
        highpass_freq = self.highpass_freq_spin.value()

        # Антиалиасинговый фильтр
        aa_enabled = self.aa_enabled_cb.isChecked()
        aa_freq = self.aa_freq_spin.value()
        aa_order = self.aa_order_spin.value()

        # Применить ко всем каналам
        for channel in self.channels:
            channel.set_notch_filter(notch_enabled, notch_freq, notch_q)
            channel.set_lowpass_filter(lowpass_enabled, lowpass_freq, lowpass_order)
            channel.set_highpass_filter(highpass_enabled, highpass_freq)
            channel.set_antialiasing_filter(aa_enabled, aa_freq, aa_order)

        print(f"Настройки фильтров применены: "
              f"Notch={notch_enabled}({notch_freq}Гц, Q={notch_q}), "
              f"ФНЧ={lowpass_enabled}({lowpass_freq}Гц, порядок={lowpass_order}), "
              f"ФВЧ={highpass_enabled}({highpass_freq}Гц), "
              f"AA={aa_enabled}({aa_freq}Гц, порядок={aa_order})")

    def _apply_preset(self, index):
        """Применить пресет фильтров"""
        presets = {
            1: {'hp': 0.5, 'lp': 45, 'notch': 50, 'lp_order': 4},   # ЭЭГ
            2: {'hp': 0.5, 'lp': 150, 'notch': 50, 'lp_order': 4},  # ЭКГ
            3: {'hp': 10.0, 'lp': 500, 'notch': 50, 'lp_order': 4}, # ЭМГ
        }
        if index not in presets:
            return
        p = presets[index]
        self.notch_enabled_cb.setChecked(True)
        self.notch_freq_spin.setValue(p['notch'])
        self.lowpass_enabled_cb.setChecked(True)
        self.lowpass_freq_spin.setValue(p['lp'])
        self.lowpass_order_spin.setValue(p['lp_order'])
        self.highpass_enabled_cb.setChecked(True)
        self.highpass_freq_spin.setValue(p['hp'])
        self._apply_filter_settings()

    # ==================== Запись ====================

    def _toggle_recording(self, checked: bool):
        """Включить/выключить запись"""
        if checked:
            self.data_recorder.start_recording()
            self.is_recording = True
            self.record_btn.setText("Stop")
            print("Запись начата")
        else:
            self.is_recording = False
            self.record_btn.setText("Rec")
            self._save_recording()

    def _save_recording(self):
        """Сохранить запись в файл"""
        if self.data_recorder.get_samples_count() == 0:
            print("Нет данных для сохранения")
            return

        suggested_name = self.data_recorder.stop_recording()
        duration = self.data_recorder.get_duration()

        # Диалог описания записи
        dialog = QDialog(self)
        dialog.setWindowTitle("Описание записи")
        dialog.setMinimumWidth(400)
        dlg_layout = QVBoxLayout()

        start_time_str = self.data_recorder.start_time.strftime('%H:%M:%S') if self.data_recorder.start_time else "—"
        dlg_layout.addWidget(QLabel(f"Время записи: {start_time_str}, длительность: {duration:.1f} сек"))

        dlg_layout.addWidget(QLabel("Описание:"))
        notes_edit = QTextEdit()
        notes_edit.setPlaceholderText("Введите описание записи...")
        notes_edit.setMaximumHeight(120)
        dlg_layout.addWidget(notes_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dlg_layout.addWidget(buttons)

        dialog.setLayout(dlg_layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            print("Сохранение отменено")
            return

        notes = notes_edit.toPlainText()

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить запись", suggested_name,
            "EEG Recording (*.eeg)"
        )

        if filepath:
            self.data_recorder.save(filepath, notes=notes)
            print(f"Запись сохранена: {filepath} ({duration:.1f} сек)")
        else:
            print("Сохранение отменено")

    def _get_filter_settings(self) -> dict:
        """Получить текущие настройки фильтров"""
        return {
            'notch': {
                'enabled': self.notch_enabled_cb.isChecked(),
                'freq': self.notch_freq_spin.value(),
                'q': self.notch_q_spin.value()
            },
            'lowpass': {
                'enabled': self.lowpass_enabled_cb.isChecked(),
                'freq': self.lowpass_freq_spin.value(),
                'order': self.lowpass_order_spin.value()
            },
            'highpass': {
                'enabled': self.highpass_enabled_cb.isChecked(),
                'freq': self.highpass_freq_spin.value()
            },
            'antialiasing': {
                'enabled': self.aa_enabled_cb.isChecked(),
                'freq': self.aa_freq_spin.value(),
                'order': self.aa_order_spin.value()
            }
        }

    # ==================== Воспроизведение ====================

    def _open_recording(self):
        """Открыть файл записи для воспроизведения"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Открыть запись", "",
            "EEG Recording (*.eeg)"
        )

        if filepath:
            try:
                self.playback_worker = PlaybackWorker()
                self.playback_worker.load(filepath)
                self.playback_worker.data_received.connect(self.update_data)
                self.playback_worker.playback_finished.connect(self._on_playback_finished)
                self.playback_worker.playback_progress.connect(self._on_playback_progress)

                # Обновляем UI
                self.play_btn.setEnabled(True)
                self.stop_playback_btn.setEnabled(True)
                self.playback_slider.setEnabled(True)
                self.playback_slider.setValue(0)

                # Показываем длительность
                duration = self.playback_worker.get_duration()
                self.time_label.setText(f"00:00 / {self._format_time(duration)}")

                metadata = self.playback_worker.get_metadata()
                file_freq = metadata.get('sample_freq', self.SAMPLE_FREQ)

                # Установить частоту из файла
                for ch in self.channels:
                    ch.set_sample_freq(file_freq)
                self._clear_channel_data()
                self.fourier_widget.sample_rate = file_freq

                # Показать имя файла
                import os
                fname = os.path.basename(filepath)
                self.file_info_btn.setText(f"[{fname}]")
                self.file_info_action.setVisible(True)
                self._loaded_filepath = filepath

                print(f"Загружен файл: {filepath}")
                print(f"  Длительность: {duration:.1f} сек, Частота: {file_freq} Гц")

            except Exception as e:
                print(f"Ошибка загрузки файла: {e}")

    def _toggle_playback(self):
        """Запуск/пауза воспроизведения"""
        if self.playback_worker is None:
            return

        if not self.playback_worker.isRunning():
            # Запускаем воспроизведение
            self.playback_mode = True
            self._clear_channel_data()
            self.playback_worker.reset()
            self.playback_worker.start()
            self.play_btn.setText("Пауза")
            # Отключаем serial controls
            self.connect_btn.setEnabled(False)
            self.port_combo.setEnabled(False)
            self.open_btn.setEnabled(False)
            print("Воспроизведение начато")
        elif self.playback_worker.paused:
            # Возобновляем
            self.playback_worker.resume()
            self.play_btn.setText("Пауза")
            print("Воспроизведение возобновлено")
        else:
            # Ставим на паузу
            self.playback_worker.pause()
            self.play_btn.setText("Продолжить")
            print("Воспроизведение приостановлено")

    def _stop_playback(self):
        """Остановить воспроизведение"""
        if self.playback_worker:
            self.playback_worker.stop()
            self.playback_worker.wait()

        self.playback_mode = False
        self.play_btn.setText("Воспр.")
        self.playback_slider.setValue(0)

        # Восстанавливаем исходную частоту
        for ch in self.channels:
            ch.set_sample_freq(self.SAMPLE_FREQ)
        self._clear_channel_data()
        self.fourier_widget.sample_rate = self.SAMPLE_FREQ

        # Включаем serial controls
        self.connect_btn.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.open_btn.setEnabled(True)

        print("Воспроизведение остановлено")

    def _on_playback_finished(self):
        """Обработка окончания воспроизведения"""
        self.playback_mode = False
        self.play_btn.setText("Воспр.")

        # Восстанавливаем исходную частоту
        for ch in self.channels:
            ch.set_sample_freq(self.SAMPLE_FREQ)
        self._clear_channel_data()
        self.fourier_widget.sample_rate = self.SAMPLE_FREQ

        # Включаем serial controls
        self.connect_btn.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.open_btn.setEnabled(True)

        print("Воспроизведение завершено")

    def _show_file_info(self):
        """Показать информацию о загруженном файле"""
        if not self.playback_worker or not self.playback_worker.metadata:
            return

        meta = self.playback_worker.metadata
        import os
        fname = os.path.basename(getattr(self, '_loaded_filepath', ''))

        dialog = QDialog(self)
        dialog.setWindowTitle("Информация о записи")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()

        info_text = (
            f"Файл: {fname}\n"
            f"Время записи: {meta.get('created_at', '—')}\n"
            f"Длительность: {meta.get('duration_seconds', 0):.1f} сек\n"
            f"Частота дискретизации: {meta.get('sample_freq', '—')} Гц\n"
            f"Каналов: {meta.get('channels', '—')}\n"
            f"Сэмплов: {meta.get('samples_count', '—')}\n"
            f"Версия формата: {meta.get('version', '—')}\n"
            f"\nОписание: {meta.get('notes', '') or '—'}"
        )

        label = QLabel(info_text)
        label.setWordWrap(True)
        layout.addWidget(label)

        btn = QPushButton("Закрыть")
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)

        dialog.setLayout(layout)
        dialog.exec()

    def _on_playback_progress(self, current_sec: float, total_sec: float):
        """Обновление прогресса воспроизведения"""
        if not self._slider_dragging:
            progress = int((current_sec / total_sec) * 1000) if total_sec > 0 else 0
            self.playback_slider.setValue(progress)
        self.time_label.setText(f"{self._format_time(current_sec)} / {self._format_time(total_sec)}")

    def _on_slider_pressed(self):
        """Пользователь начал перетаскивать слайдер"""
        self._slider_dragging = True

    def _on_slider_released(self):
        """Пользователь отпустил слайдер"""
        self._slider_dragging = False
        if self.playback_worker and self.playback_worker.is_loaded():
            position = self.playback_slider.value() / 1000.0
            self.playback_worker.seek(position)
            self._clear_channel_data()

    def _on_slider_value_changed(self, value: int):
        """Изменение значения слайдера"""
        if self._slider_dragging and self.playback_worker:
            duration = self.playback_worker.get_duration()
            current = (value / 1000.0) * duration
            self.time_label.setText(f"{self._format_time(current)} / {self._format_time(duration)}")

    def _clear_channel_data(self):
        """Очистить буферы каналов"""
        for ch in self.channels:
            ch.clear_data()

    def _disable_playback_controls(self):
        """Отключить элементы управления воспроизведением"""
        self.open_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.stop_playback_btn.setEnabled(False)
        self.playback_slider.setEnabled(False)

    def _enable_playback_controls(self):
        """Включить элементы управления воспроизведением"""
        self.open_btn.setEnabled(True)
        if self.playback_worker and self.playback_worker.is_loaded():
            self.play_btn.setEnabled(True)
            self.stop_playback_btn.setEnabled(True)
            self.playback_slider.setEnabled(True)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Форматировать время как MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def connect_sport(self):
        """Подключение к serial порту"""
        port_text = self.port_combo.currentText()
        port = port_text.split(" - ")[0]
        try:
            self.sport = serial.Serial(port, self.BAUD_RATE)
            self.serial_worker = SerialWorker(self.sport)
            self.serial_worker.data_received.connect(self.update_data)
            self.serial_worker.start()
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.port_combo.setEnabled(False)
            # Включаем запись, отключаем воспроизведение
            self.record_btn.setEnabled(True)
            self._disable_playback_controls()
            print(f"Подключено к {port}")
        except SerialException as e:
            print(f"Ошибка подключения: {e}")

    def disconnect_sport(self):
        """Отключение от serial порта"""
        # Остановить запись если активна
        if self.is_recording:
            self._toggle_recording(False)
            self.record_btn.setChecked(False)

        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker = None

        if self.sport.is_open:
            self.sport.close()

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.port_combo.setEnabled(True)
        # Отключаем запись, включаем воспроизведение
        self.record_btn.setEnabled(False)
        self._enable_playback_controls()

        print("Отключено")

    def update_data(self, data):
        """Обновление данных по событию от воркера"""
        if data is not None:
            fft_ch = self.fourier_widget.selected_channel_index
            for i, ch in enumerate(self.channels):
                if i < len(data) and (self.channel_checkboxes[i].isChecked() or i == fft_ch):
                    ch.append_data(data[i])
            # Записываем если активна запись (и не в режиме воспроизведения)
            if self.is_recording and not self.playback_mode:
                self.data_recorder.add_sample(data)

    def _update_all_channels(self):
        """Обновление всех графиков по таймеру"""
        for ch in self.channels:
            if ch.isVisible():
                ch.update_display()

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.serial_worker:
            self.serial_worker.stop()
        if self.playback_worker:
            self.playback_worker.stop()
            self.playback_worker.wait()
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
