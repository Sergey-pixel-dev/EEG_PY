import sys
import time
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
from protocol import SerProtParser, SerProtMaster, SerProtFrame

NUM_CHANNELS = 8
BYTES_PER_CHANNEL = 3

TYPE_PUSH = 0xFF


class SerialWorker(QThread):
    data_received = pyqtSignal(np.ndarray)
    response_received = pyqtSignal(int, int, bytes)  # type, seq, payload

    def __init__(self, sport, sample_freq: int = 250):
        super().__init__()
        self.sport = sport
        self.sample_freq = sample_freq
        self.running = False
        self.parser = SerProtParser()

        # Профилирование — скорость за последнюю секунду
        self.packet_count = 0
        self.last_time = time.perf_counter()
        self.last_print_time = self.last_time

        # Drift-анализ (теоретическое vs фактическое)
        self.total_packets = 0
        self.start_time = 0.0

    def run(self):
        self.running = True
        if not self.sport.is_open:
            return

        self.start_time = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] SerialWorker запущен (sample_freq={self.sample_freq})")

        while self.running:
            try:
                # Агрессивный polling без sleep — критично для 2 Мбод
                raw_data = self.sport.read(4096)
                if raw_data:
                    for b in raw_data:
                        frame = self.parser.process_byte(b)
                        if frame is not None:
                            if frame.type == TYPE_PUSH:
                                samples = SerProtMaster.decode_adc_samples(frame.payload)
                                self.packet_count += 1
                                self.total_packets += 1
                                self.data_received.emit(np.array(samples, dtype=np.float32))
                            elif frame.type in (SerProtMaster.TYPE_RESPONSE, SerProtMaster.TYPE_ERROR):
                                self.response_received.emit(frame.type, frame.seq, frame.payload)

                    current_time = time.perf_counter()
                    if current_time - self.last_print_time >= 5.0:
                        elapsed_total = current_time - self.start_time
                        expected = int(elapsed_total * self.sample_freq)
                        actual = self.total_packets
                        drift = actual - expected
                        drift_pct = (drift / expected * 100.0) if expected > 0 else 0.0

                        instant_elapsed = current_time - self.last_print_time
                        instant_rate = self.packet_count / instant_elapsed if instant_elapsed > 0 else 0

                        print(f"[DRIFT] Expected: {expected:>7} pkts | "
                              f"Actual: {actual:>7} pkts | "
                              f"Drift: {drift:+7d} ({drift_pct:+6.2f}%) | "
                              f"Instant: {instant_rate:>6.1f} pps")
                        self.packet_count = 0
                        self.last_print_time = current_time

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Ошибка: {e}")
                self.running = False

    def set_sample_freq(self, sample_freq: int):
        """Обновить эталонную частоту для drift-анализа (при смене samplerate)."""
        self.sample_freq = sample_freq

    def reset_stats(self):
        """Сбросить счётчики пакетов (при смене конфигурации)."""
        self.packet_count = 0
        self.total_packets = 0
        self.start_time = time.perf_counter()
        self.last_time = time.perf_counter()
        self.last_print_time = self.last_time

    def stop(self):
        print(f"[{time.strftime('%H:%M:%S')}] SerialWorker останавливается. "
              f"Всего пакетов получено: {self.total_packets}")
        self.running = False
        self.wait()


class EEGPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЭЭГ")
        self.setGeometry(100, 100, 1400, 800)

        self.BAUD_RATE = 921600  # FT232R стабилен на 921600, не на 2M
        self.sport = None

        self.X_AXIS_RANGE = 8  # в сек.
        self.SAMPLE_FREQ = 250  # Гц (по умолчанию)

        self.serial_worker = None
        self.serprot = SerProtMaster()

        # Активные каналы (по умолчанию только канал 0)
        self.active_channels = [0]

        # Ожидание ответа на команду
        self._pending_cmd_seq: Optional[int] = None
        self._pending_cmd_name: str = ""
        self._pending_cmd_time: float = 0.0
        self._after_response_callback = None
        self._cmd_timeout_timer = QTimer()
        self._cmd_timeout_timer.timeout.connect(self._check_cmd_timeout)
        self._cmd_timeout_timer.setInterval(100)  # 100 мс

        # Запись
        self.data_recorder = DataRecorder(self.SAMPLE_FREQ, self.active_channels)
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
            cb.toggled.connect(lambda checked, idx=i: self._on_channel_toggled(idx, checked))
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

        # Частота дискретизации
        toolbar.addWidget(QLabel("Частота:"))
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["250 Гц", "500 Гц", "1 кГц", "4 кГц"])
        self.samplerate_combo.setCurrentIndex(0)  # 250 Гц по умолчанию
        self.samplerate_combo.currentIndexChanged.connect(self._on_samplerate_changed)
        toolbar.addWidget(self.samplerate_combo)

        toolbar.addSeparator()

        # Опорное напряжение
        toolbar.addWidget(QLabel("Vref:"))
        self.vref_spin = QSpinBox()
        self.vref_spin.setRange(0, 3300)
        self.vref_spin.setValue(1000)
        self.vref_spin.setSuffix(" мВ")
        self.vref_spin.setSingleStep(100)
        self.vref_spin.valueChanged.connect(self._on_vref_changed)
        toolbar.addWidget(self.vref_spin)

        toolbar.addSeparator()

        # Усиление (Gain) для RTI
        toolbar.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.01, 10000.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        toolbar.addWidget(self.gain_spin)

        toolbar.addSeparator()

        # Статус / ошибки
        self.status_label = QLabel("Не подключено")
        self.status_label.setStyleSheet("color: gray;")
        toolbar.addWidget(self.status_label)

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

    # ==================== Команды к MCU ====================

    def _send_command(self, cmd_id: int, payload: bytes = b"", cmd_name: str = "", on_response=None):
        """Отправить команду через SerProt, если порт открыт. Ожидать ответ 100 мс."""
        if self.sport and self.sport.is_open:
            frame = self.serprot.build_command(cmd_id, payload)
            self.sport.write(frame)
            self.sport.flush()
            self._pending_cmd_seq = self.serprot._seq - 1  # seq уже инкрементирован
            self._pending_cmd_name = cmd_name or f"cmd=0x{cmd_id:02X}"
            self._pending_cmd_time = time.perf_counter()
            self._after_response_callback = on_response
            self._cmd_timeout_timer.start()
            print(f"[TX] {frame.hex()}  (seq={self._pending_cmd_seq:02X}, {self._pending_cmd_name})")
            self.status_label.setText(f"Ожидание: {self._pending_cmd_name}...")
            self.status_label.setStyleSheet("color: orange;")

    def _send_start_stream(self, on_response=None):
        """Отправить команду старта стриминга с активными каналами."""
        payload = bytes([len(self.active_channels)] + self.active_channels)
        self._send_command(0x01, payload, cmd_name="StartStream", on_response=on_response)
        print(f"[CMD] Start stream: channels={self.active_channels}")

    def _send_stop_stream(self, on_response=None):
        """Отправить команду остановки стриминга."""
        self._send_command(0x02, cmd_name="StopStream", on_response=on_response)
        print("[CMD] Stop stream")

    def _send_set_vref(self, mv: int):
        """Отправить команду установки опорного напряжения."""
        payload = bytes([mv & 0xFF, (mv >> 8) & 0xFF])
        self._send_command(0x10, payload, cmd_name="SetVref")
        print(f"[CMD] Set Vref: {mv} mV")

    def _send_set_samplerate(self, idx: int, on_response=None):
        """Отправить команду установки частоты дискретизации."""
        self._send_command(0x11, bytes([idx]), cmd_name="SetSamplerate", on_response=on_response)
        freq_map = {0: 250, 1: 500, 2: 1000, 3: 4000}
        print(f"[CMD] Set samplerate: {freq_map.get(idx, '?')} Hz (idx={idx})")

    def _check_cmd_timeout(self):
        """Проверить таймаут ответа на команду."""
        if self._pending_cmd_seq is None:
            self._cmd_timeout_timer.stop()
            return
        elapsed = (time.perf_counter() - self._pending_cmd_time) * 1000
        if elapsed > 100:  # 100 мс таймаут
            self._cmd_timeout_timer.stop()
            print(f"[TIMEOUT] Команда {self._pending_cmd_name} (seq={self._pending_cmd_seq:02X}) — нет ответа за {elapsed:.0f} мс")
            self.status_label.setText(f"ОШИБКА: таймаут {self._pending_cmd_name}")
            self.status_label.setStyleSheet("color: red;")
            self._pending_cmd_seq = None
            self._after_response_callback = None

    def _on_response_received(self, type_byte: int, seq: int, payload: bytes):
        """Обработка Response (0xDD) или Error (0xEE) от MCU."""
        self._cmd_timeout_timer.stop()
        matched = False
        if type_byte == SerProtMaster.TYPE_RESPONSE:
            print(f"[RX] Response seq={seq:02X} len={len(payload)} {payload.hex()}")
            if self._pending_cmd_seq is not None and seq == self._pending_cmd_seq:
                matched = True
                self.status_label.setText("OK: " + self._pending_cmd_name)
                self.status_label.setStyleSheet("color: green;")
                self._pending_cmd_seq = None
            else:
                self.status_label.setText(f"Response seq={seq:02X}")
                self.status_label.setStyleSheet("color: green;")
        elif type_byte == SerProtMaster.TYPE_ERROR:
            error_code = payload[0] if payload else 0xFF
            error_names = {
                0x01: "Unknown command",
                0x02: "Invalid params",
                0x03: "Device not ready",
                0x04: "HW error",
                0x05: "CRC mismatch",
                0x06: "Invalid payload size",
            }
            err_str = error_names.get(error_code, f"code=0x{error_code:02X}")
            print(f"[RX] Error seq={seq:02X} {err_str}")
            self.status_label.setText(f"ОШИБКА MCU: {err_str}")
            self.status_label.setStyleSheet("color: red;")
            self._pending_cmd_seq = None

        if matched and self._after_response_callback:
            cb = self._after_response_callback
            self._after_response_callback = None
            cb()

    def _on_channel_toggled(self, idx: int, checked: bool):
        """Обработка переключения видимости канала."""
        self.channels[idx].setVisible(checked)
        # Пересчитываем активные каналы
        new_active = [i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()]
        if not new_active:
            # Не даём отключить все каналы — оставляем текущий
            self.channel_checkboxes[idx].setChecked(True)
            return
        if new_active != self.active_channels:
            self._on_channels_changed(new_active)

    def _on_channels_changed(self, new_active: list[int]):
        """Смена активных каналов: остановка, ожидание Response, сброс, старт."""
        print(f"[UI] Channels changed: {self.active_channels} -> {new_active}")
        self.active_channels = new_active
        self.data_recorder = DataRecorder(self.SAMPLE_FREQ, self.active_channels)

        if self.sport and self.sport.is_open:
            def do_start():
                if self.serial_worker:
                    self.serial_worker.parser.reset()
                    self.serial_worker.reset_stats()
                for ch in self.channels:
                    ch.clear_data()
                self._send_start_stream()

            self._send_stop_stream(on_response=do_start)

    def _on_samplerate_changed(self, index: int):
        """Смена частоты дискретизации: остановка, ожидание Response, сброс, старт."""
        freq_map = {0: 250, 1: 500, 2: 1000, 3: 4000}
        new_freq = freq_map.get(index, 1000)
        if new_freq == self.SAMPLE_FREQ:
            return

        print(f"[UI] Samplerate changed: {self.SAMPLE_FREQ} -> {new_freq}")
        self.SAMPLE_FREQ = new_freq

        # Обновить все каналы и Fourier
        for ch in self.channels:
            ch.set_sample_freq(new_freq)
        self.fourier_widget.sample_rate = new_freq
        self.data_recorder = DataRecorder(self.SAMPLE_FREQ, self.active_channels)
        self._clear_channel_data()

        if self.sport and self.sport.is_open:
            def do_start():
                if self.serial_worker:
                    self.serial_worker.parser.reset()
                    self.serial_worker.set_sample_freq(self.SAMPLE_FREQ)
                    self.serial_worker.reset_stats()
                for ch in self.channels:
                    ch.clear_data()
                self._send_start_stream()

            def do_set_samplerate():
                self._send_set_samplerate(index, on_response=do_start)

            def after_stop():
                # Даём MCU ~150 мс выдать хвост из TX-буфера
                QTimer.singleShot(150, do_set_samplerate)

            self._send_stop_stream(on_response=after_stop)

    def _on_vref_changed(self, mv: int):
        """Смена опорного напряжения — сразу отправляем команду."""
        self._send_set_vref(mv)

    def _on_gain_changed(self, value: float):
        """Смена усиления — обновляем RTI на всех каналах."""
        for ch in self.channels:
            ch.update_noise_display(value)

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
            self.playback_worker.reset()

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
        if self.playback_worker:
            self.playback_worker.reset()

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

    def _check_latency_timer(self, port: str) -> int:
        """Проверить FTDI latency timer (критично для 2 Мбод)."""
        try:
            import glob
            for p in glob.glob('/sys/bus/usb-serial/devices/*/latency_timer'):
                if port in p or port.replace('/dev/', '') in p:
                    with open(p) as f:
                        return int(f.read().strip())
        except Exception:
            pass
        return -1

    def connect_sport(self):
        """Подключение к serial порту"""
        port_text = self.port_combo.currentText()
        port = port_text.split(" - ")[0]
        try:
            self.sport = serial.Serial(
                port=port,
                baudrate=self.BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0,
                rtscts=False,
                dsrdtr=False,
                xonxoff=False,
            )
            self.sport.rts = False
            self.sport.dtr = False
            self.sport.reset_input_buffer()
            self.sport.reset_output_buffer()
            print(f"[Serial] Opened {port} at {self.sport.baudrate} baud")

            latency = self._check_latency_timer(port)
            if latency > 1:
                print(f"[WARN] FTDI latency_timer = {latency} ms! Рекомендуется 1 мс для 2 Мбод.")
                print(f"       sudo sh -c 'echo 1 > /sys/bus/usb-serial/devices/{port.replace('/dev/', '')}/latency_timer'")
            elif latency == 1:
                print(f"[OK] FTDI latency_timer = 1 ms")
            self.serial_worker = SerialWorker(self.sport, sample_freq=self.SAMPLE_FREQ)
            self.serial_worker.data_received.connect(self.update_data)
            self.serial_worker.response_received.connect(self._on_response_received)
            self.serial_worker.start()
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.port_combo.setEnabled(False)
            # Включаем запись, отключаем воспроизведение
            self.record_btn.setEnabled(True)
            self._disable_playback_controls()

            # Инициализация MCU (интервал ~2 мс между командами)
            self._send_set_samplerate(self.samplerate_combo.currentIndex())
            time.sleep(0.002)
            self._send_set_vref(self.vref_spin.value())
            time.sleep(0.002)
            self._send_start_stream()
            self._clear_channel_data()

            self.status_label.setText("Подключено")
            self.status_label.setStyleSheet("color: green;")
            print(f"Подключено к {port}")
        except SerialException as e:
            self.status_label.setText(f"Ошибка: {e}")
            self.status_label.setStyleSheet("color: red;")
            print(f"Ошибка подключения: {e}")

    def disconnect_sport(self):
        """Отключение от serial порта"""
        # Остановить запись если активна
        if self.is_recording:
            self._toggle_recording(False)
            self.record_btn.setChecked(False)

        # Остановить стрим на MCU
        self._send_stop_stream()

        # Сбросить ожидание ответа
        self._pending_cmd_seq = None
        self._cmd_timeout_timer.stop()

        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker = None

        if self.sport and self.sport.is_open:
            self.sport.close()

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.port_combo.setEnabled(True)
        # Отключаем запись, включаем воспроизведение
        self.record_btn.setEnabled(False)
        self._enable_playback_controls()

        self.status_label.setText("Отключено")
        self.status_label.setStyleSheet("color: gray;")
        print("Отключено")

    def update_data(self, data):
        """Обновление данных по событию от воркера или плеера"""
        if data is not None:
            fft_ch = self.fourier_widget.selected_channel_index
            if self.playback_mode:
                # Playback: data — полный массив для всех NUM_CHANNELS каналов
                for ch_idx in self.active_channels:
                    if ch_idx < len(data):
                        ch = self.channels[ch_idx]
                        if self.channel_checkboxes[ch_idx].isChecked() or ch_idx == fft_ch:
                            ch.append_data(data[ch_idx])
            else:
                # Real-time: data — массив только для активных каналов
                for i, ch_idx in enumerate(self.active_channels):
                    if i < len(data):
                        ch = self.channels[ch_idx]
                        if self.channel_checkboxes[ch_idx].isChecked() or ch_idx == fft_ch:
                            ch.append_data(data[i])
            # Записываем если активна запись (и не в режиме воспроизведения)
            if self.is_recording and not self.playback_mode:
                self.data_recorder.add_sample(data)

    def _update_all_channels(self):
        """Обновление всех графиков по таймеру"""
        gain = self.gain_spin.value()
        for ch in self.channels:
            if ch.isVisible():
                ch.update_display()
                ch.update_noise_display(gain)

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.serial_worker:
            self.serial_worker.stop()
        if self.playback_worker:
            self.playback_worker.stop()
            self.playback_worker.wait()
        if self.sport and self.sport.is_open:
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
