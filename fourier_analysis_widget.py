import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QTableWidget, QTableWidgetItem, QHeaderView)
from scipy import fft


class FourierAnalysisWidget(QWidget):
    """Виджет для Фурье-анализа ЭЭГ сигналов"""

    def __init__(self, channels, sample_rate=100):
        super().__init__()

        self.channels = channels
        self.sample_rate = sample_rate
        self.selected_channel_index = 0

        self._setup_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_analysis)
        self.update_timer.start(1000)

    def _setup_ui(self):
        """Настройка интерфейса"""
        layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Выбор канала:"))

        self.channel_combo = QComboBox()
        for i, channel in enumerate(self.channels):
            self.channel_combo.addItem(f"Канал {i + 1}")
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        control_layout.addWidget(self.channel_combo)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground('w')
        self.spectrum_plot.setLabel('left', 'Амплитуда', units='мкВ')
        self.spectrum_plot.setLabel('bottom', 'Частота', units='Гц')
        self.spectrum_plot.setTitle('Частотный спектр')
        self.spectrum_plot.showGrid(x=True, y=True)
        self.spectrum_plot.setMouseEnabled(x=False, y=False)
        layout.addWidget(self.spectrum_plot)
        harmonics_label = QLabel("Доминирующие гармоники:")
        layout.addWidget(harmonics_label)

        self.harmonics_table = QTableWidget()
        self.harmonics_table.setColumnCount(3)
        self.harmonics_table.setHorizontalHeaderLabels(['№', 'Частота (Гц)', 'Амплитуда (мВ)'])
        self.harmonics_table.setRowCount(5)
        self.harmonics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.harmonics_table.setMaximumHeight(200)

        font = QFont()
        font.setPointSize(12)
        self.harmonics_table.setFont(font)

        for i in range(5):
            self.harmonics_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.harmonics_table.setItem(i, 1, QTableWidgetItem('-'))
            self.harmonics_table.setItem(i, 2, QTableWidgetItem('-'))

        layout.addWidget(self.harmonics_table)

        self.setLayout(layout)

    def on_channel_changed(self, index):
        """Обработка изменения выбранного канала"""
        self.selected_channel_index = index
        self.update_analysis()

    def update_analysis(self):
        """Обновление Фурье-анализа"""
        try:
            channel = self.channels[self.selected_channel_index]
            signal_data = channel.data_to_display.get_buffer()
            signal = np.asarray(signal_data, dtype=np.float32)

            if len(signal) < 10:
                return

            N = len(signal)
            fft_values = fft.fft(signal)
            fft_magnitude = np.abs(fft_values[:N // 2]) / N
            frequencies = fft.fftfreq(N, 1 / self.sample_rate)[:N // 2]

            start_index = np.searchsorted(frequencies, 1.0)

            self.spectrum_plot.clear()
            self.spectrum_plot.plot(
                frequencies[start_index:],
                fft_magnitude[start_index:],
                pen=pg.mkPen(color='b', width=2)
            )

            top_indices = np.argsort(fft_magnitude[start_index:])[-5:][::-1] + start_index

            for i, idx in enumerate(top_indices):
                freq = frequencies[idx]
                amplitude = fft_magnitude[idx]

                self.harmonics_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.harmonics_table.setItem(i, 1, QTableWidgetItem(f'{freq:.2f}'))
                self.harmonics_table.setItem(i, 2, QTableWidgetItem(f'{amplitude:.2f}'))

        except Exception as e:
            print(f"Ошибка при выполнении Фурье-анализа: {e}")
            import traceback
            traceback.print_exc()
