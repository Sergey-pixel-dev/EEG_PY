import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QCheckBox, QScrollArea)
from scipy import fft


class ZoomableViewBox(pg.ViewBox):
    def __init__(self, default_x_range=None, default_y_range=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(pg.ViewBox.RectMode)
        self.default_x_range = default_x_range
        self.default_y_range = default_y_range

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            if self.default_x_range is not None or self.default_y_range is not None:
                if self.default_x_range is not None:
                    self.setXRange(self.default_x_range[0], self.default_x_range[1], padding=0)
                if self.default_y_range is not None:
                    self.setYRange(self.default_y_range[0], self.default_y_range[1], padding=0)
            else:
                self.autoRange()
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class FourierAnalysisWidget(QWidget):
    """Виджет для Фурье-анализа ЭЭГ сигналов"""

    def __init__(self, channels, sample_freq=100):
        super().__init__()

        self.channels = channels
        self.sample_rate = sample_freq
        self.selected_channel_index = 0

        self.raw_visible = True
        self.filtered_visible = True

        self.display_unit = 'мкВ'
        self.display_scale = np.float32(1.0)

        self._setup_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_analysis)
        self.update_timer.start(1000)

    def _setup_ui(self):
        """Настройка интерфейса"""
        main_layout = QVBoxLayout()

        # Выбор канала
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Выбор канала:"))

        self.channel_combo = QComboBox()
        for i, channel in enumerate(self.channels):
            self.channel_combo.addItem(f"Канал {i + 1}")
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        control_layout.addWidget(self.channel_combo)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # Прокручиваемая область
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()

        # --- Спектр до фильтров ---
        self.raw_checkbox = QCheckBox("Спектр до фильтров")
        self.raw_checkbox.setChecked(True)
        self.raw_checkbox.toggled.connect(self._toggle_raw)
        scroll_layout.addWidget(self.raw_checkbox)

        self.spectrum_plot_raw = pg.PlotWidget(viewBox=ZoomableViewBox())
        self.spectrum_plot_raw.setBackground('w')
        self.spectrum_plot_raw.setLabel('left', 'Амплитуда', units='мкВ')
        self.spectrum_plot_raw.setLabel('bottom', 'Частота', units='Гц')
        self.spectrum_plot_raw.setTitle('Частотный спектр (до фильтров)')
        self.spectrum_plot_raw.getAxis('left').enableAutoSIPrefix(False)
        self.spectrum_plot_raw.getAxis('bottom').enableAutoSIPrefix(False)
        self.spectrum_plot_raw.showGrid(x=True, y=True)
        self.spectrum_plot_raw.setMinimumHeight(300)
        self.curve_raw = self.spectrum_plot_raw.plot(pen=pg.mkPen(color='b', width=2))
        scroll_layout.addWidget(self.spectrum_plot_raw)

        self.harmonics_label_raw = QLabel("Доминирующие гармоники (до фильтров):")
        scroll_layout.addWidget(self.harmonics_label_raw)

        self.harmonics_table_raw = self._create_harmonics_table()
        scroll_layout.addWidget(self.harmonics_table_raw)

        # --- Спектр после фильтров ---
        self.filtered_checkbox = QCheckBox("Спектр после фильтров")
        self.filtered_checkbox.setChecked(True)
        self.filtered_checkbox.toggled.connect(self._toggle_filtered)
        scroll_layout.addWidget(self.filtered_checkbox)

        self.spectrum_plot_filtered = pg.PlotWidget(viewBox=ZoomableViewBox())
        self.spectrum_plot_filtered.setBackground('w')
        self.spectrum_plot_filtered.setLabel('left', 'Амплитуда', units='мкВ')
        self.spectrum_plot_filtered.setLabel('bottom', 'Частота', units='Гц')
        self.spectrum_plot_filtered.setTitle('Частотный спектр (после фильтров)')
        self.spectrum_plot_filtered.getAxis('left').enableAutoSIPrefix(False)
        self.spectrum_plot_filtered.getAxis('bottom').enableAutoSIPrefix(False)
        self.spectrum_plot_filtered.showGrid(x=True, y=True)
        self.spectrum_plot_filtered.setMinimumHeight(300)
        self.curve_filtered = self.spectrum_plot_filtered.plot(pen=pg.mkPen(color='b', width=2))
        scroll_layout.addWidget(self.spectrum_plot_filtered)

        self.harmonics_label_filtered = QLabel("Доминирующие гармоники (после фильтров):")
        scroll_layout.addWidget(self.harmonics_label_filtered)

        self.harmonics_table_filtered = self._create_harmonics_table()
        scroll_layout.addWidget(self.harmonics_table_filtered)

        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def _create_harmonics_table(self):
        """Создание таблицы гармоник"""
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['№', 'Частота (Гц)', 'Амплитуда (мкВ)'])
        table.setRowCount(5)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setMaximumHeight(200)

        font = QFont()
        font.setPointSize(12)
        table.setFont(font)

        for i in range(5):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            table.setItem(i, 1, QTableWidgetItem('-'))
            table.setItem(i, 2, QTableWidgetItem('-'))

        return table

    def _toggle_raw(self, checked):
        """Показать/скрыть спектр до фильтров"""
        self.raw_visible = checked
        self.spectrum_plot_raw.setVisible(checked)
        self.harmonics_label_raw.setVisible(checked)
        self.harmonics_table_raw.setVisible(checked)

    def _toggle_filtered(self, checked):
        """Показать/скрыть спектр после фильтров"""
        self.filtered_visible = checked
        self.spectrum_plot_filtered.setVisible(checked)
        self.harmonics_label_filtered.setVisible(checked)
        self.harmonics_table_filtered.setVisible(checked)

    def set_display_unit(self, unit: str):
        """Синхронизировать единицы амплитуды с основным виджетом"""
        self.display_unit = unit
        self.display_scale = np.float32(0.001) if unit == 'мВ' else np.float32(1.0)
        for plot in (self.spectrum_plot_raw, self.spectrum_plot_filtered):
            plot.setLabel('left', 'Амплитуда', units=unit)
        header = f'Амплитуда ({unit})'
        for table in (self.harmonics_table_raw, self.harmonics_table_filtered):
            table.setHorizontalHeaderItem(2, QTableWidgetItem(header))

    def on_channel_changed(self, index):
        """Обработка изменения выбранного канала"""
        self.selected_channel_index = index
        self.spectrum_plot_raw.getViewBox().autoRange()
        self.spectrum_plot_filtered.getViewBox().autoRange()
        self.update_analysis()

    def update_analysis(self):
        """Обновление Фурье-анализа"""
        try:
            channel = self.channels[self.selected_channel_index]

            # FFT до фильтров (raw_data)
            if self.raw_visible:
                raw_signal = channel.raw_data.get_straight_buffer()
                self._compute_and_display(
                    raw_signal, self.curve_raw, self.harmonics_table_raw
                )

            # FFT после фильтров (data)
            if self.filtered_visible:
                filtered_signal = channel.data.get_straight_buffer()
                self._compute_and_display(
                    filtered_signal, self.curve_filtered, self.harmonics_table_filtered
                )

        except Exception as e:
            print(f"Ошибка при выполнении Фурье-анализа: {e}")
            import traceback
            traceback.print_exc()

    def _compute_and_display(self, signal_data, curve, table_widget):
        """Вычисление FFT и отображение результатов"""
        signal_array = np.asarray(signal_data, dtype=np.float32)

        if len(signal_array) < 10:
            return

        N = len(signal_array)
        fft_values = fft.fft(signal_array)
        fft_magnitude = 2.0 / N * np.abs(fft_values[:N // 2]) * self.display_scale
        frequencies = fft.fftfreq(N, 1 / self.sample_rate)[:N // 2]

        start_index = np.searchsorted(frequencies, 1.0)

        curve.setData(frequencies[start_index:], fft_magnitude[start_index:])

        magnitudes_slice = fft_magnitude[start_index:]
        if len(magnitudes_slice) >= 5:
            part_indices = np.argpartition(magnitudes_slice, -5)[-5:]
            part_indices = part_indices[np.argsort(magnitudes_slice[part_indices])[::-1]]
            top_indices = part_indices + start_index
        else:
            top_indices = np.argsort(magnitudes_slice)[::-1] + start_index

        for i, idx in enumerate(top_indices):
            freq = frequencies[idx]
            amplitude = fft_magnitude[idx]

            table_widget.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            table_widget.setItem(i, 1, QTableWidgetItem(f'{freq:.2f}'))
            table_widget.setItem(i, 2, QTableWidgetItem(f'{amplitude:.2f}'))
