import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from scipy import signal
from circularbuffer import CircularBuffer
from fourier_analysis_widget import ZoomableViewBox


class EEGChannelWidget(QWidget):
    """Виджет для отображения одного канала ЭЭГ"""

    def __init__(self, channel_name, x_range=15, y_range=(0, 3400), sample_freq=100):
        super().__init__()

        self.channel_name = channel_name
        self.x_range = x_range
        self.y_range = y_range
        self.sample_freq = sample_freq

        self.i = 0  # счетчик для подсчета пропускаемых данных
        self.n_i = 16  # прореживание: при 2кГц даёт ~125 точек/сек на дисплей

        # Флаги включения фильтров
        self.notch_enabled = True
        self.lowpass_enabled = False
        self.highpass_enabled = False

        # Параметры notch-фильтра 50 Гц
        self.notch_freq = 50.0
        self.quality_factor = 20.0

        # Создание notch-фильтра в SOS-формате для лучшей стабильности
        b_notch, a_notch = signal.iirnotch(
            self.notch_freq,
            self.quality_factor,
            self.sample_freq
        )

        self.sos_notch = signal.tf2sos(b_notch, a_notch)
        self.zi_notch = signal.sosfilt_zi(self.sos_notch)

        self.cutoff_freq = 40.0  # Частота среза (Гц)
        self.filter_order = 4  # Порядок фильтра (4-8)
        self.sos_lowpass = signal.butter(
            N=self.filter_order,
            Wn=self.cutoff_freq,
            btype='low',
            fs=self.sample_freq,
            output='sos'
        )
        self.zi_lowpass = signal.sosfilt_zi(self.sos_lowpass)

        # Параметры ФВЧ (высокочастотного фильтра)
        self.highpass_freq = 0.5
        self.sos_highpass = signal.butter(
            N=2,
            Wn=self.highpass_freq,
            btype='high',
            fs=self.sample_freq,
            output='sos'
        )
        self.zi_highpass = signal.sosfilt_zi(self.sos_highpass)

        # Антиалиасинговый фильтр (для дисплея)
        self.aa_enabled = True
        self.aa_cutoff_freq = 55.0
        self.aa_filter_order = 4
        self.sos_aa = signal.butter(
            N=self.aa_filter_order,
            Wn=self.aa_cutoff_freq,
            btype='low',
            fs=self.sample_freq,
            output='sos'
        )
        self.zi_aa = signal.sosfilt_zi(self.sos_aa)

        self.data_to_display = CircularBuffer(
            size=np.int32(self.x_range * self.sample_freq / self.n_i),
            dtype=np.float32
        )

        self.data = CircularBuffer(
            size=np.int32(self.x_range * self.sample_freq),
            dtype=np.float32
        )

        self.raw_data = CircularBuffer(
            size=np.int32(self.x_range * self.sample_freq),
            dtype=np.float32
        )

        self.x_axis = np.linspace(
            0,
            self.x_range,
            np.int32(self.x_range * self.sample_freq / self.n_i),
            dtype=np.float32
        )

        # Батч-буфер для фильтрации
        self.batch_size = 20
        self.batch_buffer = []

        self.curve = None
        self.cursor_line = None
        self._setup_ui()

    def _setup_ui(self):
        """Настройка интерфейса виджета"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget(viewBox=ZoomableViewBox(
            default_x_range=(0, self.x_range),
            default_y_range=(self.y_range[0], self.y_range[1])
        ))
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', self.channel_name, units='мВ')
        self.plot_widget.setLabel('bottom', 'Время', units='с')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setXRange(0, self.x_range, padding=0)
        self.plot_widget.setYRange(self.y_range[0], self.y_range[1], padding=0)
        self.plot_widget.getAxis('left').enableAutoSIPrefix(False)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot_widget.enableAutoRange(enable=False)
        self.plot_widget.setClipToView(True)

        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def update_display(self):
        """Обновление отображения графика"""
        y = self.data_to_display.get_buffer()
        cursor_position = (self.data_to_display.point / self.data_to_display.size) * self.x_range
        gap = self.x_range * 0.008
        cursor_position_with_gap = cursor_position + gap

        if self.curve is None:
            self.curve = self.plot_widget.plot(
                self.x_axis,
                y,
                pen=pg.mkPen(color='b', width=1),
                skipFiniteCheck=True
            )
            self.cursor_line = pg.InfiniteLine(
                pos=cursor_position_with_gap,
                angle=90,
                pen=pg.mkPen(color='r', style=pg.QtCore.Qt.PenStyle.DashLine, width=1),
                movable=False
            )

            self.plot_widget.addItem(self.cursor_line)
        else:
            self.curve.setData(self.x_axis, y, skipFiniteCheck=True)
            self.cursor_line.setPos(cursor_position_with_gap)

    def append_data(self, value):
        """Добавление данных в буфер с батч-фильтрацией для производительности."""
        self.batch_buffer.append(np.float32(value))
        if len(self.batch_buffer) >= self.batch_size:
            self._process_batch()

    def _process_batch(self):
        """Обработка накопленного батча данных"""
        if not self.batch_buffer:
            return

        batch = np.array(self.batch_buffer, dtype=np.float32)
        self.batch_buffer.clear()

        # Сохраняем сырые значения
        for v in batch:
            self.raw_data.append(v)

        # ФВЧ
        if self.highpass_enabled:
            batch, self.zi_highpass = signal.sosfilt(
                self.sos_highpass, batch, zi=self.zi_highpass
            )

        # ФНЧ
        if self.lowpass_enabled:
            batch, self.zi_lowpass = signal.sosfilt(
                self.sos_lowpass, batch, zi=self.zi_lowpass
            )

        # Notch
        if self.notch_enabled:
            batch, self.zi_notch = signal.sosfilt(
                self.sos_notch, batch, zi=self.zi_notch
            )

        # Антиалиасинговый фильтр для дисплея
        if self.aa_enabled:
            display_batch, self.zi_aa = signal.sosfilt(
                self.sos_aa, batch, zi=self.zi_aa
            )
        else:
            display_batch = batch

        for j, filtered_value in enumerate(batch):
            self.data.append(filtered_value)
            if self.i % self.n_i == 0:
                self.data_to_display.append(display_batch[j])
                self.i = 0
            self.i += 1

    def clear_data(self):
        """Очистка буферов данных"""
        self.batch_buffer.clear()
        self.data.clear()
        self.raw_data.clear()
        self.data_to_display.clear()
        self.i = 0
        # Сброс состояний фильтров
        self.zi_notch = signal.sosfilt_zi(self.sos_notch)
        self.zi_lowpass = signal.sosfilt_zi(self.sos_lowpass)
        self.zi_highpass = signal.sosfilt_zi(self.sos_highpass)
        self.zi_aa = signal.sosfilt_zi(self.sos_aa)

    def set_notch_filter(self, enabled, freq=50.0, quality=20.0):
        """Настройка notch-фильтра (режекторного)"""
        self.notch_enabled = enabled
        if freq != self.notch_freq or quality != self.quality_factor:
            self.notch_freq = freq
            self.quality_factor = quality
            b_notch, a_notch = signal.iirnotch(
                self.notch_freq, self.quality_factor, self.sample_freq
            )
            self.sos_notch = signal.tf2sos(b_notch, a_notch)
            self.zi_notch = signal.sosfilt_zi(self.sos_notch)

    def set_lowpass_filter(self, enabled, cutoff=40.0, order=4):
        """Настройка ФНЧ (низкочастотного фильтра)"""
        self.lowpass_enabled = enabled
        # Ограничение частоты среза частотой Найквиста
        max_cutoff = self.sample_freq / 2 - 1
        cutoff = min(cutoff, max_cutoff)
        if cutoff != self.cutoff_freq or order != self.filter_order:
            self.cutoff_freq = cutoff
            self.filter_order = order
            self.sos_lowpass = signal.butter(
                N=self.filter_order,
                Wn=self.cutoff_freq,
                btype='low',
                fs=self.sample_freq,
                output='sos'
            )
            self.zi_lowpass = signal.sosfilt_zi(self.sos_lowpass)

    def set_highpass_filter(self, enabled, cutoff=0.5):
        """Настройка ФВЧ (высокочастотного фильтра)"""
        self.highpass_enabled = enabled
        if cutoff != self.highpass_freq:
            self.highpass_freq = cutoff
            self.sos_highpass = signal.butter(
                N=2,
                Wn=self.highpass_freq,
                btype='high',
                fs=self.sample_freq,
                output='sos'
            )
            self.zi_highpass = signal.sosfilt_zi(self.sos_highpass)

    def set_antialiasing_filter(self, enabled, cutoff=55.0, order=4):
        """Настройка антиалиасингового фильтра (для дисплея)"""
        self.aa_enabled = enabled
        if cutoff != self.aa_cutoff_freq or order != self.aa_filter_order:
            self.aa_cutoff_freq = cutoff
            self.aa_filter_order = order
            self.sos_aa = signal.butter(
                N=self.aa_filter_order,
                Wn=self.aa_cutoff_freq,
                btype='low',
                fs=self.sample_freq,
                output='sos'
            )
            self.zi_aa = signal.sosfilt_zi(self.sos_aa)
