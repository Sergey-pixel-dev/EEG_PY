import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from scipy import signal
from circularbuffer import CircularBuffer


class EEGChannelWidget(QWidget):
    """Виджет для отображения одного канала ЭЭГ"""

    def __init__(self, channel_name, x_range=15, y_range=(0, 3400), sample_freq=100):
        super().__init__()

        self.channel_name = channel_name
        self.x_range = x_range
        self.y_range = y_range
        self.sample_freq = sample_freq

        self.i = 0  # счетчик для подсчета пропускаемых данных, добавляем лишь каждую 10 точку
        self.n_i = 10  # беру каждую 10-ую точку

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
        self.data_to_display = CircularBuffer(
            size=np.int32(self.x_range * self.sample_freq / self.n_i),
            dtype=np.float32
        )

        self.data = CircularBuffer(
            size=np.int32(self.x_range * self.sample_freq),
            dtype=np.float32
        )

        self.x_axis = np.linspace(
            0,
            self.x_range,
            np.int32(self.x_range * self.sample_freq / self.n_i),
            dtype=np.float32
        )

        self.curve = None
        self.cursor_line = None
        self._setup_ui()

    def _setup_ui(self):
        """Настройка интерфейса виджета"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', self.channel_name, units='мкВ')
        self.plot_widget.setLabel('bottom', 'Время', units='с')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setXRange(0, self.x_range, padding=0)
        self.plot_widget.setYRange(self.y_range[0], self.y_range[1], padding=0)
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
                pen=pg.mkPen(color='b', width=2),
                skipFiniteCheck=True
            )
            self.cursor_line = pg.InfiniteLine(
                pos=cursor_position_with_gap,
                angle=90,
                pen=pg.mkPen(color='r', style=pg.QtCore.Qt.PenStyle.DashLine, width=2),
                movable=False
            )

            self.plot_widget.addItem(self.cursor_line)
        else:
            self.curve.setData(self.x_axis, y, skipFiniteCheck=True)
            self.cursor_line.setPos(cursor_position_with_gap)

    def append_data(self, value):
        """Добавление данных в буфер с последовательной фильтрацией"""
        """value_lp, self.zi_lowpass = signal.sosfilt(
            self.sos_lowpass,
            [np.float32(value)],
            zi=self.zi_lowpass
        )
        val = [value]
        value_filtered, self.zi_notch = signal.sosfilt(
            self.sos_notch,
            val,
            zi=self.zi_notch
        )"""
        self.data.append(value)
        if self.i % self.n_i == 0:
            self.data_to_display.append(value)
            self.i = 0
        self.i += 1

    def clear_data(self):
        """Очистка буфера данных"""
        self.data_to_display.clear()
        # Сброс состояний фильтров
        self.zi_notch = signal.sosfilt_zi(self.sos_notch)
        self.zi_lowpass = signal.sosfilt_zi(self.sos_lowpass)
