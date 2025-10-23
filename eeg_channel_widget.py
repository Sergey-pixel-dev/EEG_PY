import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
from circularbuffer import CircularBuffer


class EEGChannelWidget(QWidget):
    """Виджет для отображения одного канала ЭЭГ"""

    def __init__(self, channel_name, x_range=15, y_range=(0, 3400), sample_rate=100):
        super().__init__()

        self.channel_name = channel_name
        self.x_range = x_range
        self.y_range = y_range
        self.sample_rate = sample_rate

        # Данные для отображения
        self.data_to_display = CircularBuffer(
            size=self.x_range * self.sample_rate,
            dtype=np.float32
        )

        # Предвыделенная ось X
        self.x_axis = np.linspace(
            0,
            self.x_range,
            self.x_range * self.sample_rate,
            dtype=np.float32
        )

        # Графические элементы
        self.curve = None
        self.cursor_line = None

        self._setup_ui()

    def _setup_ui(self):
        """Настройка интерфейса виджета"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Создаем PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', self.channel_name, units='мкВ')
        self.plot_widget.setLabel('bottom', 'Время', units='с')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=False, y=False)

        # Фиксируем диапазоны осей
        self.plot_widget.setXRange(0, self.x_range, padding=0)
        self.plot_widget.setYRange(self.y_range[0], self.y_range[1], padding=0)
        self.plot_widget.enableAutoRange(enable=False)

        # Оптимизации производительности
        self.plot_widget.setClipToView(True)

        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def update_display(self):
        """Обновление отображения графика"""
        y = self.data_to_display.get_buffer()

        # Вычисляем позицию курсора
        cursor_position = (self.data_to_display.point / self.data_to_display.size) * self.x_range
        gap = self.x_range * 0.008
        cursor_position_with_gap = cursor_position + gap

        if self.curve is None:
            # Первичное создание кривой
            self.curve = self.plot_widget.plot(
                self.x_axis,
                y,
                pen=pg.mkPen(color='b', width=2),
                skipFiniteCheck=True
            )

            # Создаем вертикальную пунктирную линию-курсор
            self.cursor_line = pg.InfiniteLine(
                pos=cursor_position_with_gap,
                angle=90,
                pen=pg.mkPen(color='r', style=pg.QtCore.Qt.PenStyle.DashLine, width=2),
                movable=False
            )
            self.plot_widget.addItem(self.cursor_line)
        else:
            # Обновление данных
            self.curve.setData(self.x_axis, y, skipFiniteCheck=True)
            self.cursor_line.setPos(cursor_position_with_gap)

    def append_data(self, value):
        """Добавление данных в буфер"""
        self.data_to_display.append(np.float32(value))

    def clear_data(self):
        """Очистка буфера данных"""
        self.data_to_display.clear()
