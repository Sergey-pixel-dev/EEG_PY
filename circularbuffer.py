import numpy as np


class CircularBuffer:
    def __init__(self, size, dtype=object):
        self.point = 0  # индекс, куда будет записан следующий элемент
        self.size = size
        self.count = 0  # сколько элементов реально записано (0 .. size)
        self.data = np.zeros(size, dtype=dtype)
        self._dirty = True
        self._straight_cache = None

    def append(self, item):
        if self.point == self.size:
            self.point = 0
        self.data[self.point] = item
        self.point += 1
        if self.count < self.size:
            self.count += 1
        self._dirty = True

    def get_straight_buffer(self):
        if self._dirty or self._straight_cache is None:
            if self.count < self.size:
                # Буфер ещё не заполнен — данные лежат подряд [0:count]
                self._straight_cache = self.data[:self.count].copy()
            else:
                # Буфер полный — кольцевой порядок
                self._straight_cache = np.concatenate([self.data[self.point:], self.data[:self.point]])
            self._dirty = False
        return self._straight_cache

    def get_buffer(self):
        return self.data

    def clear(self):
        self.data.fill(0)
        self.point = 0
        self.count = 0
        self._dirty = True
        self._straight_cache = None

    def set_size(self, size):
        old_data = self.get_straight_buffer()
        self.size = size
        self.data = np.zeros(size, dtype=self.data.dtype)
        copy_size = min(len(old_data), size)
        self.data[:copy_size] = old_data[:copy_size]
        self.point = copy_size % size if size > 0 else 0
        self.count = copy_size
        self._dirty = True
        self._straight_cache = None
