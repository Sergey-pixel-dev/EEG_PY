import numpy as np
class CircularBuffer:
    def __init__(self, size, dtype=object):
        self.point = 0 # индекс элемента, начиная с которого нужно обновлять дальше
        self.size = size
        self.data = np.empty(size, dtype=dtype)
        self._dirty = True
        self._straight_cache = None
    def append(self, item):
        if self.point == self.size:
            self.point = 0
        self.data[self.point] = item
        self.point += 1
        self._dirty = True
    def get_straight_buffer(self):
        if self._dirty or self._straight_cache is None:
            self._straight_cache = np.concatenate([self.data[self.point:], self.data[:self.point]])
            self._dirty = False
        return self._straight_cache
    def get_buffer(self):
        return self.data
    def clear(self):
        self.data.fill(0)
        self.point = 0
        self._dirty = True
        self._straight_cache = None
    def set_size(self, size):
        old_data = self.get_straight_buffer()
        self.size = size
        self.data = np.empty(size, dtype=self.data.dtype)
        copy_size = min(len(old_data), size)
        self.data[:copy_size] = old_data[:copy_size]
        self.point = min(self.point, size)
        self._dirty = True
        self._straight_cache = None
