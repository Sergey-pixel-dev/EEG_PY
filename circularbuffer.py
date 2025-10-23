import numpy as np
class CircularBuffer:
    def __init__(self, size, dtype=object):
        self.point = 0 # индекс элемента, начиная с которого нужно обновлять дальше
        self.size = size
        self.data = np.empty(size, dtype=dtype)
    def append(self, item):
        if self.point == self.size:
            self.point = 0
        self.data[self.point] = item
        self.point += 1
    def get_straight_buffer(self):
        return np.concatenate([self.data[self.point:], self.data[:self.point]])
    def get_buffer(self):
        return self.data
    def clear(self):
        self.data.fill(None)
        self.point = 0
    def set_size(self, size):
        old_data = self.get_straight_buffer()
        self.size = size
        self.data = np.empty(size, dtype=self.data.dtype)
        copy_size = min(len(old_data), size)
        self.data[:copy_size] = old_data[:copy_size]
        self.point = min(self.point, size)