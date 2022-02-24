import numpy as np


class DynamicArray:
    def __init__(self, size, dtype=np.uint32):
        self.dtype = dtype
        self.size = size
        self.array = np.empty(self.size, dtype=self.dtype)
        self.end = 0

    def append(self, x):
        if self.end == self.size:
            self.array = np.concatenate(
                (self.array, np.empty(self.size, dtype=self.dtype)))
            self.size *= 2
        self.array[self.end] = x
        self.end += 1

    def set_at(self, i, x):
        self.array[i] = x

    def to_numpy(self):
        return self.array[:self.end]

    def __len__(self):
        return self.end

    def __repr__(self):
        return str(self.array[:self.end])

    def __setitem__(self, i, x):
        self.array[i] = x

    def __getitem__(self, i):
        if i < self.end:
            return self.array[i]
        else:
            raise IndexError(
                f"Array index out of range. Access attempt at index {i}, but array ends at index {self.end - 1}.")
