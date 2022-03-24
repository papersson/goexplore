import cv2
import numpy as np
from scipy import ndimage


class CoarseBinarizer:
    def process(self, img):
        # Crop and resize
        img = img[34:194:2, ::2]

        # Convert to greyscale
        img = img.mean(axis=2)

        # Shrink
        img = ndimage.interpolation.zoom(img, 0.1)

        # Binarize
        img = np.round(img, 2)
        threshold = 77.7
        img[img < threshold] = 0
        img[img >= threshold] = 1

        return tuple(img.flatten())


class UberReducer:
    def __init__(self, width=12, height=12, num_colors=8):
        self.width, self.height = width, height
        self.num_colors = num_colors

    def process(self, img):
        # Widen by duplicating every other pixel colwise
        # img = cv2.resize(img, (320, 210), interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Shrink
        img = cv2.resize(img, (self.width, self.height),
                         interpolation=cv2.INTER_AREA)
        # Reduce pixel depth
        img = self._color_quantize(img)

        return tuple(img.flatten())

    def _color_quantize(self, img):
        bits = np.log2(self.num_colors)
        diff = 8 - bits  # Assumes 2^8 = 256 original depth
        k = 2**diff
        return k * (img // k)

    def __repr__(self):
        return f'Downsampler(w={self.width},h={self.height},d={self.num_colors})'
