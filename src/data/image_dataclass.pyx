import numpy as np
cimport numpy as np
import cython

cdef class ImageDataclass:
    def __init__(self, int width, int height, cython.floatcomplex z_min, cython.floatcomplex z_max):
        self.width = width 
        self.height = height
        self.aspect_ratio = max(self.width, self.height) * 1.0/min(self.width, self.height)
        self.z_min = z_min
        self.z_max = z_max
        self.z_min.real*= self.aspect_ratio
        self.z_max.real *= self.aspect_ratio
        img = np.zeros(shape = (self.width, self.height), dtype = np.intc)
        self.image_array = img 

