#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True 
#cython: nonecheck=False

import cython
import numpy as np
cimport numpy as np

from src.data.image_dataclass cimport ImageDataclass 

cdef int bounds_check(float x, float y, ImageDataclass img):
    if x >= 0 and y >= 0 and x < img.width and y < img.height:
        return 1
    else:
        return 0

cdef void point(float x, float y, float c, ImageDataclass img):
    if bounds_check(x, y, img) == 1:
        img.image_array[int(x), int(y)] = int(c * 255)

cdef float intensifyColor(float d):
    return 1 - np.power(d * 2/3., 2)

cdef float map(float x, float a, float b, float c, float d):
    return (x - a)/(b - a) * (d - c) 

cdef void line(cython.floatcomplex p0, cython.floatcomplex p1, ImageDataclass img):
    # Gupta Sprull algo for antialiased line drawing
    # Lifted from https://en.wikipedia.org/wiki/Line_drawing_algorithm#Gupta_and_Sproull_algorithm
    
    #If the two points are very close, just draw a point
    if abs(p0 - p1) <= 1e-6:
  #      point(p0.real, p0.imag, 1, img)
        return

    cdef float x0 = map(p0.real, img.z_min.real, img.z_max.real, 0, img.width)
    cdef float y0 = map(p0.imag, img.z_min.imag, img.z_max.imag, 0, img.height)
    cdef float x1 = map(p1.real, img.z_min.real, img.z_max.real, 0, img.width)
    cdef float y1 = map(p1.imag, img.z_min.imag, img.z_max.imag, 0, img.height)

    cdef float x = x0
    cdef float y = y0
    cdef float dx = x1 - x0
    cdef float dy = y1 - y0
    cdef float d = 2 * dy - dx
    
    cdef float D = 0
    
    cdef float length = np.sqrt(dx * dx + dy * dy) 
    
    cdef float sin = dx / length    
    cdef float cos = dy / length
    while (x <= x1):
        point(x, y - 1, intensifyColor(D + cos), img)
        point(x, y, intensifyColor(D), img)
        point(x, y + 1, intensifyColor(D - cos), img)
        x = x + 1
        if (d <= 0):
            D = D + sin
            d = d + 2 * dy
        else:
            D = D + sin - cos
            d = d + 2 * (dy - dx)
            y = y + 1
