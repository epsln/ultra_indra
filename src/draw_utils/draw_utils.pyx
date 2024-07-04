#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True 
#cython: nonecheck=False

import cython
import numpy as np
cimport numpy as np

cdef int bounds_check(float x, float y, cython.floatcomplex[:] bounds):
    if x >= 0 and y >= 0 and x < bounds[2].real and y < bounds[2].imag:
        return 1
    else:
        return 0

cdef void point(float x, float y, float c, int[:, :] img, cython.floatcomplex[:] bounds):
    if bounds_check(x, y, bounds) == 1:
        img[int(x), int(y)] = int(c * 255)

cdef float intensifyColor(float d):
    return 1 - np.power(d * 2/3., 2)

cpdef void line(cython.floatcomplex p0, cython.floatcomplex p1, int[:, :] img, cython.floatcomplex[:] bounds):
    # Gupta Sprull algo for antialiased line drawing
    # Lifted from https://en.wikipedia.org/wiki/Line_drawing_algorithm#Gupta_and_Sproull_algorithm

    cdef float x0 = (p0.real - bounds[0].real)/(bounds[1].real - bounds[0].real) * (bounds[2].real) 
    cdef float y0 = (p0.imag - bounds[0].imag)/(bounds[1].imag - bounds[0].imag) * (bounds[2].imag) 
    cdef float x1 = (p1.real - bounds[0].real)/(bounds[1].real - bounds[0].real) * (bounds[2].real) 
    cdef float y1 = (p1.imag - bounds[0].imag)/(bounds[1].imag - bounds[0].imag) * (bounds[2].imag) 

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
        point(x, y - 1, intensifyColor(D + cos), img, bounds)
        point(x, y, intensifyColor(D), img, bounds)
        point(x, y + 1, intensifyColor(D - cos), img, bounds)
        x = x + 1
        if (d <= 0):
            D = D + sin
            d = d + 2 * dy
        else:
            D = D + sin - cos
            d = d + 2 * (dy - dx)
            y = y + 1
