import cython

cdef class ImageDataclass:
    cdef int width 
    cdef int height 
    cdef int[:, :] image_array 
    cdef cython.floatcomplex z_min 
    cdef cython.floatcomplex z_max
    cdef float aspect_ratio 
