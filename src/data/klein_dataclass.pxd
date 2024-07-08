import cython

cdef class KleinDataclass:
    # fields can be declared using annotations
    cdef int[:] tag
    cdef int[:] state
    cdef int[:, :] FSA 
    cdef cython.floatcomplex[:, :, :] words 
    cdef cython.floatcomplex[:, :, :] generators 
    cdef cython.floatcomplex[:, :] fixed_points 
    cdef int[:] fixed_points_shape 
    cdef int level 
    cdef float epsilon
    cdef int max_depth 
