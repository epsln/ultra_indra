import cython 
cimport numpy as np
import numpy as np

cdef struct t_node:
    cdef np.ndarray word
    cdef np.ndarray FSA 
    cdef int level
    cdef int tag 
    cdef int state

