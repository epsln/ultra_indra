import cython 
cimport numpy as np
import numpy as np

cdef class tree_node():
    cdef np.ndarray word
    cdef int level
    cdef int tag 

    def __cinit__(self, int tag, int level, np.ndarray word, np.ndarray fix_points):
        self.tag = tag 
        self.level = level
        self.word = word
        self.fix_points = fix_points 
