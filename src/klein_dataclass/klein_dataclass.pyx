cimport cython
import numpy as np
cimport numpy as np

cdef class KleinDataclass:
    # fields can be declared using annotations
    def __init__(self, np.ndarray tag, np.ndarray state, np.ndarray FSA, np.ndarray words, np.ndarray generators, np.ndarray fixed_points, np.ndarray fixed_points_shape, float epsilon, int max_depth):
        self.tag = tag
        self.state = state
        self.FSA = FSA
        self.words = words
        self.generators = generators
        self.fixed_points = fixed_points
        self.fixed_points_shape = fixed_points_shape 
        self.level = 0 
        self.epsilon = epsilon
        self.max_depth = max_depth
