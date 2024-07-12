import numpy as np


class KleinModel:
    # Wrapper model to pass arguments to cython without being a mess
    def __init__(
        self,
        start_word,
        start_tag,
        start_state,
        FSA,
        generators,
        fixed_points,
        fixed_points_shape,
        epsilon,
        max_depth,
    ):
        self.tag = np.empty((max_depth), dtype=np.int32)
        self.state = np.empty((max_depth), dtype=np.int32)
        self.words = np.zeros((max_depth, 2, 2), dtype=np.complex64)
        self.FSA = FSA
        self.generators = generators
        self.fixed_points = fixed_points
        self.fixed_points_shape = fixed_points_shape
        self.level = 0
        self.epsilon = epsilon
        self.max_depth = max_depth

        self.tag[0] = start_tag
        self.words[0] = start_word
        self.state[0] = start_state
