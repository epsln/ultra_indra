# cython: profile=True
#cython: boundscheck=False
import cython 
from cython.parallel import prange
cimport numpy as np
import numpy as np
import logging
from multiprocessing import current_process

_logger = logging.getLogger(__name__)

cdef class tree_explorer:
    cdef np.ndarray generators
    cdef np.ndarray FSA
    cdef int level
    cdef int start_tag 
    cdef int max_depth
    cdef int curr_state 
    cdef int last_idx 
    cdef int last_idx_points
    cdef int precomputing 
    cdef float epsilon
    cdef np.ndarray tag
    cdef np.ndarray state
    cdef np.ndarray words
    cdef np.ndarray last_words
    cdef np.ndarray last_state
    cdef np.ndarray last_tags
    cdef np.ndarray fixed_points 
    cdef np.ndarray start_word
    cdef np.ndarray points 

    def __init__(self, int max_d, float epsilon, np.ndarray gen, np.ndarray fsa, np.ndarray fix_pt):
        self.level = 0 
        self.precomputing = 0
        self.max_depth = max_d 
        self.epsilon = epsilon
        self.generators = gen 
        self.FSA = fsa
        self.fixed_points = fix_pt 
        
        self.words = np.empty((self.max_depth, 2, 2), dtype=complex)
        self.tag       = np.empty((self.max_depth), dtype=int)
        self.state     = np.empty((self.max_depth), dtype=int)
        self.last_words = np.zeros((4 * np.power(3, self.max_depth ), 2, 2), dtype=complex)
        self.last_state = np.zeros((4 * np.power(3, self.max_depth )), dtype=int)
        self.last_tags = np.zeros((4 * np.power(3, self.max_depth )), dtype=int)
        self.points = np.zeros((fix_pt.shape[0] * fix_pt.shape[1] * 4 * np.power(3, self.max_depth)), dtype=complex)

        self.last_idx = 0
        self.last_idx_points = 0


    cdef void set_next_state(self, int idx_gen):
        self.state[self.level + 1] = self.FSA[self.state[self.level]][idx_gen]

    cdef int get_next_state(self, int idx_gen):
        return self.FSA[self.state[self.level]][idx_gen]

    cdef int get_right_gen(self):
        cdef int idx_gen = 0
        cdef int i = 1
        while True:
            idx_gen = (self.tag[self.level] + i) % 4
            i += 1
            if self.get_next_state(idx_gen) != 0:
                break
        return idx_gen

    cdef int get_left_gen(self):
        cdef int idx_gen = 0
        cdef int i = 1
        while True:
            idx_gen = (self.tag[self.level + 1] - i) % 4
            i += 1
            if self.get_next_state(idx_gen) != 0:
                break
        return idx_gen

    cdef complex mobius(self, np.ndarray[np.complex128_t, ndim = 2] m, np.complex128_t z):
        return (m[0, 0] * z + m[0, 1])/(m[1, 0] * z + m[1, 1])

    cdef int branch_terminated(self):
        if self.level == self.max_depth - 1:
            if self.precomputing == 1:
                self.last_words[self.last_idx] = self.words[self.level]
                self.last_state[self.last_idx] = self.state[self.level]
                self.last_tags[self.last_idx] = self.tag[self.level]
                self.last_idx += 1
            return 1 

        cdef int idx_gen = self.tag[self.level]
        cdef cython.complex p = self.mobius(self.words[self.level], self.fixed_points[idx_gen][0])
        cdef cython.complex old_p = p
        self.points[self.last_idx] = p
        self.last_idx_points += 1

        for i, fp in enumerate(self.fixed_points[idx_gen][1:]):
            if p == 0 + 0j:
                return 1 

            old_p = p 
            p = self.mobius(self.words[self.level], fp)
            if abs(old_p - p) > self.epsilon:
                self.last_idx_points -= i + 1 
                return 0 

            self.points[self.last_idx_points] = p
            self.last_idx_points += 1

        return 1 

    cdef void backward_move(self):
        self.level -= 1

    cdef int available_turn(self):
        cdef int idx_gen = self.get_right_gen() 
        if self.level == -1:
            return 1 

        #TODO: Check this line
        if self.FSA[self.state[self.level + 1]][idx_gen] == 0:
            return 0 
        else:
            return 1 

    cdef void turn_forward_move(self):
        self.curr_state = self.state[self.level]
        cdef int idx_gen = self.get_left_gen()
        self.set_next_state(idx_gen)
        self.curr_state = self.state[self.level + 1]
        self.tag[self.level + 1] = idx_gen
        if self.level == -1:
            self.words[0] = self.generators[idx_gen] 
        else:
            self.words[self.level + 1] = np.matmul(self.words[self.level], self.generators[idx_gen])
            
        self.level += 1

    cdef void forward_move(self):
        cdef int idx_gen = self.get_right_gen()

        self.set_next_state(idx_gen)
        self.curr_state = self.state[self.level + 1]
        self.tag[self.level + 1] = idx_gen
        if self.level == -1:
            self.words[0] = self.generators[idx_gen]
        else:
            self.words[self.level + 1] = np.matmul(self.words[self.level], self.generators[idx_gen])

        self.level += 1

    cpdef tuple compute_tree(self):
        self.words[0] = self.generators[0] 
        self.tag[0] = 0 
        self.state[0] = 1 
        self.precomputing = 0
        while not (self.level == -1 and self.tag[0] == 1):
            while self.branch_terminated() == 0:
                self.forward_move()
            while True:
                self.backward_move() 
                if self.available_turn() == 1 or self.level == -1:
                    break
            if self.level == -1 and self.tag[0] == 1:
                break
            self.turn_forward_move()
        self.last_idx_points = 0
        self.last_idx  = 0
        return self.last_words, self.last_state, self.last_tags

    cpdef np.ndarray compute_leaf(self, int start_tag, int start_state, int start_level, np.ndarray start_word):
        self.words[0] = start_word
        self.tag[0] = start_tag
        self.state[0] = start_state
        while not (self.level == -1 and self.tag[0] == start_tag):
            while self.branch_terminated() == 0:
                self.forward_move()
            while True:
                self.backward_move() 
                if self.level == -1 or self.available_turn():
                    break
            if self.level == -1 and self.tag[0] == start_tag:
                break
            self.turn_forward_move()
        return self.points
