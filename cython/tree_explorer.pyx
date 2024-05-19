import cython 
cimport numpy as np
import numpy as np
import logging

_logger = logging.getLogger(__name__)

cdef class tree_explorer():
    cdef np.ndarray words
    cdef np.ndarray generators
    cdef np.ndarray FSA
    cdef int level
    cdef int start_tag 
    cdef int max_depth
    cdef int curr_state 
    cdef np.ndarray tag
    cdef np.ndarray state
    cdef np.ndarray last_tags
    cdef np.ndarray fixed_points 
    cdef np.ndarray start_word

    def __cinit__(self, int start_tg, int start_lvl, int max_d, np.ndarray gen, np.ndarray fsa, np.ndarray fix_pt, np.ndarray start_word):
        self.level = start_lvl 
        self.max_depth = max_d 
        self.words = np.empty((self.max_depth, 2, 2), dtype=complex)
        self.generators = gen 
        self.FSA = fsa
        self.fixed_points = fix_pt 
        
        self.tag       = np.empty((self.max_depth), dtype=int)
        self.state     = np.empty((self.max_depth), dtype=int)
        self.last_tags = np.empty((self.max_depth), dtype=int)
        self.last_words = np.empty((4 * np.pow(3, self.max_depth)), dtype=int)

        self.words[0] = start_word
        self.tag[0] = start_tg


    cpdef void set_next_state(self, int idx_gen):
        self.state[self.level + 1] = self.FSA[self.state[self.level]][idx_gen]

    cpdef int get_next_state(self, int idx_gen):
        return self.FSA[self.state[self.level]][idx_gen]

    cpdef int get_right_gen(self):
        cdef int idx_gen = 0
        cdef int i = 1
        while True:
            idx_gen = (self.tag[self.level] + i) % 4
            i += 1
            if self.get_next_state(idx_gen) != 0:
                break
        return idx_gen

    cpdef int get_left_gen(self):
        cdef int idx_gen = 0
        cdef int i = 1
        while True:
            idx_gen = (self.tag[self.level + 1] - i) % 4
            i += 1
            if self.get_next_state(idx_gen) != 0:
                break
        return idx_gen


    cpdef int branch_terminated(self):
        points = []
        if self.level == self.max_depth - 1:
            idx = (self.last_words != 0).argmax()
            self.last_words[idx] = self.words[self.level]
            return 1 

        #for fp in fixed_points[1:]:
        #    #Append p to some array and return it
        #    points.append(np.matmul(word, fp))
        #    if np.isclose(points[i], points[i - 1], atol = self.epsilon):
        #        return True

        return 0 

    def print_word(self):
        word = ""
        for i in range(self.level + 1):
            t = self.tag[i]
            if i == self.level:
                word += "\x1b[31;20m"
            else:
                word += "\x1b[0m" 
            if t == 0:
                word += "a"
            elif t == 1:
                word += "b"
            elif t == 2:
                word += "A"
            elif t == 3:
                word += "B"
        _logger.debug(f"Word: {word}")

    cpdef void backward_move(self):
        self.level -= 1
        _logger.debug("Backward move")
        _logger.debug(f"level {self.level}")

    cpdef int available_turn(self):
        cdef int idx_gen = self.get_right_gen() 
        if self.level == -1:
            return 1 

        #TODO: Check this line
        if self.FSA[self.state[self.level + 1]][idx_gen] == 0:
            return 0 
        else:
            return 1 

    cpdef void turn_forward_move(self):
        self.curr_state = self.state[self.level]
        _logger.debug(f"curr_state : {self.curr_state}")
        cdef int idx_gen = self.get_left_gen()
        self.set_next_state(idx_gen)
        self.curr_state = self.state[self.level + 1]
        self.tag[self.level + 1] = idx_gen

        self.words[self.level] = np.matmul(self.words[self.level], self.generators[idx_gen])
            
        self.level += 1

        self.print_word()

    cpdef void forward_move(self):
        cdef int idx_gen = self.get_right_gen()

        self.set_next_state(idx_gen)
        self.curr_state = self.state[self.level + 1]
        self.tag[self.level + 1] = idx_gen


        self.words[self.level + 1] = np.matmul(self.words[self.level], self.generators[idx_gen])

        self.level += 1

        _logger.debug("Forward move")
        _logger.debug(f"nex_gen : {idx_gen}")
        _logger.debug(f"curr_state : {self.curr_state}")
        self.print_word()
        _logger.debug(f"level {self.level}")

    cpdef np.ndarray compute_leaf(self):
        fuckout = 0
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

        #TODO: Adapt to explore all branch and returns list of reached tags + words
        if self.start_word == np.identity(2): 
            return self.last_words, self.last_tags 
        #else:

