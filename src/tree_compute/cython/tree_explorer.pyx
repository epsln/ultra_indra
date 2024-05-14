import cython 
cimport numpy as np
import numpy as np

cdef class tree_explorer():
    cdef np.ndarray words
    cdef np.ndarray generators
    cdef np.ndarray FSA
    cdef int level
    cdef int start_tag 
    cdef int max_depth
    cdef np.ndarray tag
    cdef np.ndarray state
    cdef np.ndarray last_tags

    def __cinit__(self, int start_tg, int start_lvl, int max_d, np.ndarray gen, np.ndarray fsa):
        self.start_tag = start_tg
        self.level = start_lvl 
        self.max_depth = max_d 
        self.words = np.empty((self.max_depth * 2 * 2), dtype=complex)
        self.generators = gen 
        self.FSA = fsa 
        self.tag       = np.empty((self.max_depth), dtype=int)
        self.state     = np.empty((self.max_depth), dtype=int)
        self.last_tags = np.empty((self.max_depth), dtype=int)


    cpdef void set_next_state(self, int idx_gen):
        self.state[self.level + 1] = self.FSA[self.state[self.level] * 4 + idx_gen]

    cpdef int get_next_state(self, int idx_gen):
        return self.FSA[self.state[self.level] * 4 + idx_gen]

    cpdef int get_next_gen(self):
        cdef int idx_gen = 0
        cdef int i = 1
        while True:
            idx_gen = (self.tag[self.level + 1] - i) % 4
            i -= 1
            if self.get_next_state(idx_gen) != 0:
                break
        return idx_gen

    cpdef int branch_terminated(self):
        points = []
        if self.level == self.max_depth:
            return 1 

        #for fp in fixed_points[1:]:
        #    #Append p to some array and return it
        #    points.append(np.matmul(word, fp))
        #    if np.isclose(points[i], points[i - 1], atol = self.epsilon):
        #        return True

        return 0 

    cpdef void backward_move(self):
        self.level -= 1

    cpdef int available_turn(self):
        if self.level == -1:
            return 0 

        #TODO: Check this line
        if self.FSA[self.curr_state * 4 + (self.tag[self.level + 1] - 1)] % 4 == 0:
            return 0 
        else:
            return 1 

    cpdef void turn_forward_move(self):
        cdef int idx_gen = self.get_next_gen()
        self.set_next_state(idx_gen)
        self.curr_state = self.state[self.level + 1]
        self.tag[self.level + 1] = idx_gen

        if self.level == -1:
            self.words[0] = self.generators[self.tag[0]]
        else:
            self.words[0] = np.matmul(self.word, self.generators[idx_gen])
            
        self.level += 1

    cpdef void forward_move(self):
        cdef int idx_gen = self.get_next_gen()

        self.set_next_state(idx_gen)
        self.curr_state = self.state[self.level + 1]
        self.tag[self.level + 1] = idx_gen

        self.level += 1

        self.word = np.matmul(self.word, self.generators[idx_gen])

    cpdef np.ndarray compute_leaf(self):
        while self.level != -1 or self.tag[0] == self.start_tag:
            while self.branch_terminated() == False:
                self.forward_move()
            while True:
                self.backward_move() 
                if self.available_turn() == True or self.level == -1:
                    break
            self.turn_forward_move()

        #TODO: Adapt to explore all branch and returns list of reached tags + words
        #if start_word == np.identity(2): 
        #    return self.last_words, self.last_tags 
        #else:
        return self.points

