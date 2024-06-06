cimport numpy as np
cdef class tree_explorer:
    cdef void set_next_state(self, int idx_gen)
    cdef int get_next_state(self, int idx_gen)
    cdef int get_next_gen(self)
    cdef int branch_terminated(self)
    cdef void backward_move(self)
    cdef int available_turn(self)
    cdef void turn_forward_move(self)
    cdef void forward_move(self)
    cpdef tuple compute_leaf(self, int max_depth, int start_depth, int start_tag)
    cpdef np.array compute_leaf(self, int max_depth, int start_depth, int start_tag)
