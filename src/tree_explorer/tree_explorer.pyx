#cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True 
#cython: nonecheck=False

foo = 'fuck'

import cython 
from cython.parallel import prange
import numpy as np
cimport numpy as np

cdef int mod(int a, int b):
    cdef int r = a % b;
    if r < 0:
        return r + b
    else:
        return r

cdef void set_next_state(int[:] state, int[:, :] FSA, int[:] level, int idx_gen) noexcept :
    state[level[0] + 1] = FSA[state[level[0]]][idx_gen]

cdef int get_next_state(int[:] state, int[:, :] FSA, int[:] level, int idx_gen) noexcept :
    return FSA[state[level[0]]][idx_gen]

cdef int get_right_gen(int[:] tag, int[:] state, int[:, :] FSA, int[:] level) noexcept :
    cdef int idx_gen = 0
    cdef int i = 1
    while True:
        idx_gen = (tag[level[0]] + i) % 4
        i += 1
        if get_next_state(state, FSA, level, idx_gen) != 0:
            break
    return idx_gen

cdef int get_left_gen(int[:] tag, int[:] state, int[:, :] FSA, int[:] level) noexcept :
    cdef int idx_gen = 0
    cdef int i = 1
    while True:
        idx_gen = mod(tag[level[0] + 1] - i, 4)
        i += 1
        if get_next_state(state, FSA, level, idx_gen) != 0:
            break
    return idx_gen

cdef cython.floatcomplex mobius(cython.floatcomplex[:, :] m, cython.floatcomplex z) noexcept :
    return (m[0, 0] * z + m[0, 1])/(m[1, 0] * z + m[1, 1])

cdef void matmul(cython.floatcomplex[:, :, :] a, cython.floatcomplex[:, :] b, int level) noexcept :
    a[level + 1, 0, 0] = a[level, 0, 0] * b[0, 0] + a[level, 0, 1] * b[1, 0]
    a[level + 1, 0, 1] = a[level, 0, 0] * b[0, 1] + a[level, 0, 1] * b[1, 1]
    a[level + 1, 1, 0] = a[level, 1, 0] * b[0, 0] + a[level, 1, 1] * b[1, 0]
    a[level + 1, 1, 1] = a[level, 1, 0] * b[0, 1] + a[level, 1, 1] * b[1, 1]
    

cdef int branch_terminated(int[:] tag, int[:] state, int[:, :] FSA, cython.floatcomplex[:, :, :] words, cython.floatcomplex[:, :] fixed_points, int[:] fixed_points_shape, int[:, :] img, cython.floatcomplex[:] bounds, int[:] level, float epsilon, int max_depth) noexcept :
    if level[0] == max_depth - 1:
        return 1 

    cdef int idx_gen = tag[level[0]]

    cdef cython.floatcomplex p 
    cdef cython.floatcomplex fp 
    cdef cython.floatcomplex comp_p 
    cdef cython.floatcomplex comp_fp 

    cdef int i = 0 
    cdef int num_fp = fixed_points_shape[idx_gen] - 1
    for i from 0 <= i < num_fp:
        fp = fixed_points[idx_gen, i]
        comp_fp = fixed_points[idx_gen, i + 1]

        p = mobius(words[level[0]], fp)
        comp_p = mobius(words[level[0]], comp_fp)
        if abs(p - comp_p) > epsilon:
            return 0 
        
    for i from 0 <= i < num_fp:
        #TODO: Memoization
        #TODO: Line tracing
        fp = fixed_points[idx_gen, i]
        comp_fp = fixed_points[idx_gen, i + 1]
        p = mobius(words[level[0]], fp)
        comp_p = mobius(words[level[0]], comp_fp)

        x0 = (p.real - bounds[0].real)/(bounds[1].real - bounds[0].real) * (bounds[2].real) 
        y0 = (p.imag - bounds[0].imag)/(bounds[1].imag - bounds[0].imag) * (bounds[2].imag) 
        x1 = (comp_p.real - bounds[0].real)/(bounds[1].real - bounds[0].real) * (bounds[2].real) 
        y1 = (comp_p.imag - bounds[0].imag)/(bounds[1].imag - bounds[0].imag) * (bounds[2].imag) 
        img[int(x0), int(y0)] = 255 
        img[int(x1), int(y1)] = 255 

    return 1 

cdef void backward_move(int[:] level) noexcept :
    level[0] -= 1

cdef int available_turn(int[:] tag, int[:] state, int[:, :] FSA, cython.floatcomplex[:, :, :] words, int[:] level, float epsilon) noexcept :
    cdef int idx_gen = get_right_gen(tag,  state, FSA, level) 
    if level[0] == -1:
        return 1 

    if FSA[state[level[0] + 1]][idx_gen] == 0:
        return 0 
    else:
        return 1 

cdef void turn_forward_move(int[:] tag, int[:] state, int[:, :] FSA, cython.floatcomplex[:, :, :] words, cython.floatcomplex[:, :, :] generators, int[:] level, float epsilon, int max_depth) noexcept :
    cdef int idx_gen = get_left_gen(tag, state, FSA, level)

    set_next_state(state, FSA, level, idx_gen)
    tag[level[0] + 1] = idx_gen

    cdef int i = 0
    cdef int j = 0
    if level[0] == -1:
        for i from 0 <= i < 2:
            for j from 0 <= j < 2:
                words[0, i, j] = generators[idx_gen, i, j] 
    else:
        matmul(words, generators[idx_gen], level[0])
        
    level[0] += 1

cdef void forward_move(int[:] tag, int[:] state, int[:, :] FSA, cython.floatcomplex[:, :, :] words, cython.floatcomplex[:, :, :] generators, int[:] level, float epsilon, int max_depth) noexcept :
    cdef int idx_gen = get_right_gen(tag, state, FSA, level)

    set_next_state(state, FSA, level, idx_gen)

    tag[level[0] + 1] = idx_gen
    cdef int i = 0
    cdef int j = 0
    if level[0] == -1:
        for i from 0 <= i < 2:
            for j from 0 <= j < 2:
                words[0, i, j] = generators[idx_gen, i, j] 
    else:
        matmul(words, generators[idx_gen], level[0])
    level[0] += 1

cpdef list compute_start_points(int max_depth, float epsilon, np.ndarray generators, np.ndarray FSA, np.ndarray fix_pt, np.ndarray fix_pt_shape):
    cdef np.ndarray[np.complex64_t, ndim = 3] words = np.zeros((max_depth, 2, 2), dtype=np.complex64)
    cdef np.ndarray[np.int32_t, ndim = 1] tag   = np.empty((max_depth), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] state = np.empty((max_depth), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] level = np.zeros((1), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 2] img = np.zeros((1080, 1080), dtype=np.int32)
    cdef np.ndarray[np.complex64_t, ndim = 1] bounds = np.zeros((4), dtype=np.complex64)

    bounds[0] = -1 - 1j
    bounds[1] = +1 + 1j
    bounds[2] = +1080 + 1080j

    words[0] = generators[0] 
    tag[0] = 0 
    state[0] = 1 

    cdef int[:] p_tag = tag
    cdef int[:] p_state = state 
    cdef int[:] p_level = level 
    cdef int[:] p_fix_pt_shape = fix_pt_shape.astype(np.int32) 
    cdef int[:, :] p_fsa = FSA.astype(np.int32)
    cdef cython.floatcomplex [:, :, :] p_words = words.astype(np.complex64)
    cdef cython.floatcomplex [:, :, :] p_generators = generators.astype(np.complex64)
    cdef cython.floatcomplex [:, :] p_fix_pt   = fix_pt.astype(np.complex64)


    cdef int[:, :] p_img = img.astype(np.intc)
    cdef cython.floatcomplex[:] p_bounds = bounds.astype(np.complex64)
    p_tag[0] = 0 
    p_state[0] = 1 

    last_points = []
    
    while not (p_level[0] == -1 and p_tag[0] == 1):
        while branch_terminated(p_tag, p_state, p_fsa, p_words, p_fix_pt, p_fix_pt_shape, p_img, p_bounds, p_level, epsilon, max_depth) == 0:
            forward_move(p_tag, p_state, p_fsa, p_words, p_generators, p_level, epsilon, max_depth)
        last_points.append((tag[level[0]], state[level[0]], words[level[0], :, :]))
        while True:
            backward_move(p_level) 
            if available_turn(p_tag, p_state, p_fsa, p_words, p_level, epsilon) == 1 or p_level[0] == -1:
                break
        if p_level[0] == -1 and p_tag[0] == 1:
            break
        turn_forward_move(p_tag, p_state, p_fsa, p_words, p_generators, p_level, epsilon, max_depth)
    return last_points 

cpdef np.ndarray compute_tree(int start_tag, int start_state, np.ndarray start_word, int max_depth, float epsilon, np.ndarray generators, np.ndarray FSA, np.ndarray fix_pt, np.ndarray fix_pt_shape, np.ndarray img_):
    cdef np.ndarray[np.complex64_t, ndim = 3] words = np.zeros((max_depth, 2, 2), dtype=np.complex64)
    cdef np.ndarray[np.int32_t, ndim = 1] tag   = np.empty((max_depth), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] state = np.empty((max_depth), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] level = np.zeros((1), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 2] img = np.zeros((1080, 1080), dtype=np.int32)
    cdef np.ndarray[np.complex64_t, ndim = 1] bounds = np.zeros((3), dtype=np.complex64)

    bounds[0] = -1 - 1j
    bounds[1] = +1 + 1j
    bounds[2] = +1080 + 1080j

    tag[0] = start_tag 
    words[0] = start_word 
    state[0] = start_state 

    cdef int[:] p_tag = tag
    cdef int[:] p_state = state 
    cdef int[:] p_level = level 
    cdef int[:] p_fix_pt_shape = fix_pt_shape.astype(np.int32) 
    cdef int[:, :] p_fsa = FSA.astype(np.int32)
    cdef cython.floatcomplex [:, :, :] p_words = words.astype(np.complex64)
    cdef cython.floatcomplex [:, :, :] p_generators = generators.astype(np.complex64)
    cdef int[:, :] p_img = img.astype(np.intc)
    cdef cython.floatcomplex[:] p_bounds = bounds.astype(np.complex64)
    cdef cython.floatcomplex [:, :] p_fix_pt   = fix_pt.astype(np.complex64)

    while not (p_level[0] == -1 and p_tag[0] == start_tag):
        while branch_terminated(p_tag, p_state, p_fsa, p_words, p_fix_pt, p_fix_pt_shape, p_img, p_bounds, p_level, epsilon, max_depth) == 0:
            forward_move(p_tag, p_state, p_fsa, p_words, p_generators, p_level, epsilon, max_depth)
        while True:
            backward_move(p_level) 
            if available_turn(p_tag, p_state, p_fsa, p_words, p_level, epsilon) == 1 or p_level[0] == -1:
                break
        if p_level[0] == -1 and p_tag[0] == start_tag:
            break
        turn_forward_move(p_tag, p_state, p_fsa, p_words, p_generators, p_level, epsilon, max_depth)
    
    print(tag)
    print(np.count_nonzero(np.asarray(p_img)))
    return np.asarray(p_img) 
