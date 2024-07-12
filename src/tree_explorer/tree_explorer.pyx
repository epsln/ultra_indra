#cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True 
#cython: nonecheck=False

import cython 
from cython.parallel import prange
import numpy as np
cimport numpy as np
from src.draw_utils.draw_utils cimport line
from src.data.klein_dataclass cimport KleinDataclass
from src.data.image_dataclass cimport ImageDataclass 

cdef int mod(int a, int b):
    cdef int r = a % b;
    if r < 0:
        return r + b
    else:
        return r

cdef void set_next_state(KleinDataclass kc, int idx_gen) noexcept :
    kc.state[kc.level + 1] = kc.FSA[kc.state[kc.level]][idx_gen]

cdef int get_next_state(KleinDataclass kc, int idx_gen) noexcept :
    return kc.FSA[kc.state[kc.level]][idx_gen]

cdef int get_right_gen(KleinDataclass kc) noexcept :
    cdef int idx_gen = 0
    cdef int i = 1
    while True:
        idx_gen = (kc.tag[kc.level] + i) % 4
        i += 1
        if get_next_state(kc, idx_gen) != 0:
            break
    return idx_gen

cdef int get_left_gen(KleinDataclass kc) noexcept :
    cdef int idx_gen = 0
    cdef int i = 1
    while True:
        idx_gen = mod(kc.tag[kc.level + 1] - i, 4)
        i += 1
        if get_next_state(kc, idx_gen) != 0:
            break
    return idx_gen

cdef cython.floatcomplex mobius(cython.floatcomplex[:, :] m, cython.floatcomplex z) noexcept :
    return (m[0, 0] * z + m[0, 1])/(m[1, 0] * z + m[1, 1])

cdef void matmul(cython.floatcomplex[:, :, :] a, cython.floatcomplex[:, :] b, int level) noexcept :
    a[level + 1, 0, 0] = a[level, 0, 0] * b[0, 0] + a[level, 0, 1] * b[1, 0]
    a[level + 1, 0, 1] = a[level, 0, 0] * b[0, 1] + a[level, 0, 1] * b[1, 1]
    a[level + 1, 1, 0] = a[level, 1, 0] * b[0, 0] + a[level, 1, 1] * b[1, 0]
    a[level + 1, 1, 1] = a[level, 1, 0] * b[0, 1] + a[level, 1, 1] * b[1, 1]
    

cdef int branch_terminated(KleinDataclass kc, ImageDataclass img) noexcept :
    if kc.level == kc.max_depth - 1:
        return 1 

    cdef int idx_gen = kc.tag[kc.level]

    cdef cython.floatcomplex p 
    cdef cython.floatcomplex fp 
    cdef cython.floatcomplex comp_p 
    cdef cython.floatcomplex comp_fp 

    cdef int i = 0 
    cdef int num_fp = kc.fixed_points_shape[idx_gen] - 1
    for i from 0 <= i < num_fp:
        fp = kc.fixed_points[idx_gen, i]
        comp_fp = kc.fixed_points[idx_gen, i + 1]

        p = mobius(kc.words[kc.level], fp)
        comp_p = mobius(kc.words[kc.level], comp_fp)
        if abs(p - comp_p) > kc.epsilon:
            return 0 
        
    for i from 0 <= i < num_fp:
        #TODO: Memoization
        fp = kc.fixed_points[idx_gen, i]
        comp_fp = kc.fixed_points[idx_gen, i + 1]
        p = mobius(kc.words[kc.level], fp)
        comp_p = mobius(kc.words[kc.level], comp_fp)

        line(p, comp_p, img)

    return 1 

cdef void backward_move(KleinDataclass kc) noexcept :
    kc.level -= 1

cdef int available_turn(KleinDataclass kc) noexcept :
    cdef int idx_gen = get_right_gen(kc) 
    if kc.level == -1:
        return 1 

    if kc.FSA[kc.state[kc.level + 1]][idx_gen] == 0:
        return 0 
    else:
        return 1 

cdef void turn_forward_move(KleinDataclass kc) noexcept :
    cdef int idx_gen = get_left_gen(kc)

    set_next_state(kc, idx_gen)
    kc.tag[kc.level + 1] = idx_gen

    cdef int i = 0
    cdef int j = 0
    if kc.level == -1:
        for i from 0 <= i < 2:
            for j from 0 <= j < 2:
                kc.words[0, i, j] = kc.generators[idx_gen, i, j] 
    else:
        matmul(kc.words, kc.generators[idx_gen], kc.level)

    kc.level += 1

cdef void forward_move(KleinDataclass kc) noexcept :
    cdef int idx_gen = get_right_gen(kc)

    set_next_state(kc, idx_gen)

    kc.tag[kc.level + 1] = idx_gen
    cdef int i = 0
    cdef int j = 0
    if kc.level == -1:
        for i from 0 <= i < 2:
            for j from 0 <= j < 2:
                kc.words[0, i, j] = kc.generators[idx_gen, i, j] 
    else:
        matmul(kc.words, kc.generators[idx_gen], kc.level)
    kc.level += 1

cpdef np.ndarray compute_tree(int start_tag, int start_state, np.ndarray start_word, int max_depth, float epsilon, np.ndarray generators, np.ndarray FSA, np.ndarray fix_pt, np.ndarray fix_pt_shape, tuple image_dim, complex z_min, complex z_max):
    cdef np.ndarray[np.complex64_t, ndim = 3] words = np.zeros((max_depth, 2, 2), dtype=np.complex64)
    cdef np.ndarray[np.int32_t, ndim = 1] tag   = np.empty((max_depth), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] state = np.empty((max_depth), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim = 1] level = np.zeros((1), dtype=np.int32)
    cdef np.ndarray[np.complex64_t, ndim = 1] bounds = np.zeros((3), dtype=np.complex64)

    tag[0] = start_tag 
    words[0] = start_word 
    state[0] = start_state 
    cdef KleinDataclass kc = KleinDataclass(
            tag = tag,
            state = state,
            FSA = FSA.astype(np.int32),
            words = words.astype(np.complex64),
            generators = generators.astype(np.complex64),
            fixed_points = fix_pt.astype(np.complex64),
            fixed_points_shape = fix_pt_shape.astype(np.int32),
            epsilon = epsilon,
            max_depth = max_depth
    )

    cdef ImageDataclass img = ImageDataclass(
            width = image_dim[0],
            height = image_dim[1],
            z_min = z_min,
            z_max = z_max 
            )
    
    while not (kc.level == -1 and kc.tag[0] == start_tag):
        while branch_terminated(kc, img) == 0:
            forward_move(kc)
        while True:
            backward_move(kc) 
            if available_turn(kc) == 1 or kc.level == -1:
                break
        if kc.level == -1 and kc.tag[0] == start_tag:
            break
        turn_forward_move(kc)

    return np.asarray(img.image_array) 
