import pytest

import numpy as np
import src.utils 

def test_pad_to_dense():
    arr = [[1 + 2j], [1j, 2j, 3j], [1j + 2, 2j], [3]]
    t_arr = src.utils.pad_to_dense(arr)
    assert t_arr.shape == (4, 3)
