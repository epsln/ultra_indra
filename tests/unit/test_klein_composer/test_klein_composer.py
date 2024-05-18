import pytest

import numpy as np
from src.klein_composer import KleinComposer

def test_init():
    kc = KleinComposer()
    assert kc.num_threads == 4
    assert kc.word_length == 0
    assert kc.gen.shape == (4, 2, 2)
    assert kc.fsa.shape == (5, 4)

def test_fix_pts():
    kc = KleinComposer()
    
    test_word = np.matmul(kc.gen[0], kc.gen[1]) 
    test_word = np.matmul(test_word, kc.gen[2]) 
    test_word = np.matmul(test_word, kc.gen[3]) 
    assert kc.mobius_fixed_point(test_word)[0] in kc.fix_pts  
