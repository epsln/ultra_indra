import pytest
import numpy as np

from src.recipe_manager import RecipeManager 
import numpy as np

def test_maskit():
    rm = RecipeManager("maskit")
    rm = RecipeManager("grandma_recipe")
    with pytest.raises(ValueError):
        rm = RecipeManager("foobar")

def test_maskit():
    mu = 2j
    rm = RecipeManager("maskit")
    gen_a = np.array([[mu * 1j, 1j],[1j, 0]])
    gen_b = np.array([[1, 2],[ 0, 1]])
    gen_A = np.linalg.inv(gen_a)
    gen_B = np.linalg.inv(gen_b)
    expected = np.array([gen_a, gen_b, gen_A, gen_B])
    output = rm.generate(mu) 

    assert expected.all() == output.all()
    
    #Checking if b(z) = z + 2
    gen_test = output[1]
    z = 2 + 1j
    output_test = (gen_test[0, 0] * z + gen_test[0, 1])/(gen_test[1, 0] * z + gen_test[1, 1])
    assert output_test == z + 2 
    #Checking if a(z) = u + 1/z 
    gen_test = output[0]
    output_test = (gen_test[0, 0] * z + gen_test[0, 1])/(gen_test[1, 0] * z + gen_test[1, 1])
    z_out = mu + 1/z 
    assert output_test == z_out 

def test_grandma():
    ta = 2
    tb = 2
    rm = RecipeManager("grandma_recipe")
    with pytest.raises(ValueError):
        rm.generate(0, 0) 
    output = rm.generate(ta, tb) 
