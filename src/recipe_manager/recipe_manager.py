import numpy as np
import logging 

_logger = logging.getLogger(__name__)

class RecipeManager:
    def __init__(self, recipe_name):
        self.recipe_name = recipe_name
        if self.recipe_name == "maskit":
            self.generate = self.maskit
        elif self.recipe_name == "grandma_recipe":
            self.generate = self.grandma_recipe
        else:
            raise ValueError(f"No recipe named {recipe_name} ! Crashing.")

    def generate(ta, tb):
        pass

    @staticmethod
    def maskit(ta):
        mu = ta * -1j
        gen_a = np.zeros((2, 2), dtype = complex)
        gen_b = np.zeros((2, 2), dtype = complex)

        gen_a[0, 0] = mu 
        gen_a[0, 1] = -1j 
        gen_a[1, 0] = -1j

        gen_b[0, 0] = 1 
        gen_b[0, 1] = 2 
        gen_b[1, 0] = 0 
        gen_b[1, 1] = 1 

        return np.array([gen_a, gen_b, np.linalg.inv(gen_a), np.linalg.inv(gen_b)])

    @staticmethod
    def grandma_recipe(ta, tb):
        if ta == 0 and tb == 0:
            raise ValueError("ta and tb cannot be 0 ! Crashing")
        a = complex(1)
        b = complex(-ta * tb)
        c = complex(ta * ta + tb * tb)
        delta = b * b - 4 * a * c
        tab = (-b - np.sqrt(delta))/(2 * a)
        z0 = ((tab - 2) * tb)/(tb * tab - 2 * ta + 2j * tab) 

        gen_a = np.zeros((2, 2), dtype = complex)
        gen_b = np.zeros((2, 2), dtype = complex)

        gen_a[0, 0] = ta/2 
        gen_a[0, 1] = (ta * tab - 2 * tb + 4j) / (z0 * (2 * tab + 4)) 
        gen_a[1, 0] = (z0 * (ta * tab - 2 * tb - 4j)) / (2 * tab - 4) 
        gen_a[1, 1] = ta/2 

        gen_b[0, 0] = (tb - 2j)/2 
        gen_b[0, 1] = tb/2 
        gen_b[1, 0] = tb/2 
        gen_b[1, 1] = (tb - 2j)/2 

        _logger.debug(f"{gen_a}")
        _logger.debug(f"{gen_b}")
        return np.array([gen_a, gen_b, np.linalg.inv(gen_a), np.linalg.inv(gen_b)])
