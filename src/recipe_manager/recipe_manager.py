import numpy as np
from src.models import FractalModel
import logging

_logger = logging.getLogger(__name__)


class RecipeManager:
    def __init__(self, recipe_name):
        self.recipe_name = recipe_name
        if self.recipe_name == "maskit":
            self.compute_generators = self.maskit
        elif self.recipe_name == "grandma_recipe":
            self.compute_generators = self.grandma_recipe
        elif self.recipe_name == "grandma_special_recipe":
            self.compute_generators = self.grandma_special_recipe
        elif self.recipe_name == "jorgensen":
            self.compute_generators = self.jorgensen_recipe
        else:
            raise ValueError(f"No recipe named {recipe_name} ! Crashing.")

    def compute_generators(ta, tb, tab):
        pass

    def generate(self, ta, tb=None, tab=None):
        gens = self.compute_generators(ta, tb)
        return FractalModel(generators=gens)

    @staticmethod
    def maskit(ta, tb=None, tab=None):
        mu = ta * -1j
        gen_a = np.zeros((2, 2), dtype=complex)
        gen_b = np.zeros((2, 2), dtype=complex)

        gen_a[0, 0] = mu
        gen_a[0, 1] = -1j
        gen_a[1, 0] = -1j

        gen_b[0, 0] = 1
        gen_b[0, 1] = 2
        gen_b[1, 0] = 0
        gen_b[1, 1] = 1

        return np.array([gen_a, gen_b, np.linalg.inv(gen_a), np.linalg.inv(gen_b)])

    @staticmethod
    def grandma_recipe(ta, tb, tab=None):
        if ta == 0 and tb == 0:
            raise ValueError("ta and tb cannot be 0 ! Crashing")
        a = complex(1)
        b = complex(-ta * tb)
        c = complex(ta * ta + tb * tb)
        delta = b * b - 4 * a * c
        tab = (-b + np.sqrt(delta)) / (2 * a)
        z0 = ((tab - 2) * tb) / (tb * tab - 2 * ta + 2j * tab)

        gen_a = np.zeros((2, 2), dtype=complex)
        gen_b = np.zeros((2, 2), dtype=complex)

        gen_a[0, 0] = ta / 2
        gen_a[0, 1] = (ta * tab - 2 * tb + 4j) / (z0 * (2 * tab + 4))
        gen_a[1, 0] = (z0 * (ta * tab - 2 * tb - 4j)) / (2 * tab - 4)
        gen_a[1, 1] = ta / 2

        gen_b[0, 0] = (tb - 2j) / 2
        gen_b[0, 1] = tb / 2
        gen_b[1, 0] = tb / 2
        gen_b[1, 1] = (tb + 2j) / 2

        _logger.debug(f"{gen_a}")
        _logger.debug(f"{gen_b}")
        return np.array([gen_a, gen_b, np.linalg.inv(gen_a), np.linalg.inv(gen_b)])

    @staticmethod
    def grandma_special_recipe(ta, tb, tab):
        if ta == 0 and tb == 0:
            raise ValueError("ta and tb cannot be 0 ! Crashing")
        elif abs(tab) == 2:
            raise ValueError("taB cannot be +/-2 ! Crashing")

        tc = complex(ta * ta + tb * tb + tab * tab - ta * tb * tab - 2)
        Q = np.sqrt(2 - tc)

        if abs(tc + Q * np.sqrt(tc + 2) * 1j) >= 2:
            R = np.sqrt(tc + 2)
        else:
            R = -np.sqrt(tc + 2)

        z0 = ((tab - 2) * (tb + R)) / (tb * tab - 2 * ta + 1j * Q * tab)

        gen_a = np.zeros((2, 2), dtype=complex)
        gen_b = np.zeros((2, 2), dtype=complex)

        gen_a[0, 0] = ta / 2
        gen_a[0, 1] = (ta * tab - 2 * tb + 2 * 1j * Q) / (z0 * (2 * tab + 4))
        gen_a[1, 0] = (z0 * (ta * tab - 2 * tb - 2 * 1j * Q)) / (2 * tab - 4)
        gen_a[1, 1] = ta / 2

        gen_b[0, 0] = (tb - 1j * Q) / 2
        gen_b[0, 1] = (tb * tab - 2 * ta - 1j * Q * tab) / (z0 * (2 * tab + 4))
        gen_b[1, 0] = (z0 * (tb * tab - 2 * ta + 1j * Q * tab)) / (2 * tab + 4)
        gen_b[1, 1] = (tb + 1j * Q) / 2

        return np.array([gen_a, gen_b, np.linalg.inv(gen_a), np.linalg.inv(gen_b)])

    @staticmethod
    def jorgensen_recipe(ta, tb):
        if ta == 0 and tb == 0:
            raise ValueError("ta and tb cannot be 0 ! Crashing")
        z = 0.5 * np.sqrt(complex(ta * ta * tb * tb - 4 * ta * ta - 4 * tb * tb))
        tab = 0.5 * ta * tb - z

        gen_a = np.zeros((2, 2), dtype=complex)
        gen_b = np.zeros((2, 2), dtype=complex)

        gen_a[0, 0] = ta - tb / tab
        gen_a[0, 1] = ta / (tab * tab)
        gen_a[1, 0] = ta
        gen_a[1, 1] = tb / tab

        gen_b[0, 0] = tb - ta / tab
        gen_b[0, 1] = -tb / (tab * tab)
        gen_b[1, 0] = -tb
        gen_b[1, 1] = ta / tab

        return np.array([gen_a, gen_b, np.linalg.inv(gen_a), np.linalg.inv(gen_b)])
