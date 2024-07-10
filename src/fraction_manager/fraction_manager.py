from fractions import Fraction
import cmath
import logging

from src.recipe_manager import RecipeManager 
from src.fraction_manager.fraction_math import trace_equation, gcd

_logger = logging.getLogger(__name__)

class FractionManager:
    def __init__(self, fractions: list[Fraction],  root_iter = 100, root_epsilon = 1e-6):
        if root_iter < 10:
            raise ValueError(f"Value root_iter must be > 10 !")
        if root_epsilon < 0 or root_epsilon > 1:
            raise ValueError(f"Value root_epsilon must be in ]0; 1[ !")
        
        self.fractions = fractions 
        self.root_iter = root_iter 
        self.root_epsilon = root_epsilon
        self.recipe = RecipeManager("grandma_recipe")
        self.mu = 2j
        if Fraction(1, 1) in self.fractions:
            self.fractions.remove(Fraction(1, 1))

    @classmethod
    def from_farey_sequence(cls, max_denum: int):
        v = []
     
        for i in range(0, max_denum + 1):
            for j in range(i, max_denum + 1):
                if gcd(i, j) == 1:
                    v.append(Fraction(i, j))
        v = sorted(v)
        return cls(fractions = v)

    def generate(self):
        for fract in self.fractions:
            self.mu = self.trace_solver(fract, self.mu)
            yield self.recipe.generate(self.mu * -1j, 2)

    def trace_solver(self, fract: Fraction, ta: complex):
        #Need to implement custom newton root finder
        #As the function to optimize needs a fraction and a z value
        z = ta
        if cmath.isinf(trace_equation(fract, ta)):
            raise ValueError("Newton method failed ! ta is inf")
        for i in range(self.root_iter):
            real_v = (trace_equation(fract, z + self.root_epsilon) - trace_equation(fract, z - self.root_epsilon))/(2 * self.root_epsilon)
            imag_v = (trace_equation(fract, z + self.root_epsilon *1j) - trace_equation(fract, z - self.root_epsilon * 1j))/(2j * self.root_epsilon)
            deriv = (real_v + imag_v)/2.
            trace_eq_val = trace_equation(fract, z)
            z -= trace_eq_val/deriv
            if abs(trace_eq_val) <= self.root_epsilon:
                return z

        raise ValueError(f"Newton method failed after {self.root_iter} iterations ! Last values: f({z}) = {trace_eq_val}")


