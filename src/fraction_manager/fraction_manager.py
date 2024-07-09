from fractions import Fraction

from src.recipe_manager import RecipeManager 
from src.fraction_manager.fraction_math import trace_solver

class FractionManager:
    def __init__(self, root_iter = 1000, root_epsilon = 1e-6):
        if root_iter < 10:
            raise ValueError(f"Value root_iter must be > 10 !")
        if root_epsilon < 0 or root_epsilon > 1:
            raise ValueError(f"Value root_epsilon must be in ]0; 1[ !")
        
        self.fractions = [] 
        self.root_iter = root_iter 
        self.root_epsilon = root_epsilon
        self.recipe = RecipeManager("grandma_recipe")

    def generate(self):
        for fract in self.fractions:
            yield self.recipe.generate(trace_solver(fract), 2)
