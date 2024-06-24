import cProfile
import logging
import os

from src.models import FractalModel, ComputeModel, OutputModel
from src.klein_composer import KleinComposer 
from klein_compute.tree_explorer import tree_explorer
from src.recipe_manager import RecipeManager 
from pstats import Stats

rm = RecipeManager("grandma_recipe")
fm = rm.generate(2, 2)
cm = ComputeModel(max_depth = 25,
                    num_threads = 1000 
    )
tree_exp = tree_explorer(
    15,
    0.001,
    fm.generators,
    fm.FSA,
    fm.fixed_points,
)
with cProfile.Profile() as pr:
    tree_exp.compute_tree()
    
    pr.print_stats(sort = "time")
    old_stats = Stats('profile.prof')
    pr.dump_stats("profile.prof")
    pr = Stats(pr)

    print(f"New time: {pr.total_tt:.3f}")
    print(f"Old_time: {old_stats.total_tt:.3f}")
    print(f"Diff {old_stats.total_tt - pr.total_tt:.3f}, ({(old_stats.total_tt - pr.total_tt) * 1.0/old_stats.total_tt:.3f}%)")
