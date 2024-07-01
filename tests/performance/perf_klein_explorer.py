import cProfile
import logging
import os

from src.models import FractalModel, ComputeModel, OutputModel
from src.klein_composer import KleinComposer 
from klein_compute.tree_explorer import tree_explorer
from klein_compute.tree_exp import compute_start_points
from src.recipe_manager import RecipeManager 
import numpy as np
from pstats import Stats
import time 

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
fix_pt_shape = []
for fp in fm.fixed_points:
    fix_pt_shape.append(np.count_nonzero(fp))
with cProfile.Profile() as pr:
    p = compute_start_points(2, 0.001, fm.generators, fm.FSA, fm.fixed_points, np.array(fix_pt_shape))
with cProfile.Profile() as po:
    tree_exp.compute_tree()


print(p)
pr.print_stats(sort = "time")
po.dump_stats("profile_old.prof")
ref_stats = Stats('profile_old.prof')
old_stats = Stats('profile.prof')
pr.dump_stats("profile.prof")
pr = Stats(pr)

print(f"New time: {pr.total_tt:.3f}")
print(f"Old time: {old_stats.total_tt:.3f}")
print(f"Ref time: {ref_stats.total_tt:.3f}")
print(f"Diff {old_stats.total_tt - pr.total_tt:.3f}, ({(old_stats.total_tt - pr.total_tt) * 100./old_stats.total_tt:.3f}%)")
