from src.models import FractalModel, ComputeModel
from src.klein_composer import KleinComposer
from src.output_manager import OutputManager

from fractions import Fraction
import numpy as np


if __name__ == "__main__":
    fm = FractalModel(
        generators=np.random.uniform(-1, 1, (4, 2, 2))
        + 1.0j * np.random.uniform(-1, 1, (4, 2, 2)),
        FSA=np.array(
            [[1, 2, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0], [0, 2, 3, 4], [1, 0, 3, 4]]
        ),
        special_fract=Fraction(0, 1),
    )

    cm = ComputeModel(max_depth=8, num_threads=4)

    kc = KleinComposer(fm, cm)

    om = OutputManager(file_type="jpg", output_dir="output")

    p = kc.compute_thread()

    om.save_image(p)
