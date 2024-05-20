from dataclasses import dataclass


@dataclass
class ComputeModel:
    num_threads: int = 4
    max_depth: int = 3
    start_word: np.ndarray = np.identity(2, dtype = complex)
    start_tag: int = 0
    start_level: int = 0
