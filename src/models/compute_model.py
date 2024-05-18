from dataclasses import dataclass
@dataclass
class ComputeModel:
    num_threads:int = 4
    max_depth: int = 3
