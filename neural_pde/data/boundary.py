import numpy as np
from typing import Callable



def sample_square_boundary(B: int, lb: float, ub: float):
    x = np.random.uniform(lb, ub, size=(B, 2))
    return x


def create_boundary(sample: Callable, kwargs):
    pass