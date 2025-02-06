import pickle
from pathlib import Path
import numpy as np
from typing import Callable



def sample_square_interior(B: int, lb: float, ub: float):
    x = np.random.uniform(lb, ub, size=(B, 2))
    return x


def create_interior(sample: Callable, kwargs):


    root = Path(f"./datasets")
    cases = [(100_000, "train"), (500, "val"), (500, "test")]
    dtype = np.float32
    root.mkdir(parents=True, exist_ok=True)

    info = {}
    for B, split in cases:
        X = sample(B, **kwargs)
        # How do we get output values?

        print(X.shape)

        info[split] = {"shape": X.shape, "dtype": dtype}

        X_memmap = np.memmap(root / f"X_{split}.dat", dtype=dtype, mode="w+", shape=X.shape)
        X_memmap[:] = X[:]
        X_memmap.flush()
    
    pickle.dump(info, open(root / "info.pkl", "wb"))
    

SAMPLE_TO_SHAPE = {
    "square": sample_square_interior
}

