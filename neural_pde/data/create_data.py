"""
We need to sample both interior and boundary points. We can break this into two functions

- run this from root directory
"""

from neural_pde.utils.parser import get_data_parser
from neural_pde.data.interior import create_interior
from neural_pde.data.boundary import create_boundary


if __name__ == "__main__":
    parser = get_data_parser()
    spec_args, func_args = parser.parse_known_args()

    shape_args = spec_args.shape

    # Collect function arguments
    kwargs = {}

    for item in func_args:
        if item.startswith("--"):
            key, value = item[2:].split("=", maxsplit=1)
            kwargs[key] = value
    
    for k, v in kwargs.items():
        if v.isdigit():
            kwargs[k] = int(v)
        elif v.replace(".", "").isdigit():
            kwargs[k] = float(v)