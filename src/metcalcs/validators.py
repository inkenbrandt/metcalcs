import numpy as np
import warnings


def validate_inputs(**kwargs) -> None:
    """Validate input parameters"""
    for name, value in kwargs.items():
        if isinstance(value, (np.ndarray, list, tuple)):
            if not all(np.isfinite(x) for x in np.asarray(value).flatten()):
                raise (f"Invalid {name}: contains non-finite values")
        elif not np.isfinite(value):
            raise warnings.warn(f"Invalid {name}: {value} is not finite")


def to_array(*args):
    """Convert inputs to numpy arrays while preserving single values"""
    results = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            results.append(np.array(arg))
        else:
            results.append(arg)
    return results[0] if len(results) == 1 else results
