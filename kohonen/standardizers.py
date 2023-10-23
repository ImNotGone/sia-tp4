from typing import Any, Callable, Dict
import numpy as np
from numpy import ndarray as NDArray

def _z_score(data: NDArray) -> NDArray:

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    return (data - means) / stds


def _min_max(data: NDArray) -> NDArray:
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)

    return (data - mins) / (maxs - mins)

def _unit_length(data: NDArray) -> NDArray:
    lengths = np.linalg.norm(data, axis=1, keepdims=True)

    return data / lengths

StandardizationFunction = Callable[[NDArray], NDArray]

def get_standardizer(config: Dict[str, Any]) -> StandardizationFunction:
    name = config['standardizer']

    if name == 'z_score':
        return _z_score
    elif name == 'min_max':
        return _min_max
    elif name == 'unit_length':
        return _unit_length
    else:
        raise ValueError(f'Unknown standardizer: {name}')
