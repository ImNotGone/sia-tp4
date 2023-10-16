import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Any

def _euclidean_distance(x: NDArray, y: NDArray) -> float:
    return np.linalg.norm(x - y)

def _exponential_distance(x: NDArray, y: NDArray) -> float:
    return np.exp(-np.linalg.norm(x - y) ** 2)

DistanceFunction = Callable[[NDArray, NDArray], float]

def get_distance_function(config: Dict[str, Any]) -> DistanceFunction:
    distance_type = config["distance"]

    if distance_type == "euclidean":
        return _euclidean_distance
    elif distance_type == "exponential":
        return _exponential_distance

    raise ValueError(f"Unknown distance type: {distance_type}")


