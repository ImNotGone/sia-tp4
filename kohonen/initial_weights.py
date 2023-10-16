import numpy as np
from typing import Callable, Dict, Any
from numpy.typing import NDArray

def _random_weigths(som_dimension: int, dataset: NDArray) -> NDArray:
    # K x K matrix of weights
    # each weight is a vector of the same dimension as the input
    # each weight is initialized randomly according to the inputs range
    # (min and max values)
    empty_weights = np.zeros((som_dimension, som_dimension, len(dataset[0])))
    for i in range(len(empty_weights)):
        for j in range(len(empty_weights[i])):
            empty_weights[i][j] = np.random.uniform(
                np.min(dataset), np.max(dataset), len(dataset[0])
            )

    return empty_weights

def _random_weights_from_input(som_dimension: int, dataset: NDArray) -> NDArray:
    # K x K matrix of weights
    # each weight is a vector of the same dimension as the input
    # each weight is initialized to have the same value as a randomly chosen input
    empty_weights = np.zeros((som_dimension, som_dimension, len(dataset[0])))
    for i in range(len(empty_weights)):
        for j in range(len(empty_weights[i])):
            empty_weights[i][j] = dataset[np.random.randint(0, len(dataset))]

    return empty_weights

InitialWeightsFunction = Callable[[int, NDArray], NDArray]

def get_initial_weights_function(config: Dict[str, Any]) -> InitialWeightsFunction:
    initial_weights_type = config["initial_weights"]

    if initial_weights_type == "random":
        return _random_weigths
    elif initial_weights_type == "input":
        return _random_weights_from_input

    raise ValueError(f"Unknown initial weights type: {initial_weights_type}")
