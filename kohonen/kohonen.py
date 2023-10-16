from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from eta import EtaFunction
from initial_weights import InitialWeightsFunction
from radius import RadiusFunction
from distance import DistanceFunction


def train_kohonen(
    som_dimension: int,
    get_initial_weights: InitialWeightsFunction,
    get_neighbour_radius: RadiusFunction,
    get_eta: EtaFunction,
    get_distance: DistanceFunction,
    max_epochs: int,
    dataset: NDArray,
) -> NDArray:
    # K x K matrix of weights
    weights = get_initial_weights(som_dimension, dataset)

    for epoch in range(max_epochs):
        selected_input = dataset[np.random.randint(0, len(dataset))]

        winner_pos = get_winner_pos(selected_input, weights, get_distance)

        neighbour_radius = get_neighbour_radius(epoch)
        eta = get_eta(epoch)

        _update_weights(weights, selected_input, winner_pos, neighbour_radius, eta)

    return weights


def get_winner_pos(
    selected_input: NDArray, weights: NDArray, get_distance: Callable
) -> Tuple[int, int]:
    winner_pos = -1, -1
    winner_distance = -1

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            distance = get_distance(selected_input, weights[i][j])
            if winner_distance == -1 or distance < winner_distance:
                winner_distance = distance
                winner_pos = i, j

    return winner_pos


def _update_weights(
    weights: NDArray,
    selected_input: NDArray,
    winner_pos: Tuple[int, int],
    neighbour_radius: float,
    eta: float,
):
    winner_row, winner_col = winner_pos

    start_row = max(0, int(winner_row - neighbour_radius))
    end_row = min(len(weights), int(winner_row + neighbour_radius + 1))
    start_col = max(0, int(winner_col - neighbour_radius))
    end_col = min(len(weights[0]), int(winner_col + neighbour_radius + 1))

    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            distance = np.linalg.norm(np.array(winner_pos) - np.array([i, j]))

            if distance <= neighbour_radius:
                weights[i][j] += eta * (selected_input - weights[i][j])
