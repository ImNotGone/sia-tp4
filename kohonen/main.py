import json
from dataset_loaders import load_countries
from distance import get_distance_function
from eta import get_eta_function

from initial_weights import get_initial_weights_function
from draw import (
    create_neuron_activations_heatmap,
    create_unified_distance_matrix,
    create_average_values_heatmaps,
)
from radius import get_radius_function

from kohonen import train_kohonen, get_winner_pos


def main():
    config_path = "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

        som_dimension = config["k"]
        max_epochs_multiplier = config["max_epochs_multiplier"]

        inital_weights_function = get_initial_weights_function(config)
        radius_function = get_radius_function(config)
        eta_function = get_eta_function(config)
        distance_function = get_distance_function(config)

        input_labels, variable_labels, dataset = load_countries("../data/europe.csv")

        max_epochs = max_epochs_multiplier * len(variable_labels)

        trained_kohonen_weights = train_kohonen(
            som_dimension,
            inital_weights_function,
            radius_function,
            eta_function,
            distance_function,
            max_epochs,
            dataset,
        )

        create_neuron_activations_heatmap(
            dataset, trained_kohonen_weights, distance_function
        )
        create_average_values_heatmaps(
            trained_kohonen_weights, variable_labels, dataset, distance_function
        )

        radius = radius_function(max_epochs)
        create_unified_distance_matrix(trained_kohonen_weights, radius)


if __name__ == "__main__":
    main()
