import json
from dataset_loaders import load_countries
from distance import get_distance_function
from eta import get_eta_function

from initial_weights import get_initial_weights_function
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

        # Count how many times each country was assigned to each neuron in a matrix
        # of size K x K
        neuron_assignments = [
            [0 for _ in range(som_dimension)] for _ in range(som_dimension)
        ]

        for country in dataset:
            winner_pos = get_winner_pos(
                country, trained_kohonen_weights, distance_function
            )
            neuron_assignments[winner_pos[0]][winner_pos[1]] += 1

        print("Neuron assignments:")
        for row in neuron_assignments:
            print(row)


if __name__ == "__main__":
    main()
