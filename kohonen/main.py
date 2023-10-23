import json
import csv
from dataset_loaders import load_countries
from distance import get_distance_function
from eta import get_eta_function

from initial_weights import get_initial_weights_function
from draw import (
    create_neuron_activations_heatmap,
    create_neuron_activations_heatmap_with_labels,
    create_unified_distance_matrix,
    create_average_values_heatmaps,
    create_k_dead_neurons_chart,
)
from standardizers import get_standardizer
from radius import get_radius_function

from kohonen import train_kohonen, get_winner_pos


def k_test(
    dataset,
    inital_weights_function,
    radius_function,
    eta_function,
    distance_function,
    max_epochs,
):
    min_k = 3
    max_k = 6
    iterations = 10

    results = []
    for k in range(min_k, max_k + 1):
        print(f"Testing k={k}")
        for _ in range(iterations):
            trained_kohonen_weights = train_kohonen(
                k,
                inital_weights_function,
                radius_function,
                eta_function,
                distance_function,
                max_epochs,
                dataset,
            )
            neuron_activations = [
                [0 for _ in range(k)] for _ in range(k)
            ]
            for country in dataset:
                winner_pos = get_winner_pos(country, trained_kohonen_weights, distance_function)
                neuron_activations[winner_pos[0]][winner_pos[1]] += 1

            dead_neurons = 0
            for i in range(k):
                for j in range(k):
                    if neuron_activations[i][j] == 0:
                        dead_neurons += 1

            results.append((k, dead_neurons))

    # Save to csv
    with open("k_test.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "dead_neurons"])
        writer.writerows(results)


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

        standardizer = get_standardizer(config)
        dataset = standardizer(dataset)

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

        create_neuron_activations_heatmap_with_labels(
            dataset, trained_kohonen_weights, distance_function, input_labels
        )

        create_average_values_heatmaps(
            trained_kohonen_weights, variable_labels, dataset, distance_function
        )

        radius = radius_function(max_epochs)
        create_unified_distance_matrix(trained_kohonen_weights, radius)

        if config["run_k_test"]:
            k_test(
                dataset,
                inital_weights_function,
                radius_function,
                eta_function,
                distance_function,
                max_epochs,
            )
    create_k_dead_neurons_chart("k_test.csv")

if __name__ == "__main__":
    main()
