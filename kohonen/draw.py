from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import csv
from kohonen import get_winner_pos
from numpy.typing import NDArray

from distance import DistanceFunction


def create_neuron_activations_heatmap(
    dataset: NDArray,
    trained_kohonen_weights: NDArray,
    distance_function: DistanceFunction,
):
    # Count how many times each country was assigned to each neuron in a matrix
    som_dimension = len(trained_kohonen_weights)
    neuron_activations = [
        [0 for _ in range(som_dimension)] for _ in range(som_dimension)
    ]
    for country in dataset:
        winner_pos = get_winner_pos(country, trained_kohonen_weights, distance_function)
        neuron_activations[winner_pos[0]][winner_pos[1]] += 1

    # Create a heatmap
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.imshow(neuron_activations, cmap="Blues", interpolation="nearest", aspect="auto")

    # Add black lines between cells
    for i in range(som_dimension + 1):
        plt.axhline(y=i - 0.5, color="black", linewidth=1.5)
        plt.axvline(x=i - 0.5, color="black", linewidth=1.5)

    plt.colorbar()  # Add a colorbar to the plot
    plt.clim(0, np.max(neuron_activations))
    plt.title("Neuron Activations")  # Set the title
    plt.yticks([])
    plt.xticks([])
    plt.savefig("neuron_activations.png")  # Save the figure
    plt.close()


def create_neuron_activations_heatmap_with_labels(
    dataset: NDArray,
    trained_kohonen_weights: NDArray,
    distance_function: DistanceFunction,
    labels: List[str],
):
    # Count how many times each country was assigned to each neuron in a matrix
    som_dimension = len(trained_kohonen_weights)
    neuron_activations = [
        [[] for _ in range(som_dimension)] for _ in range(som_dimension)
    ]
    for country_name, country_data in zip(labels, dataset):
        winner_pos = get_winner_pos(
            country_data, trained_kohonen_weights, distance_function
        )
        neuron_activations[winner_pos[0]][winner_pos[1]].append(country_name)

    # Create a heatmap
    neuron_activations_number = [
        [len(neuron_activations[i][j]) for j in range(som_dimension)]
        for i in range(som_dimension)
    ]
    plt.figure(figsize=(8, 6))  # Set the figure size
    im = plt.imshow(
        neuron_activations_number, cmap="Blues", interpolation="nearest", aspect="auto"
    )

    # Add black lines between cells
    for i in range(som_dimension + 1):
        plt.axhline(y=i - 0.5, color="black", linewidth=1.5)
        plt.axvline(x=i - 0.5, color="black", linewidth=1.5)

    plt.colorbar(im)  # Add a colorbar to the plot
    plt.clim(0, np.max(neuron_activations_number))
    plt.title("Neuron Activations")  # Set the title
    plt.yticks([])
    plt.xticks([])

    # Add labels for each neuron
    for i in range(som_dimension):
        for j in range(som_dimension):
            string = "\n".join(neuron_activations[i][j])
            color = "white" if neuron_activations_number[i][j] > 5 else "black"
            plt.text(
                j,
                i,
                string,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )

    plt.savefig("neuron_activations_with_labels.png")  # Save the figure
    plt.close()


def create_unified_distance_matrix(
    trained_kohonen_weights: NDArray, neighbour_radius: float
):
    som_dimension = len(trained_kohonen_weights)
    unified_distance_matrix = [
        [0 for _ in range(som_dimension)] for _ in range(som_dimension)
    ]

    for i in range(som_dimension):
        for j in range(som_dimension):
            start_row = max(0, int(i - neighbour_radius))
            end_row = min(len(trained_kohonen_weights), int(i + neighbour_radius + 1))
            start_col = max(0, int(j - neighbour_radius))
            end_col = min(
                len(trained_kohonen_weights[0]), int(j + neighbour_radius + 1)
            )

            total_distance = 0.0
            total_cells = 0
            for row in range(start_row, end_row):
                for col in range(start_col, end_col):
                    distance_between_cells = np.linalg.norm(
                        np.array([i, j]) - np.array([row, col])
                    )

                    if distance_between_cells <= neighbour_radius:
                        distance_between_weights = np.linalg.norm(
                            trained_kohonen_weights[i][j]
                            - trained_kohonen_weights[row][col]
                        )
                        total_distance += distance_between_weights
                        total_cells += 1

            unified_distance_matrix[i][j] = total_distance / total_cells

    # Create a heatmap
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.imshow(
        unified_distance_matrix, cmap="Blues", interpolation="nearest", aspect="auto"
    )

    # Add black lines between cells
    for i in range(som_dimension + 1):
        plt.axhline(y=i - 0.5, color="black", linewidth=1.5)
        plt.axvline(x=i - 0.5, color="black", linewidth=1.5)

    plt.colorbar()  # Add a colorbar to the plot
    plt.title("Unified Distance Matrix")  # Set the title
    plt.yticks([])
    plt.xticks([])
    plt.savefig("unified_distance_matrix.png")  # Save the figure
    plt.close()


def create_average_values_heatmaps(
    trained_kohonen_weights: NDArray,
    variables: List[str],
    dataset: NDArray,
    distance_function: DistanceFunction,
):
    # Create a matrix of the average_values of all the inputs assigned to each neuron
    som_dimension = len(trained_kohonen_weights)
    neuron_activations = [
        [(np.zeros(len(variables)), 0) for _ in range(som_dimension)]
        for _ in range(som_dimension)
    ]
    for country in dataset:
        winner_i, winner_j = get_winner_pos(
            country, trained_kohonen_weights, distance_function
        )
        neuron_activations[winner_i][winner_j] = (
            neuron_activations[winner_i][winner_j][0] + country,
            neuron_activations[winner_i][winner_j][1] + 1,
        )

    average_values = [
        [np.zeros(len(variables)) for _ in range(som_dimension)]
        for _ in range(som_dimension)
    ]
    for i in range(som_dimension):
        for j in range(som_dimension):
            if neuron_activations[i][j][1] == 0:
                continue

            average_values[i][j] = (
                neuron_activations[i][j][0] / neuron_activations[i][j][1]
            )

    # Create a heatmap for each variable
    for variable_index in range(len(variables)):
        variable_values = [
            [average_values[i][j][variable_index] for j in range(som_dimension)]
            for i in range(som_dimension)
        ]

        plt.figure(figsize=(8, 6))
        plt.imshow(
            variable_values, cmap="Blues", interpolation="nearest", aspect="auto"
        )

        # Add black lines between cells
        for i in range(som_dimension + 1):
            plt.axhline(y=i - 0.5, color="black", linewidth=1.5)
            plt.axvline(x=i - 0.5, color="black", linewidth=1.5)

        plt.colorbar()  # Add a colorbar to the plot
        plt.title(f"Average {variables[variable_index]}")  # Set the title
        plt.yticks([])
        plt.xticks([])
        plt.savefig(f"average_{variables[variable_index]}.png")  # Save the figure
        plt.close()


def create_k_dead_neurons_chart(csv_file_path: str):
    with open(csv_file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        k_values = [int(row[0]) for row in reader]
        k_values = list(set(k_values))

        f.seek(0)
        next(reader)  # Skip the header

        dead_neurons: Dict[int, List[int]] = dict.fromkeys(k_values, [])
        for k in k_values:
            dead_neurons[k] = []

        for row in reader:
            dead_neurons[int(row[0])].append(int(row[1]))

        average_dead_neurons_as_percentages = [
            (100 * (sum(dead_neurons[k]) / (k * k))) / len(dead_neurons[k])
            for k in k_values
        ]
        std_dead_neurons_as_percentages = [
            (100 * np.std(dead_neurons[k]) / (k * k)) for k in k_values
        ]

        plt.figure(figsize=(8, 6))

        # Bar chart
        plt.bar(
            k_values,
            average_dead_neurons_as_percentages,
            yerr=std_dead_neurons_as_percentages,
            capsize=5,
        )
        plt.title(
            "Average Dead Neurons as percentage of total neurons"
        )  # Set the title
        plt.xticks(k_values)
        plt.ylabel("Average Dead Neurons (%)")
        plt.xlabel("k")

        # Add percentages to the y ticks
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        # Error bars
        plt.savefig("average_dead_neurons.png")  # Save the figure
