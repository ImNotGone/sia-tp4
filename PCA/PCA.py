from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------

def load_countries(file_path: str) -> Tuple[List[str], List[str], NDArray]:
    country_data = []
    countries = []
    variables = []

    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        variables = next(csv_reader)[1:]

        for row in csv_reader:
            countries.append(row[0])

            # Convert strings to floats
            country_data.append([float(x) for x in row[1:]])

    return countries, variables, np.array(country_data)


def create_original_vs_standardized_boxplots(
    original_data: NDArray, scaled_data: NDArray, variables: List[str]
):
    # Set the colors
    variable_colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Create a figure for the original data boxplot
    plt.figure(figsize=(10, 6))

    # Iterate through each variable and create a custom boxplot
    for i in range(original_data.shape[1]):
        plt.boxplot(
            original_data[:, i],
            positions=[i],
            patch_artist=True,
            boxprops=dict(
                facecolor=variable_colors[i], edgecolor=variable_colors[i], alpha=0.5
            ),
            capprops=dict(color=variable_colors[i]),
            whiskerprops=dict(color=variable_colors[i]),
            medianprops=dict(color=variable_colors[i]),
            flierprops=dict(
                color=variable_colors[i],
                markeredgecolor=variable_colors[i],
                marker="o",
                alpha=0.75,
            ),
        )

    plt.title("Original Data")
    plt.xticks(range(original_data.shape[1]), variables)
    plt.xlabel("Indicators")
    plt.ylabel("Values")

    # Add background color to the plot and lines
    plt.axvspan(-0.5, 6.5, facecolor="lightsteelblue", alpha=0.5)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig("original_data_boxplot.png")

    # Create a figure for the standardized data boxplot
    plt.figure(figsize=(10, 6))

    # Iterate through each variable and create a custom boxplot
    for i in range(scaled_data.shape[1]):
        plt.boxplot(
            scaled_data[:, i],
            positions=[i],
            patch_artist=True,
            boxprops=dict(
                facecolor=variable_colors[i], edgecolor=variable_colors[i], alpha=0.5
            ),
            capprops=dict(color=variable_colors[i]),
            whiskerprops=dict(color=variable_colors[i]),
            medianprops=dict(color=variable_colors[i]),
            flierprops=dict(
                color=variable_colors[i],
                markeredgecolor=variable_colors[i],
                marker="o",
                alpha=0.75,
            ),
        )

    plt.title("Standardized Data")
    plt.xticks(range(scaled_data.shape[1]), variables)
    plt.xlabel("Indicators")
    plt.ylabel("Standardized Values")

    plt.axvspan(-0.5, 6.5, facecolor="lightsteelblue", alpha=0.5)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig("standardized_data_boxplot.png")

# -----------------------------------------------------------------------------

# Load data
country_names, variables, country_data = load_countries("../data/europe.csv")

# Standardize data
scaler = StandardScaler()

scaler.fit(country_data)
scaled_data = scaler.transform(country_data)

# Create boxplots
create_original_vs_standardized_boxplots(country_data, scaled_data, variables)


# TODO check numero de componentes
pca = PCA(n_components=7)
pca.fit(scaled_data)
x = pca.transform(scaled_data)
