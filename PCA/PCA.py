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


def create_first_component_biplot(
    pca_result: NDArray, pca: PCA, countries: List[str], variables: List[str]
):
    pc1 = pca_result[:, 0]

    # Pesos
    pc1_loadings = pca.components_[0, :]

    plt.figure(figsize=(10, 6))
    plt.scatter(pc1, np.zeros_like(pc1), marker="o", label="PC1 Scores", color="blue")

    # Plot the variable loadings as vectors
    y_offset = 0.02
    for i, (loading, variable) in enumerate(zip(pc1_loadings, variables)):
        plt.arrow(0, y_offset * (i + 1), loading, 0, color="red", alpha=0.7)

        text_x = loading + 0.02 if loading > 0 else loading - 0.09 * len(variable)

        plt.text(text_x, y_offset * (i + 1), variable, fontsize=12, color="red")

    for i, score in enumerate(pc1):
        plt.annotate(
            countries[i],
            (score, 0),
            xytext=(0, -70),
            textcoords="offset points",
            fontsize=8,
            ha="right",
            rotation=90,
        )

    plt.xlabel("PC1 Scores")
    plt.ylim(-0.05, 0.2)
    plt.legend()
    plt.grid()
    plt.title("Biplot: PC1 Scores and Variable Loadings")
    plt.savefig("first_component_biplot.png")


def create_first_second_component_biplot(
    pca_result: NDArray, pca: PCA, countries: List[str], variables: List[str]
):
    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]

    # Set the colors
    variable_colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Pesos
    pc1_loadings = pca.components_[0, :]
    pc2_loadings = pca.components_[1, :]

    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.scatter(pc1, pc2, marker="o", label="PCA Scores", color="blue")

    # Plot the variable loadings as vectors
    for pc1_loading, pc2_loading, variable, color in zip(
        pc1_loadings, pc2_loadings, variables, variable_colors
    ):
        plt.arrow(0, 0, pc1_loading, pc2_loading, color=color, alpha=0.7)

        text_x = (
            pc1_loading + 0.02
            if pc1_loading > 0
            else pc1_loading - 0.09 * len(variable)
        )
        text_y = pc2_loading + 0.02

        plt.text(text_x, text_y, variable, fontsize=12, color=color)

    for i, (score1, score2) in enumerate(zip(pc1, pc2)):
        plt.annotate(
            countries[i],
            (score1, score2),
            xytext=(25, -10),
            textcoords="offset points",
            fontsize=8,
            ha="right",
        )

    plt.xlabel("PC1 Scores")
    plt.ylabel("PC2 Scores")
    plt.legend()
    plt.title("Biplot: PC1, PC2 Scores and Variable Loadings")
    plt.savefig("first_second_component_biplot.png")


# -----------------------------------------------------------------------------

# Load data
country_names, variables, country_data = load_countries("../data/europe.csv")

# Standardize data
scaler = StandardScaler()

scaler.fit(country_data)
scaled_data = scaler.transform(country_data)

# Create boxplots
create_original_vs_standardized_boxplots(country_data, scaled_data, variables)


pca = PCA(n_components=2)
pca.fit(scaled_data)
x = pca.transform(scaled_data)

create_first_component_biplot(x, pca, country_names, variables)
create_first_second_component_biplot(x, pca, country_names, variables)
