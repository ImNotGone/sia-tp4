import json
from typing import Dict
import itertools
import numpy as np
from hopfield import Hopfield, NDArray
import matplotlib.pyplot as plt

def main():

    with open("letters.json", 'r') as f:
        letters = json.load(f)

    with open("config.json", 'r') as f:
        config = json.load(f)

    # NOTA: Siguiendo lo que vimos en clase
    # de utilizar aproximadamente el 15% de los patrones
    # como hay 26 letras, 15% de 26 = 3.9 ~= 4
    selected_letters = config['selected_pattern']

    print(f"Using: {selected_letters}")

    for key, value in letters.items():
        letters[key] = np.reshape(value, 25)

    combination_values = generate_orthogonal_patterns(letters, 4)
    print(combination_values)

    patterns = []
    for letter in selected_letters:
        patterns += [letters[letter]]

    patterns = np.array(patterns)

    noise_percentage = config["noise_percentage"]
    limit = config["limit"]

    hopfield = Hopfield(patterns)

    pattern = noisify_pattern(patterns[0], noise_percentage)

    # print_letter(pattern)

    pattern, energy, patterns = hopfield.calculate_output(np.array(pattern), limit)

    #print_letter(pattern)

    print(f"Is spurious: {hopfield.is_spurious(pattern)}")
    print_letter(pattern)
    print(energy, len(energy))

    if (config["display_steps"]):
        display_steps(patterns)
    if (config["display_energy"]):
        display_energy(energy)

    return 0

# agrega ruido a un patron
def noisify_pattern(pattern: NDArray, noise_percentage: float) -> NDArray:
    pattern = np.copy(pattern)
    # a cuantos les hago flip
    qty_to_flip = int(noise_percentage * len(pattern))
    # a cuales les hago flip
    index_to_flip = np.random.choice(len(pattern), qty_to_flip, replace = False)
    # flip
    pattern[index_to_flip] *= -1
    return pattern

def generate_orthogonal_patterns(letters: Dict[int, NDArray], qty_per_group: int):
    combinations = itertools.combinations(letters.keys(), qty_per_group)

    # TODO: podria usarlo para no tener q calcular el dot en todos
    # sino usar la mat y es O[1] (obviamente sin contar q tengo q hacer la mat primero)
    # mat = calculate_orthogonality_matrix(letters.values())

    output = []

    for combination in combinations:
        k = 0
        ortogonality = 0
        max = 0
        qty = 0
        for i in range(qty_per_group):
            for j in range(i+1, qty_per_group):
                normalized_dot_prod = np.dot(letters[combination[i]], letters[combination[j]])
                normalized_dot_prod = abs(normalized_dot_prod)
                ortogonality += normalized_dot_prod
                k += 1
                if max < normalized_dot_prod:
                    max = normalized_dot_prod
                    qty = 1
                elif max == normalized_dot_prod:
                    qty += 1
        normalized_ortogonality = ortogonality / (k)
        output += [[list(combination), normalized_ortogonality, max, qty]]

    # ordenar x maximo, despues x cantidad de maximos y por ultimo x normalized_ortogonality
    output.sort(key=lambda entry: (entry[2], entry[3], entry[1]), reverse=True)

    return output

# calcula la matriz de ortogonalidad para todos los valores entre si
def calculate_orthogonality_matrix(patterns):
    num_patterns = len(patterns)
    orthogonality_matrix = np.zeros((num_patterns, num_patterns))

    for i in range(num_patterns):
        for j in range(i, num_patterns):
            pattern1 = patterns[i]
            pattern2 = patterns[j]
            dot_product = np.dot(pattern1, pattern2)
            orthogonality = dot_product / len(pattern1)
            orthogonality_matrix[i, j] = orthogonality
            orthogonality_matrix[j, i] = orthogonality

    return orthogonality_matrix


def print_letter(pattern):
    i = 0
    line = "["
    for value in pattern:
        if(value == 1):
            line += '#'
        else:
            line += ' '
        i+=1
        if (i%5==0):
            print(line + "]")
            line = "["
    print()

# muestra como cambia la energia en funcion de las epocas
def display_energy(energy: NDArray):
    epochs = np.arange(len(energy))

    plt.plot(epochs, energy, marker='o', linestyle='-', color='b')
    plt.title("Energy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.show()

# muestra los pasos en matrices de 5x5
def display_steps(patterns: NDArray):
    num_patterns = len(patterns)
    cols = int(np.ceil(np.sqrt(num_patterns)))
    rows = int(np.ceil(num_patterns / cols))

    _, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, pattern in enumerate(patterns):
        ax = axes[i // cols, i % cols]
        ax.imshow(pattern.reshape(5, 5), cmap='binary')
        ax.axis('off')

    for i in range(num_patterns, rows*cols):
        axes[i // cols, i % cols].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
