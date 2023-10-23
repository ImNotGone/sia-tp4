import json
import numpy as np
from hopfield import Hopfield, NDArray
import matplotlib.pyplot as plt

def main():

    with open("letters.json", 'r') as f:
        letters = json.load(f)

    # TODO: move this to config
    # NOTA: Siguiendo lo que vimos en clase
    # de utilizar aproximadamente el 15% de los patrones
    # como hay 26 letras, 15% de 26 = 3.9 ~= 4
    selected_letters = ["A", "G", "J", "Z"]

    patterns = []
    for letter in selected_letters:
        mat = letters[letter]
        pattern = []
        for row in mat:
            for value in row:
                pattern += [value]
        patterns += [pattern]

    patterns = np.array(patterns)

    # TODO: move to conf
    noise_percentage = 0.4
    limit = 100

    hopfield = Hopfield(patterns)

    pattern = noisify_pattern(patterns[0], noise_percentage)

    print_letter(pattern)

    pattern, energy, patterns = hopfield.calculate_output(np.array(pattern), limit)

    print_letter(pattern)

    print(hopfield.is_spurious(pattern))

    print(pattern, energy, len(energy))

    display_steps(patterns)

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

# TODO: implementar esto
def generate_orthogonal_patterns(patterns: NDArray) -> NDArray:
    pass

def print_letter(pattern):
    i = 0
    line = []
    for value in pattern:
        if(value == 1):
            line += ['#']
        else:
            line += [' ']
        i+=1
        if (i%5==0):
            print(line)
            line = []

# muestra los pasos en matrices de 5x5
def display_steps(patterns):
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
