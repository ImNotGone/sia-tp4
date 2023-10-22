import json
import numpy as np
from hopfield import Hopfield
import copy

def main():

    with open("letters.json", 'r') as f:
        letters = json.load(f)

    # TODO: move this to config
    selected_letters = ["A", "G", "J", "L", "Z"]

    patterns = []
    for letter in selected_letters:
        mat = letters[letter]
        pattern = []
        for row in mat:
            for value in row:
                pattern += [value]
        patterns += [pattern]

    patterns = np.array(patterns)

    hopfield = Hopfield(patterns)

    print(hopfield.calculate_energy(patterns[0]))

    pattern = [+1, +1, +1, +1, -1,
               +1, -1, -1, -1, +1,
               +1, +1, -1, +1, +1,
               +1, -1, -1, -1, +1,
               +1, -1, -1, -1, +1]

    pattern, energy = hopfield.calculate_output(np.array(pattern))

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

    print(pattern, energy, len(energy))

    return 0

if __name__ == "__main__":
    main()
