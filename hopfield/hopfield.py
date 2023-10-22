
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class Hopfield:
    def __init__(self, patterns: NDArray):
        # ya vienen en lista
        # si fuese [[-1, 1], [1, 1]]
        # aca recibo [-1, 1, 1, 1]
        self._patterns = patterns
        self._weigths = self.create_weight_matrix()

    def create_weight_matrix(self) -> NDArray:
        k_t_mat = self._patterns
        k_mat = np.transpose(k_t_mat)

        N = len(k_t_mat)
        # @ para multiplicar matrices
        weigths = (k_mat @ k_t_mat) / N

        # lleno la diagonal con 0's para evitar conexiones sobre si mismos
        np.fill_diagonal(weigths, 0)

        return weigths


    def calculate_output(self, input: NDArray) -> Tuple[NDArray, NDArray]:
        # TODO: hacer limit parametro?
        limit = 100
        i = 0

        prev = None
        output = input

        energy_eta = []
        energy_eta.append(self.calculate_energy(output))

        while(i < limit and not output.all(prev)):
            prev = output
            # @ para multiplicar matrices
            # TODO: revisar lo del 0
            output = np.sign(self._weigths @ prev)
            energy_eta.append(self.calculate_energy(output))
            i += 1

        return output, np.array(energy_eta)

    # TODO: ver si es mas rapido hacer esto o usar la matriz entera
    # y multiplicaciones con numpy
    def calculate_energy(self, pattern: NDArray) -> int:
        energy = 0
        for i in range(len(self._patterns[0])):
            for j in range(i+1, len(self._patterns[0])):
                energy -= self._weigths[i][j] * pattern[i] * pattern[j]
        return energy

