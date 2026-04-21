"""Individual representation. A chromosome is a permutation of customers."""

import numpy as np


class Individual:
    __slots__ = ("perm", "fitness")

    def __init__(self, perm, fitness=None):
        self.perm = np.asarray(perm, dtype=np.int64)
        self.fitness = fitness

    def copy(self):
        return Individual(self.perm.copy(), self.fitness)

    def __len__(self):
        return len(self.perm)

    def __repr__(self):
        return f"Ind(fit={self.fitness:.2f})" if self.fitness is not None else "Ind(?)"