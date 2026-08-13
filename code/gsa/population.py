"""Population container."""

import numpy as np
from .individual import Individual


class Population:
    def __init__(self, inds):
        self.inds = list(inds)

    def evaluate_all(self, problem):
        for ind in self.inds:
            if ind.fitness is None:
                ind.fitness = problem.fast_evaluate(ind.perm)

    def sort(self):
        self.inds.sort(key=lambda x: x.fitness)

    def best(self):
        return min(self.inds, key=lambda x: x.fitness)

    def worst_index(self):
        wi = 0
        wf = self.inds[0].fitness
        for i in range(1, len(self.inds)):
            if self.inds[i].fitness > wf:
                wf = self.inds[i].fitness
                wi = i
        return wi

    def __len__(self):
        return len(self.inds)


def random_population(problem, size, rng):
    inds = []
    n = problem.n
    for _ in range(size):
        perm = np.arange(1, n + 1, dtype=np.int64)
        rng.shuffle(perm)
        ind = Individual(perm)
        ind.fitness = problem.fast_evaluate(ind.perm)
        inds.append(ind)
    return Population(inds)