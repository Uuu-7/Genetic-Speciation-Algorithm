import random
from typing import List

from .individual import Individual


class Population:
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @classmethod
    def initialize(cls, problem, size: int) -> "Population":
        individuals = []
        for _ in range(size):
            chromosome = problem.random_chromosome()
            individuals.append(Individual(chromosome))
        return cls(individuals)

    def evaluate(self, problem) -> None:
        for ind in self.individuals:
            ind.fitness = problem.evaluate(ind.chromosome)

    def best(self) -> Individual:
        return min(self.individuals, key=lambda ind: ind.fitness)

    def worst(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.fitness)

    def sort_by_fitness(self) -> None:
        self.individuals.sort(key=lambda ind: ind.fitness)

    def get_elite(self, ratio: float) -> List[Individual]:
        self.sort_by_fitness()
        elite_size = max(1, int(len(self.individuals) * ratio))
        return [ind.copy() for ind in self.individuals[:elite_size]]

    def sample(self, k: int) -> List[Individual]:
        return random.sample(self.individuals, k)

    def extend(self, new_individuals: List[Individual]) -> None:
        self.individuals.extend(new_individuals)

    def subset(self, individuals: List[Individual]) -> "Population":
        return Population(individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]