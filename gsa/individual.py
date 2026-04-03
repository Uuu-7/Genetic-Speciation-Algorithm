from typing import List, Optional


class Individual:
    def __init__(self, chromosome: List[int]):
        self.chromosome = chromosome
        self.fitness: Optional[float] = None
        self.species_id: Optional[int] = None

    def copy(self) -> "Individual":
        new_ind = Individual(self.chromosome[:])
        new_ind.fitness = self.fitness
        new_ind.species_id = self.species_id
        return new_ind

    def __repr__(self) -> str:
        return (
            f"Individual(fitness={self.fitness}, "
            f"species_id={self.species_id}, "
            f"chromosome={self.chromosome})"
        )