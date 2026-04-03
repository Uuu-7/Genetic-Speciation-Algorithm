import random
from typing import List, Optional

from .individual import Individual
from .population import Population
from .selection import tournament_selection
from .operators import (
    order_crossover,
    swap_mutation,
    robust_order_crossover,
    directed_swap_mutation,
    inject_segment,
    inject_by_mask,
    inject_edges_from_elite_to_non_elite,
)
from .mask import (
    build_edge_frequency_mask,
    build_position_informative_mask_from_robust,
    build_adjacency_pair_mask,
)
from .speciator import (
    speciate_by_rank,
    speciate_by_kmeans,
    speciate_by_wishart,
)
from .local_search import local_improve_chromosome


class GSA:
    def __init__(
        self,
        problem,
        population_size=60,
        generations=100,
        crossover_rate=0.7,
        mutation_rate=0.3,
        elite_ratio=0.3,
        tournament_size=3,
        mask_threshold=0.8,
        num_species=3,
        immigrant_rate=0.05,
        speciation_method="wishart",
        kmeans_random_state=42,
        # ===== Wishart speciation =====
        wishart_neighbors=7,
        wishart_significance_level=0.25,
        wishart_min_cluster_size=3,
        wishart_noise_policy="nearest",
        seed: Optional[int] = None,
        # ===== GEA scenarios =====
        scenario1_rate=0.2,
        scenario2_rate=0.2,
        scenario3_rate=0.4,
        cross_species_rate=0.3,
        non_elite_injection_bias=0.9,
        worst_injection_ratio=0.3,
        # ===== Local search =====
        local_search_rate=0.25,
        local_search_top_ratio=0.3,
        local_search_rounds=1,
    ):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        self.mask_threshold = mask_threshold
        self.num_species = num_species
        self.immigrant_rate = immigrant_rate

        self.speciation_method = speciation_method
        self.kmeans_random_state = kmeans_random_state

        self.wishart_neighbors = wishart_neighbors
        self.wishart_significance_level = wishart_significance_level
        self.wishart_min_cluster_size = wishart_min_cluster_size
        self.wishart_noise_policy = wishart_noise_policy

        self.seed = seed

        self.scenario1_rate = scenario1_rate
        self.scenario2_rate = scenario2_rate
        self.scenario3_rate = scenario3_rate
        self.cross_species_rate = cross_species_rate
        self.non_elite_injection_bias = non_elite_injection_bias
        self.worst_injection_ratio = worst_injection_ratio

        self.local_search_rate = local_search_rate
        self.local_search_top_ratio = local_search_top_ratio
        self.local_search_rounds = local_search_rounds

        self.population = None
        self.best_individual = None
        self.history = []
        self.species_history = []

        self.expected_gene_set = None
        self.expected_length = None

        self._validate_hyperparameters()
        if self.seed is not None:
            random.seed(self.seed)

    def _validate_hyperparameters(self):
        total = self.scenario1_rate + self.scenario2_rate + self.scenario3_rate

        plain_gea_mode = (
            self.num_species == 1
            and self.immigrant_rate == 0.0
            and self.cross_species_rate == 0.0
            and self.local_search_rate == 0.0
        )

        if (not plain_gea_mode) and total > 1.0 + 1e-12:
            raise ValueError(
                "scenario1_rate + scenario2_rate + scenario3_rate must be <= 1.0 "
                "when using probabilistic scenario routing"
            )

        if not (0.0 <= self.cross_species_rate <= 1.0):
            raise ValueError("cross_species_rate must be in [0, 1]")
        if not (0.0 <= self.immigrant_rate <= 1.0):
            raise ValueError("immigrant_rate must be in [0, 1]")
        if not (0.0 <= self.local_search_rate <= 1.0):
            raise ValueError("local_search_rate must be in [0, 1]")
        if not (0.0 <= self.local_search_top_ratio <= 1.0):
            raise ValueError("local_search_top_ratio must be in [0, 1]")
        if not (0.0 <= self.worst_injection_ratio <= 1.0):
            raise ValueError("worst_injection_ratio must be in [0, 1]")
        if not (0.0 <= self.non_elite_injection_bias <= 1.0):
            raise ValueError("non_elite_injection_bias must be in [0, 1]")

        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.num_species <= 0:
            raise ValueError("num_species must be positive")

        if self.wishart_neighbors <= 0:
            raise ValueError("wishart_neighbors must be positive")
        if self.wishart_significance_level < 0.0:
            raise ValueError("wishart_significance_level must be >= 0")
        if self.wishart_min_cluster_size <= 0:
            raise ValueError("wishart_min_cluster_size must be positive")

        valid_methods = {"rank", "kmeans", "wishart"}
        if self.speciation_method not in valid_methods:
            raise ValueError(
                f"speciation_method must be one of {sorted(valid_methods)}"
            )


    def initialize(self):
        self.population = Population.initialize(self.problem, self.population_size)
        self.population.evaluate(self.problem)
        self.population.sort_by_fitness()
        self.best_individual = self.population.best().copy()

        self.expected_gene_set = set(self.problem.customers)
        self.expected_length = len(self.problem.customers)

    def _repair_chromosome(self, chromosome):
        if self.expected_gene_set is None:
            self.expected_gene_set = set(self.problem.customers)
        if self.expected_length is None:
            self.expected_length = len(self.problem.customers)

        seen = set()
        repaired = []

        for gene in chromosome:
            if gene is None:
                continue
            if gene not in self.expected_gene_set:
                continue
            if gene in seen:
                continue
            repaired.append(gene)
            seen.add(gene)

        missing = [c for c in self.problem.customers if c not in seen]
        repaired.extend(missing)

        if len(repaired) > self.expected_length:
            repaired = repaired[:self.expected_length]

        return repaired

    def _make_safe_individual(self, chromosome):
        chromosome = self._repair_chromosome(chromosome)
        ind = Individual(chromosome)
        ind.fitness = self.problem.evaluate(ind.chromosome)
        return ind

    def _sanitize_individual(self, ind: Individual) -> Individual:
        repaired = self._repair_chromosome(ind.chromosome)
        if repaired != ind.chromosome:
            return self._make_safe_individual(repaired)

        if ind.fitness is None:
            ind.fitness = self.problem.evaluate(ind.chromosome)
        return ind

    def _is_plain_gea_mode(self) -> bool:
        return (
            self.num_species == 1
            and self.immigrant_rate == 0.0
            and self.cross_species_rate == 0.0
            and self.local_search_rate == 0.0
        )

    def _speciate(self, population):
        if self.speciation_method == "wishart":
            return speciate_by_wishart(
                population=population,
                problem=self.problem,
                wishart_neighbors=self.wishart_neighbors,
                significance_level=self.wishart_significance_level,
                min_cluster_size=self.wishart_min_cluster_size,
                attach_noise=self.wishart_noise_policy,
                fallback_to_rank=True,
            )

        if self.speciation_method == "kmeans":
            return speciate_by_kmeans(
                population=population,
                problem=self.problem,
                num_species=self.num_species,
                random_state=self.kmeans_random_state,
                fallback_to_rank=True,
            )

        return speciate_by_rank(population, num_species=self.num_species)

    def _species_quota(self, species_dict, min_quota=5):
        species_ids = list(species_dict.keys())
        num_species = len(species_ids)

        if num_species == 0:
            return {}

        if min_quota * num_species > self.population_size:
            base = self.population_size // num_species
            quotas = {sid: base for sid in species_ids}
            remainder = self.population_size - base * num_species
            for sid in species_ids[:remainder]:
                quotas[sid] += 1
            return quotas

        quotas = {sid: min_quota for sid in species_ids}
        remaining = self.population_size - min_quota * num_species

        best_values = {
            sid: min(ind.fitness for ind in members)
            for sid, members in species_dict.items()
        }

        weights = {sid: 1.0 / (best_values[sid] + 1e-8) for sid in species_ids}
        total_weight = sum(weights.values())

        assigned = 0
        extra_quota = {}

        for sid in species_ids[:-1]:
            q = int(remaining * weights[sid] / total_weight)
            extra_quota[sid] = q
            assigned += q

        last_sid = species_ids[-1]
        extra_quota[last_sid] = remaining - assigned

        for sid in species_ids:
            quotas[sid] += extra_quota[sid]

        return quotas

    def _select_parent_from_population(self):
        return tournament_selection(
            self.population.individuals,
            tournament_size=min(self.tournament_size, len(self.population.individuals)),
        )

    def _select_parent_from_species(self, members: List[Individual]):
        return tournament_selection(
            members,
            tournament_size=min(self.tournament_size, len(members)),
        )

    def _build_species_knowledge(self, species_members):
        species_members = sorted(species_members, key=lambda x: x.fitness)
        elite_count = max(1, int(round(len(species_members) * self.elite_ratio)))
        elite = [
            self._sanitize_individual(ind.copy())
            for ind in species_members[:elite_count]
        ]

        _, robust_individual = build_edge_frequency_mask(
            elite_individuals=elite,
            problem=self.problem,
            threshold=self.mask_threshold,
        )

        if robust_individual is None:
            robust_individual = elite[0]

        robust_individual = self._sanitize_individual(robust_individual.copy())

        informative_mask = build_position_informative_mask_from_robust(
            elite_individuals=elite,
            robust_individual=robust_individual,
            threshold=self.mask_threshold,
        )

        return elite, robust_individual.copy(), informative_mask

    def _apply_base_ga_step(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child1, child2 = order_crossover(parent1, parent2)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        child1 = swap_mutation(child1, mutation_rate=self.mutation_rate)
        child2 = swap_mutation(child2, mutation_rate=self.mutation_rate)

        child1 = self._sanitize_individual(child1)
        child2 = self._sanitize_individual(child2)

        return child1, child2

    def _get_worst_pool(self, sorted_members):
        n = len(sorted_members)
        if n == 0:
            return []

        worst_size = max(1, int(round(n * self.worst_injection_ratio)))
        return [
            self._sanitize_individual(ind.copy())
            for ind in sorted_members[-worst_size:]
        ]

    def _apply_local_search_to_individual(self, ind: Individual) -> Individual:
        ind = self._sanitize_individual(ind)

        original_fitness = ind.fitness
        new_chromosome, improved = local_improve_chromosome(
            chromosome=ind.chromosome,
            problem=self.problem,
            use_two_opt=True,
            use_relocate=True,
            rounds=self.local_search_rounds,
        )

        new_chromosome = self._repair_chromosome(new_chromosome)

        if improved:
            new_fitness = self.problem.evaluate(new_chromosome)
            if new_fitness < original_fitness:
                child = Individual(new_chromosome)
                child.fitness = new_fitness
                return child

        return ind

    def _refine_offspring_with_local_search(self, offspring: List[Individual]) -> List[Individual]:
        if not offspring:
            return offspring

        offspring = [self._sanitize_individual(ind) for ind in offspring]
        offspring.sort(key=lambda x: x.fitness)

        top_k = max(1, int(round(len(offspring) * self.local_search_top_ratio)))

        refined = []
        for idx, ind in enumerate(offspring):
            if idx < top_k and random.random() < self.local_search_rate:
                refined.append(self._apply_local_search_to_individual(ind))
            else:
                refined.append(ind)

        return refined

    def _evolve_one_generation_plain_gea(self):
        self.population.sort_by_fitness()
        pop = [self._sanitize_individual(ind.copy()) for ind in self.population.individuals]

        elite_count = max(1, int(round(len(pop) * self.elite_ratio)))
        elite = [self._sanitize_individual(ind.copy()) for ind in pop[:elite_count]]
        non_elite = [self._sanitize_individual(ind.copy()) for ind in pop[elite_count:]]

        _, robust_individual = build_edge_frequency_mask(
            elite_individuals=elite,
            problem=self.problem,
            threshold=self.mask_threshold,
        )
        if robust_individual is None:
            robust_individual = elite[0].copy()

        informative_mask = build_position_informative_mask_from_robust(
            elite_individuals=elite,
            robust_individual=robust_individual,
            threshold=self.mask_threshold,
        )

        offspring = []

        # 1) Classical GA offspring
        while len(offspring) < self.population_size:
            parent1 = tournament_selection(
                pop, tournament_size=min(self.tournament_size, len(pop))
            )
            parent2 = tournament_selection(
                pop, tournament_size=min(self.tournament_size, len(pop))
            )

            child1, child2 = self._apply_base_ga_step(parent1, parent2)
            offspring.append(self._sanitize_individual(child1))
            if len(offspring) < self.population_size:
                offspring.append(self._sanitize_individual(child2))

        # 2) Scenario 1 offspring
        s1_count = int(round(self.population_size * self.scenario1_rate))
        for _ in range(s1_count):
            parent2 = tournament_selection(
                pop, tournament_size=min(self.tournament_size, len(pop))
            )
            child = robust_order_crossover(robust_individual, parent2)
            child = swap_mutation(child, mutation_rate=self.mutation_rate)
            offspring.append(self._sanitize_individual(child))

        # 3) Scenario 2 offspring
        s2_count = int(round(self.population_size * self.scenario2_rate))
        for _ in range(s2_count):
            base = tournament_selection(
                elite, tournament_size=min(self.tournament_size, len(elite))
            )
            child = directed_swap_mutation(
                individual=base,
                problem=self.problem,
                informative_mask=informative_mask,
                mutation_rate=1.0,
            )
            offspring.append(self._sanitize_individual(child))

        # 4) Scenario 3 offspring
        s3_count = int(round(self.population_size * self.scenario3_rate))
        worst_pool_size = max(1, int(round(len(pop) * self.worst_injection_ratio)))
        worst_pool = [self._sanitize_individual(ind.copy()) for ind in pop[-worst_pool_size:]]

        informative_pairs = build_adjacency_pair_mask(
            elite_individuals=elite,
            problem=self.problem,
            threshold=self.mask_threshold,
        )

        for _ in range(s3_count):
            donor = random.choice(elite).copy()

            if non_elite:
                target = random.choice(non_elite).copy()
            else:
                target = random.choice(pop).copy()

            child = inject_edges_from_elite_to_non_elite(
                target=target,
                donor=donor,
                informative_pairs=informative_pairs,
                problem=self.problem,
            )
            offspring.append(self._sanitize_individual(child))

        merged = pop + offspring
        merged = [self._sanitize_individual(ind) for ind in merged]
        merged.sort(key=lambda x: x.fitness)

        self.population = Population(merged[:self.population_size])

        current_best = self.population.best()
        if current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best.copy()

        species_summary = [{
            "species_id": 0,
            "size": len(self.population.individuals),
            "best": self.population.best().fitness,
            "worst": self.population.worst().fitness,
            "quota": len(self.population.individuals),
        }]
        self.species_history.append(species_summary)

    def _evolve_species(self, sid, species_members, quota):
        del sid

        species_pop = Population(
            [self._sanitize_individual(ind.copy()) for ind in species_members]
        )
        species_pop.sort_by_fitness()

        elite, robust_individual, informative_mask = self._build_species_knowledge(
            species_pop.individuals
        )

        sorted_members = species_pop.individuals
        worst_pool = self._get_worst_pool(sorted_members)

        new_individuals = []

        elite_keep = min(len(elite), max(1, quota // 3))
        for ind in elite[:elite_keep]:
            new_individuals.append(self._sanitize_individual(ind.copy()))

        while len(new_individuals) < quota:
            r = random.random()

            # Scenario 1
            if r < self.scenario1_rate:
                if random.random() < self.cross_species_rate:
                    parent2 = self._select_parent_from_population()
                else:
                    parent2 = self._select_parent_from_species(species_pop.individuals)

                child = robust_order_crossover(robust_individual, parent2)
                child = swap_mutation(child, mutation_rate=self.mutation_rate)
                child = self._sanitize_individual(child)
                new_individuals.append(child)
                continue

            # Scenario 2
            if r < self.scenario1_rate + self.scenario2_rate:
                if elite:
                    base = self._select_parent_from_species(elite)
                else:
                    base = self._select_parent_from_species(species_pop.individuals)

                child = directed_swap_mutation(
                    individual=base,
                    problem=self.problem,
                    informative_mask=informative_mask,
                    mutation_rate=1.0,
                )
                child = self._sanitize_individual(child)
                new_individuals.append(child)
                continue

            # Scenario 3
            if r < self.scenario1_rate + self.scenario2_rate + self.scenario3_rate:
                donor = random.choice(elite) if elite else robust_individual.copy()

                use_worst = worst_pool and (
                    random.random() < self.non_elite_injection_bias
                )
                if use_worst:
                    target = random.choice(worst_pool).copy()
                else:
                    target = self._select_parent_from_species(
                        species_pop.individuals
                    ).copy()

                child = inject_by_mask(
                    target=target,
                    donor=donor,
                    informative_mask=informative_mask,
                    problem=self.problem,
                )
                child = self._sanitize_individual(child)
                new_individuals.append(child)
                continue

            # Base GA
            parent1 = self._select_parent_from_species(species_pop.individuals)

            if random.random() < self.cross_species_rate:
                parent2 = self._select_parent_from_population()
            else:
                parent2 = self._select_parent_from_species(species_pop.individuals)

            child1, child2 = self._apply_base_ga_step(parent1, parent2)
            new_individuals.append(child1)
            if len(new_individuals) < quota:
                new_individuals.append(child2)

        new_individuals = new_individuals[:quota]
        new_individuals = [self._sanitize_individual(ind) for ind in new_individuals]

        if self.local_search_rate > 0 and self.local_search_top_ratio > 0:
            new_individuals = self._refine_offspring_with_local_search(new_individuals)

        new_individuals = [self._sanitize_individual(ind) for ind in new_individuals]
        new_individuals.sort(key=lambda x: x.fitness)

        return new_individuals

    def _inject_immigrants(self, offspring):
        num_immigrants = int(self.population_size * self.immigrant_rate)
        if num_immigrants <= 0:
            return offspring

        offspring = [self._sanitize_individual(ind) for ind in offspring]
        offspring.sort(key=lambda ind: ind.fitness, reverse=True)

        for i in range(min(num_immigrants, len(offspring))):
            chromosome = self.problem.random_chromosome()
            offspring[i] = self._make_safe_individual(chromosome)

        return offspring

    def _summarize_species(self, species_dict, quotas=None):
        summary = []
        for sid, members in species_dict.items():
            fitness_values = [ind.fitness for ind in members]
            item = {
                "species_id": sid,
                "size": len(members),
                "best": min(fitness_values),
                "worst": max(fitness_values),
            }
            if quotas is not None and sid in quotas:
                item["quota"] = quotas[sid]
            summary.append(item)

        summary.sort(key=lambda x: x["species_id"])
        return summary

    def evolve_one_generation(self):
        if self._is_plain_gea_mode():
            self._evolve_one_generation_plain_gea()
            return

        species_dict = self._speciate(self.population)
        quotas = self._species_quota(species_dict)

        offspring = []
        for sid, members in species_dict.items():
            species_offspring = self._evolve_species(sid, members, quotas[sid])
            offspring.extend(species_offspring)

        while len(offspring) < self.population_size:
            chromosome = self.problem.random_chromosome()
            offspring.append(self._make_safe_individual(chromosome))

        offspring = self._inject_immigrants(offspring)
        offspring = offspring[:self.population_size]
        offspring = [self._sanitize_individual(ind) for ind in offspring]

        self.population = Population(offspring)
        self.population.sort_by_fitness()

        current_best = self.population.best()
        if current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best.copy()

        species_dict_after = self._speciate(self.population)
        quotas_after = self._species_quota(species_dict_after)
        species_summary = self._summarize_species(species_dict_after, quotas=quotas_after)
        self.species_history.append(species_summary)

    def run(self, verbose=True, verbose_species=False):
        self.initialize()

        for gen in range(self.generations):
            self.evolve_one_generation()

            best_fitness = self.population.best().fitness
            self.history.append(best_fitness)

            if verbose:
                print(f"Generation {gen + 1:03d} | Best fitness: {best_fitness}")

            if verbose_species:
                species_summary = self.species_history[-1]
                for item in species_summary:
                    print(
                        f"  Species {item['species_id']}: "
                        f"size={item['size']}, "
                        f"best={item['best']}, "
                        f"worst={item['worst']}, "
                        f"quota={item['quota']}"
                    )

        return self.best_individual