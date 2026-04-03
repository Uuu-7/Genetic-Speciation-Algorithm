import math
import random
from typing import List


class VRP:
    def __init__(self, path: str):
        self.path = path
        self.name = ""
        self.dimension = 0
        self.capacity = 0
        self.coords = {}   # node_id -> (x, y)
        self.demands = {}  # node_id -> demand
        self.depot = 1

        self._load_vrp(path)
        self.customers = [i for i in range(1, self.dimension + 1) if i != self.depot]
        self.distance_matrix = self._build_distance_matrix()

    def _load_vrp(self, path: str) -> None:
        section = None

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("NAME"):
                    self.name = line.split(":")[-1].strip()
                elif line.startswith("DIMENSION"):
                    self.dimension = int(line.split(":")[-1].strip())
                elif line.startswith("CAPACITY"):
                    self.capacity = int(line.split(":")[-1].strip())
                elif line.startswith("NODE_COORD_SECTION"):
                    section = "coords"
                    continue
                elif line.startswith("DEMAND_SECTION"):
                    section = "demands"
                    continue
                elif line.startswith("DEPOT_SECTION"):
                    section = "depot"
                    continue
                elif line.startswith("EOF"):
                    break

                if section == "coords":
                    parts = line.split()
                    if len(parts) == 3:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        self.coords[node_id] = (x, y)

                elif section == "demands":
                    parts = line.split()
                    if len(parts) == 2:
                        node_id = int(parts[0])
                        demand = int(parts[1])
                        self.demands[node_id] = demand

                elif section == "depot":
                    if line == "-1":
                        section = None
                    else:
                        self.depot = int(line)

        if not self.coords:
            raise ValueError("VRP file parse failed: NODE_COORD_SECTION not found or empty.")
        if not self.demands:
            raise ValueError("VRP file parse failed: DEMAND_SECTION not found or empty.")
        if self.dimension <= 0:
            raise ValueError("VRP file parse failed: invalid DIMENSION.")
        if self.capacity <= 0:
            raise ValueError("VRP file parse failed: invalid CAPACITY.")

    def _euclidean(self, a: int, b: int) -> float:
        x1, y1 = self.coords[a]
        x2, y2 = self.coords[b]
        return round(math.hypot(x1 - x2, y1 - y2))

    def _build_distance_matrix(self) -> List[List[float]]:
        size = self.dimension + 1
        dist = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(1, self.dimension + 1):
            for j in range(1, self.dimension + 1):
                dist[i][j] = self._euclidean(i, j)
        return dist

    def random_chromosome(self) -> List[int]:
        chromosome = self.customers[:]
        random.shuffle(chromosome)
        return chromosome

    def decode(self, chromosome: List[int]) -> List[List[int]]:
        routes = []
        current_route = [self.depot]
        current_load = 0

        for customer in chromosome:
            if customer not in self.demands:
                continue

            demand = self.demands[customer]

            if current_load + demand <= self.capacity:
                current_route.append(customer)
                current_load += demand
            else:
                current_route.append(self.depot)
                routes.append(current_route)

                current_route = [self.depot, customer]
                current_load = demand

        current_route.append(self.depot)
        routes.append(current_route)

        return routes

    def route_distance(self, route: List[int]) -> float:
        total = 0.0
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i]][route[i + 1]]
        return total

    def validate_solution(self, chromosome: List[int]) -> List[str]:
        customers = list(self.customers)
        expected = set(customers)

        actual = list(chromosome)
        actual_set = set(actual)

        errors = []

        if len(actual) != len(customers):
            errors.append(f"Length mismatch: expected {len(customers)}, got {len(actual)}")

        if any(g is None for g in actual):
            errors.append("Chromosome contains None")

        if len(actual_set) != len(actual):
            errors.append("Duplicate customers found in chromosome")

        missing = expected - actual_set
        extra = actual_set - expected

        if missing:
            errors.append(f"Missing customers: {sorted(missing)}")
        if extra:
            errors.append(f"Unexpected customers: {sorted(extra)}")

        if errors:
            return errors

        try:
            routes = self.decode(chromosome)
        except Exception as e:
            errors.append(f"Decode failed: {e}")
            return errors

        served = []
        for route in routes:
            route_customers = [c for c in route if c != self.depot]

            load = sum(self.demands[c] for c in route_customers)
            if load > self.capacity:
                errors.append(f"Capacity violation: load={load} > capacity={self.capacity}")

            served.extend(route_customers)

        served_set = set(served)

        if served_set != expected:
            errors.append(
                f"Decoded routes mismatch. Missing in routes: {sorted(expected - served_set)}, "
                f"extra in routes: {sorted(served_set - expected)}"
            )

        if len(served) != len(served_set):
            errors.append("Duplicate customers found across decoded routes")

        return errors

    def evaluate(self, chromosome: List[int]) -> float:
        errors = self.validate_solution(chromosome)
        if errors:
            return 10**9

        routes = self.decode(chromosome)
        total_distance = sum(self.route_distance(route) for route in routes)
        return total_distance

    def is_feasible(self, chromosome: List[int]) -> bool:
        return len(self.validate_solution(chromosome)) == 0

    def pretty_routes(self, chromosome: List[int]) -> List[str]:
        routes = self.decode(chromosome)
        result = []
        for idx, route in enumerate(routes, start=1):
            load = sum(self.demands[node] for node in route if node != self.depot)
            dist = self.route_distance(route)
            result.append(f"Route {idx}: {route} | load={load} | dist={dist}")
        return result