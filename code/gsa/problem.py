"""Capacitated VRP problem definition.

Key design choice: `fast_evaluate` does a single pass to compute total
distance (depot included on capacity overflow) and does NOT run any
O(n^2) feasibility check. Validation is only performed when a solution
is reported to the user.
"""

import numpy as np


class VRP:
    """Capacitated Vehicle Routing Problem.

    - coords[0] is the depot, coords[1..n] are customers.
    - demands[0] = 0, demands[1..n] are customer demands.
    - capacity is the vehicle capacity (all vehicles identical).
    - A solution is a permutation of [1..n]; routes are produced greedily
      by splitting the permutation every time capacity would be exceeded.
    """

    def __init__(self, coords, demands, capacity):
        self.coords = np.asarray(coords, dtype=float)
        self.demands = np.asarray(demands, dtype=float)
        self.capacity = float(capacity)
        self.n = len(coords) - 1  # number of customers
        # Precompute distance matrix (Euclidean).
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.sqrt((diff ** 2).sum(axis=-1))

    def fast_evaluate(self, perm):
        """Return total route length. Single pass, no validation.

        This is the hot function called by every operator. It must stay
        cheap: no sorted() checks, no decode_routes().
        """
        D = self.dist
        dem = self.demands
        cap = self.capacity
        total = 0.0
        cap_used = 0.0
        prev = 0  # depot
        for c in perm:
            d = dem[c]
            if cap_used + d > cap:
                # close current route back to depot, open a new one
                total += D[prev, 0]
                prev = 0
                cap_used = 0.0
            total += D[prev, c]
            prev = c
            cap_used += d
        total += D[prev, 0]
        return total

    # Convenience alias; some code paths may call `evaluate`.
    def evaluate(self, perm):
        return self.fast_evaluate(perm)

    def decode_routes(self, perm):
        """Split a permutation into a list of routes (customers only, no depot)."""
        routes = []
        current = []
        cap_used = 0.0
        for c in perm:
            d = self.demands[c]
            if cap_used + d > self.capacity:
                if current:
                    routes.append(current)
                current = [int(c)]
                cap_used = d
            else:
                current.append(int(c))
                cap_used += d
        if current:
            routes.append(current)
        return routes

    def validate_solution(self, perm):
        """Return True if perm is a valid permutation of [1..n] and every
        route respects capacity. Used only for final reporting."""
        perm = list(perm)
        if sorted(perm) != list(range(1, self.n + 1)):
            return False
        for route in self.decode_routes(perm):
            if sum(self.demands[c] for c in route) > self.capacity + 1e-9:
                return False
        return True