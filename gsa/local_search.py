from typing import List, Tuple


def _infer_depot(problem):
    if hasattr(problem, "depot"):
        return problem.depot
    return 1


def _strip_depot(route: List[int], depot: int) -> List[int]:
    if not route:
        return []

    stripped = list(route)
    if stripped and stripped[0] == depot:
        stripped = stripped[1:]
    if stripped and stripped[-1] == depot:
        stripped = stripped[:-1]
    return stripped


def _route_distance(problem, route: List[int]) -> float:
    depot = _infer_depot(problem)

    if not route:
        return 0.0

    full = [depot] + list(route) + [depot]
    dist = 0.0
    for a, b in zip(full[:-1], full[1:]):
        dist += problem.distance_matrix[a][b]
    return dist


def _route_load(problem, route: List[int]) -> float:
    if not hasattr(problem, "demands"):
        return 0.0
    return sum(problem.demands[c] for c in route if c in problem.demands)


def _encode_routes(routes: List[List[int]], original_customers: List[int]) -> List[int]:
    chromosome = []
    seen = set()
    valid_customers = set(original_customers)

    for route in routes:
        for c in route:
            if c in valid_customers and c not in seen:
                chromosome.append(c)
                seen.add(c)

    for c in original_customers:
        if c not in seen:
            chromosome.append(c)

    return chromosome


def two_opt_route(route: List[int], problem) -> Tuple[List[int], bool]:
    n = len(route)
    if n < 4:
        return route[:], False

    best = route[:]
    best_cost = _route_distance(problem, best)
    improved = False

    changed = True
    while changed:
        changed = False
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                candidate = best[:]
                candidate[i:j + 1] = reversed(candidate[i:j + 1])
                cost = _route_distance(problem, candidate)

                if cost < best_cost:
                    best = candidate
                    best_cost = cost
                    improved = True
                    changed = True
                    break
            if changed:
                break

    return best, improved


def two_opt_solution(chromosome: List[int], problem) -> Tuple[List[int], bool]:
    depot = _infer_depot(problem)
    original_customers = list(chromosome)
    decoded_routes = problem.decode(chromosome)
    routes = [_strip_depot(route, depot) for route in decoded_routes]

    improved = False
    new_routes = []

    for route in routes:
        new_route, changed = two_opt_route(route, problem)
        new_routes.append(new_route)
        improved = improved or changed

    return _encode_routes(new_routes, original_customers), improved


def relocate_between_routes(
    chromosome: List[int],
    problem,
    max_rounds: int = 3,
) -> Tuple[List[int], bool]:
    depot = _infer_depot(problem)
    routes = [_strip_depot(r, depot) for r in problem.decode(chromosome)]
    original_customers = list(chromosome)

    if len(routes) < 2:
        return chromosome[:], False

    capacity = getattr(problem, "capacity", None)
    if capacity is None or not hasattr(problem, "demands"):
        return chromosome[:], False

    improved = False

    for _ in range(max_rounds):
        route_loads = [_route_load(problem, r) for r in routes]
        current_total = sum(_route_distance(problem, r) for r in routes)

        best_delta = 0.0
        best_move = None

        for i in range(len(routes)):
            if len(routes[i]) <= 1:
                continue

            for pos in range(len(routes[i])):
                customer = routes[i][pos]
                if customer not in problem.demands:
                    continue

                demand = problem.demands[customer]

                for j in range(len(routes)):
                    if i == j:
                        continue

                    if route_loads[j] + demand > capacity:
                        continue

                    source_route = routes[i]
                    target_route = routes[j]

                    source_removed = source_route[:pos] + source_route[pos + 1:]

                    for insert_pos in range(len(target_route) + 1):
                        target_inserted = target_route[:insert_pos] + [customer] + target_route[insert_pos:]

                        old_cost = _route_distance(problem, source_route) + _route_distance(problem, target_route)
                        new_cost = _route_distance(problem, source_removed) + _route_distance(problem, target_inserted)

                        delta = old_cost - new_cost
                        if delta > best_delta + 1e-9:
                            best_delta = delta
                            best_move = (i, pos, j, insert_pos)

        if best_move is None:
            break

        i, pos, j, insert_pos = best_move
        customer = routes[i].pop(pos)
        routes[j].insert(insert_pos, customer)

        new_total = sum(_route_distance(problem, r) for r in routes)
        if new_total < current_total - 1e-9:
            improved = True
        else:
            break

    routes = [route for route in routes if route]
    return _encode_routes(routes, original_customers), improved


def local_improve_chromosome(
    chromosome: List[int],
    problem,
    use_two_opt: bool = True,
    use_relocate: bool = True,
    rounds: int = 1,
) -> Tuple[List[int], bool]:
    best = chromosome[:]
    improved = False

    for _ in range(max(1, rounds)):
        changed = False

        if use_two_opt:
            cand, ok = two_opt_solution(best, problem)
            if ok:
                best = cand
                improved = True
                changed = True

        if use_relocate:
            cand, ok = relocate_between_routes(best, problem)
            if ok:
                best = cand
                improved = True
                changed = True

        if not changed:
            break

    return best, improved