import random


def tournament_selection(individuals, tournament_size=3):
    if not individuals:
        raise ValueError("individuals must not be empty")

    tournament_size = min(tournament_size, len(individuals))
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")

    competitors = random.sample(individuals, tournament_size)
    winner = min(competitors, key=lambda ind: ind.fitness)
    return winner.copy()