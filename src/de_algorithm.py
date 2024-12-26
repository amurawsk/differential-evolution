import numpy as np
import matplotlib.pyplot as plt

def initialize_population(pop_size: int, dim: int, bounds: tuple[float, float]) -> np.ndarray:
    return np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))

def differential_evolution(
        func: callable,
        dim: int,
        bounds: tuple[float, float],
        pop_size: int = 20,
        F: float = 0.5,
        CR: float = 0.9,
        max_iter: int = 100
) -> tuple[np.ndarray, float]:
    population = initialize_population(pop_size, dim, bounds)
    fitness = np.array([func(*ind) for ind in population])
    trace = []

    for iteration in range(max_iter):
        new_population = np.copy(population)
        for i in range(pop_size):

            idxs = np.random.choice(pop_size, 3, replace=False)
            while i in idxs:
                idxs = np.random.choice(pop_size, 3, replace=False)

            a, b, c = population[idxs]

            mutant = a + F * (b - c)
            mutant = np.clip(mutant, bounds[0], bounds[1])

            crossover = np.copy(population[i])
            for d in range(dim):
                if np.random.rand() < CR:
                    crossover[d] = mutant[d]

            if func(*crossover) < fitness[i]:
                new_population[i] = crossover
                fitness[i] = func(*crossover)

        population = new_population
        best_idx = np.argmin(fitness)
        trace.append(population[best_idx])

    plot_de(trace, func)
    return population[best_idx], fitness[best_idx]


def plot_de(trace, f2):

    x = np.arange(-2, 2, 0.05)
    y = np.arange(-3, 2, 0.05)
    x, y = np.meshgrid(x, y)
    z = f2(x, y)
    plt.figure()
    plt.contour(x, y, z, 50)
    trace_x = [ind[0] for ind in trace]
    trace_y = [ind[1] for ind in trace]
    plt.scatter(trace_x, trace_y, s=10)
    plt.scatter(trace_x[-10:], trace_y[-10:], s=10, color='red')
    plt.title(f'siema')
    plt.show()

if __name__ == "__main__":
    f2 = lambda x, y: 1.5 - np.exp(-x ** 2 - y ** 2) - 0.5 * np.exp(-(x - 1) ** 2 - (y + 2) ** 2)
    dim = 2
    bounds = (-30, 30)
    best_solution, best_fitness = differential_evolution(
        f2, dim, bounds, pop_size=30, max_iter=100
    )

