from utils import get_average, get_variance

import numpy as np


class DifferentialEvolution:
    def __init__(self, func, dim, bounds, pop_size=20, F=0.5, CR=0.9, max_iter=100, mode="normal"):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        self.population = self.initialize_population()
        self.fitness = self.func(self.population)
        self.avg_fitnesses = [get_average(self.fitness)]
        self.result_vars = [get_variance(self.population)]
        best_idx = np.argmin(self.fitness)
        self.trace = [self.population[best_idx]]
        self.mode = mode

    def initialize_population(self) -> np.ndarray:
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def run(self, amplifier=1.1, reductor=0.9):
        for _ in range(self.max_iter):
            idxs = np.array([np.random.choice(self.pop_size, 3, replace=False) for _ in range(self.pop_size)])

            # Mutation and Crossover
            a, b, c = self.population[idxs[:, 0]], self.population[idxs[:, 1]], self.population[idxs[:, 2]]
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

            mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trial_population = np.where(mask, mutant, self.population)

            # Evaluate new population
            trial_fitness = self.func(trial_population)
            improved = trial_fitness < self.fitness
            self.population = np.where(improved[:, None], trial_population, self.population)
            self.fitness = np.where(improved, trial_fitness, self.fitness)
            success_count = np.sum(improved)

            if self.mode == "PSR":
                self.F = self.update_F_psr(success_count, amplifier=amplifier, reductor=reductor)
            elif self.mode == "MSR":
                self.F = self.update_F_msr(amplifier, reductor)

            best_idx = np.argmin(self.fitness)
            self.trace.append(self.population[best_idx])
            self.avg_fitnesses.append(get_average(self.fitness))
            self.result_vars.append(get_variance(self.population))

    def update_F_msr(self, amplifier=1.1, reductor=0.9) -> float:
        diffs = self.fitness - self.func(self.population)
        median = np.median(diffs)
        if median < 0:
            self.F *= amplifier
        else:
            self.F *= reductor
        return np.clip(self.F, 0.1, 1.0)

    def update_F_psr(self, success_count, target_ratio=0.5, amplifier=1.1, reductor=0.9) -> float:
        success_ratio = success_count / self.pop_size
        if success_ratio < target_ratio:
            self.F *= reductor
        else:
            self.F *= amplifier
        return np.clip(self.F, 0.1, 1.0)
