from utils import get_average, get_variance

import numpy as np


class DifferentialEvolution:
    def __init__(self, func, dim, bounds, population = None, pop_size=20, F=0.5, CR=0.9, max_iter=100, mode="normal", amplifier=1.05, reductor=0.95, target_ratio=0.2, threshold_percentage = None):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        self.population = self.initialize_population() if population is None else population
        self.fitness = self.func(self.population)
        self.avg_fitnesses = [get_average(self.fitness)]
        self.result_vars = [get_variance(self.population)]
        best_idx = np.argmin(self.fitness)
        self.trace = [self.population[best_idx]]
        self.trace_f = [self.F]
        self.mode = mode
        self.amplifier = amplifier
        self.reductor = reductor
        self.target_ratio = target_ratio
        if threshold_percentage is None:
            self.threshold_percentage = 0.08 if self.dim == 10 else 0.02
        else:
            self.threshold_percentage = threshold_percentage

    def initialize_population(self) -> np.ndarray:
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def run(self):
        for _ in range(self.max_iter):
            idxs = np.array([np.random.choice(self.pop_size, 3, replace=False) for _ in range(self.pop_size)])

            # Mutation and Crossover
            a, b, c = self.population[idxs[:, 0]], self.population[idxs[:, 1]], self.population[idxs[:, 2]]
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

            mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trial_population = np.where(mask, mutant, self.population)

            old_median = np.median(self.fitness)
            # Evaluate new population
            trial_fitness = self.func(trial_population)
            improved = trial_fitness < self.fitness
            self.population = np.where(improved[:, None], trial_population, self.population)
            self.fitness = np.where(improved, trial_fitness, self.fitness)
            success_count = np.sum(improved)

            if self.mode == "PSR":
                self.F = self.update_F_psr(success_count)
            elif self.mode == "MSR":
                self.F = self.update_F_msr(old_median)

            best_idx = np.argmin(self.fitness)
            self.trace.append(self.population[best_idx])
            self.trace_f.append(self.F)
            self.avg_fitnesses.append(get_average(self.fitness))
            self.result_vars.append(get_variance(self.population))

    def update_F_msr(self, old_median) -> float:
        median = np.median(self.fitness)
        
        delta = median - old_median
        threshold = np.abs(self.threshold_percentage * old_median)

        if delta < -threshold:
            self.F *= self.amplifier
        else:
            self.F *= self.reductor
        
        return np.clip(self.F, 0.1, 1.0)


    def update_F_psr(self, success_count) -> float:
        success_ratio = success_count / self.pop_size
        if success_ratio < self.target_ratio:
            self.F *= self.reductor
        else:
            self.F *= self.amplifier
        return np.clip(self.F, 0.1, 1.0)
