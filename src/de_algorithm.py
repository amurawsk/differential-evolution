import functools

import numpy as np
import matplotlib.pyplot as plt
from cec2017.functions import all_functions
import time
from functools import lru_cache
import torch
import pandas as pd
import pprint
from scipy.optimize import differential_evolution
import seaborn as sns

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
        self.trace = []
        self.mode = mode



    def initialize_population(self) -> np.ndarray:
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))


    def run(self, amplifier=1.1, reductor=0.9, F=0.5, CR=0.9):
        self.population = self.initialize_population()
        self.F = F
        self.CR = CR
        for iteration in range(self.max_iter):

            idxs = np.array([np.random.choice(self.pop_size, 3, replace=False) for _ in range(self.pop_size)])

            # Mutation and Crossover
            a, b, c = self.population[idxs[:, 0]], self.population[idxs[:, 1]], self.population[
                idxs[:, 2]]
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

        return self.population[best_idx], self.fitness[best_idx]

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

def DE_loop(results_df, function, dim, bounds, F, CR, index, i, pop_size, max_iter):
    modes = ["normal", "PSR", "MSR"]
    for mode in modes:
        # Initialize Differential Evolution instance
        de_instance = DifferentialEvolution(
            function, dim, bounds, pop_size=pop_size, max_iter=max_iter,
            mode=mode if mode != "normal" else None
        )

        # Run the DE instance and retrieve results
        best_solution, best_fitness = de_instance.run(F=F, CR=CR)

        # Create a new row for the results
        results_df.loc[len(results_df)] = {
            'function': f"function {index + 1}",
            'dim': dim,
            'F': F,
            'CR': CR,
            'mode': mode,
            'best_fitness': best_fitness,
            'iteration': i
        }

if __name__ == "__main__":
    dimensions = [10, 30]
    bounds = (-100, 100)
    start = time.time()
    columns = ['function', 'dim', 'F', 'CR', 'mode', 'best_fitness', 'iteration']
    results_df = pd.DataFrame(columns=columns)


    for index, function in enumerate(all_functions):
        print(index)
        for dim in dimensions:
            for F in range(0, 11):
                for CR in range(0, 11):
                    DE_loop(results_df, function, dim, bounds, F/10, CR / 10, index, 1, 100, 100)
            print(f"Function {index + 1}, dim:{dim} done")




    agg_results = results_df.groupby(['function', 'dim', 'mode', 'F', 'CR']).agg(['mean', 'median']).reset_index()
    results_df.to_csv('output.csv', index=False)


