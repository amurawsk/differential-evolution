import numpy as np
import matplotlib.pyplot as plt
import cec2017


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
        self.fitness = np.array([self.func(*ind) for ind in self.population])
        self.trace = []
        self.mode = mode

    def initialize_population(self) -> np.ndarray:
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def run(self, amplifier=1.1, reductor=0.9):
        for iteration in range(self.max_iter):
            new_population = np.copy(self.population)
            success_count = 0

            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(self.pop_size, 3, replace=False)

                a, b, c = self.population[idxs]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.copy(self.population[i])
                for d in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[d] = mutant[d]

                trial_fitness = self.func(*trial)
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness
                    success_count += 1

            self.population = new_population

            if self.mode == "PSR":
                self.F = self.update_F_psr(success_count, amplifier, reductor)
            elif self.mode == "MSR":
                self.F = self.update_F_msr(amplifier, reductor)

            best_idx = np.argmin(self.fitness)
            self.trace.append(self.population[best_idx])

        return self.population[best_idx], self.fitness[best_idx]

    def update_F_msr(self, amplifier=1.1, reductor=0.9) -> float:
        diffs = self.fitness - np.array([self.func(*ind) for ind in self.population])
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

    def plot_trace(self, title="Optimization Trace"):
        x = np.arange(self.bounds[0], self.bounds[1], 0.05)
        y = np.arange(self.bounds[0], self.bounds[1], 0.05)
        x, y = np.meshgrid(x, y)
        z = self.func(x, y)

        plt.figure()
        plt.contour(x, y, z, 50)
        trace_x = [ind[0] for ind in self.trace]
        trace_y = [ind[1] for ind in self.trace]
        plt.scatter(trace_x, trace_y, s=10)
        plt.scatter(trace_x[-10:], trace_y[-10:], s=10, color='red')
        plt.title(title)
        plt.show()





if __name__ == "__main__":
    f2 = lambda x, y: 1.5 - np.exp(-x ** 2 - y ** 2) - 0.5 * np.exp(-(x - 1) ** 2 - (y + 2) ** 2)
    dim = 2
    bounds = (-3, 3)
    DE = DifferentialEvolution(f2, dim, bounds, pop_size=30, max_iter=30)
    DE_normal = DifferentialEvolution(f2, dim, bounds, pop_size=30, max_iter=30)
    DE_psr = DifferentialEvolution(f2, dim, bounds, pop_size=30, max_iter=30, mode="PSR")
    DE_msr = DifferentialEvolution(f2, dim, bounds, pop_size=30, max_iter=30, mode="MSR")

    best_solution, best_fitness = DE.run(
    )
    DE.plot_trace(title="DE")

