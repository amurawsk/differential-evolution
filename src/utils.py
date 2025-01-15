import numpy as np


def get_average(values):
    return np.average(values)


def get_variance(population):
    population = np.array(population)
    mean_point = np.mean(population, axis=0)
    total_variance = np.mean(np.sum((population - mean_point) ** 2, axis=1))
    return total_variance
