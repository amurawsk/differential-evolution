import pandas as pd
import numpy as np
from tqdm import tqdm
from cec2017.functions import all_functions

from run_de import *
from generate_plots import *


def demonstration():
    dimensions = [10, 30]
    bounds = (-100, 100)
    columns = ['function', 'dim', 'F', 'CR', 'mode', 'avg_fitnesses', 'result_vars', 'f_trace']
    results_df = pd.DataFrame(columns=columns)

    for index, function in tqdm(enumerate(all_functions[:1])):
        for dim in dimensions:
            pop_size = 10*dim
            max_iter = 40*dim
            population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))
            for F in [0.1, 0.5, 0.9]:
                get_single_run_for_all_modes(results_df=results_df, population=population, function=function, dim=dim, bounds=bounds, F=F, CR=0.9, index=index, pop_size=pop_size, max_iter=max_iter)
    plot_demonstration(results_df)
    results_df.to_json('../records/demonstration.json', orient='records', lines=True)


def different_amplifier_reductor_msr():
    dimensions = [10, 30]
    bounds = (-100, 100)
    columns = ['function', 'dim', 'F', 'CR', 'mode', 'step', 'avg_fitnesses', 'result_vars', 'f_trace']
    results_df = pd.DataFrame(columns=columns)
    
    for index, function in tqdm(enumerate(all_functions[:1])): # todo
        for dim in dimensions:
            pop_size = 10*dim
            max_iter = 40*dim
            population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))
            for step in [0.05, 0.1, 0.3]:
                get_different_amplifier_reductor_data(results_df=results_df, population=population, function=function, dim=dim, bounds=bounds, F=0.5, CR=0.9, index=index, pop_size=pop_size, max_iter=max_iter, step=step)
    plot_amplifier_reductor(results_df)
    results_df.to_json('../records/different_amplifier.json', orient='records', lines=True)


def different_target_ratio():
    dimensions = [10, 30]
    bounds = (-100, 100)
    columns = ['function', 'dim', 'F', 'CR', 'mode', 'target_ratio', 'avg_fitnesses', 'result_vars', 'f_trace']
    results_df = pd.DataFrame(columns=columns)

    for index, function in tqdm(enumerate(all_functions[:1])):
        for dim in dimensions:
            pop_size = 10*dim
            max_iter = 40*dim
            population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))
            for target_ratio in [0.1, 0.2, 0.35, 0.5]:
                get_different_target_ratio_data(results_df=results_df, population=population, function=function, dim=dim, bounds=bounds, F=0.5, CR=0.9, index=index, pop_size=pop_size, max_iter=max_iter, target_ratio=target_ratio)
    plot_target_ratio(results_df)
    results_df.to_json('../records/different_target_ratio.json', orient='records', lines=True)


def different_threshold_percentage():
    dimensions = [10, 30]
    bounds = (-100, 100)
    columns = ['function', 'dim', 'F', 'CR', 'mode', 'threshold', 'avg_fitnesses', 'result_vars', 'f_trace']
    results_df = pd.DataFrame(columns=columns)

    for index, function in tqdm(enumerate(all_functions[:1])):
        for dim in dimensions:
            pop_size = 10*dim
            max_iter = 40*dim
            population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))
            for threshold in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
                get_different_threshold_data(results_df=results_df, population=population, function=function, dim=dim, bounds=bounds, F=0.5, CR=0.9, index=index, pop_size=pop_size, max_iter=max_iter, threshold=threshold)
    plot_threshold(results_df)
    results_df.to_json('../records/different_threshold.json', orient='records', lines=True)
