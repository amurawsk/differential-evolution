from differential_evolution import DifferentialEvolution
import json


def get_single_run_for_all_modes(results_df, population, function, dim, bounds, F, CR, index, pop_size, max_iter):
    modes = ["normal", "PSR", "MSR"]
    for mode in modes:
        de_instance = DifferentialEvolution(function, dim, bounds, population=population.copy(), pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, mode=mode)
        de_instance.run()

        results_df.loc[len(results_df)] = {
            'function': f"function {index + 1}",
            'dim': dim,
            'F': F,
            'CR': CR,
            'mode': mode,
            'avg_fitnesses': json.dumps(list(de_instance.avg_fitnesses)),
            'result_vars': json.dumps(list(de_instance.result_vars)),
            'f_trace': json.dumps(list(de_instance.trace_f))
        }


def get_different_amplifier_reductor_data(results_df, population, function, dim, bounds, F, CR, index, pop_size, max_iter, step):
    modes = ["PSR", "MSR"]
    for mode in modes:
        de_instance = DifferentialEvolution(function, dim, bounds, population=population.copy(), pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, mode=mode, amplifier=1+step, reductor=1-step)
        de_instance.run()

        results_df.loc[len(results_df)] = {
            'function': f"function {index + 1}",
            'dim': dim,
            'F': F,
            'CR': CR,
            'mode': mode,
            'step': step,
            'avg_fitnesses': json.dumps(list(de_instance.avg_fitnesses)),
            'result_vars': json.dumps(list(de_instance.result_vars)),
            'f_trace': json.dumps(list(de_instance.trace_f))
        }


def get_different_target_ratio_data(results_df, population, function, dim, bounds, F, CR, index, pop_size, max_iter, target_ratio):
    de_instance = DifferentialEvolution(function, dim, bounds, population=population.copy(), pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, mode="PSR", target_ratio=target_ratio)
    de_instance.run()

    results_df.loc[len(results_df)] = {
        'function': f"function {index + 1}",
        'dim': dim,
        'F': F,
        'CR': CR,
        'mode': "PSR",
        'target_ratio': target_ratio,
        'avg_fitnesses': json.dumps(list(de_instance.avg_fitnesses)),
        'result_vars': json.dumps(list(de_instance.result_vars)),
        'f_trace': json.dumps(list(de_instance.trace_f))
    }


def get_different_threshold_data(results_df, population, function, dim, bounds, F, CR, index, pop_size, max_iter, threshold):
    de_instance = DifferentialEvolution(function, dim, bounds, population=population.copy(), pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, mode="MSR", threshold_percentage=threshold)
    de_instance.run()

    results_df.loc[len(results_df)] = {
        'function': f"function {index + 1}",
        'dim': dim,
        'F': F,
        'CR': CR,
        'mode': "MSR",
        'threshold': threshold,
        'avg_fitnesses': json.dumps(list(de_instance.avg_fitnesses)),
        'result_vars': json.dumps(list(de_instance.result_vars)),
        'f_trace': json.dumps(list(de_instance.trace_f))
    }
