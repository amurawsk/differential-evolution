from differential_evolution import DifferentialEvolution

import json


def get_data(results_df, function, dim, bounds, F, CR, index, i, pop_size, max_iter):
    modes = ["normal", "PSR", "MSR"]
    for mode in modes:
        de_instance = DifferentialEvolution(function, dim, bounds, pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, mode=mode)
        de_instance.run()

        results_df.loc[len(results_df)] = {
            'function': f"function {index + 1}",
            'dim': dim,
            'F': F,
            'CR': CR,
            'mode': mode,
            'avg_fitnesses': json.dumps(de_instance.avg_fitnesses),
            'result_vars': json.dumps(de_instance.result_vars),
            'iteration': i
        }
