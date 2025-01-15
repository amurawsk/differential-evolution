from differential_evolution import DifferentialEvolution
import numpy as np

import json


def get_data(results_df, function, dim, bounds, F, CR, index, i, pop_size, max_iter):
    modes = ["normal", "PSR", "MSR"]
    for mode in modes:
        avg_fitnesses = []
        result_vars = []
        for _ in range(5):
            de_instance = DifferentialEvolution(function, dim, bounds, pop_size=pop_size, max_iter=max_iter, F=F, CR=CR, mode=mode)
            de_instance.run()
            avg_fitnesses.append(de_instance.avg_fitnesses)
            result_vars.append(de_instance.result_vars)
            
        avg_fitnesses = np.average(avg_fitnesses, axis=0)
        result_vars = np.average(result_vars, axis=0)
        
        results_df.loc[len(results_df)] = {
            'function': f"function {index + 1}",
            'dim': dim,
            'F': F,
            'CR': CR,
            'mode': mode,
            'avg_fitnesses': json.dumps(list(avg_fitnesses)),
            'result_vars': json.dumps(list(result_vars)),
        }
