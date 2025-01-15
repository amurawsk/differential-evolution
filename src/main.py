import pandas as pd
from tqdm import tqdm
from cec2017.functions import all_functions
from datetime import datetime
from data import get_data


from generate_plots import *


def main():
    dimensions = [10]
    bounds = (-100, 100)
    columns = ['function', 'dim', 'F', 'CR', 'mode', 'avg_fitnesses', 'result_vars']
    results_df = pd.DataFrame(columns=columns)

    for index, function in tqdm(enumerate(all_functions[:1])):
        for dim in dimensions:
            for F in tqdm(range(1, 10)):
                for CR in range(1, 10):
                    get_data(results_df=results_df, function=function, dim=dim, bounds=bounds, F=F/10, CR=CR/10, index=index, pop_size=100, max_iter=500)
        # print(f"Function {index + 1}, dim:{dim} done")

    # agg_results = results_df.groupby(['function', 'dim', 'mode', 'F', 'CR']).agg(['mean', 'median']).reset_index()
    results_df.to_csv(f'output-{datetime.now()}.csv', index=False)
    plot_avg_fitness_per_iteration_logscale(results_df)
    plot_result_vars_per_iteration_logscale(results_df)

if __name__ == "__main__":
    main()
