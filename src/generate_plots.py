import matplotlib.pyplot as plt
import json
import seaborn as sns


def plot_demonstration(results_df):
    sns.set_style("whitegrid")
    functions = results_df['function'].unique()
    modes = results_df['mode'].unique()
    dims = results_df['dim'].unique()
    fs = results_df['F'].unique()
    
    for func in functions:
        for dim in dims:
            for F in fs:
                fig, axes = plt.subplots(len(modes), 3, figsize=(18, 4 * len(modes)))

                for i, mode in enumerate(modes):
                    subset = results_df[(results_df['function'] == func) & (results_df['mode'] == mode) & (results_df['dim'] == dim) & (results_df['F'] == F)]

                    if subset.empty:
                        continue

                    avg_fitnesses = json.loads(subset.iloc[0]['avg_fitnesses'])
                    f_trace_value = subset.iloc[0]['f_trace']
                    if isinstance(f_trace_value, str):  
                        f_trace = json.loads(f_trace_value)  
                    else:  
                        f_trace = [f_trace_value]

                    result_vars_value = subset.iloc[0]['result_vars']
                    if isinstance(result_vars_value, str):  
                        result_vars = json.loads(result_vars_value)  
                    else:  
                        result_vars = [result_vars_value]

                    dim = subset.iloc[0]['dim']
                    filename = f"../plots/demonstration/{func}_{dim}_{F}.png"

                    ax1 = axes[i, 0]
                    ax1.plot(range(len(avg_fitnesses)), avg_fitnesses, label=f"{mode} - avg fitness")
                    ax1.set_yscale('log')
                    ax1.set_title(f"{func} - {mode} (avg fitness)")
                    ax1.set_xlabel("Iteration")
                    ax1.set_ylabel("Avg Fitness (log scale)")
                    ax1.legend()

                    min_fitness = min(avg_fitnesses)
                    min_fitness_index = avg_fitnesses.index(min_fitness)

                    ax1.text(min_fitness_index, min_fitness, f"Min: {min_fitness:.2e}", color='blue', fontsize=10, ha='center', va='bottom')

                    ax2 = axes[i, 1]
                    ax2.plot(range(len(f_trace)), f_trace, label=f"{mode} - f_trace", color='r')
                    ax2.set_title(f"{func} - {mode} (f_trace)")
                    ax2.set_xlabel("Iteration")
                    ax2.set_ylabel("F")
                    ax2.legend()

                    ax3 = axes[i, 2]
                    ax3.plot(range(len(result_vars)), result_vars, label=f"{mode} - variance", color='g')
                    ax3.set_title(f"{func} - {mode} (variance)")
                    ax3.set_xlabel("Iteration")
                    ax3.set_ylabel("Variance")
                    ax3.legend()

                plt.tight_layout()
                plt.savefig(filename)


def plot_amplifier_reductor(results_df):
    sns.set_style("whitegrid")
    functions = results_df['function'].unique()
    modes = results_df['mode'].unique()
    dims = results_df['dim'].unique()
    steps = results_df['step'].unique()

    for func in functions:
        for dim in dims:
            for mode in modes:
                fig, axes = plt.subplots(len(steps), 3, figsize=(18, 4 * len(steps)))
                for i, step in enumerate(steps):
                    subset = results_df[(results_df['function'] == func) & (results_df['mode'] == mode) & (results_df['dim'] == dim) & (results_df['step'] == step)]

                    if subset.empty:
                        continue

                    avg_fitnesses = json.loads(subset.iloc[0]['avg_fitnesses'])
                    f_trace_value = subset.iloc[0]['f_trace']
                    if isinstance(f_trace_value, str):  
                        f_trace = json.loads(f_trace_value)  
                    else:  
                        f_trace = [f_trace_value]

                    result_vars_value = subset.iloc[0]['result_vars']
                    if isinstance(result_vars_value, str):  
                        result_vars = json.loads(result_vars_value)  
                    else:  
                        result_vars = [result_vars_value]

                    dim = subset.iloc[0]['dim']
                    filename = f"../plots/amp_red/{func}_{dim}_{mode}.png"

                    ax1 = axes[i, 0]
                    ax1.plot(range(len(avg_fitnesses)), avg_fitnesses, label=f"{mode} - avg fitness")
                    ax1.set_yscale('log')
                    ax1.set_title(f"{func} - {mode} (avg fitness) - step {step}")
                    ax1.set_xlabel("Iteration")
                    ax1.set_ylabel("Avg Fitness (log scale)")
                    ax1.legend()

                    min_fitness = min(avg_fitnesses)
                    min_fitness_index = avg_fitnesses.index(min_fitness)

                    ax1.text(min_fitness_index, min_fitness, f"Min: {min_fitness:.2e}", color='blue', fontsize=10, ha='center', va='bottom')

                    ax2 = axes[i, 1]
                    ax2.plot(range(len(f_trace)), f_trace, label=f"{mode} - f_trace", color='r')
                    ax2.set_title(f"{func} - {mode} (f_trace)")
                    ax2.set_xlabel("Iteration")
                    ax2.set_ylabel("F")
                    ax2.legend()

                    ax3 = axes[i, 2]
                    ax3.plot(range(len(result_vars)), result_vars, label=f"{mode} - variance", color='g')
                    ax3.set_title(f"{func} - {mode} (variance)")
                    ax3.set_xlabel("Iteration")
                    ax3.set_ylabel("Variance")
                    ax3.legend()

                plt.tight_layout()
                plt.savefig(filename)


def plot_target_ratio(results_df):
    sns.set_style("whitegrid")
    functions = results_df['function'].unique()
    modes = results_df['mode'].unique()
    dims = results_df['dim'].unique()
    target_ratios = results_df['target_ratio'].unique()

    for func in functions:
        for dim in dims:
            for mode in modes:
                fig, axes = plt.subplots(len(target_ratios), 3, figsize=(18, 4 * len(target_ratios)))
                for i, target_ratio in enumerate(target_ratios):
                    subset = results_df[(results_df['function'] == func) & (results_df['mode'] == mode) & (results_df['dim'] == dim) & (results_df['target_ratio'] == target_ratio)]

                    if subset.empty:
                        continue

                    avg_fitnesses = json.loads(subset.iloc[0]['avg_fitnesses'])
                    f_trace_value = subset.iloc[0]['f_trace']
                    if isinstance(f_trace_value, str):  
                        f_trace = json.loads(f_trace_value)  
                    else:  
                        f_trace = [f_trace_value]

                    result_vars_value = subset.iloc[0]['result_vars']
                    if isinstance(result_vars_value, str):  
                        result_vars = json.loads(result_vars_value)  
                    else:  
                        result_vars = [result_vars_value]

                    dim = subset.iloc[0]['dim']
                    filename = f"../plots/target_ratio/{func}_{dim}_{mode}.png"

                    ax1 = axes[i, 0]
                    ax1.plot(range(len(avg_fitnesses)), avg_fitnesses, label=f"{mode} - avg fitness")
                    ax1.set_yscale('log')
                    ax1.set_title(f"{func} - {mode} (avg fitness) - target_ratio {target_ratio}")
                    ax1.set_xlabel("Iteration")
                    ax1.set_ylabel("Avg Fitness (log scale)")
                    ax1.legend()

                    min_fitness = min(avg_fitnesses)
                    min_fitness_index = avg_fitnesses.index(min_fitness)

                    ax1.text(min_fitness_index, min_fitness, f"Min: {min_fitness:.2e}", color='blue', fontsize=10, ha='center', va='bottom')

                    ax2 = axes[i, 1]
                    ax2.plot(range(len(f_trace)), f_trace, label=f"{mode} - f_trace", color='r')
                    ax2.set_title(f"{func} - {mode} (f_trace)")
                    ax2.set_xlabel("Iteration")
                    ax2.set_ylabel("F")
                    ax2.legend()

                    ax3 = axes[i, 2]
                    ax3.plot(range(len(result_vars)), result_vars, label=f"{mode} - variance", color='g')
                    ax3.set_title(f"{func} - {mode} (variance)")
                    ax3.set_xlabel("Iteration")
                    ax3.set_ylabel("Variance")
                    ax3.legend()

                plt.tight_layout()
                plt.savefig(filename)


def plot_threshold(results_df):
    sns.set_style("whitegrid")
    functions = results_df['function'].unique()
    modes = results_df['mode'].unique()
    dims = results_df['dim'].unique()
    thresholds = results_df['threshold'].unique()

    for func in functions:
        for dim in dims:
            for mode in modes:
                fig, axes = plt.subplots(len(thresholds), 3, figsize=(18, 4 * len(thresholds)))
                for i, threshold in enumerate(thresholds):
                    subset = results_df[(results_df['function'] == func) & (results_df['mode'] == mode) & (results_df['dim'] == dim) & (results_df['threshold'] == threshold)]

                    if subset.empty:
                        continue

                    avg_fitnesses = json.loads(subset.iloc[0]['avg_fitnesses'])
                    f_trace_value = subset.iloc[0]['f_trace']
                    if isinstance(f_trace_value, str):  
                        f_trace = json.loads(f_trace_value)  
                    else:  
                        f_trace = [f_trace_value]

                    result_vars_value = subset.iloc[0]['result_vars']
                    if isinstance(result_vars_value, str):  
                        result_vars = json.loads(result_vars_value)  
                    else:  
                        result_vars = [result_vars_value]

                    dim = subset.iloc[0]['dim']
                    filename = f"../plots/threshold/{func}_{dim}_{mode}.png"

                    ax1 = axes[i, 0]
                    ax1.plot(range(len(avg_fitnesses)), avg_fitnesses, label=f"{mode} - avg fitness")
                    ax1.set_yscale('log')
                    ax1.set_title(f"{func} - {mode} (avg fitness) - threshold {threshold}")
                    ax1.set_xlabel("Iteration")
                    ax1.set_ylabel("Avg Fitness (log scale)")
                    ax1.legend()

                    min_fitness = min(avg_fitnesses)
                    min_fitness_index = avg_fitnesses.index(min_fitness)

                    ax1.text(min_fitness_index, min_fitness, f"Min: {min_fitness:.2e}", color='blue', fontsize=10, ha='center', va='bottom')

                    ax2 = axes[i, 1]
                    ax2.plot(range(len(f_trace)), f_trace, label=f"{mode} - f_trace", color='r')
                    ax2.set_title(f"{func} - {mode} (f_trace)")
                    ax2.set_xlabel("Iteration")
                    ax2.set_ylabel("F")
                    ax2.legend()

                    ax3 = axes[i, 2]
                    ax3.plot(range(len(result_vars)), result_vars, label=f"{mode} - variance", color='g')
                    ax3.set_title(f"{func} - {mode} (variance)")
                    ax3.set_xlabel("Iteration")
                    ax3.set_ylabel("Variance")
                    ax3.legend()

                plt.tight_layout()
                plt.savefig(filename)
