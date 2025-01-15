import matplotlib.pyplot as plt
import json
import os


def plot_avg_fitness_per_iteration_logscale(results_df, output_dir="plots/fit"):
    os.makedirs(output_dir, exist_ok=True)
    for (F, CR, dim) in results_df.groupby(['F', 'CR', 'dim']).groups:
        plt.figure(figsize=(12, 8))
        min_fitness_annotations = []
        for mode in ['normal', 'PSR', 'MSR']:
            subset = results_df[(results_df['F'] == F) & 
                                (results_df['CR'] == CR) & 
                                (results_df['dim'] == dim) & 
                                (results_df['mode'] == mode)]

            min_fitness = float('inf')
            for _, row in subset.iterrows():
                avg_fitnesses = json.loads(row['avg_fitnesses'])
                plt.plot(range(1, len(avg_fitnesses) + 1), avg_fitnesses, label=f"{mode}")
                min_fitness = min(min_fitness, min(avg_fitnesses))

            min_fitness_annotations.append((mode, min_fitness))

        annotation_text = "\n".join([f"{mode}: min fitness = {min_fitness:.2e}" for mode, min_fitness in min_fitness_annotations])
        plt.text(0.05, 0.95, annotation_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='lightyellow'))

        plt.title(f"Avg Fitness per Iteration (F={F}, CR={CR}, dim={dim}, Log Scale)")
        plt.xlabel("Iteration")
        plt.ylabel("Average Fitness (Log Scale)")
        plt.yscale('log')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)

        filename = f"F_{F}_CR_{CR}_dim_{dim}_logscale.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def plot_result_vars_per_iteration_logscale(results_df, output_dir="plots/var"):
    os.makedirs(output_dir, exist_ok=True)
    for (F, CR, dim) in results_df.groupby(['F', 'CR', 'dim']).groups:
        plt.figure(figsize=(12, 8))
        
        min_variance_annotations = []
        for mode in ['normal', 'PSR', 'MSR']:
            subset = results_df[(results_df['F'] == F) & 
                                (results_df['CR'] == CR) & 
                                (results_df['dim'] == dim) & 
                                (results_df['mode'] == mode)]

            min_variance = float('inf')
            for _, row in subset.iterrows():
                result_vars = json.loads(row['result_vars'])
                plt.plot(range(1, len(result_vars) + 1), result_vars, label=f"{mode}")
                min_variance = min(min_variance, min(result_vars))
            
            min_variance_annotations.append((mode, min_variance))

        annotation_text = "\n".join([f"{mode}: min variance = {min_variance:.2e}" for mode, min_variance in min_variance_annotations])
        plt.text(0.05, 0.95, annotation_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='lightyellow'))

        plt.title(f"Result Vars per Iteration (F={F}, CR={CR}, dim={dim}, Log Scale)")
        plt.xlabel("Iteration")
        plt.ylabel("Result Vars (Log Scale)")
        plt.yscale('log')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)

        filename = f"F_{F}_CR_{CR}_dim_{dim}_logscale_result_vars.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
