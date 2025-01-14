import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("output.csv")
    sns.set_theme()
    print(df.head())
    agg_results = (
        df.groupby(['function', 'dim', 'mode', 'F', 'CR'])['best_fitness']
        .min()
        .reset_index()
    )

    for dim in agg_results['dim'].unique():
        for mode in agg_results['mode'].unique():
            for function in agg_results['function'].unique():
                filtered_results = agg_results[
                    (agg_results['function'] == 'function 1') &
                    (agg_results['dim'] == dim) &
                    (agg_results['mode'] == mode)
                    ]

                filtered_results = filtered_results[filtered_results['CR'] > 0.6]
                filtered_results = filtered_results[filtered_results['F'] > 0.6]

            # Pivot the DataFrame to create a matrix for the heatmap
                heatmap_data = filtered_results.pivot(index='F', columns='CR', values='best_fitness')

                # Plot the heatmap
                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data)
                plt.title('Heatmap of Best Fitness by F and CR')
                plt.xlabel('CR')
                plt.ylabel('F')
                plt.show()