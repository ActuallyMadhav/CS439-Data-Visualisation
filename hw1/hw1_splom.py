import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def task3(filename: str, attrs: list):
    df = pd.read_excel(filename)
    n = len(attrs)

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n), squeeze=False)
    
    regions = sorted(df['Region'].dropna().unique())
    colors = {'America': 'gold', 'Asia': 'mediumvioletred', 'Europe': 'darkblue'}

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            row_attr = attrs[i]
            col_attr = attrs[j]
            
            for region in regions:
                subset = df[df['Region'] == region]
                ax.scatter(
                    subset[col_attr], 
                    subset[row_attr], 
                    c=colors.get(region, 'gray'), 
                    label=region, 
                    alpha=0.7, 
                    edgecolors='black', 
                    linewidth=0.5
                )
            
            # labels
            if j == 0:
                ax.set_ylabel(row_attr)
            
            if i == n - 1:
                ax.set_xlabel(col_attr)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Production Origin', loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    # Display
    fig.suptitle(f'Scatter Plot Matrix of {len(attrs)} EV Attributes', weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.85) 
    plt.show()

if __name__ == '__main__':
    #argparse
    parser = argparse.ArgumentParser(description='SPLOM')

    parser.add_argument('-i', '--input', dest='filename', required=True, help='Path to data file')
    parser.add_argument('-a', dest='attrs', required=True, action='append', help='attribute list')

    args = parser.parse_args()

    task3(args.filename, args.attrs)