import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import argparse

def task1(filename: str):
    df = pd.read_excel(filename)
    # print(df.head())

    bins = [100, 140, 180, 220, 260, 280]
    intervals = ['[100, 140)', '[140, 180)', '[180, 220)', '[220, 260)', '[260, 280)']

    df['Efficiency Intervals'] = pd.cut(df['Efficiency'], bins=bins, labels=intervals, right=False) # left inclusive, right exclusive
    counts = df.groupby(['Region','Efficiency Intervals'], observed=False).size().unstack(fill_value=0)
    fractoins = counts.div(counts.sum(axis=1), axis=0)

    fig, (chart1, chart2) = plt.subplots(1, 2, figsize=(15,6))

    regions = sorted(df['Region'].dropna().unique()) # Typically ['America', 'Asia', 'Europe']
    x = np.arange(len(intervals))
    width = 0.25
    
    # Set custom colors for each region
    colors = {'America': 'gold', 'Asia': 'mediumvioletred', 'Europe': 'darkblue'}
    
    for i, region in enumerate(regions):
        # Offset bars based on index to group them together
        offset = (i - 1) * width
        
        # Absolute counts plot
        chart1.bar(x + offset, counts.loc[region], width, label=region, color=colors.get(region, 'gray'), edgecolor='black', linewidth=1)
        
        # Relative fractions plot
        chart2.bar(x + offset, fractoins.loc[region], width, label=region, color=colors.get(region, 'gray'), edgecolor='black', linewidth=1)
        
#   number of models
    chart1.set_xticks(x)
    chart1.set_xticklabels(intervals)
    chart1.set_xlabel('Efficiency (Wh/km)')
    chart1.set_ylabel('Number of EV Models')
    chart1.legend(title='Production Origin')
    
#   proportion of models
    chart2.set_xticks(x)
    chart2.set_xticklabels(intervals)
    chart2.set_xlabel('Efficiency (Wh/km)')
    chart2.set_ylabel('Prportion of EV Models')
    chart2.legend(title='Production Origin')
    
    # Render both charts in a single window
    plt.title('Efficiency of EVs produced between 2010 and 2024')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='Grouped Bar Chart')
    parser.add_argument('-i', '--input', dest='filename', required=True, help='include path to data file')
    args = parser.parse_args()
    task1(args.filename)
