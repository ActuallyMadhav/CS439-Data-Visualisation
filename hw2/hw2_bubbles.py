# Deliverable and API: hw2_bubbles.py -i <dataset path> [-x X] [-y Y] [-s SIZE] [-c COLOR]
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse

import math
from scipy.interpolate import make_interp_spline


def n_orders(sorted_vals):
    min_ = np.min(sorted_vals)
    max_ = np.max(sorted_vals)
    if min_ <= 0:
        print(f'WARNING: values are not strictly positive, min={min_}')
    return math.log10(min_), math.log10(max_)

#copied from brightspace 
class SizeLegend:

    def __init__(self, values, sizes, nstops=4, log_scale=True, verbose=False,
                 shape='o', spacing=0.5, ax=None, facecolor='white',
                 edgecolor='black', title=None):
        self.verbose = verbose
        self.shape = shape
        self.spacing = spacing
        self.ax = ax
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.title = title
        self.values = values
        self.sizes = sizes
        self.nstops = nstops
        self.log_scale = log_scale
        self.uvals, self.unique_indices = np.unique(values, return_index=True)
        self.usizes = [sizes[i] for i in self.unique_indices]

        self.all_vals = np.array([self.uvals, self.usizes]).transpose()
        self.all_vals = np.sort(self.all_vals, axis=0)

        if self.verbose:
            print(f'unique sorted values are\n{self.uvals}')
            print(f'corresponding sizes are\n{self.usizes}')
        self.val_to_size = make_interp_spline(self.all_vals[:, 0], self.all_vals[:, 1])
        self.min_order, self.max_order = n_orders(self.all_vals[:, 0])
        if self.verbose:
            print(f'order range: {self.min_order} to {self.max_order}')

        if not log_scale:
            self.stops = np.linspace(self.all_vals[0, 0], self.all_vals[-1, 0], nstops)
        else:
            orders = np.linspace(self.min_order, self.max_order, nstops)
            self.stops = np.power(10, orders)

        self.size_stops = self.val_to_size(self.stops)
        if self.verbose:
            print(f'value stops: {self.stops}')
            print(f'size stops: {self.size_stops}')

        self.make_size_labels()
        self.make_size_legend()

    def make_size_labels(self):
        self.labels = []
        if self.verbose: print(f'size_text: stops are {self.stops}')
        if self.min_order >= 0 and self.max_order <= 4:
            self.labels = [f'{s}' for s in self.stops]
            for i, l in enumerate(self.labels):
                if len(l) > 4:
                    self.labels[i] = f'{self.stops[i]:.1f}'
        else:
            self.labels = [f'{s:.1e}' for s in self.stops]
        return self.labels

    def make_size_legend(self):
        custom_circles = [plt.Line2D(range(1), range(1), markersize=math.sqrt(s),
                                      color='white', marker=self.shape,
                                      markerfacecolor=self.facecolor,
                                      markeredgecolor=self.edgecolor)
                           for s in self.size_stops]
        heights = np.sqrt(self.size_stops)
        if self.verbose: print(f'maxsize = {heights}')
        # spacing will be multiplied by fontsize = 10
        dist = 0.5 * (heights[-2] + heights[-1])
        if self.verbose: print(f'distance is {dist}')
        spacing = 0.8 * dist / 10

        if self.ax is None:
            self.size_legend = plt.legend(custom_circles,
                                           [f"{s}" for s in self.labels],
                                           title=self.title, title_fontproperties={'weight': 'bold'}, loc='upper right',
                                           bbox_to_anchor=(1, 1), handletextpad=2.0,
                                           labelspacing=spacing)
        else:
            self.size_legend = self.ax.legend(custom_circles,
                                               [f"{s}" for s in self.labels],
                                               title=self.title, title_fontproperties={'weight': 'bold'}, loc='upper right',
                                               bbox_to_anchor=(1, 1), handletextpad=2.0,
                                               labelspacing=spacing)
        if self.verbose: print(f'fontsize = {self.size_legend._fontsize}')
        return self.size_legend

# Defaults used when -x/-y/-s/-c are not provided on the command line.
DEFAULT_X = 'GDP_per_capita'
DEFAULT_Y = 'military_expenditures'
DEFAULT_SIZE = 'population'
DEFAULT_COLOR = 'life_expectancy'

NON_NUMERIC_COLUMNS = {'name', 'region'}


def load_data(filename):
    
    df = pd.read_csv(filename)
    for col in df.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def clean_subset(df, x_attr, y_attr, size_attr, color_attr):

    cols = list(dict.fromkeys(['name', x_attr, y_attr, size_attr, color_attr]))
    sub = df[cols].dropna(subset=[x_attr, y_attr, size_attr, color_attr])
    return sub.reset_index(drop=True)


def scaled_sizes(values, scale=2500.0):
    
    values = np.asarray(values, dtype=float)
    max_val = np.nanmax(values) if len(values) else 1.0
    if max_val <= 0:
        max_val = 1.0
    return (values / max_val) * scale


def task1(filename: str, x_attr: str, y_attr: str, size: str, color: str):
    df = load_data(filename)
    sub = clean_subset(df, x_attr, y_attr, size, color)

    scaled_size = scaled_sizes(sub[size])

    fig, chart = plt.subplots(figsize=(10, 7))

    # Create bubble chart
    scatter = chart.scatter(
        sub[x_attr],
        sub[y_attr],
        s=scaled_size,
        c=sub[color],
        cmap='viridis',
        alpha=0.75,
        edgecolors='black',
        linewidth=1
    )

    # labels
    chart.set_xlabel(x_attr)
    chart.set_ylabel(y_attr)

    cbar = plt.colorbar(scatter, ax=chart)
    cbar.set_label(color)

    # Size legend (bubble area -> attribute value), per legend.py's example
    log_scale = bool((sub[size] > 0).all())
    size_legend = SizeLegend(
        sub[size].to_numpy(), scaled_size.to_numpy() if hasattr(scaled_size, 'to_numpy') else scaled_size,
        nstops=4, log_scale=log_scale, shape='o',
        ax=chart, facecolor='white', edgecolor='black', title=size
    )
    chart.add_artist(size_legend.size_legend)

    chart.set_title(f'Bubble Chart Representation of {x_attr}, {y_attr}, {size}, {color}', weight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bubble Chart')
    parser.add_argument('-i', '--input', dest='filename', required=True, help='Path to dataset')
    parser.add_argument('-x', dest='x_attr', default=DEFAULT_X, help='X attribute')
    parser.add_argument('-y', dest='y_attr', default=DEFAULT_Y, help='Y attribute')
    parser.add_argument('-s', dest='size', default=DEFAULT_SIZE, help='Size attribute')
    parser.add_argument('-c', dest='color', default=DEFAULT_COLOR, help='Color attribute')

    args = parser.parse_args()

    task1(args.filename, args.x_attr, args.y_attr, args.size, args.color)