from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import sys
import json
import numpy as np
import math
from scipy.interpolate import make_interp_spline

def n_orders(sorted_vals):
    min_ = np.min(sorted_vals)
    max_ = np.max(sorted_vals)
    if min_ <= 0:
        print(f'WARNING: values are not strictly positive, min={min_}')
    return math.log10(min_), math.log10(max_)

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
        print(self.unique_indices)
        print(self.uvals)
        self.usizes = [ sizes[i] for i in self.unique_indices]

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
            self.labels= [ f'{s}' for s in self.stops ]
            for i, l in enumerate(self.labels):
                if len(l) > 4:
                    self.labels[i] = f'{self.stops[i]:.1f}'
        else:
            self.labels = [ f'{s:.1e}' for s in self.stops ]
        return self.labels

    def make_size_legend(self):
        custom_circles = [ plt.Line2D(range(1), range(1), markersize=math.sqrt(s),
                                    color='white', marker=self.shape,
                                    markerfacecolor=self.facecolor,
                                    markeredgecolor=self.edgecolor)
                        for s in self.size_stops ]
        heights = np.sqrt(self.size_stops)
        if self.verbose: print(f'maxsize = {heights}')
        # spacing will be multiplied by fontsize = 10
        dist = 0.5*(heights[-2] + heights[-1])
        if self.verbose: print(f'distance is {dist}')
        spacing = 0.8*(dist)/10

        if self.ax is None:
            self.size_legend = plt.legend(custom_circles,
                            [ f"{s}" for s in self.labels ],
                            title=self.title, title_fontproperties={'weight': 'bold'},loc='upper right',
                            bbox_to_anchor=(1, 1), handletextpad=2.0,
                            labelspacing=spacing)
        else:
            self.size_legend = self.ax.legend(custom_circles,
                            [ f"{s}" for s in self.labels ],
                            title=self.title, title_fontproperties={'weight': 'bold'},loc='upper right',
                            bbox_to_anchor=(1, 1), handletextpad=2.0,
                            labelspacing=spacing)
        if self.verbose: print(f'fontsize = {self.size_legend._fontsize}')
        return self.size_legend

if __name__ == '__main__':
    data = np.random.random((100, 4)) # x, y, size variable, color variable
    sizes = 500*data[:,2]
    acmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1,1,figsize=(12,6))
    axes.scatter(x=data[:,0], y=data[:,1], s=sizes, c=[acmap(v) for v in data[:,3]], edgecolor='black')

    legend = SizeLegend(data[:,2], sizes, nstops=4, log_scale=True, verbose=True, 
                 shape='o', spacing=1.5, ax=None, facecolor='white',
                 edgecolor='black', title='Size')
    axes.add_artist(legend.size_legend)
    plt.title('Bubble chart of random dataset')

    plt.show()