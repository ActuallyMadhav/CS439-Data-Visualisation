# Deliverable and API: hw2_widgets.py -i <dataset path> [-x X] [-y Y] [-s SIZE] [-c COLOR]
import sys
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, QtCore

import math
from scipy.interpolate import make_interp_spline


def n_orders(sorted_vals):
    min_ = np.min(sorted_vals)
    max_ = np.max(sorted_vals)
    if min_ <= 0:
        print(f'WARNING: values are not strictly positive, min={min_}')
    return math.log10(min_), math.log10(max_)

# copied from birghtspace
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

MIN_SCALE = 200
MAX_SCALE = 5000
DEFAULT_SCALE = 2500


def load_data(filename):

    df = pd.read_csv(filename)
    for col in df.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_numeric_attributes(df):
    return sorted(c for c in df.columns if c not in NON_NUMERIC_COLUMNS)


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


class BubbleWidgetWindow(QtWidgets.QMainWindow):
    def __init__(self, filename, x_attr, y_attr, size_attr, color_attr):
        super().__init__()
        self.setWindowTitle('HW2 - Task 2: Widgets')

        self.df = load_data(filename)
        self.attributes = get_numeric_attributes(self.df)

        self.x_attr = x_attr
        self.y_attr = y_attr
        self.size_attr = size_attr
        self.color_attr = color_attr
        self.scale = DEFAULT_SCALE

        self._build_ui()
        self.redraw()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        #Left: matplotlib canvas + navigation toolbar
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_container, stretch=4)

        # Right: control panel
        controls = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(controls)

        self.x_combo = self._make_combo(self.x_attr)
        self.y_combo = self._make_combo(self.y_attr)
        self.size_combo = self._make_combo(self.size_attr)
        self.color_combo = self._make_combo(self.color_attr)

        self.x_combo.currentTextChanged.connect(self._on_x_changed)
        self.y_combo.currentTextChanged.connect(self._on_y_changed)
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        self.color_combo.currentTextChanged.connect(self._on_color_changed)

        form.addRow('X', self.x_combo)
        form.addRow('Y', self.y_combo)
        form.addRow('Size', self.size_combo)
        form.addRow('Color', self.color_combo)

        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setMinimum(MIN_SCALE)
        self.scale_slider.setMaximum(MAX_SCALE)
        self.scale_slider.setValue(DEFAULT_SCALE)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        form.addRow('Scaling factor', self.scale_slider)

        main_layout.addWidget(controls, stretch=1)
        self.resize(1200, 700)

    def _make_combo(self, current):
        combo = QtWidgets.QComboBox()
        combo.addItems(self.attributes)
        combo.setCurrentText(current)
        return combo

    def _on_x_changed(self, text):
        self.x_attr = text
        self.redraw()

    def _on_y_changed(self, text):
        self.y_attr = text
        self.redraw()

    def _on_size_changed(self, text):
        self.size_attr = text
        self.redraw()

    def _on_color_changed(self, text):
        self.color_attr = text
        self.redraw()

    def _on_scale_changed(self, value):
        self.scale = value
        self.redraw()

    def redraw(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        sub = clean_subset(self.df, self.x_attr, self.y_attr, self.size_attr, self.color_attr)
        sizes = scaled_sizes(sub[self.size_attr], scale=self.scale)

        scatter = self.ax.scatter(
            sub[self.x_attr],
            sub[self.y_attr],
            s=sizes,
            c=sub[self.color_attr],
            cmap='viridis',
            alpha=0.75,
            edgecolors='black',
            linewidth=1,
        )

        self.ax.set_xlabel(self.x_attr)
        self.ax.set_ylabel(self.y_attr)

        cbar = self.figure.colorbar(scatter, ax=self.ax)
        cbar.set_label(self.color_attr)

        log_scale = bool((sub[self.size_attr] > 0).all())
        size_legend = SizeLegend(
            sub[self.size_attr].to_numpy(), sizes.to_numpy() if hasattr(sizes, 'to_numpy') else sizes,
            nstops=4, log_scale=log_scale, shape='o',
            ax=self.ax, facecolor='white', edgecolor='black', title=self.size_attr,
        )
        self.ax.add_artist(size_legend.size_legend)

        self.ax.set_title(
            f'Bubble Chart Representation of {self.x_attr}, {self.y_attr}, '
            f'{self.size_attr}, {self.color_attr}', weight='bold'
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()


def task2(filename, x_attr, y_attr, size, color):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = BubbleWidgetWindow(filename, x_attr, y_attr, size, color)
    window.show()
    app.exec_()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Widgets')
    parser.add_argument('-i', '--input', dest='filename', required=True, help='Dataset filepath')
    parser.add_argument('-x', dest='x_attr', default=DEFAULT_X, help='X attribute')
    parser.add_argument('-y', dest='y_attr', default=DEFAULT_Y, help='Y attribute')
    parser.add_argument('-s', dest='size', default=DEFAULT_SIZE, help='Size attribute')
    parser.add_argument('-c', dest='color', default=DEFAULT_COLOR, help='Color attribute')

    args = parser.parse_args()

    task2(args.filename, args.x_attr, args.y_attr, args.size, args.color)