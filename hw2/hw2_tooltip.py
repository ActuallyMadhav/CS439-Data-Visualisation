# Deliverable and API: hw2_tooltip.py -i <dataset path>
import sys
import argparse

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from PyQt5 import QtWidgets, QtCore
from matplotlib import pyplot as plt

import math
from scipy.interpolate import make_interp_spline


def n_orders(sorted_vals):
    min_ = np.min(sorted_vals)
    max_ = np.max(sorted_vals)
    if min_ <= 0:
        print(f'WARNING: values are not strictly positive, min={min_}')
    return math.log10(min_), math.log10(max_)

# copied from brightspace
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

NON_NUMERIC_COLUMNS = {'name', 'region'}

MIN_SCALE = 200
MAX_SCALE = 5000
DEFAULT_SCALE = 2500

LEFT_DEFAULTS = dict(x='birth_rate', y='life_expectancy', size='population', color='median_age')
RIGHT_DEFAULTS = dict(x='imports', y='exports', size='labor_force', color='unemployment')

EXCLUDED_COLOR = (0.75, 0.75, 0.75)

TOOLTIP_FIELDS = [
    ('region', 'Region', 'str'),
    ('population', 'Population', 'int'),
    ('area', 'Area', 'int'),
    ('birth_rate', 'Birth rate', 'float'),
    ('life_expectancy', 'Life expectancy', 'float'),
    ('median_age', 'Median age', 'float'),
    ('imports', 'Imports', 'int'),
    ('exports', 'Exports', 'int'),
    ('labor_force', 'Labor force', 'int'),
    ('unemployment', 'Unemployment', 'float'),
    ('inflation', 'Inflation', 'float'),
]

MIN_HOVER_RADIUS_PX = 7


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


def format_field(value, kind):
    if pd.isna(value):
        return 'N/A'
    if kind == 'int':
        return f'{value:,.0f}'
    if kind == 'float':
        return f'{value:.2f}'
    return str(value)


def build_tooltip_text(row):
    lines = [f"Name: {row['name']}"]
    for col, label, kind in TOOLTIP_FIELDS:
        lines.append(f'{label}: {format_field(row.get(col), kind)}')
    return '\n'.join(lines)


class BubblePanel(QtWidgets.QWidget):
   
    def __init__(self, df, attributes, defaults, on_brush, on_hover, parent=None):
        super().__init__(parent)
        self.df = df
        self.attributes = attributes
        self.on_brush = on_brush
        self.on_hover = on_hover

        self.x_attr = defaults['x']
        self.y_attr = defaults['y']
        self.size_attr = defaults['size']
        self.color_attr = defaults['color']
        self.scale = DEFAULT_SCALE

        self.last_sub = None    # last drawn (cleaned) subset, including 'name'
        self.last_sizes = None  # marker areas matching last_sub, same order

        self.highlight_artist = None
        self.tooltip_annotation = None

        self._build_ui()
        self.redraw(selected_names=None)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('figure_leave_event', lambda e: self.on_hover(self, None))

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=1)

        grid = QtWidgets.QGridLayout()

        self.x_combo = self._make_combo(self.x_attr)
        self.y_combo = self._make_combo(self.y_attr)
        self.size_combo = self._make_combo(self.size_attr)
        self.color_combo = self._make_combo(self.color_attr)

        self.x_combo.currentTextChanged.connect(self._on_x_changed)
        self.y_combo.currentTextChanged.connect(self._on_y_changed)
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        self.color_combo.currentTextChanged.connect(self._on_color_changed)

        grid.addWidget(QtWidgets.QLabel('X'), 0, 0)
        grid.addWidget(self.x_combo, 0, 1)
        grid.addWidget(QtWidgets.QLabel('Y'), 0, 2)
        grid.addWidget(self.y_combo, 0, 3)

        grid.addWidget(QtWidgets.QLabel('Size'), 1, 0)
        grid.addWidget(self.size_combo, 1, 1)
        grid.addWidget(QtWidgets.QLabel('Color'), 1, 2)
        grid.addWidget(self.color_combo, 1, 3)

        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setMinimum(MIN_SCALE)
        self.scale_slider.setMaximum(MAX_SCALE)
        self.scale_slider.setValue(DEFAULT_SCALE)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        grid.addWidget(QtWidgets.QLabel('Scaling factor'), 2, 0)
        grid.addWidget(self.scale_slider, 2, 1, 1, 3)

        layout.addLayout(grid)

        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=False,
            button=[1], minspanx=0, minspany=0,
            spancoords='data', interactive=False,
        )

    def _make_combo(self, current):
        combo = QtWidgets.QComboBox()
        combo.addItems(self.attributes)
        combo.setCurrentText(current)
        return combo

    def _on_x_changed(self, text):
        self.x_attr = text
        self.redraw(self.on_brush(None, query=True))

    def _on_y_changed(self, text):
        self.y_attr = text
        self.redraw(self.on_brush(None, query=True))

    def _on_size_changed(self, text):
        self.size_attr = text
        self.redraw(self.on_brush(None, query=True))

    def _on_color_changed(self, text):
        self.color_attr = text
        self.redraw(self.on_brush(None, query=True))

    def _on_scale_changed(self, value):
        self.scale = value
        self.redraw(self.on_brush(None, query=True))

    def _on_select(self, eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])

        # near zero area drag (a click) clears the current selection.
        if self.last_sub is None or (x1 - x0) < 1e-12 or (y1 - y0) < 1e-12:
            selected_names = None
        else:
            sub = self.last_sub
            mask = (
                (sub[self.x_attr] >= x0) & (sub[self.x_attr] <= x1) &
                (sub[self.y_attr] >= y0) & (sub[self.y_attr] <= y1)
            )
            selected_names = set(sub.loc[mask, 'name'])

        self.on_brush(selected_names)

    def _on_motion(self, event):
        if event.inaxes != self.ax or self.last_sub is None or len(self.last_sub) == 0:
            self.on_hover(self, None)
            return

        xy_data = self.last_sub[[self.x_attr, self.y_attr]].to_numpy()
        xy_px = self.ax.transData.transform(xy_data)
        dist = np.hypot(xy_px[:, 0] - event.x, xy_px[:, 1] - event.y)

        radius_px = np.sqrt(np.maximum(self.last_sizes, 1.0) / np.pi) * (self.figure.dpi / 72.0)
        pickup_radius = np.maximum(radius_px, MIN_HOVER_RADIUS_PX)

        within = dist <= pickup_radius
        if not within.any():
            self.on_hover(self, None)
            return

        idx = np.argmin(np.where(within, dist, np.inf))
        name = self.last_sub.iloc[idx]['name']
        self.on_hover(self, name)

    def set_highlight(self, name, show_tooltip):
        changed = False
        if name is not None and self.last_sub is not None:
            match = self.last_sub[self.last_sub['name'] == name]
        else:
            match = None

        if match is not None and len(match):
            x, y = match.iloc[0][self.x_attr], match.iloc[0][self.y_attr]
            row_idx = match.index[0]
            size = self.last_sizes[row_idx] if self.last_sizes is not None else 200
            self.highlight_artist.set_offsets([[x, y]])
            self.highlight_artist.set_sizes([max(size * 1.3, 120)])
            self.highlight_artist.set_visible(True)

            if show_tooltip:
                full_row = self.df[self.df['name'] == name]
                if len(full_row):
                    text = build_tooltip_text(full_row.iloc[0])
                    self.tooltip_annotation.set_text(text)
                    self.tooltip_annotation.xy = (x, y)
                    self.tooltip_annotation.set_visible(True)
            else:
                self.tooltip_annotation.set_visible(False)
            changed = True
        else:
            if self.highlight_artist.get_visible() or self.tooltip_annotation.get_visible():
                self.highlight_artist.set_visible(False)
                self.tooltip_annotation.set_visible(False)
                changed = True

        if changed:
            self.canvas.draw_idle()

    def redraw(self, selected_names):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        sub = clean_subset(self.df, self.x_attr, self.y_attr, self.size_attr, self.color_attr)
        self.last_sub = sub
        sizes = scaled_sizes(sub[self.size_attr], scale=self.scale)
        self.last_sizes = sizes.to_numpy() if hasattr(sizes, 'to_numpy') else np.asarray(sizes)

        if not selected_names:
            scatter = self.ax.scatter(
                sub[self.x_attr], sub[self.y_attr], s=sizes, c=sub[self.color_attr],
                cmap='viridis', alpha=0.75, edgecolors='black', linewidth=1, zorder=2,
            )
            color_mappable = scatter
        else:
            is_selected = sub['name'].isin(selected_names)
            excluded, selected = sub[~is_selected], sub[is_selected]
            excluded_sizes, selected_sizes = sizes[~is_selected], sizes[is_selected]

            self.ax.scatter(
                excluded[self.x_attr], excluded[self.y_attr], s=excluded_sizes,
                c=[EXCLUDED_COLOR], alpha=0.4, edgecolors='none', zorder=1,
            )
            scatter = self.ax.scatter(
                selected[self.x_attr], selected[self.y_attr], s=selected_sizes,
                c=selected[self.color_attr], cmap='viridis', vmin=sub[self.color_attr].min(),
                vmax=sub[self.color_attr].max(), alpha=0.9, edgecolors='black',
                linewidth=1, zorder=2,
            )
            color_mappable = scatter

        self.ax.set_xlabel(self.x_attr)
        self.ax.set_ylabel(self.y_attr)

        cbar = self.figure.colorbar(color_mappable, ax=self.ax)
        cbar.set_label(self.color_attr)

        log_scale = bool((sub[self.size_attr] > 0).all())
        size_legend = SizeLegend(
            sub[self.size_attr].to_numpy(), self.last_sizes,
            nstops=4, log_scale=log_scale, shape='o',
            ax=self.ax, facecolor='white', edgecolor='black', title=self.size_attr,
        )
        self.ax.add_artist(size_legend.size_legend)

        self.ax.set_title('CIA Factbook 2023', weight='bold')
        self.figure.tight_layout()

        self.ax.set_xlim(self.ax.get_xlim())
        self.ax.set_ylim(self.ax.get_ylim())

        # Hover highlight ring 
        self.highlight_artist = self.ax.scatter(
            [], [], s=[], facecolors='none', edgecolors='red',
            linewidths=2.5, zorder=5, visible=False,
        )
        # Details-on-demand tooltip annotation.
        self.tooltip_annotation = self.ax.annotate(
            '', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
            fontsize=8, family='monospace', visible=False, zorder=6,
            bbox=dict(boxstyle='round', fc='mistyrose', ec='red', alpha=0.9),
        )

        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=False,
            button=[1], minspanx=0, minspany=0,
            spancoords='data', interactive=False,
        )
        self.canvas.draw_idle()


class TooltipWindow(QtWidgets.QMainWindow):
    def __init__(self, filename):
        super().__init__()
        self.setWindowTitle('HW2 - Task 4: Details on Demand')

        self.df = load_data(filename)
        attributes = get_numeric_attributes(self.df)
        self.selected_names = None
        self.hovered_name = None 

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.left_panel = BubblePanel(self.df, attributes, LEFT_DEFAULTS, self.handle_brush, self.handle_hover)
        self.right_panel = BubblePanel(self.df, attributes, RIGHT_DEFAULTS, self.handle_brush, self.handle_hover)

        layout.addWidget(self.left_panel)
        layout.addWidget(self.right_panel)

        self.resize(1400, 750)

    def handle_brush(self, selected_names, query=False):
        if query:
            return self.selected_names
        self.selected_names = selected_names
        self.left_panel.redraw(self.selected_names)
        self.right_panel.redraw(self.selected_names)
        return self.selected_names

    def handle_hover(self, source_panel, name):
        self.hovered_name = name
        for panel in (self.left_panel, self.right_panel):
            panel.set_highlight(name, show_tooltip=(panel is source_panel))


def task4(filename):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TooltipWindow(filename)
    window.show()
    app.exec_()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Details on Demand')
    parser.add_argument('-i', '--input', dest='filename', required=True, help='dataset filepath')

    args = parser.parse_args()

    task4(args.filename)