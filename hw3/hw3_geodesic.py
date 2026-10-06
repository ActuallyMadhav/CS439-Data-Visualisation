# Deliverable and API: hw3_geodesic.py -m <worldmap> -a <airports> -f <flights>
import sys
import argparse
import json
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, hsv_to_rgb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import textwrap

from PyQt5 import QtWidgets, QtCore
from geographiclib.geodesic import Geodesic

ALL_AIRLINES = 'All'
GEODESIC_STEP_KM = 250   # one sample every ~250 km along each route
GEODESIC_MAX_POINTS = 64
NODE_SLIDER_STEPS = 1000  # importance slider is logarithmic

# Node size (marker area, points^2) is proportional to the importance factor
NODE_MIN_AREA = 2.0
NODE_MAX_AREA = 350.0
NODE_CMAP = plt.get_cmap('viridis')

# Edge encoding: width and saturation both grow with the weight.
# Hue and value stay fixed so that only saturation ("purity") changes.
EDGE_HUE = 0.6
EDGE_ALPHA = 0.8
EDGE_VALUE = 0.8
EDGE_MIN_SAT, EDGE_MAX_SAT = 0.3, 1.0
EDGE_MIN_WIDTH, EDGE_MAX_WIDTH = 0.3, 2.5


def load_airports(filename):
    df = pd.read_excel(filename)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith('Unnamed')])
    # text cells sometimes carry leading non-breaking spaces
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    # Some coordinates are stored as text (e.g. with a leading non-breaking space)
    for col in ('Latitude', 'Longitude'):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r'(-?\d+(?:\.\d+)?)', expand=False),
            errors='coerce')
    df = df.dropna(subset=['IATA', 'Latitude', 'Longitude'])
    df = df.drop_duplicates(subset='IATA').reset_index(drop=True)
    return df


def load_flights(filename):
    with open(filename) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_world(filename):
    return gpd.read_file(filename)


class FlightGraph:
    """Airports (vertices) and routes (edges) as numpy arrays.

    Edge i connects node src[i] and node dst[i]; weight[i] is the number of
    flights on that route.
    """

    def __init__(self, airports, flights):
        self.airports = airports
        index = {code: i for i, code in enumerate(airports['IATA'])}

        keep = (flights['origin'].isin(index) & flights['destination'].isin(index)
                & (flights['origin'] != flights['destination']))
        flights = flights[keep].reset_index(drop=True)
        self.flights = flights

        self.lon = airports['Longitude'].to_numpy(float)
        self.lat = airports['Latitude'].to_numpy(float)
        self.xy = np.column_stack([self.lon, self.lat])

        self.src = flights['origin'].map(index).to_numpy()
        self.dst = flights['destination'].map(index).to_numpy()
        self.weight = flights['number_of_flights'].to_numpy(float)
        self.edge_airlines = [list(a) for a in flights['airlines']]

        n = len(airports)
        self.n_nodes = n
        self.n_edges = len(flights)
        # importance factor = total flights touching an airport
        self.importance = (np.bincount(self.src, self.weight, n)
                           + np.bincount(self.dst, self.weight, n))

        # number of distinct airlines serving each airport
        node_airlines = [set() for _ in range(n)]
        for s, d, airlines in zip(self.src, self.dst, self.edge_airlines):
            node_airlines[s].update(airlines)
            node_airlines[d].update(airlines)
        self.n_airlines = np.array([len(a) for a in node_airlines], dtype=float)

        # degree = number of distinct airports connected to each airport
        neighbours = [set() for _ in range(n)]
        for s, d in zip(self.src, self.dst):
            neighbours[s].add(d)
            neighbours[d].add(s)
        self.degree = np.array([len(nb) for nb in neighbours])

        # routes operated by each airline
        self.n_edge_airlines = np.array([max(len(a), 1) for a in self.edge_airlines], dtype=float)
        routes = {}
        for e, airlines in enumerate(self.edge_airlines):
            for code in airlines:
                routes.setdefault(code, []).append(e)
        self.airline_routes = {code: np.array(ids) for code, ids in routes.items()}
        self.airlines = sorted(self.airline_routes)

    def edge_weights(self, airline=ALL_AIRLINES):
        if airline == ALL_AIRLINES or airline not in self.airline_routes:
            return self.weight.copy(), np.ones(self.n_edges, dtype=bool)
        operated = np.zeros(self.n_edges, dtype=bool)
        operated[self.airline_routes[airline]] = True
        return np.where(operated, self.weight / self.n_edge_airlines, 0.0), operated

    def segments(self, edge_ids):
        return np.stack([self.xy[self.src[edge_ids]], self.xy[self.dst[edge_ids]]], axis=1)


def node_areas(importance, max_importance):
    return NODE_MIN_AREA + (NODE_MAX_AREA - NODE_MIN_AREA) * np.asarray(importance, float) / max_importance


def node_color_norm(n_airlines):
    return LogNorm(vmin=1, vmax=max(2, np.max(n_airlines)))


def edge_fraction(weights, vmin, vmax):
    weights = np.asarray(weights, float)
    if vmax <= vmin:  # all visible edges share one weight: use a mid-range style
        return np.full_like(weights, 0.5)
    return np.clip((weights - vmin) / (vmax - vmin), 0, 1)


def edge_colors(weights, vmin, vmax):
    t = edge_fraction(weights, vmin, vmax)
    hsv = np.column_stack([np.full_like(t, EDGE_HUE),
                           EDGE_MIN_SAT + (EDGE_MAX_SAT - EDGE_MIN_SAT) * t,
                           np.full_like(t, EDGE_VALUE)])
    return np.column_stack([hsv_to_rgb(hsv), np.full(len(t), EDGE_ALPHA)])


def edge_widths(weights, vmin, vmax):
    t = edge_fraction(weights, vmin, vmax)
    return EDGE_MIN_WIDTH + (EDGE_MAX_WIDTH - EDGE_MIN_WIDTH) * t


def nice_log_stops(vmin, vmax, steps=(1, 3, 10, 30, 100, 300, 1000, 3000)):
    stops = [s for s in steps if vmin <= s < vmax * 0.8]
    return stops + [vmax]


def fmt_number(v):
    return f'{v:.0f}' if float(v).is_integer() else f'{v:.1f}'


class SizeLegend:

    def __init__(self, stops, sizes, ax, title=None, shape='o', facecolor='white',
                 edgecolor='black', anchor=(1.01, 1.0)):
        self.stops = stops
        self.size_stops = np.asarray(sizes, float)
        self.labels = [fmt_number(s) for s in stops]
        handles = [plt.Line2D([], [], linestyle='none', marker=shape,
                              markersize=math.sqrt(s), markerfacecolor=facecolor,
                              markeredgecolor=edgecolor)
                   for s in self.size_stops]
        self.size_legend = ax.legend(handles, self.labels, title=title,
                                     title_fontproperties={'weight': 'bold'},
                                     loc='upper left', bbox_to_anchor=anchor,
                                     handletextpad=1.5, labelspacing=1.0, borderpad=0.8)


def node_color_legend(ax, n_airlines, anchor=(1.01, 0.60)):
    norm = node_color_norm(n_airlines)
    stops = nice_log_stops(1, float(np.max(n_airlines)))
    handles = [plt.Line2D([], [], linestyle='none', marker='o', markersize=8,
                          markerfacecolor=NODE_CMAP(norm(s)), markeredgecolor='black')
               for s in stops]
    return ax.legend(handles, [fmt_number(s) for s in stops], title='Number of airlines',
                     title_fontproperties={'weight': 'bold'}, loc='upper left',
                     bbox_to_anchor=anchor)


def edge_legend(ax, vmin, vmax, nstops=5, anchor=(1.01, 0.30)):
    if vmax > vmin:
        decimals = 0 if vmax - vmin >= nstops else 1
        stops = np.unique(np.round(np.linspace(vmin, vmax, nstops), decimals))
    else:
        stops = np.array([vmin])
    colors = edge_colors(stops, vmin, vmax)
    widths = edge_widths(stops, vmin, vmax)
    handles = [plt.Line2D([], [], color=c, linewidth=w) for c, w in zip(colors, widths)]
    return ax.legend(handles, [fmt_number(s) for s in stops], title='Number of flights',
                     title_fontproperties={'weight': 'bold'}, loc='upper left',
                     bbox_to_anchor=anchor, handlelength=3)


MIN_HOVER_RADIUS_PX = 5
EDGE_HOVER_PX = 4


def text_or_na(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return 'N/A'
    return str(value)


def node_tooltip_text(graph, i):
    row = graph.airports.iloc[i]
    lines = [
        f"Name:       {row['Airport name']} ({row['IATA']})",
        f"City:       {text_or_na(row.get('City'))}",
        f"Province:   {text_or_na(row.get('Province'))}",
        f"Country:    {text_or_na(row.get('Country'))}",
        f"Degree:     {graph.degree[i]}",
        f"Importance: {graph.importance[i]:.0f}",
    ]
    return '\n'.join(lines)


def apply_filters(graph, min_importance, min_weight, airline):
    weights, operated = graph.edge_weights(airline)
    node_ok = graph.importance >= min_importance
    edge_ok = (operated & node_ok[graph.src] & node_ok[graph.dst]
               & (weights >= min_weight - 1e-9))
    edges = np.flatnonzero(edge_ok)
    edges = edges[np.argsort(weights[edges], kind='stable')]  # heaviest drawn last
    nodes = np.unique(np.concatenate([graph.src[edges], graph.dst[edges]]))
    nodes = nodes[np.argsort(graph.importance[nodes], kind='stable')]
    return nodes, edges, weights


def edge_tooltip_text(graph, e, extra_lines=()):
    a, b = graph.airports.iloc[graph.src[e]], graph.airports.iloc[graph.dst[e]]
    airlines = ', '.join(sorted(graph.edge_airlines[e]))
    wrapped = textwrap.wrap(airlines, 60) or ['N/A']
    lines = [
        f"Airport 1: {a['Airport name']} ({a['IATA']})",
        f"Airport 2: {b['Airport name']} ({b['IATA']})",
        f"Flights:   {graph.weight[e]:.0f}",
        *extra_lines,
        f"Airlines:  {wrapped[0]}",
        *[f"           {line}" for line in wrapped[1:]],
    ]
    return '\n'.join(lines)


class HoverTooltip:
    
    def __init__(self, fig, ax, node_text, edge_text):
        self.fig, self.ax, self.canvas = fig, ax, fig.canvas
        self.node_text, self.edge_text = node_text, edge_text
        self.background = None
        self.current = None

        self.annotation = ax.annotate(
            '', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
            fontsize=8, family='monospace', multialignment='left', zorder=10, annotation_clip=False,
            bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.92),
            visible=False, animated=True)
        self.node_ring = ax.scatter([], [], s=[], facecolors='none', edgecolors='red',
                                    linewidths=2, zorder=9, visible=False, animated=True)
        self.edge_highlight = LineCollection([], colors='red', linewidths=2.5, zorder=8,
                                             visible=False, animated=True)
        ax.add_collection(self.edge_highlight)
        self.artists = (self.edge_highlight, self.node_ring, self.annotation)

        self.set_data(np.empty((0, 2)), np.empty(0), np.empty(0, int),
                      np.empty((0, 2, 2)), np.empty(0, int))

        self.canvas.mpl_connect('draw_event', self._on_draw)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('figure_leave_event', lambda e: self._show(None))

    def set_data(self, node_xy, node_area, node_ids, segments, seg_edge):
        """Set what can currently be hovered (only visible elements)."""
        self.node_xy = np.asarray(node_xy, float).reshape(-1, 2)
        self.node_area = np.asarray(node_area, float)
        self.node_ids = np.asarray(node_ids, int)
        self.segments = np.asarray(segments, float).reshape(-1, 2, 2)
        self.seg_edge = np.asarray(seg_edge, int)
        self.seg_lo = self.segments.min(axis=1)
        self.seg_hi = self.segments.max(axis=1)
        self.current = None
        for artist in self.artists:
            artist.set_visible(False)

    def _on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.fig.bbox)
        for artist in self.artists:
            if artist.get_visible():
                self.ax.draw_artist(artist)

    def _blit(self):
        if self.background is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self.background)
        for artist in self.artists:
            if artist.get_visible():
                self.ax.draw_artist(artist)
        self.canvas.blit(self.fig.bbox)

    def _pixels_per_unit(self):
        (x0, y0), (x1, y1) = self.ax.transData.transform([[0, 0], [1, 1]])
        return abs(x1 - x0), abs(y1 - y0)

    def _hit_node(self, x, y, sx, sy):
        if len(self.node_xy) == 0:
            return None
        dist = np.hypot((self.node_xy[:, 0] - x) * sx, (self.node_xy[:, 1] - y) * sy)
        radius = np.sqrt(self.node_area / np.pi) * self.fig.dpi / 72.0
        within = dist <= np.maximum(radius, MIN_HOVER_RADIUS_PX)
        if not within.any():
            return None
        return int(np.argmin(np.where(within, dist, np.inf)))

    def _hit_edge(self, x, y, sx, sy):
        if len(self.segments) == 0:
            return None
        tx, ty = EDGE_HOVER_PX / sx, EDGE_HOVER_PX / sy
        near = ((self.seg_lo[:, 0] - tx <= x) & (self.seg_hi[:, 0] + tx >= x)
                & (self.seg_lo[:, 1] - ty <= y) & (self.seg_hi[:, 1] + ty >= y))
        candidates = np.flatnonzero(near)
        if len(candidates) == 0:
            return None
        scale = np.array([sx, sy])
        a = self.segments[candidates, 0] * scale
        b = self.segments[candidates, 1] * scale
        p = np.array([x, y]) * scale
        ab = b - a
        length2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * ab).sum(axis=1) / length2, 0, 1)
        closest = a + t[:, None] * ab
        dist = np.hypot(*(closest - p).T)
        k = np.argmin(dist)
        if dist[k] > EDGE_HOVER_PX:
            return None
        return int(self.seg_edge[candidates[k]])

    def _on_motion(self, event):
        if event.inaxes is not self.ax or event.button is not None:
            self._show(None)
            return
        sx, sy = self._pixels_per_unit()
        node = self._hit_node(event.xdata, event.ydata, sx, sy)
        if node is not None:
            self._show(('node', node), event)
            return
        edge = self._hit_edge(event.xdata, event.ydata, sx, sy)
        self._show(('edge', edge) if edge is not None else None, event)

    def _show(self, target, event=None):
        if target == self.current:
            return
        self.current = target
        for artist in self.artists:
            artist.set_visible(False)

        if target is not None:
            kind, k = target
            if kind == 'node':
                xy = self.node_xy[k]
                self.node_ring.set_offsets([xy])
                self.node_ring.set_sizes([max(self.node_area[k] * 1.6, 60)])
                self.node_ring.set_visible(True)
                text = self.node_text(self.node_ids[k])
            else:
                xy = (event.xdata, event.ydata)
                self.edge_highlight.set_segments(list(self.segments[self.seg_edge == k]))
                self.edge_highlight.set_visible(True)
                text = self.edge_text(k)

            # keep the box inside the axes: flip it towards the centre
            bbox = self.ax.bbox
            px, py = self.ax.transData.transform(xy)
            right = px > bbox.x0 + bbox.width * 0.5
            top = py > bbox.y0 + bbox.height * 0.5
            self.annotation.xy = xy
            self.annotation.set_text(text)
            self.annotation.set_position((-15 if right else 15, -15 if top else 15))
            self.annotation.set_horizontalalignment('right' if right else 'left')
            self.annotation.set_verticalalignment('top' if top else 'bottom')
            self.annotation.set_visible(True)
        self._blit()


def draw_world(ax, world):
    world.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


def split_antimeridian(points):
    pieces, current = [], [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        jump = b[0] - a[0]
        if abs(jump) > 180:
            edge = -180.0 if jump > 0 else 180.0
            b_lon = b[0] - 360 if jump > 0 else b[0] + 360
            t = (edge - a[0]) / (b_lon - a[0])
            lat = a[1] + t * (b[1] - a[1])
            current.append((edge, lat))
            pieces.append(np.array(current))
            current = [(-edge, lat), b]
        else:
            current.append(b)
    pieces.append(np.array(current))
    return pieces


def geodesic_path(lat1, lon1, lat2, lon2):
    line = Geodesic.WGS84.InverseLine(lat1, lon1, lat2, lon2,
                                      Geodesic.LATITUDE | Geodesic.LONGITUDE
                                      | Geodesic.DISTANCE_IN)
    n = int(np.clip(np.ceil(line.s13 / 1000.0 / GEODESIC_STEP_KM), 1, GEODESIC_MAX_POINTS - 1)) + 1
    points = []
    for k in range(n):
        p = line.Position(line.s13 * k / (n - 1), Geodesic.LATITUDE | Geodesic.LONGITUDE)
        points.append((p['lon2'], p['lat2']))
    return split_antimeridian(points)


class GeodesicPaths:
    
    def __init__(self, graph):
        self.pieces = []        # one list of (k, 2) arrays per edge
        segments, seg_edge = [], []
        for e in range(graph.n_edges):
            s, d = graph.src[e], graph.dst[e]
            pieces = geodesic_path(graph.lat[s], graph.lon[s], graph.lat[d], graph.lon[d])
            self.pieces.append(pieces)
            for piece in pieces:
                if len(piece) > 1:
                    segments.append(np.stack([piece[:-1], piece[1:]], axis=1))
                    seg_edge.append(np.full(len(piece) - 1, e))
        # every small segment of every geodesic, for hit-testing the tooltip
        self.segments = np.concatenate(segments)
        self.seg_edge = np.concatenate(seg_edge)

    def lines(self, edges):
        lines, line_edge = [], []
        for e in edges:
            for piece in self.pieces[e]:
                lines.append(piece)
                line_edge.append(e)
        return lines, np.array(line_edge, dtype=int)

    def hover_segments(self, edge_mask):
        keep = edge_mask[self.seg_edge]
        return self.segments[keep], self.seg_edge[keep]


class FilterWindow(QtWidgets.QMainWindow):
    window_title = 'HW3 - Task 4: Interactive Filtering'

    def __init__(self, world, graph):
        super().__init__()
        self.setWindowTitle(self.window_title)
        self.graph = graph
        self.airline = ALL_AIRLINES
        self.min_importance = 0.0
        self.min_weight = 1.0
        self.weights = graph.weight
        self.edge_legend = None

        self._build_ui()
        self._build_plot(world)
        self.update_view()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.figure = Figure(figsize=(15, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        grid = QtWidgets.QGridLayout()

        self.node_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.node_slider.setRange(0, NODE_SLIDER_STEPS)
        self.node_slider.setTracking(False)  # filter when released, label while dragging
        self.node_slider.sliderMoved.connect(self._on_node_slider_moved)
        self.node_slider.valueChanged.connect(self._on_node_slider_changed)
        self.node_label = QtWidgets.QLabel()
        self.node_label.setMinimumWidth(60)

        self.edge_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.edge_slider.setTracking(False)
        self.edge_slider.sliderMoved.connect(self._on_edge_slider_moved)
        self.edge_slider.valueChanged.connect(self._on_edge_slider_changed)
        self.edge_label = QtWidgets.QLabel()
        self.edge_label.setMinimumWidth(60)
        self._reset_edge_slider()

        self.airline_combo = QtWidgets.QComboBox()
        self.airline_combo.addItems([ALL_AIRLINES] + self.graph.airlines)
        self.airline_combo.currentTextChanged.connect(self._on_airline_changed)

        grid.addWidget(QtWidgets.QLabel('Min. airport importance'), 0, 0)
        grid.addWidget(self.node_slider, 0, 1)
        grid.addWidget(self.node_label, 0, 2)
        grid.addWidget(QtWidgets.QLabel('Min. edge weight (flights)'), 1, 0)
        grid.addWidget(self.edge_slider, 1, 1)
        grid.addWidget(self.edge_label, 1, 2)
        grid.addWidget(QtWidgets.QLabel('Airline'), 0, 3)
        grid.addWidget(self.airline_combo, 1, 3)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self._on_node_slider_moved(self.node_slider.value())
        self.resize(1500, 900)

    def _build_plot(self, world):
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.05, right=0.84, top=0.94, bottom=0.07)
        draw_world(self.ax, world)

        g = self.graph
        self.edge_artist = LineCollection([], zorder=3)
        self.ax.add_collection(self.edge_artist)
        self.node_norm = node_color_norm(g.n_airlines)
        self.node_artist = self.ax.scatter(np.empty(0), np.empty(0), s=np.empty(0),
                                           edgecolors='black', linewidths=0.3, zorder=4)

        # node legends do not depend on the filters
        max_imp = g.importance.max()
        imp_stops = nice_log_stops(10, max_imp, steps=(10, 100, 1000, 10000))
        size_legend = SizeLegend(imp_stops, node_areas(imp_stops, max_imp), self.ax,
                                 title='Airport significance')
        self.ax.add_artist(size_legend.size_legend)
        self.ax.add_artist(node_color_legend(self.ax, g.n_airlines))

        self.tooltip = HoverTooltip(self.figure, self.ax,
                                    node_text=lambda i: node_tooltip_text(g, i),
                                    edge_text=self._edge_text)

    def edge_geometry(self, edges):
        """Lines to draw, the edge of each line, and the same for hit-testing."""
        segments = self.graph.segments(edges)
        return list(segments), edges, segments, edges

    def _importance_from_slider(self, value):
        if value <= 0:
            return 0.0
        max_imp = self.graph.importance.max()
        return float(np.round(10 ** (value / NODE_SLIDER_STEPS * np.log10(max_imp))))

    def _on_node_slider_moved(self, value):
        self.node_label.setText(fmt_number(self._importance_from_slider(value)))

    def _on_node_slider_changed(self, value):
        self._on_node_slider_moved(value)
        self.min_importance = self._importance_from_slider(value)
        self.update_view()

    def _on_edge_slider_moved(self, value):
        self.edge_label.setText(str(value))

    def _on_edge_slider_changed(self, value):
        self._on_edge_slider_moved(value)
        self.min_weight = float(value)
        self.update_view()

    def _reset_edge_slider(self):
        weights, operated = self.graph.edge_weights(self.airline)
        max_w = int(np.ceil(weights[operated].max())) if operated.any() else 1
        self.edge_slider.blockSignals(True)
        self.edge_slider.setRange(1, max(max_w, 1))
        self.edge_slider.setValue(min(max(int(self.min_weight), 1), max(max_w, 1)))
        self.edge_slider.blockSignals(False)
        self.min_weight = float(self.edge_slider.value())
        self._on_edge_slider_moved(self.edge_slider.value())

    def _on_airline_changed(self, text):
        self.airline = text
        self.min_weight = 1.0  # weights change meaning with the airline: start unfiltered
        self._reset_edge_slider()
        self.update_view()

    def _edge_text(self, e):
        extra = []
        if self.airline != ALL_AIRLINES:
            extra.append(f'{self.airline} flights: {fmt_number(round(self.weights[e], 1))}')
        return edge_tooltip_text(self.graph, e, extra)

    def update_view(self):
        g = self.graph
        nodes, edges, weights = apply_filters(g, self.min_importance, self.min_weight, self.airline)
        self.weights = weights

        lines, line_edge, hover_segments, hover_edge = self.edge_geometry(edges)
        if len(edges):
            vmin, vmax = weights[edges].min(), weights[edges].max()
            w = weights[line_edge]
            self.edge_artist.set_segments(lines)
            self.edge_artist.set_color(edge_colors(w, vmin, vmax))
            self.edge_artist.set_linewidths(edge_widths(w, vmin, vmax))
        else:
            self.edge_artist.set_segments([])

        areas = node_areas(g.importance[nodes], g.importance.max())
        self.node_artist.set_offsets(g.xy[nodes] if len(nodes) else np.empty((0, 2)))
        self.node_artist.set_sizes(areas)
        self.node_artist.set_facecolors(NODE_CMAP(self.node_norm(g.n_airlines[nodes])))

        if self.edge_legend is not None:
            self.edge_legend.remove()
            self.edge_legend = None
        if len(edges):
            self.edge_legend = edge_legend(self.ax, vmin, vmax)
            self.ax.legend_ = None  # keep it only as an artist so it can be removed
            self.ax.add_artist(self.edge_legend)

        who = 'All Airlines' if self.airline == ALL_AIRLINES else f'Airline {self.airline}'
        self.ax.set_title(f'Major Air Routes for {who}  ({len(edges)} routes, {len(nodes)} airports)',
                          weight='bold')

        self.tooltip.set_data(g.xy[nodes], areas, nodes, hover_segments, hover_edge)
        self.canvas.draw_idle()


def run_window(window_class, map_file, airports_file, flights_file):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    world = load_world(map_file)
    graph = FlightGraph(load_airports(airports_file), load_flights(flights_file))
    window = window_class(world, graph)
    window.show()
    app.exec_()


class GeodesicWindow(FilterWindow):
    window_title = 'HW3 - Task 5: Geodesics'

    def __init__(self, world, graph):
        print('Computing geodesic flight paths...', flush=True)
        self.paths = GeodesicPaths(graph)
        super().__init__(world, graph)

    def edge_geometry(self, edges):
        lines, line_edge = self.paths.lines(edges)
        mask = np.zeros(self.graph.n_edges, dtype=bool)
        mask[edges] = True
        segments, seg_edge = self.paths.hover_segments(mask)
        return lines, line_edge, segments, seg_edge


def task5(map_file, airports_file, flights_file):
    run_window(GeodesicWindow, map_file, airports_file, flights_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geodesics')
    parser.add_argument('-m', '--map', dest='map_file', required=True, help='world map (GeoJSON)')
    parser.add_argument('-a', '--airports', dest='airports_file', required=True, help='airports (Excel)')
    parser.add_argument('-f', '--flights', dest='flights_file', required=True, help='flights (JSON)')
    args = parser.parse_args()

    task5(args.map_file, args.airports_file, args.flights_file)
