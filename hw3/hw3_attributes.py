# Deliverable and API: hw3_attributes.py -m <worldmap> -a <airports> -f <flights>
import argparse
import json
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, hsv_to_rgb

N_HEAVIEST = 200

# Node size (marker area, points^2) is proportional to the importance factor
NODE_MIN_AREA = 2.0
NODE_MAX_AREA = 350.0
NODE_CMAP = plt.get_cmap('viridis')

# Edge encoding: width and saturation both grow with the weight.
# Hue and value stay fixed so that only saturation ("purity") changes.
EDGE_HUE = 0.6
EDGE_VALUE = 0.8
EDGE_MIN_SAT, EDGE_MAX_SAT = 0.3, 1.0
EDGE_MIN_WIDTH, EDGE_MAX_WIDTH = 0.4, 4.0


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

    def heaviest_edges(self, k=N_HEAVIEST):
        order = np.argsort(-self.weight, kind='stable')[:k]
        return order[::-1]  # lightest first so the heaviest are drawn on top

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
    return hsv_to_rgb(hsv)


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


def add_legends(ax, graph, edge_vmin, edge_vmax):
    max_imp = graph.importance.max()
    imp_stops = nice_log_stops(10, max_imp, steps=(10, 100, 1000, 10000))
    size_legend = SizeLegend(imp_stops, node_areas(imp_stops, max_imp), ax,
                             title='Airport significance')
    ax.add_artist(size_legend.size_legend)
    ax.add_artist(node_color_legend(ax, graph.n_airlines))
    ax.add_artist(edge_legend(ax, edge_vmin, edge_vmax))


def draw_world(ax, world):
    world.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


def draw_nodes(ax, graph, node_ids):
    # largest nodes last so they are not hidden by small ones
    node_ids = node_ids[np.argsort(graph.importance[node_ids], kind='stable')]
    return ax.scatter(graph.lon[node_ids], graph.lat[node_ids],
                      s=node_areas(graph.importance[node_ids], graph.importance.max()),
                      c=graph.n_airlines[node_ids], cmap=NODE_CMAP,
                      norm=node_color_norm(graph.n_airlines),
                      edgecolors='black', linewidths=0.3, zorder=2)


def task2(map_file, airports_file, flights_file):
    world = load_world(map_file)
    graph = FlightGraph(load_airports(airports_file), load_flights(flights_file))

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(left=0.05, right=0.84, top=0.94, bottom=0.07)
    draw_world(ax, world)

    edges = graph.heaviest_edges()
    w = graph.weight[edges]
    vmin, vmax = w.min(), w.max()
    ax.add_collection(LineCollection(graph.segments(edges), colors=edge_colors(w, vmin, vmax),
                                     linewidths=edge_widths(w, vmin, vmax), zorder=3))

    draw_nodes(ax, graph, np.flatnonzero(graph.importance > 0))
    add_legends(ax, graph, vmin, vmax)

    ax.set_title('Major Air Routes', weight='bold')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encoding additional attributes')
    parser.add_argument('-m', '--map', dest='map_file', required=True, help='world map (GeoJSON)')
    parser.add_argument('-a', '--airports', dest='airports_file', required=True, help='airports (Excel)')
    parser.add_argument('-f', '--flights', dest='flights_file', required=True, help='flights (JSON)')
    args = parser.parse_args()

    task2(args.map_file, args.airports_file, args.flights_file)
