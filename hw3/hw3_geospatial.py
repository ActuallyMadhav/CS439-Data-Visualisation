# Deliverable and API: hw3_geospatial.py -m <worldmap> -a <airports> -f <flights>
import argparse
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

N_HEAVIEST = 200


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

        n = len(airports)
        self.n_nodes = n
        self.n_edges = len(flights)
        # importance factor = total flights touching an airport
        self.importance = (np.bincount(self.src, self.weight, n)
                           + np.bincount(self.dst, self.weight, n))

    def heaviest_edges(self, k=N_HEAVIEST):
        order = np.argsort(-self.weight, kind='stable')[:k]
        return order[::-1]  # lightest first so the heaviest are drawn on top

    def segments(self, edge_ids):
        return np.stack([self.xy[self.src[edge_ids]], self.xy[self.dst[edge_ids]]], axis=1)


def draw_world(ax, world):
    world.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3, zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


def task1(map_file, airports_file, flights_file):
    world = load_world(map_file)
    graph = FlightGraph(load_airports(airports_file), load_flights(flights_file))

    fig, ax = plt.subplots(figsize=(15, 8))
    draw_world(ax, world)

    edges = graph.heaviest_edges()
    ax.add_collection(LineCollection(graph.segments(edges), colors='blue',
                                     linewidths=1.0, alpha=0.7, zorder=2))

    active = graph.importance > 0
    ax.scatter(graph.lon[active], graph.lat[active], s=4, c='darkred',
               edgecolors='black', linewidths=0.2, zorder=3)

    ax.set_title(f'World Airports and {N_HEAVIEST} Busiest Routes', weight='bold')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geospatial graph visualization')
    parser.add_argument('-m', '--map', dest='map_file', required=True, help='world map (GeoJSON)')
    parser.add_argument('-a', '--airports', dest='airports_file', required=True, help='airports (Excel)')
    parser.add_argument('-f', '--flights', dest='flights_file', required=True, help='flights (JSON)')
    args = parser.parse_args()

    task1(args.map_file, args.airports_file, args.flights_file)
