import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# tree.py uses Tree before defining it
_b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
_b['Tree'] = object
from tree import Tree  # noqa: E402
from draw import buchheim  # noqa: E402
del _b['Tree']

FONT = 8
COLORS = {'internal': 'blue', 'folded': 'red', 'leaf': 'black'}


def node_kind(n):
    if n.is_folded():
        return 'folded'
    if len(n.children) > 0:
        return 'internal'
    return 'leaf'


class RadialNodeLink:
    def __init__(self, root, fig, ax):
        self.root = root
        self.fig = fig
        self.ax = ax
        self.last_click = None
        fig.canvas.mpl_connect('pick_event', self.on_pick)

    def layout(self):
        buchheim(self.root)
        self.nodes = list(self.root.iterator())
        xs = [n.x for n in self.nodes]
        # +1 keeps first and last apart
        xmin, span = min(xs), max(xs) - min(xs) + 1.0
        # x -> angle, depth -> radius
        for n in self.nodes:
            n.theta = 2 * np.pi * (n.x - xmin) / span
            n.radius = n.y
            n.px = n.radius * np.cos(n.theta)
            n.py = n.radius * np.sin(n.theta)

    def redraw(self):
        self.layout()
        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        ax.set_aspect('equal')

        edges = [[(n.px, n.py), (c.px, c.py)]
                 for n in self.nodes for c in n.children]
        ax.add_collection(LineCollection(edges, colors='black',
                                         linewidths=1.5, zorder=1))

        self.scatter = ax.scatter([n.px for n in self.nodes],
                                  [n.py for n in self.nodes],
                                  c=[COLORS[node_kind(n)] for n in self.nodes],
                                  s=25, zorder=3, picker=True, pickradius=5)

        self.text_node = {}
        for n in self.nodes:
            self.text_node[self.make_label(n)] = n

        lim = max(n.radius for n in self.nodes) + 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        self.fig.canvas.draw_idle()

    def make_label(self, n):
        if n.is_root():
            return self.ax.text(n.px, n.py, n.name, ha='right', va='center',
                                fontsize=FONT + 2, zorder=4, picker=True,
                                bbox=dict(fc='white', ec='none', pad=0.5))
        deg = np.degrees(n.theta) % 360
        # flip left half so text is upright
        flip = 90 < deg < 270
        rotation = deg - 180 if flip else deg
        # expanded nodes: label toward the root
        inward = node_kind(n) == 'internal'
        step = -0.08 if inward else 0.08
        r = n.radius + step
        if inward:
            ha = 'left' if flip else 'right'
        else:
            ha = 'right' if flip else 'left'
        return self.ax.text(r * np.cos(n.theta), r * np.sin(n.theta), n.name,
                            rotation=rotation, rotation_mode='anchor',
                            ha=ha, va='center', fontsize=FONT, zorder=4,
                            picker=True)

    def on_pick(self, event):
        # one click may pick dot and label
        if event.mouseevent is self.last_click:
            return
        self.last_click = event.mouseevent
        if event.artist is self.scatter:
            node = self.nodes[event.ind[0]]
        else:
            node = self.text_node.get(event.artist)
        if node is None or node.is_leaf():
            return
        node.toggle_fold()
        self.redraw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        tree = Tree(data=json.load(f))
    # show first three levels
    tree.set_max_depth(2)

    fig, ax = plt.subplots(figsize=(13, 13))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    # keep a reference so clicks work
    view = RadialNodeLink(tree, fig, ax)
    view.redraw()
    plt.show()


if __name__ == '__main__':
    main()
