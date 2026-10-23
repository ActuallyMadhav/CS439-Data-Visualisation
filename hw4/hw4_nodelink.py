import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# tree.py uses Tree before defining it
_b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
_b['Tree'] = object
from tree import Tree  # noqa: E402
from draw import buchheim  # noqa: E402
del _b['Tree']

LEVEL_SEP = 4.0
FONT = 9
COLORS = {'internal': 'blue', 'folded': 'red', 'leaf': 'black'}


def node_kind(n):
    if n.is_folded():
        return 'folded'
    if len(n.children) > 0:
        return 'internal'
    return 'leaf'


class NodeLink:
    def __init__(self, root, fig, ax):
        self.root = root
        self.fig = fig
        self.ax = ax
        self.last_click = None
        fig.canvas.mpl_connect('pick_event', self.on_pick)

    def redraw(self):
        buchheim(self.root)
        self.nodes = list(self.root.iterator())
        # vertical layout -> horizontal, stretched
        for n in self.nodes:
            n.px = n.y * LEVEL_SEP
            n.py = n.x

        ax = self.ax
        ax.clear()
        ax.set_axis_off()

        edges = [[(n.px, n.py), (c.px, c.py)]
                 for n in self.nodes for c in n.children]
        ax.add_collection(LineCollection(edges, colors='black',
                                         linewidths=2, zorder=1))

        self.scatter = ax.scatter([n.px for n in self.nodes],
                                  [n.py for n in self.nodes],
                                  c=[COLORS[node_kind(n)] for n in self.nodes],
                                  s=30, zorder=3, picker=True, pickradius=5)

        self.text_node = {}
        for n in self.nodes:
            kind = node_kind(n)
            if kind == 'internal':
                # expanded nodes: label left of node
                t = ax.text(n.px - 0.15, n.py, n.name, ha='right',
                            va='center', fontsize=FONT, zorder=4,
                            picker=True,
                            bbox=dict(fc='white', ec='none', pad=0.5))
            else:
                t = ax.text(n.px + 0.15, n.py, n.name, ha='left',
                            va='center', fontsize=FONT, zorder=4,
                            picker=True)
            self.text_node[t] = n

        xs = [n.px for n in self.nodes]
        ys = [n.py for n in self.nodes]
        ax.set_xlim(min(xs) - LEVEL_SEP, max(xs) + LEVEL_SEP)
        ax.set_ylim(min(ys) - 1, max(ys) + 1)
        self.fig.canvas.draw_idle()

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

    fig, ax = plt.subplots(figsize=(12, 15))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    # keep a reference so clicks work
    view = NodeLink(tree, fig, ax)
    view.redraw()
    plt.show()


if __name__ == '__main__':
    main()
