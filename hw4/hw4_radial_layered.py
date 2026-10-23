import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

_b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
_b['Tree'] = object
from tree import Tree 
del _b['Tree']

MAX_FONT = 10
CHAR_W = 0.6
ROOT_COLOR = '#dcdcdc'


def compute_weights(node):
    # directory weight = sum of its files
    if len(node.children) == 0:
        node.weight = float(node.value)
    else:
        node.weight = sum(compute_weights(c) for c in node.children)
    return node.weight


def partition(node, start, end, depth, maxdepth, color, out):
    # split [start, end) among children by weight
    out.append((node, depth, start, end, color))
    if depth >= maxdepth or node.weight <= 0:
        return
    cmap = plt.get_cmap('tab20c')
    k = len(node.children)
    pos = start
    for i, c in enumerate(node.children):
        size = (end - start) * c.weight / node.weight
        # one color per main branch
        col = cmap((k - 1 - i) % 20) if depth == 0 else color
        partition(c, pos, pos + size, depth + 1, maxdepth, col, out)
        pos += size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('--maxdepth', type=int, default=-1)
    args = parser.parse_args()

    with open(args.input) as f:
        tree = Tree(data=json.load(f))
    compute_weights(tree)
    # negative or missing means whole tree
    maxdepth = args.maxdepth if args.maxdepth >= 0 else 10 ** 9

    cells = []
    partition(tree, 0.0, 1.0, 0, maxdepth, ROOT_COLOR, cells)
    depth = max(c[1] for c in cells)

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    lim = depth + 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # points per data unit
    ppu = ax.get_window_extent().width / (2 * lim) * 72 / fig.dpi
    # fraction -> angle, depth d -> ring [d, d+1]
    for node, d, s, e, col in cells:
        if d == 0:
            ax.add_patch(Wedge((0, 0), 1, 0, 360, facecolor=col,
                               edgecolor='black', linewidth=1))
            ax.text(0, 0, node.name, ha='center', va='center', fontsize=12)
            continue
        ax.add_patch(Wedge((0, 0), d + 1, 360 * s, 360 * e, width=1,
                           facecolor=col, edgecolor='black', linewidth=1))
        mid = 2 * np.pi * (s + e) / 2
        deg = np.degrees(mid)
        # flip left half so text is upright
        rotation = deg - 180 if 90 < deg < 270 else deg
        # fit text in wedge width and ring
        arc = (d + 0.5) * 2 * np.pi * (e - s)
        fs = min(MAX_FONT, 0.7 * arc * ppu,
                 0.9 * ppu / (CHAR_W * len(node.name)))
        r = d + 0.5
        ax.text(r * np.cos(mid), r * np.sin(mid), node.name,
                rotation=rotation, rotation_mode='anchor',
                ha='center', va='center', fontsize=fs)

    plt.show()


if __name__ == '__main__':
    main()
