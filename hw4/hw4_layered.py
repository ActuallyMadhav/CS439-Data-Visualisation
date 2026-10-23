import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


_b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
_b['Tree'] = object
from tree import Tree 
del _b['Tree']

MAX_FONT = 14
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

    fig, ax = plt.subplots(figsize=(12, 13))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, depth + 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # points per unit of height
    ppu = ax.get_window_extent().height * 72 / fig.dpi
    # depth d is the column [d, d+1]
    for node, d, s, e, col in cells:
        rect = Rectangle((d, s), 1.0, e - s, facecolor=col,
                         edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        fs = min(MAX_FONT, 0.7 * (e - s) * ppu)
        t = ax.text(d + 0.01, e, node.name, ha='left', va='top', fontsize=fs)
        t.set_clip_path(rect)

    plt.show()


if __name__ == '__main__':
    main()
