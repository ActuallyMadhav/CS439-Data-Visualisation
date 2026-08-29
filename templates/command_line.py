import matplotlib.pyplot as plt
import numpy as np
import argparse as ap

def bubble_chart(ax, N, max_size=20.0, colormap='viridis', edgecolor='black'):
    # N points in square domain [0, 100] x [0, 100]
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100
    # random radii
    radii = np.random.random(size=N) * max_size
    # some values to map to colors
    values = x + y
    # "size" of bubbles is set to square of radius since it controls the area of the circle.
    plot = ax.scatter(x, y, c=values, cmap=colormap, edgecolor=edgecolor, s=radii**2, alpha=0.5)
    # Always add a color bar when using colors to represent values
    cb = plt.colorbar(plot)
    cb.set_label('x+y', fontsize='medium')
    ax.set_xlabel('x', fontsize='medium')
    ax.set_ylabel('y', fontsize='medium')
    ax.set_title('Bubble Chart', fontweight='bold', fontsize='large')
    return plot, cb

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Visualize tabular data as bubble chart')
    parser.add_argument('-n', '--number', type=int, required=True, help='Number of points')
    parser.add_argument('-s', '--size', type=float, default=20.0, help='Maximum radius of bubbles')
    parser.add_argument('-c', '--color', type=str, default='viridis', help='Colormap to use')
    parser.add_argument('-e', '--edge', type=str, default='black', help='Edge color')
    parser.add_argument('-r', '--random', type=float, nargs=3, default=[0., 100., 100.], help='Random triplet of floats')
    args = parser.parse_args()
    fig, ax = plt.subplots()
    bubble_chart(ax, N=args.number, max_size=args.size, colormap=args.color, edgecolor=args.edge)
    print(f'The entered numbers were {args.random}')
    plt.show()
