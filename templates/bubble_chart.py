import matplotlib.pyplot as plt
import numpy as np

def bubble_chart(ax, N=200):
    # N points in square domain [0, 100] x [0, 100]
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100
    # random radii
    radii = np.random.random(size=N) * 20
    # some values to map to colors
    values = x + y
    # "size" of bubbles is set to square of radius since it controls the area of the circle.
    plot = ax.scatter(x, y, c=values, cmap='viridis', edgecolor='black', s=radii**2, alpha=0.5)
    # Always add a color bar when using colors to represent values
    cb = plt.colorbar(plot)
    cb.set_label('x+y', fontsize='medium')
    ax.set_xlabel('x', fontsize='medium')
    ax.set_ylabel('y', fontsize='medium')
    ax.set_title('Bubble Chart', fontweight='bold', fontsize='large')
    return plot, cb

if __name__ == '__main__':
    fig, ax = plt.subplots()
    bubble_chart(ax)
    plt.show()
