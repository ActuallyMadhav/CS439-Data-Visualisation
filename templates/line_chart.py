import matplotlib.pyplot as plt
import numpy as np

def line_chart(ax, dt=0.01):
    t = np.arange(-5.0*np.pi, 5.0*np.pi, dt)
    s = np.sin(t)/t
    plot = ax.plot(t, s)
    ax.set(
        xlabel='x',
        ylabel='sin(x)/x')
    ax.set_title('Line Chart', fontweight='bold', fontsize='large')
    return plot

if __name__ == '__main__':
    fig, ax = plt.subplots()
    line_chart(ax)
    plt.show()