import numpy as np
import matplotlib.pyplot as plt
import bar_chart as bar
import line_chart as line
import templates.bubble_chart as bubble

def multiple_charts(axs, N=200, dt=0.01):
    bar.bar_chart(axs[0])
    line.line_chart(axs[1], dt=dt)
    bubble.bubble_chart(axs[2], N=N)
    return axs

if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, figsize=(12,4), layout='constrained')
    multiple_charts(axs)
    plt.suptitle('Multiple Charts', fontweight='bold', fontsize='x-large')
    plt.show()
    