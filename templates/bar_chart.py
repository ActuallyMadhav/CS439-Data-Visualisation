import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def bar_chart(ax):
    xkcd = mcolors.XKCD_COLORS
    fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
    counts = [5, 3, 4, 2, 4, 6]
    colors = [ 
            xkcd['xkcd:apple green'], 
            xkcd['xkcd:pear'], 
            xkcd['xkcd:orangey yellow'], 
            xkcd['xkcd:plum'], 
            xkcd['xkcd:grape'], 
            xkcd['xkcd:strawberry']
        ]
    ax.bar(fruits, counts, width=0.9, color=colors)
    ax.set_ylabel('Fruit count')
    ax.set_title('Bar Chart', fontweight='bold', fontsize='large')
    ax.tick_params(axis='x', which='both', labelsize=7)
    return ax

if __name__ == '__main__':
    fig, ax = plt.subplots()
    bar_chart(ax)
    plt.show()
    