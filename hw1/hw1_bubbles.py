import argparse
import pandas as pd
import matplotlib.pyplot as plt

def task2(filename: str, x_attr: str, y_attr: str, color_attr: str, size_attr: str):
    df = pd.read_excel(filename)
    # print(df.head())
    max_val = df[size_attr].max()
    scaled_size = (df[size_attr] / max_val) * 2500

    fig, chart = plt.subplots(figsize=(10, 7))
    
    # Create bubble chart
    scatter = chart.scatter(
        df[x_attr], 
        df[y_attr], 
        s=scaled_size, 
        c=df[color_attr], 
        cmap='viridis', 
        alpha=0.75, 
        edgecolors='black', 
        linewidth=1
    )

    # labels
    chart.set_xlabel(x_attr)
    chart.set_ylabel(y_attr)
    
    cbar = plt.colorbar(scatter, ax=chart)
    cbar.set_label(color_attr)
    
    # Display
    # Replace your current plt.title(...) with this:
    chart.set_title(f'Bubble Chart Representation of {x_attr}, {y_attr}, {size_attr}, {color_attr}', weight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #argparse
    parser = argparse.ArgumentParser(description='Bubble Chart')
    
    parser.add_argument('-i', '--input', dest='filename', required=True, help='Path to the data file')
    parser.add_argument('-x', dest='x_attr', required=True, help='x-axis attribute')
    parser.add_argument('-y', dest='y_attr', required=True, help='y-axis attribute')
    parser.add_argument('-c', dest='color_attr', required=True, help='color scale attribute')
    parser.add_argument('-s', dest='size_attr', required=True, help='bubble size attribute')
    
    args = parser.parse_args()
    
    task2(args.filename, args.x_attr, args.y_attr, args.color_attr, args.size_attr)