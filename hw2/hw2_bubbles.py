# Deliverable and API: hw2_bubbles.py -i <dataset path> [-x X] [-y Y] [-s SIZE] [-c COLOR]
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse

def task1(filename: str, x_attr: str, y_attr: str, size: str, color: str):
    df = pd.read_csv(filename)
    # print(x_attr + ' ' + y_attr + ' ' + size + ' ' + color)
    # print()
    # print(df.head(3))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bubble Chart')
    parser.add_argument('-i', '--input', dest='filename', required=True, help='Path to dataset')
    parser.add_argument('-x', dest='x_attr', required=True, help='X attribute')
    parser.add_argument('-y', dest='y_attr', required=True, help='Y attribute')
    parser.add_argument('-s', dest='size', required=True, help='Size')
    parser.add_argument('-c', dest='color', required=True, help='Color')

    args = parser.parse_args()

    task1(args.filename, args.x_attr, args.y_attr, args.size, args.color)