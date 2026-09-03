import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import argparse

# parser for argparse


df = pd.read_excel("evs_assignment1.xlsx")
# print(df.head())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')