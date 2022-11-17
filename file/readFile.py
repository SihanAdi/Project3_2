import numpy as np
import pandas as pd


def read_file(path):
    data = pd.read_csv(path)
    return data
