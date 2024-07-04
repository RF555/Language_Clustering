import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'


def open_pkl(path):
    return pkl.load(open(ROOT_DIR + path, 'rb'))
