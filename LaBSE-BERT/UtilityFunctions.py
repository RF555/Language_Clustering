import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pklToDF(path: str):
    with open(path, "rb") as file:
        loaded_dict = pickle.load(file)

    word_keys = pd.DataFrame(loaded_dict.keys())
    vector_arr = np.array(loaded_dict[list(loaded_dict.keys())[0]])
    for key in loaded_dict.keys():

        if key != list(loaded_dict.keys())[0]:
            vector_arr = np.vstack((vector_arr, loaded_dict[key]))

    vector_df = pd.DataFrame(vector_arr)

    return word_keys, vector_df


def plotGraph3D(curr_data):
    # Creating figure
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(curr_data[:, 0], curr_data[:, 1], curr_data[:, 2], color="green")
    plt.title("3D scatter plot")
    ax.set_xlabel('First principle component', fontweight='bold')
    ax.set_ylabel('Second principle component', fontweight='bold')
    ax.set_zlabel('Third principle component', fontweight='bold')

    # show plot
    plt.show()


def plotGraph2D(curr_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(curr_data[:, 0], curr_data[:, 1])
    plt.title("2D scatter plot")
    plt.xlabel('First principle component', fontweight='bold')
    plt.ylabel('Second principle component', fontweight='bold')

    plt.show()
