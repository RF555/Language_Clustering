import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objs as go


def DFtoDict(word_keys, vector_df):
    word_list = list(word_keys['Words'])
    word_dict = dict()

    for i, vec in vector_df.iterrows():
        word_dict[word_list[i]] = vec.to_numpy()

    return word_dict


def pklToDF(input_path: str):
    if not input_path.endswith('.pkl'):
        input_path += '.pkl'
    with open(input_path, "rb") as file:
        loaded_dict = pickle.load(file)

    word_keys = pd.DataFrame(loaded_dict.keys())
    word_keys.rename(columns={0: 'Words'}, inplace=True)
    vector_arr = np.array(loaded_dict[list(loaded_dict.keys())[0]])
    for key in loaded_dict.keys():

        if key != list(loaded_dict.keys())[0]:
            vector_arr = np.vstack((vector_arr, loaded_dict[key]))

    vector_df = pd.DataFrame(vector_arr)

    return word_keys, vector_df


def dictToPkl(word_dict: dict, output_path: str):
    if not output_path.endswith('.pkl'):
        output_path += '.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump(word_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def plotlyGraph3D(curr_data, labels):
    Scene = dict(xaxis=dict(title='X -->'), yaxis=dict(title='Y -->'),
                 zaxis=dict(title='Z -->'))
    trace = go.Scatter3d(x=curr_data[0], y=curr_data[1], z=curr_data[2],
                         mode='markers',
                         marker=dict(color=labels,
                                     size=5,
                                     colorscale='Viridis',  # change colors
                                     line=dict(color='black', width=10))
                         )
    layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=800, width=800)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def pyplotGraph3D(curr_data):
    x = np.array(curr_data[0])
    y = np.array(curr_data[1])
    z = np.array(curr_data[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker="s", c=curr_data['Cluster'], s=40, cmap="Set1")

    plt.show()


def plotGraph2D(curr_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(curr_data[:, 0], curr_data[:, 1])
    plt.title("2D scatter plot")
    plt.xlabel('First principle component', fontweight='bold')
    plt.ylabel('Second principle component', fontweight='bold')

    plt.show()
