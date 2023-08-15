import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import plotly.graph_objs


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


# def plotlyGraph2D(curr_data, labels):
def plotlyGraph3D2(curr_data, labels):
    fig = px.scatter_3d(curr_data, x=0, y=1, z=2,
                        color=curr_data['Cluster'],
                        opacity=0.5,
                        hover_name='Cluster'
                        )
    fig.show()


def plotlyGraph3D(curr_data, labels):
    axes_label = dict(xaxis=dict(title='X -->'), yaxis=dict(title='Y -->'),
                      zaxis=dict(title='Z -->'))
    marker = dict(color=curr_data['Cluster'],
                  size=5,
                  colorscale='Viridis',  # change colors
                  line=dict(color='black', width=10))
    trace = go.Scatter3d(x=curr_data[0], y=curr_data[1], z=curr_data[2],
                         mode='markers',
                         marker=marker,
                         # hovertemplate=['<b>{0}</b><br>'
                         #                'Cluster: {1}<br>'
                         #                'x: {2}<br>'
                         #                'y: {3}<br>'
                         #                'z: {4}<br>'.format(w, c, _x, _y, _z)
                         #                for w, c, _x, _y, _z in zip(curr_data['word'],
                         #                                            curr_data['Cluster'],
                         #                                            curr_data[0],
                         #                                            curr_data[1],
                         #                                            curr_data[2])]
                         hovertext=['<b>{0}</b><br>'
                                    'Cluster: {1}'.format(w, c)
                                    for w, c in zip(curr_data['word'],
                                                    curr_data['Cluster'])]
                         )
    data = [trace]
    layout = go.Layout(margin=dict(l=0, r=0, t=0), scene=axes_label, height=800, width=800)
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
