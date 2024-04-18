import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def split_full_df(og_df: pd.DataFrame):
    words_df: pd.DataFrame = og_df.iloc[:, 0]
    vectors_df: pd.DataFrame = og_df.iloc[:, 1:]
    return words_df, vectors_df


def dataframe_to_dict(word_keys, vector_df):
    word_list = list(word_keys['Word'])
    word_dict = dict()

    for i, vec in vector_df.iterrows():
        word_dict[word_list[i]] = vec.to_numpy()

    return word_dict


def pkl_to_dataframe(input_path: str):
    if not input_path.endswith('.pkl'):
        input_path += '.pkl'
    with open(input_path, "rb") as file:
        loaded_dict = pickle.load(file)

    word_keys = pd.DataFrame(loaded_dict.keys())
    word_keys.rename(columns={0: 'word'}, inplace=True)
    vector_arr = np.array(loaded_dict[list(loaded_dict.keys())[0]])

    for key in loaded_dict.keys():

        if key != list(loaded_dict.keys())[0]:
            # print(f'[{len(vector_arr)}] {key}')
            vector_arr = np.vstack((vector_arr, loaded_dict[key]))

    vector_df = pd.DataFrame(vector_arr)

    return word_keys, vector_df


def dict_to_pkl(word_dict: dict, output_path: str):
    if not output_path.endswith('.pkl'):
        output_path += '.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump(word_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def plotly_graph_2d(data, x_label='X -->', y_label='Y -->'):
    trace = go.Scatter(x=data[0], y=data[1],
                       mode='lines+markers',
                       hovertext=['Num of Clusters: {0}<br>'
                                  'WCSS: {1}'.format(clusters, wcss)
                                  for clusters, wcss in zip(data[0],
                                                            data[1])]
                       )
    fig = go.Figure(data=[trace])
    fig.update_layout(xaxis_title=x_label,
                      yaxis_title=y_label)

    fig.show()


def plotly_graph_2d_scatter(data=None, subplot=None, x_label='X -->', y_label='Y -->', cluster=False):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if data is not None:
        marker = dict()
        if cluster:
            marker = dict(color=data['Cluster'],
                          size=7,
                          colorscale='Viridis'  # change colors
                          )
        if 'word' in data:
            hovertext = ['<b>{0}</b><br>'
                         'Cluster: {1}'.format(w, c)
                         for w, c in zip(data['word'],
                                         data['Cluster'])]
        else:
            hovertext = ['<b>{0}</b><br>'
                         'Cluster: {0}'.format(c)
                         for c in data['Cluster']]

        trace1 = go.Scatter(x=data[0], y=data[1],
                            mode='markers',
                            marker=marker,
                            hovertext=hovertext
                            )

        fig.add_trace(trace1)

    if subplot is not None:
        marker2 = dict(size=10)
        for i, (discription, subplot_i) in enumerate(subplot.items()):
            trace2 = go.Scatter(x=subplot_i[0], y=subplot_i[1],
                                mode='markers',
                                marker=marker2,
                                hovertext='<b>' + discription + '</b>')
            fig.add_trace(trace2)

    fig.update_layout(xaxis_title=x_label,
                      yaxis_title=y_label)

    fig.show()


def plotly_graph_3d_clusters(data):
    axes_label = dict(xaxis=dict(title='X -->'), yaxis=dict(title='Y -->'),
                      zaxis=dict(title='Z -->'))
    marker = dict(color=data['Cluster'],
                  size=5,
                  colorscale='Viridis',  # change colors
                  line=dict(color='black', width=10))
    trace = go.Scatter3d(x=data[0], y=data[1], z=data[2],
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
                                    for w, c in zip(data['word'],
                                                    data['Cluster'])]
                         )
    layout = go.Layout(margin=dict(b=0, t=0), scene=axes_label)
    fig = go.Figure(data=[trace], layout=layout)

    fig.show()


def pyplot_graph_3d(curr_data):
    x = np.array(curr_data[0])
    y = np.array(curr_data[1])
    z = np.array(curr_data[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker="s", c=curr_data['Cluster'], s=40, cmap="Set1")

    plt.show()


def pyplot_graph_2d(data, x_label, y_label, title='2D Graph Plot'):
    plt.plot(data[0], data[1])
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squers')

    plt.show()


def pyplot_graph_2d_scatter(data, x_label, y_label, title='2D Scatter Plot'):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')

    plt.show()
