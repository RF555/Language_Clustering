import pickle
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import sklearn.preprocessing as Scaler
import matplotlib.pyplot as plt

demo_vec = 'demo-vec'
test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

curr_in = test1

_input = 'VECTOR-files/OG-vectors/' + curr_in
_output = 'VECTOR-files/reduct/' + curr_in


def reductPCA():
    # preprocessing data (scaling the data)
    my_scaler = Scaler.StandardScaler()
    # my_scaler=Scaler.MinMaxScaler() # another method of scaling
    my_scaler.fit(vector_df)
    scaled_data = my_scaler.transform(vector_df)  # transform the data (transform again later)
    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)  # transform back

    return data_pca


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


if __name__ == '__main__':
    with open(_input + '.pkl', "rb") as file:
        loaded_dict = pickle.load(file)
    # print(loaded_dict)
    # print(loaded_dict.keys())
    # print(loaded_dict['hi'][0])

    # word_df = pd.DataFrame()
    word_keys = pd.DataFrame(loaded_dict.keys())

    vector_arr = np.array(loaded_dict[list(loaded_dict.keys())[0]])
    for key in loaded_dict.keys():
        # df = pd.DataFrame(loaded_dict[key])
        # df.insert(loc=0, column='Word', value=key)
        # word_df = pd.concat([word_df, df])

        if key != list(loaded_dict.keys())[0]:
            vector_arr = np.vstack((vector_arr, loaded_dict[key]))

    vector_df = pd.DataFrame(vector_arr)

    # print('\nvector_df\n')
    # print(vector_df)
    # print('\n\n')
    # print(word_keys)

    reducted_data = reductPCA()

    plotGraph2D(reducted_data)
    plotGraph3D(reducted_data)
