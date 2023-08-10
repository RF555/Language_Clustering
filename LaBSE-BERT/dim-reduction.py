from UtilityFunctions import *

from sklearn.decomposition import PCA
import sklearn.preprocessing as skscaler

demo_vec = 'demo-vec'
test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

curr_in = test1

_input = 'VECTOR-files/OG-vectors/' + curr_in
_output = 'VECTOR-files/DIM-reduction/' + curr_in


def reductPCA(n_components):
    # preprocessing data (scaling the data)
    my_scaler = skscaler.StandardScaler()
    # my_scaler=Scaler.MinMaxScaler() # another method of scaling

    my_scaler.fit(vector_df)
    scaled_data = my_scaler.transform(vector_df)  # transform the data (transform again later)

    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)  # transform back

    return data_pca


if __name__ == '__main__':
    word_keys, vector_df = pklToDF(_input + '.pkl')

    reducted_data = reductPCA(3)

    plotGraph2D(reducted_data)
    plotGraph3D(reducted_data)
