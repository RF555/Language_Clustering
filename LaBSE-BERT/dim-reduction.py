from UtilityFunctions import *

from sklearn.decomposition import PCA
import sklearn.preprocessing as skscaler

demo_vec = 'demo-vec'
test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

# Choose the current file to use as input
curr_in = test1
reduct_dim_to = 3

# Path to the current file used as input/output
_input = 'VECTOR-files/BERT-vectors/' + curr_in + '-dim768'
_output = 'VECTOR-files/Reduct-to-' + str(reduct_dim_to) + 'D/' + curr_in


def reduct_pca(n_components):
    """
    Reduce the dimensionality of the voctors using the PCA algorithm.
    @param n_components: Reduce the dimensionality of the vector to.
    @return: The data with reduced dimensionality.
    """
    # Preprocessing data (scaling the data)
    my_scaler = skscaler.StandardScaler()
    # my_scaler=Scaler.MinMaxScaler() # another method of scaling

    my_scaler.fit(vector_df)
    scaled_data = my_scaler.transform(vector_df)  # transform the data (transform again later)

    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)  # transform back

    global _output
    _output += '(PCA)'

    return data_pca


if __name__ == '__main__':
    if not _input.endswith('.pkl'):
        _input += '.pkl'
    word_keys, vector_df = pkl_to_dataframe(_input)

    # Reduce the dimension of the original vectors
    reducted_data = reduct_pca(reduct_dim_to)

    # pyplot_graph_2d_scatter(data=reducted_data,
    #                         x_label='First principle component',
    #                         y_label='Second principle component')
    # pyplot_graph_3d(reducted_data)

    updated_output = _output + '-dim' + str(reducted_data[0].size)
    dict_to_pkl(word_dict=dataframe_to_dict(word_keys=word_keys, vector_df=pd.DataFrame(reducted_data)),
                output_path=updated_output)
