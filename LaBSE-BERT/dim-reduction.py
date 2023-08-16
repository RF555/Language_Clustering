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
reduct_to = 3

_input = 'VECTOR-files/BERT-vectors/' + curr_in + '-dim768'
_output = 'VECTOR-files/Reduct-to-' + str(reduct_to) + 'D/' + curr_in


def reduct_pca(n_components):
    # preprocessing data (scaling the data)
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
    word_keys, vector_df = pkl_to_dataframe(_input + '.pkl')

    reducted_data = reduct_pca(reduct_to)

    # plotGraph2D(reducted_data)
    # pyplotGraph3D(reducted_data)

    updated_output = _output + '-dim' + str(reducted_data[0].size)
    dict_to_pkl(word_dict=dataframe_to_dict(word_keys=word_keys,
                                            vector_df=pd.DataFrame(reducted_data)),
                output_path=updated_output)
