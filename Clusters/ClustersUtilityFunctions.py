from LaBSE_BERT.UtilityFunctions import split_full_df, pd, plotly_graph_2d, pkl_to_dataframe, plotly_graph_3d_clusters
# from LaBSE_BERT.LaBSE_try import get_vec

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import os

cur_path = os.path.dirname(__file__)

nltk_df = 'VECTOR_files/BERT_vectors/nltk(236736_words)_dim768_df.pkl'
nltk_dict = 'VECTOR_files/BERT_vectors/nltk(236736_words)_dim768_dict.pkl'

nltk_df_path = os.path.relpath('..\LaBSE_BERT\\' + nltk_df, cur_path)
nltk_dict_path = os.path.relpath('..\LaBSE_BERT\\' + nltk_dict, cur_path)


def get_words_vectors_df(df_path):
    full_df = pd.read_pickle(df_path)
    return split_full_df(full_df)


def get_vec_from_dict(word: str):
    full_dict = pd.read_pickle(nltk_dict_path)
    return full_dict[word]


def sentence_to_words_vectors_df(sentence: str):
    words_list = sentence.split(' ')
    words_df = pd.DataFrame()
    vectors_df = pd.DataFrame()
    full_dict = pd.read_pickle(nltk_dict_path)
    i = 0
    for word in words_list:
        temp_word_df = pd.DataFrame({"word": [word]}, index=[i])
        temp_vector_df = pd.DataFrame(full_dict[word], index=[i])

        # if word in full_dict:
        #     temp_vector_df = pd.DataFrame(full_dict[word], index=[i])
        # else:
        #     temp_vector_df = pd.DataFrame(get_vec(word), index=[i])

        i += 1
        vectors_df = pd.concat([vectors_df, temp_vector_df])
        words_df = pd.concat([words_df, temp_word_df])
    return words_df, vectors_df
