from wordfreq import word_frequency
from UtilityFunctions import split_full_df
import pandas as pd
import pickle as pkl

nltk_df = 'VECTOR_files/BERT_vectors/nltk(236736_words)_dim768_df.pkl'
_output = 'VECTOR_files/BERT_vectors/nltk_dim768_df(filtered2).pkl'

# def get_freq_vectors(words_df, vectors_df):
#     vectors_np = vectors_df.to_numpy()
#     freqls = []
#     freq_vec = []
#     i = 0
#     for word in words_df:
#         wf = zipf_frequency(word, 'en')
#         if wf > 0.1:
#             freqls.append(word)
#             freq_vec.append([vectors_np[i]])
#         i = i + 1
#     freq_vec = pd.DataFrame(freq_vec)
#     return freq_vec


# Import DataFrames
word_df, vector_df = split_full_df(pd.read_pickle(nltk_df))
drop_count = 0
for i, vec in vector_df.iterrows():
    wf = word_frequency(word_df[i], 'en')
    if wf <= 0.0000001:
        drop_count += 1
        print(f'(droped){i}: {word_df[i]}\t\t\t\tdrop_count = {drop_count}, wf={wf}')
        word_df.drop([i])
        vector_df.drop([i])
    else:
        print(f'{i}: {word_df[i]}\t\t\t\tdrop_count = {drop_count}, wf={wf}')

pkl.dump([word_df, vector_df], open(_output, "wb"))
