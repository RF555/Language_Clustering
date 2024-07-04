from wordfreq import zipf_frequency
import pandas as pd


def get_freq_vectors(words_df, vectors_df):
    vectors_np = vectors_df.to_numpy()
    freqls = []
    freq_vec = []
    i = 0
    for word in words_df:
        wf = zipf_frequency(word, 'en')
        if wf > 0.1:
            freqls.append(word)
            freq_vec.append([vectors_np[i]])
        i = i + 1
    freq_vec = pd.DataFrame(freq_vec)
    return freq_vec
