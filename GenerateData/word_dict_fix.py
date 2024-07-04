from UtilityFunctions import pkl, np, pd, open_pkl

freq_dict_path = 'GenerateData/pkl_files/nltk(236736_words)_freq_dict.pkl'
word_df_path = 'GenerateData/pkl_files/nltk(236736_words)BERT_dim768_df.pkl'
if __name__ == '__main__':
    freq_dict = open_pkl(freq_dict_path)
    words_df, vectors_df = open_pkl(word_df_path)
    print(words_df[0][0])

    drop_count = 0
    filtered_words_df = list()
    filtered_vectors_df = pd.DataFrame()

    for i, vec in vectors_df.iterrows():
        wf = freq_dict[words_df[i]]
        if wf <= 0.0000001:
            drop_count += 1
            print(f'(droped){i}: {words_df[i]}\t\t\t\tdrop_count = {drop_count}, wf={wf}')
            # words_df.drop([i])
            # vectors_df.drop([i])
        else:
            print(f'{i}: {words_df[i]}\t\t\t\tdrop_count = {drop_count}, wf={wf}')
            filtered_words_df.append(words_df[i])
            filtered_vectors_df = filtered_vectors_df.append(vec, ignore_index=True)

    print(filtered_words_df)
    print(filtered_vectors_df)
