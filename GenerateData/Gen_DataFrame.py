from UtilityFunctions import pd, open_pkl

freq_dict_path = 'GenerateData/pkl_files/nltk(236736_words)_freq_dict.pkl'
word_dict_path = 'GenerateData/pkl_files/nltk(236736_words)BERT_dim768_dict.pkl'
word_dict_path2 = 'GenerateData/pkl_files/nltk(235892_words)BERT_dim768_dict2.pkl'

if __name__ == '__main__':
    word_dict = open_pkl(word_dict_path)
    for word in word_dict:
        word_dict[word] = word_dict[word][0]

    full_word_df = pd.DataFrame.from_dict(word_dict, orient='index')

    words_df = pd.DataFrame(full_word_df.index)[0]
    vectors_df = full_word_df
    vectors_df.reset_index(drop=True, inplace=True)

    pd.to_pickle([words_df, vectors_df], 'pkl_files/nltk(236736_words)BERT_dim768_df.pkl')

    freq_dict = open_pkl(freq_dict_path)
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

    filtered_words_df = pd.DataFrame(filtered_words_df)
    print(filtered_words_df)
    print(filtered_vectors_df)

    # pd.to_pickle([words_df, vectors_df], 'pkl_files/nltk(filtered)BERT_dim768_df.pkl')
    pd.to_pickle([filtered_words_df, filtered_vectors_df], 'pkl_files/nltk(filtered)BERT_dim768_df.pkl')
