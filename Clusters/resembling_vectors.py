from ClustersUtilityFunctions import split_full_df, pd, KMeans, pairwise_distances_argmin_min, \
    sentence_to_words_vectors_df, get_vec_from_dict

words_600_df, vectors_600_df = split_full_df(pd.read_csv('OLD_DATA/600_centroid_words_vectors_works.csv'))
print(f'words_600_df:\n{words_600_df}')
print(f'vectors_600_df:\n{vectors_600_df}\n\n')

sentence = 'we go to the cinema tonight'

sentence_words, sentence_vectors = sentence_to_words_vectors_df(sentence)
print(f'sentence_words:\n{sentence_words}')
print(f'sentence_vectors:\n{sentence_vectors}\n\n')

# Index list of vectors (from the 600 words) closest to each vector (from the given sentence)
resembling_vectors_index, _ = pairwise_distances_argmin_min(sentence_vectors, vectors_600_df)
print(f'resembling_vectors_index:\n{resembling_vectors_index}')

words_600_np = words_600_df.to_numpy()
vectors_600_np = vectors_600_df.to_numpy()
resembling_vectors = pd.DataFrame([words_600_np[i] for i in resembling_vectors_index])
print(f'resembling_vectors:\n{resembling_vectors}')
