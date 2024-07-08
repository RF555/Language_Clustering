from UtilityFunctions import pd, open_pkl, dump_pkl

# Inputs
from Pathways import filtered_600_DF_path, full_600_DF_path, \
    full_nltk_PCA_DICT_path

from sklearn.metrics import pairwise_distances_argmin_min

words_600_df, vectors_600_df = open_pkl(filtered_600_DF_path)
print(f'words_600_df:\n{words_600_df}')
print(f'vectors_600_df:\n{vectors_600_df}\n\n')

word_dict = open_pkl(full_nltk_PCA_DICT_path)

sentence1 = 'we go to the cinema tonight'
sentence2 = 'yesterday i went to eat with a friend'

curr_sentence = sentence1

sentence_words = list()
sentence_vectors = pd.DataFrame()

for word in curr_sentence.split():
    sentence_words.append(word)
    vector = pd.DataFrame([word_dict[word]])
    # print(f'vector:\n\t{vector}')
    sentence_vectors = pd.concat([sentence_vectors, vector], ignore_index=True)
print(f'sentence_words:\n{sentence_words}')
print(f'sentence_vectors:\n{sentence_vectors}\n\n')

# Index list of vectors (from the 600 words) closest to each vector (from the given sentence)
resembling_vectors_index, _ = pairwise_distances_argmin_min(sentence_vectors, vectors_600_df)
print(f'resembling_vectors_index:\n{resembling_vectors_index}')

words_600_np = words_600_df.to_numpy()
vectors_600_np = vectors_600_df.to_numpy()
resembling_vectors = pd.DataFrame([words_600_np[i] for i in resembling_vectors_index])
print(f'resembling_vectors:\n{resembling_vectors}')
