from UtilityFunctions import pd, open_pkl, dump_pkl

# Inputs
from Pathways import full_nltk_PCA_DICT_path, full_nltk_FREQ_DICT_path
# Output
from Pathways import filtered_nltk_PCA_DF_path

word_dict = open_pkl(full_nltk_PCA_DICT_path)
freq_dict = open_pkl(full_nltk_FREQ_DICT_path)

drop_count = 0
filtered_words_df = list()
filtered_vectors_df = pd.DataFrame()
i = 0

for word, vector in word_dict.items():
    wf = freq_dict[word]
    if wf <= 0.0000001:
        drop_count += 1
        print(f'(droped){i}: {word}\t\t\t\tdrop_count = {drop_count}, wf={wf}')
    else:
        print(f'{i}: {word}\t\t\t\tdrop_count = {drop_count}, wf={wf}')
        filtered_words_df.append(word)
        filtered_vectors_df = filtered_vectors_df.append(vector, ignore_index=True)
    i += 1

filtered_words_df = pd.DataFrame(filtered_words_df)
print(filtered_words_df)
print(filtered_vectors_df)

dump_pkl([filtered_words_df, filtered_vectors_df], filtered_nltk_PCA_DF_path)
