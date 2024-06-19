import pickle as pkl
import pandas as pd

input_path = 'VECTOR_files/BERT_vectors/nltk(236736_words)_dim768_dict.pkl'
output_path = 'VECTOR_files/BERT_vectors/nltk(236736_words)_dim768_df.pkl'

with open(input_path, "rb") as file:
    loaded_dict = pkl.load(file)
# print(loaded_dict.keys())
full_df = pd.DataFrame()

i = 0
for word, vector in loaded_dict.items():
    temp_vector_df = pd.DataFrame(vector, index=[i])
    temp_word_df = pd.DataFrame({"word": [word]}, index=[i])
    temp_full_df = pd.concat([temp_word_df, temp_vector_df], axis=1)
    i += 1
    # print(f'[{i}] {word}:\n{temp_vectors_df}')
    full_df = pd.concat([full_df, temp_full_df])
    print(f'[{i}] {word}')

    # if i == 5:
    #     print(full_df)
    #     break

# vectors_df.insert(0, 'words', words_df)
# print(vectors_df)
# print(words_df)

full_df.to_pickle(output_path)
