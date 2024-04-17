import pickle as pkl
import pandas as pd

input_path = 'VECTOR-files/BERT-vectors/nltk(236736_words-dim768.pkl'
output_path = 'VECTOR-files/BERT-vectors/nltk(236736_words-dim768_df.pkl'

with open(input_path, "rb") as file:
    loaded_dict = pkl.load(file)
# print(loaded_dict.keys())
words_df = pd.DataFrame()
vectors_df = pd.DataFrame()

i = 0
for word, vector in loaded_dict.items():
    temp_vectors_df = pd.DataFrame(vector, index=[i])
    temp_words_df = pd.DataFrame({"words": [word]}, index=[i])
    i += 1
    # print(f'[{i}] {word}:\n{temp_vectors_df}')
    vectors_df = pd.concat([vectors_df, temp_vectors_df])
    words_df = pd.concat([words_df, temp_words_df])
    print(f'[{i}] {word}')

    # if i == 5:
    #     break

# vectors_df.insert(0, 'words', words_df)
# print(vectors_df)
# print(words_df)

vectors_df.to_pickle(output_path)