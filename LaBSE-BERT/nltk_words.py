import nltk
from nltk.corpus import words
from LaBSE_try import get_vec, add_to_word_dict, word_dict
from UtilityFunctions import dict_to_pkl

_output = 'VECTOR-files/BERT-vectors/nltk'
# word_dict = {}

if __name__ == '__main__':

    # nltk.download('words')  # first time download the word corpus
    _words = words.words()
    _words_size = len(_words)
    print(f'_words_size = {_words_size}')

    for word in _words:
        add_to_word_dict(word)

    first_word = list(word_dict.keys())[0]
    vec_dim = word_dict[first_word].size
    updated_output = _output + '(' + str(_words_size) + '_words-dim' + str(vec_dim)
    dict_to_pkl(word_dict=word_dict, output_path=updated_output)
