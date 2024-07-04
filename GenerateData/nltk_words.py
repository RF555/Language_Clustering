from nltk.corpus import words
from TransformersUtilityFunctions import get_vec
from UtilityFunctions import pkl
from wordfreq import word_frequency

if __name__ == '__main__':

    # nltk.download('words')  # first time download the word corpus
    _words = words.words()
    # _words = ['hello', 'world', 'hi', 'go', 'bye'] # Example to check if works
    _words_size = str(len(_words))
    print(f'_words_size = {_words_size}')
    word_dict = {}
    freq_dict = {}

    i = 1
    for word in _words:
        word_dict[word] = get_vec(word)
        freq_dict[word] = word_frequency(word, 'en')
        print(f'{i}: {word}')
        i += 1

    first_word = list(word_dict.keys())[0]
    vec_dim = word_dict[first_word].size
    pkl.dump(word_dict, open('pkl_files/nltk(' + _words_size + '_words)BERT_dim' + str(vec_dim) + '_dict.pkl', 'wb'))
    pkl.dump(freq_dict, open('pkl_files/nltk(' + _words_size + '_words)_freq_dict.pkl', 'wb'))
