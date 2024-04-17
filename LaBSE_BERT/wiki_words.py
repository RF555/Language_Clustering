import io
from LaBSE_try import add_to_word_dict, word_dict, check_sign
from UtilityFunctions import dict_to_pkl

test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

# Choose the current file to use as input
curr_in = test1

# Path to the current file used as input/output
_input = 'split_output/split_output-' + curr_in
_output = 'VECTOR_files/BERT_vectors/' + curr_in

if __name__ == '__main__':
    # Open the relevant file (text file already split to rows by sentences)
    f_in = io.open(_input, 'r', encoding='utf-8', newline='\n', errors='ignore')

    # Read the first sentence
    sentences = f_in.readline()

    # Add all signs to the dictionary
    signs = "./?!\"\':;-(),"
    for sign in signs:
        add_to_word_dict(sign)

    # Loop over all sentences
    while sentences:
        # print(sentence)

        # Split sentence to words by ' '
        words = sentences.split()

        # Loop over all words in sentence
        for w in words:
            check_sign(w)

        # Read the next sentence
        sentences = f_in.readline()

    # print(word_df)
    # print(word_dict)

    first_word = list(word_dict.keys())[0]
    vec_dim = word_dict[first_word].size
    updated_output = _output + '_dim' + str(vec_dim)
    dict_to_pkl(word_dict=word_dict, output_path=updated_output)
