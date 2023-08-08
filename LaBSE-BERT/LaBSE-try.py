import io

import pandas as pd
from transformers import *
import torch
from sklearn.preprocessing import normalize

import pickle

test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

curr_in = old_first_par

_input = 'split-output/split-output-' + curr_in
_output = 'VECTOR-files/' + curr_in
word_dict = {}
# word_df = pd.DataFrame()

# Load Pre-Trained Model
tokenizer = AutoTokenizer.from_pretrained("setu4993/LaBSE", do_lower_case=False)
model = AutoModel.from_pretrained("setu4993/LaBSE")


def get_vec(word):
    # max_seq_length = 64
    english_sentences = [word]  # ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]

    tok_res = tokenizer(english_sentences, add_special_tokens=True,
                        padding='max_length')  # , max_length=max_seq_length)
    input_ids = tok_res['input_ids']
    token_type_ids = tok_res['token_type_ids']
    attention_mask = tok_res['attention_mask']

    device = torch.device('cpu')
    # device=torch.device('cuda') # uncomment if using gpu
    input_ids = torch.tensor(input_ids).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    curr_model = model.to(device)

    output = curr_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    english_embeddings = output[1].cpu().detach().numpy()
    english_embeddings = normalize(english_embeddings)

    return english_embeddings


def add_to_word_dict(word, pre_sign='none', post_sign='none'):
    if word not in word_dict:
        word_dict[word] = get_vec(word)
        # df = pd.DataFrame(get_vec(word))
        # df.insert(loc=0, column='Word', value=word)
        # global word_df
        # word_df = pd.concat([word_df, df])
        if pre_sign != 'none' and post_sign != 'none':
            print(word, "\t(Without surrounding '", pre_sign, "')")
        elif pre_sign != 'none':
            print(word, "\t(Without pre '", pre_sign, "')")
        elif post_sign != 'none':
            print(word, "\t(Without post '", post_sign, "')")
        else:
            print(word)


def check_sign(word):
    add_to_word_dict(word=word)

    if len(word) > 1:
        if (word[0] == '\"' or word[0] == '\'') and word[0] == word[-1]:
            surrounded_word = word.split(word[0])[1]
            add_to_word_dict(word=surrounded_word)
            check_pre_sign(surrounded_word)
            check_post_sign(surrounded_word)
        elif word[0] == '(' and word[-1] == ')':
            surrounded_word = word.split(word[0])[1]
            surrounded_word = surrounded_word.split(word[-1])[0]
            add_to_word_dict(word=surrounded_word)
            check_pre_sign(surrounded_word)
            check_post_sign(surrounded_word)
        else:
            check_pre_sign(word)
            check_post_sign(word)


def check_pre_sign(word):
    # check prefix signs
    if word[0] == '(' or word[0] == '\"' or word[0] == '\'':
        w_pre = word.split(word[0])[1]
        check_sign(w_pre)
        if word[0] == word[-1] or (word[0] == '(' and word[-1] == ')'):  # check surround sign
            add_to_word_dict(word=w_pre, pre_sign=word[0], post_sign=word[-1])
        else:
            add_to_word_dict(word=w_pre, pre_sign=word[0])


def check_post_sign(word):
    # check postfix signs
    if word[-1] == '.' or word[-1] == '!' or word[-1] == '?' or word[-1] == ',' or word[-1] == ';' or word[-1] == ':' \
            or word[-1] == ')' or word[-1] == '\'' or word[-1] == '\"':
        w_post = word.split(word[-1])[0]
        check_sign(w_post)
        add_to_word_dict(word=w_post, post_sign=word[-1])


if __name__ == '__main__':
    fin = io.open(_input, 'r', encoding='utf-8', newline='\n', errors='ignore')
    sentences = fin.readline()

    # add all signs
    signs = "./?!\"\':;-(),"
    for sign in signs:
        add_to_word_dict(sign)

    while sentences:
        # print(sentence)

        words = sentences.split()

        for w in words:
            check_sign(w)

        sentences = fin.readline()

    # print(word_df)
    # print(word_dict)

    with open(_output + '.pkl', 'wb') as file:
        pickle.dump(word_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    # word_df.to_csv(_output+'.csv')
