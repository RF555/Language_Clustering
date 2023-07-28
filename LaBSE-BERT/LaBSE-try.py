import io

import pandas as pd
from transformers import *
import torch
from sklearn.preprocessing import normalize

test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'
old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

curr_in = old_first_par

_input = 'split-output/split-output-' + curr_in
_output = 'CSVs/' + curr_in + '.csv'
word_dict = {}
word_df = pd.DataFrame()

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


if __name__ == '__main__':
    fin = io.open(_input, 'r', encoding='utf-8', newline='\n', errors='ignore')
    sentences = fin.readline()

    while sentences:
        # print(sentence)

        words = sentences.split()

        for w in words:
            if w[-1]=='.':
                w2=w.split('.')[0]
                if w2 not in word_dict:
                    word_dict[w2] = w2
                    df2 = pd.DataFrame(get_vec(w2))
                    df2.insert(loc=0, column='Word', value=w2)
                    word_df = pd.concat([word_df, df2])
                    print(w2,"\t(Without '.')")

            if w not in word_dict:
                word_dict[w] = w
                df = pd.DataFrame(get_vec(w))
                df.insert(loc=0, column='Word', value=w)
                word_df = pd.concat([word_df, df])
                print(w)

        sentences = fin.readline()

    print(word_df)
    word_df.to_csv(_output)
