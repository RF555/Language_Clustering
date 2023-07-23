import io
import torch
from transformers import BertModel, BertTokenizerFast

fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

curr_in = first_par

_input = 'split-output/split-output-' + curr_in

word_dict = {}

# Load Pre-Trained Model
tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()


def check_key(key, val):
    if key not in word_dict:
        word_dict[key] = val
        # print(key, ": ", val)


if __name__ == '__main__':
    fin = io.open(_input, 'r', encoding='utf-8', newline='\n', errors='ignore')
    x = fin.readline()

    while x:
        # Tokenize 'english_sentences' with the BERT tokenizer.
        english_token_ids = tokenizer(x, return_tensors="pt", padding=True)
        english_tokens = tokenizer.convert_ids_to_tokens(english_token_ids['input_ids'][0])

        print(english_tokens)

        for i in range(len(english_tokens)):
            check_key(english_tokens[i], english_token_ids['input_ids'][0][i].item())
        x = fin.readline()

    # print(word_dict)
