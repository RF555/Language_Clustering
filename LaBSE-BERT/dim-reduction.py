import pickle
import torch

test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

curr_in = test1

_input = 'VECTOR-files/OG-vectors/' + curr_in
_output = 'VECTOR-files/reduct/' + curr_in

if __name__ == '__main__':
    with open(_input + '.pkl', "rb") as file:
        loaded_dict = pickle.load(file)
    print(loaded_dict)
