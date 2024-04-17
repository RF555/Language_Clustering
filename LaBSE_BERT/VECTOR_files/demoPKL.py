import pickle
import numpy

demo_vec = 'demo-vec'

curr_in = demo_vec

_output = 'BERT_vectors/' + curr_in

if __name__ == '__main__':
    word_dict = {}
    hi_vec = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    how_vec = numpy.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    are_vec = numpy.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    you_vec = numpy.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
    word_dict['hi'] = hi_vec
    word_dict['how'] = how_vec
    word_dict['are'] = are_vec
    word_dict['you'] = you_vec

    with open(_output + '.pkl', 'wb') as file:
        pickle.dump(word_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
