import pickle
import numpy
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

demo_vec = 'demo-vec'
test1 = 'test1'
fil9 = 'fil9'
first_par = 'wiki-first-paragraph'

old_fil9 = 'fil9-old'
old_first_par = 'wiki-first-paragraph-old'

curr_in = demo_vec

_input = 'VECTOR-files/OG-vectors/' + curr_in
_output = 'VECTOR-files/reduct/' + curr_in


def reductPCA():
    # scaled_data = preprocessing.scale(word_df.T)
    scaled_data = word_df.drop(columns=['Word'])
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    # print(scaled_data)

    # scree graph plotting
    per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
    labels=[x for x in word_df['Word']]

    plt.bar(x=range(1,len(per_var)+1),height=per_var)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    # scatter plot of PCA
    pca_df=pd.DataFrame(pca_data,index=)




if __name__ == '__main__':
    with open(_input + '.pkl', "rb") as file:
        loaded_dict = pickle.load(file)
    # print(loaded_dict)
    # print(loaded_dict.keys())
    # print(loaded_dict['hi'][0])

    word_df = pd.DataFrame()
    for key in loaded_dict.keys():
        df = pd.DataFrame(loaded_dict[key])
        df.insert(loc=0, column='Word', value=key)
        word_df = pd.concat([word_df, df])

    # print(word_df)

    reductPCA()
