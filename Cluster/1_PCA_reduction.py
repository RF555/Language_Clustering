from UtilityFunctions import pd, open_pkl, dump_pkl
from Pathways import full_nltk_word_DF_path, traind_PCA_path, \
    full_nltk_PCA_DF_path, full_nltk_PCA_DICT_path

from sklearn.decomposition import PCA
import sklearn.preprocessing as skscaler

words_df, vectors_df = open_pkl(full_nltk_word_DF_path)
print(vectors_df)
# Preprocessing data (scaling the data)
my_scaler = skscaler.StandardScaler()
# my_scaler=Scaler.MinMaxScaler() # another method of scaling

my_scaler.fit(vectors_df)
scaled_data = my_scaler.transform(vectors_df)  # transform the data (transform again later)

pca = PCA(n_components=0.95)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)  # transform back
data_pca = pd.DataFrame(data_pca)
dump_pkl(pca, traind_PCA_path)

print(data_pca)

dump_pkl([words_df, data_pca], full_nltk_PCA_DF_path)

word_pca_dict = {}

for i, vec in data_pca.iterrows():
    word_pca_dict[words_df[i]] = vec
    if i % 50000 == 0: print(f'{i}: {words_df[i]}')

print(dict(list(word_pca_dict.items())[0:10]))

dump_pkl(word_pca_dict, full_nltk_PCA_DICT_path)
