from ClustersUtilityFunctions import nltk_df_path, split_full_df, pd, KMeans, pairwise_distances_argmin_min, \
    get_words_vectors_df
from sklearn.decomposition import PCA
import sklearn.preprocessing as skscaler
import pickle as pkl

traind_PCA_path = 'pkl_files/traind_PCA.pkl'

words_df, vectors_df = get_words_vectors_df(nltk_df_path)

# Preprocessing data (scaling the data)
my_scaler = skscaler.StandardScaler()
# my_scaler=Scaler.MinMaxScaler() # another method of scaling

my_scaler.fit(vectors_df)
scaled_data = my_scaler.transform(vectors_df)  # transform the data (transform again later)

pca = PCA(n_components=0.95)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)  # transform back
data_pca=pd.DataFrame(data_pca)
pkl.dump(pca, open(traind_PCA_path, "wb"))

print(data_pca)

pkl.dump([words_df,data_pca], open('pkl_files/nltk(236736_words)_PCA_word_vector_df.pkl', "wb"))
