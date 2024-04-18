from LaBSE_BERT.UtilityFunctions import plotly_graph_2d_scatter, split_full_df, pd, np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import os

cur_path = os.path.dirname(__file__)

nltk_df = 'VECTOR_files/BERT_vectors/nltk(236736_words_dim768_df.pkl'

new_path = os.path.relpath('..\LaBSE_BERT\\' + nltk_df, cur_path)

_input = new_path

full_df = pd.read_pickle(_input)
print(full_df.head())

words_df, vectors_df = split_full_df(full_df)

# Split to clusters using KMeans
vectors_df = pd.DataFrame(vectors_df)
kmeans = KMeans(n_clusters=600, random_state=0, n_init="auto")
kmeans.fit(vectors_df)
print(f'kmeans.labels_:\n{kmeans.labels_[:100]}')
print(f'kmeans.cluster_centers_:\n{kmeans.cluster_centers_}\n')

# cluster_labels = kmeans.fit_predict(vectors_df)
cluster_labels = kmeans.labels_
clustered_data = vectors_df.copy()
clustered_data['Cluster'] = cluster_labels
print(f'data:\n{vectors_df}\n')

centroids_ = pd.DataFrame(kmeans.cluster_centers_)
print(f'num of centroids = {len(centroids_)}\n')
print(f'centroids:\n{centroids_}\n\n')

# Index list of points closest to each centroid
centroids_closest_points_index, _ = pairwise_distances_argmin_min(centroids_, vectors_df)

vectors_np = vectors_df.to_numpy()
words_np = words_df.to_numpy()
# closest_points = [X_np[i] for i in closest_ids]
closest_points = pd.DataFrame([vectors_np[i] for i in centroids_closest_points_index])
closest_words = pd.DataFrame([words_np[i] for i in centroids_closest_points_index])
print(f'closest points to centroids:\n{closest_points}\n\n')
print(f'closest words to centroids:\n{closest_words}')

closest_words.rename(columns={0: 'word'}, inplace=True)  # rename the wors column

closest_points.to_csv('600_closest_points.csv')
closest_words.to_csv('600_closest_words.csv')

closest_words_vectors = pd.concat([closest_words, closest_points], axis=1)
closest_words_vectors.to_csv('600_closest_words_vectors.csv')
