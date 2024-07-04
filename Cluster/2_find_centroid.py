from UtilityFunctions import pd, open_pkl, dump_pkl

# Inputs
from Pathways import filtered_nltk_PCA_DF_path, full_nltk_PCA_DF_path
# Outputs
from Pathways import filtered_600_DF_path, full_600_DF_path

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

words_df, vectors_df = open_pkl(filtered_nltk_PCA_DF_path)

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

# The actual centroids of the clusters (not part of the data)
centroids = pd.DataFrame(kmeans.cluster_centers_)
print(f'num of centroids = {len(centroids)}\n')
print(f'centroids:\n{centroids}\n\n')

# Index list of vectors closest to each centroid
centroids_closest_points_index, _ = pairwise_distances_argmin_min(centroids, vectors_df)

vectors_np = vectors_df.to_numpy()
words_np = words_df.to_numpy()

closest_points = pd.DataFrame([vectors_np[i] for i in centroids_closest_points_index])
closest_words = pd.DataFrame([words_np[i] for i in centroids_closest_points_index])
print(f'closest points to centroids:\n{closest_points}\n\n')
print(f'closest words to centroids:\n{closest_words}')

closest_words.rename(columns={0: 'word'}, inplace=True)  # rename the wors column

dump_pkl([closest_words, closest_points], filtered_600_DF_path)
closest_words.to_csv('(filtered)600_centroid_words.csv')
