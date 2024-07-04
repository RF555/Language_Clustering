from ClustersUtilityFunctions import nltk_df_path, nltk_df_filtered_path, split_full_df, pd, KMeans, \
    pairwise_distances_argmin_min, \
    get_words_vectors_df
import pickle as pkl

# words_df, vectors_df = pkl.load(open('pkl_files/nltk(236736_words)_PCA_word_vector_df.pkl', 'rb'))
# words_df, vectors_df = pkl.load(open('pkl_files/nltk_filtered_PCA_word_vector_df.pkl', 'rb'))
words_df, vectors_df = pkl.load(open('pkl_files/nltk_filtered_PCA_word_vector_df2.pkl', 'rb'))

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

# pkl.dump([closest_words, closest_points], open('pkl_files/600_centroid_PCA_word_vector_df.pkl', "wb"))
# closest_words.to_csv('600_centroid_words.csv')

# pkl.dump([closest_words, closest_points], open('pkl_files/(filtered)600_centroid_PCA_word_vector_df.pkl', "wb"))
# closest_words.to_csv('(filtered)600_centroid_words.csv')

pkl.dump([closest_words, closest_points], open('pkl_files/(filtered)600_centroid_PCA_word_vector_df2.pkl', "wb"))
closest_words.to_csv('(filtered)600_centroid_words2.csv')

# closest_points.iloc[: , 1:].to_csv('600_centroid_vectors.csv')
# closest_words.iloc[: , 1:].to_csv('600_centroid_words.csv')
#
# closest_words_vectors = pd.concat([closest_words, closest_points], axis=1)
# closest_words_vectors.to_csv('600_centroid_words_vectors2.csv')
# closest_words_vectors.to_pickle('600_centroid_words_vectors2.pkl')
