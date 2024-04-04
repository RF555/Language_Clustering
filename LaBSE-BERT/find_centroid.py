from UtilityFunctions import plotly_graph_2d_scatter, pd, np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

X = np.array([[1, 1], [2, 4], [1.5, 0], [3, 3], [2.5, 0.5],
              [5, 3], [4.5, 2], [6, 2.5], [6.5, 1],
              [7, 2], [8, 4], [9, 0], [10, 2], [8.5, 3]])
X = pd.DataFrame(X)
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
# print(f'kmeans.labels_:\n{kmeans.labels_}')
# print(f'kmeans.predict([[0, 0], [12, 3]]):\n{kmeans.predict([[0, 0], [12, 3]])}')
# print(f'kmeans.cluster_centers_:\n{kmeans.cluster_centers_}')

id_clusters = kmeans.fit_predict(X)
clustered_data = X.copy()
clustered_data['Cluster'] = id_clusters
print(f'data:\n{X}')

centroids_ = pd.DataFrame(kmeans.cluster_centers_)
print(f'num of centroids = {len(centroids_)}')
print(f'centroids:\n{centroids_}')

plotly_graph_2d_scatter(data=clustered_data, subplot={"centroids": centroids_},
                        cluster=True)

closest_ids, _ = pairwise_distances_argmin_min(centroids_, X)
# X_=X[1][0],X[1][1]
X_np = X.to_numpy()
closest_points = [X_np[i] for i in closest_ids]
closest_points = pd.DataFrame(closest_points)
print(f'closest points to centroids:\n{closest_points}')

plotly_graph_2d_scatter(data=clustered_data,
                        subplot={"centroids": centroids_,
                                 "closest points to centroids": closest_points},
                        cluster=True)
