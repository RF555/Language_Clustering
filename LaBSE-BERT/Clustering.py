import matplotlib.pyplot as plt

from UtilityFunctions import *
from sklearn.cluster import KMeans

demo_vec = 'demo-vec'
test1 = 'test1'

curr_in = test1

_input = 'VECTOR-files/DIM-reduction/' + curr_in + '(PCA)-dim3'


def kmeansElbowMethod(data, top_range):
    # choose number of clusters using the 'Elbow Method'
    wcss = []
    curr_range = range(1, top_range)

    for i in curr_range:
        kmeans_elbow = KMeans(n_clusters=i, n_init="auto")
        kmeans_elbow.fit(data)
        wcss_iter = kmeans_elbow.inertia_
        wcss.append(wcss_iter)

    plt.plot(curr_range, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squers')

    plt.show()


if __name__ == '__main__':
    word_keys, vector_df = pklToDF(_input)
    # kmeansElbowMethod(data=vector_df, top_range=20)

    kmeans = KMeans(n_clusters=10, n_init="auto")
    kmeans.fit(vector_df)
    id_clusters = kmeans.fit_predict(vector_df)
    clustered_data = vector_df.copy()
    clustered_data['Cluster'] = id_clusters
    plotlyGraph3D(clustered_data, kmeans.labels_)
