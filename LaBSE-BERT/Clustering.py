from UtilityFunctions import *
from sklearn.cluster import KMeans

demo_vec = 'demo-vec'
test1 = 'test1'
first_par = 'wiki-first-paragraph'
old_first_par = 'wiki-first-paragraph-old'

curr_in = first_par

_input = 'VECTOR-files/Reduct-to-3D/' + curr_in + '(PCA)-dim3'


def kmeans_elbow_method(data):
    # choose number of clusters using the 'Elbow Method'
    wcss = []
    # curr_range = range(1, top_range)
    curr_range = 1

    kmeans_elbow = KMeans(n_clusters=curr_range, n_init="auto")
    kmeans_elbow.fit(data)
    wcss_iter = kmeans_elbow.inertia_
    wcss.append(wcss_iter)
    curr_range += 1
    kmeans_elbow = KMeans(n_clusters=curr_range, n_init="auto")
    kmeans_elbow.fit(data)
    wcss_iter = kmeans_elbow.inertia_
    wcss.append(wcss_iter)
    curr_range += 1

    while wcss[-1] - wcss[-2] < -1 / pow(curr_range <= data[0].size, 100):
        kmeans_elbow = KMeans(n_clusters=curr_range, n_init="auto")
        kmeans_elbow.fit(data)
        wcss_iter = kmeans_elbow.inertia_
        wcss.append(wcss_iter)
        curr_range += 1

    plotly_graph_2d(curr_data=[list(range(1, curr_range)), wcss],
                    x_label='Number of clusters',
                    y_label='Within-cluster Sum of Squers (WCSS)')


if __name__ == '__main__':
    word_keys, vector_df = pkl_to_dataframe(_input)

    # kmeans_elbow_method(data=vector_df)

    kmeans = KMeans(n_clusters=20, n_init="auto")
    kmeans.fit(vector_df)
    id_clusters = kmeans.fit_predict(vector_df)
    clustered_data = vector_df.copy()
    clustered_data['word'] = word_keys
    clustered_data['Cluster'] = id_clusters
    plotly_graph_3d_clusters(clustered_data)
