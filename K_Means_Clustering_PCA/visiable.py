from matplotlib import pyplot as plt


def show(data, k, centroids, cluster):
    features_number, col_num = data.shape
    if col_num != 2:
        print('dimension is not 2')
        return 1
    # Use different color shapes to represent each category
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'dr', '<r', 'pr']
    for i in range(features_number):
        markIndex = int(cluster[i])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])
    # Use different color shapes to represent each category
    mark = ['*g', '*k', '*r', '*b', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=20)
    plt.show()
