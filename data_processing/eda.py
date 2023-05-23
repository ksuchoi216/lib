from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib


def do_pca(x_data, y_data, info_ratio):
    N, dim = x_data.shape
    x_data = x_data.reshape((N, -1))  # flatten the image data
    print(x_data.shape)
    print(y_data.shape)

    # standarize features
    scaler = StandardScaler()
    scaler.fit(x_data)
    pca_x = scaler.transform(x_data)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(pca_x)

    pca_x = pca.transform(pca_x)

    fig = plt.figure(1, figsize=(20, 20))
    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=y_data)
    plt.show()

    scaler = StandardScaler()
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)

    _pca = PCA(n_components=dim)
    _pca.fit(x_data)
    projected = _pca.transform(x_data)
    print("data:", x_data.shape)
    print("projected:", projected.shape)

    plt.plot(_pca.explained_variance_)
    plt.grid()
    plt.xlabel('Explained Variance')
    plt.xticks(np.arange(0, dim))
    plt.figure()

    plt.plot(np.arange(len(_pca.explained_variance_ratio_))+1,
             np.cumsum(_pca.explained_variance_ratio_), 'o-')  # plot the scree graph
    plt.axis([1, len(_pca.explained_variance_ratio_), 0, 1])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Scree Graph')
    plt.xticks(np.arange(0, dim))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid()
    plt.show()
    percent = np.cumsum(_pca.explained_variance_ratio_)
    print("The number of total dimension:", len(percent))

    if info_ratio > 1:
        raise ValueError
    the_number_of_dimension = np.where((percent >= info_ratio))[0]

    print("The number of dimension to keep 95%:", the_number_of_dimension[0])

# do_pca(x_data, labels, 0.8)