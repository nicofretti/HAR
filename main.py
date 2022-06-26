if __name__ == "__main__":
    # %%
    # Load data and show distribution of activities
    #
    import pandas as pd
    import numpy as np
    import plots as plts
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D

    train = pd.read_csv("data/train.csv")
    activities = train["Activity"].value_counts()
    activities = {
        "index": activities.index,
        "count": activities.values
    }

    plts.activities_distribution(activities)
    x_train = train.drop(["subject", "Activity"], axis=1)
    x_train.astype(np.float64)
    y_train = train.Activity
    w, h = train.shape
    # %%
    # [task] Calculate covariance matrix with eigenvalues and eigenvectors
    #
    # 1. Calculate mean vector and remove the mean from the dataset
    mu = x_train.mean(axis=0)
    # 2. Calculate the scatter matrix
    cov_mat = np.cov(x_train - mu, rowvar=False)
    # 3. Calculate eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vecs = eig_vecs.T
    # 4. Calculate the percentage of variance explained by each eigenvector
    tot = sum(eig_vals)
    norm_eig_vals = [(i / tot) for i in sorted(eig_vals, reverse=True)]
    # 5. Plot the cumulative sum of the eigenvalues
    plts.cumulative_eigenvalues(norm_eig_vals)
    # %%
    # [task] Apply PCA to the dataset and visualize the results
    #
    n_eigenvectors = 100
    sorted_eig_vals = np.argsort(eig_vals)[::-1]
    p_matrix = eig_vecs[sorted_eig_vals[:n_eigenvectors]]
    p_data = np.matmul(x_train - mu, p_matrix.T)
    p_data.astype(np.float64)
    # Plot the first three principal components
    plts.scatter_with_labels(p_data, y_train)
    # %%
    # [task] LDA to the p_data
    #
    # 1. Grouping every feature by activity
    k = len(activities["index"])
    c_data = [[]] * k
    for i in range(k):
        c_data[i] = p_data[train["Activity"] == activities["index"][i]]
    # 2. Compute the mean for each activity and store the cardinality of each activity
    mu_c = [np.mean(c_data[i], axis=0) for i in range(k)]
    shape_c = [c_data[i].shape[0] for i in range(k)]
    # 3. Calculate the between and within-class scatter matrix
    sw = np.zeros((n_eigenvectors, n_eigenvectors), dtype=np.complex128)
    sb = np.zeros((n_eigenvectors, n_eigenvectors), dtype=np.complex128)
    mu = np.mean(p_data, axis=0)
    for i in range(k):
        sw += np.dot((c_data[i] - mu_c[i]).T, c_data[i] - mu_c[i]) / shape_c[i]
        sb += np.dot(mu_c[i] - mu, (mu_c[i] - mu).T)

    # %%
    # [task] Project the data to the new space formed by K-1 eigenvectors
    #
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(sw) * sb)
    eig_vecs = eig_vecs.T
    sorted_eig_vals = np.argsort(eig_vals)[::-1]
    p_matrix = eig_vecs[sorted_eig_vals[:k - 1]]
    p_data = np.matmul(p_data - mu, p_matrix.T)
    # Plot the first three principal components
    plts.scatter_with_labels(p_data, y_train)
    # p_data = p_data.astype(np.float64)
    # %%
    # [task] Use K-means to cluster the data
    #
    from random import uniform
    from math import dist

    n_clusters = k
    max_iter = 6;
    iteration = 0;
    prev_centroids = None
    p_data = p_data.astype(np.float64)
    # 1. Initialize the centroids
    min = np.min(p_data, axis=0)
    max = np.max(p_data, axis=0)
    centroids = []
    for _ in range(n_clusters):
        centroids.append(
            np.random.uniform(low=min, high=max, size=p_data.shape[1])
        )
    centroids = [uniform(np.min(p_data), np.max(p_data)) for i in range(n_clusters)]
    datas = [list(x)[1:] for x in p_data.itertuples()]
    while np.not_equal(centroids, prev_centroids).any() and iteration < max_iter:
        # 2. Sort point and assign to the nearest centroid
        prev_centroids = centroids
        sorted_points = [[] for _ in range(n_clusters)]
        for x in datas:
            dists = [dist(x, centroids[i]) for i in range(n_clusters)]
            sorted_points[dists.index(np.min(dists))].append(x)
            centroids = [np.mean(sorted_points[i], axis=0) for i in range(n_clusters)]
            for i, centroid in enumerate(centroids):
                if (np.isnan(centroid).any()):
                    centroids[i] = prev_centroids[i]
        iteration += 1
        print("K-means status: {}/{}".format(iteration, max_iter))
    print("K-mean done")
    # %% Prediction after training
    kmeans = KMeans(n_clusters=k).fit(p_data.astype(np.float64))
    y_pred = kmeans.labels_
    #y_pred = np.empty((0))
    #for x in datas:
    #    dists = [dist(x, centroids[i]) for i in range(n_clusters)]
    #    y_pred = np.append(y_pred, dists.index(np.min(dists)))
    # Associate the labels to the clusters
    labels = {
        'LAYING': 0,
        'SITTING': 1,
        'STANDING': 2,
        'WALKING': 3,
        'WALKING_DOWNSTAIRS': 4,
        'WALKING_UPSTAIRS': 5
    }
    y_train_id = [labels[x] for x in y_train]
    # Test the shifting of the labels to find the best shift
    fit = np.zeros(k)
    for i in range(k):
        # Find how much labels are equals with shift of i
        fit[i] = np.sum(np.equal(y_train_id, (y_pred + i) % k))

    # %%
    from sklearn.metrics import confusion_matrix

    y_pred = (y_pred + np.argmax(fit)) % k
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_train_id, y_pred), fmt=".3g", annot=True)
    plt.title("Confusion Matrix")
    plt.show()
