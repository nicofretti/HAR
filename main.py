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

    train = pd.read_csv("data/train.csv")
    activities = train["Activity"].value_counts()
    activities = {
        "index": activities.index,
        "count": activities.values
    }
    plts.activities_distribution(activities)
    x_train = train.drop(["subject", "Activity"], axis=1)
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
    # Plot the first three principal components
    plts.scatter_pca(p_data, y_train)
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
        sw += np.dot((c_data[i]-mu_c[i]).T, c_data[i]-mu_c[i])/shape_c[i]
        sb += np.dot(mu_c[i]-mu, (mu_c[i]-mu).T)

    '''
    c_scatter = []
    swx = np.zeros((n_eigenvectors, n_eigenvectors), dtype=np.complex128)
    for i in range(k):
        c_scatter.append(np.matmul((c_data[i] - mu_c[i]).T, (c_data[i] - mu_c[i]))/shape_c[i])
        swx += c_scatter[i]
    # 4. Calculate scatter between matrix
    sbx = np.zeros((n_eigenvectors, n_eigenvectors), dtype=np.complex128)
    m = np.mean(p_data, axis=0)
    for i in range(k):
        sbx += shape_c[i] * np.matmul(mu_c[i] - m, (mu_c[i] - m).T)
    '''
    # 5. Calculate the eigenvectors and eigenvalues and project the data
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(sw) * sb)
    eig_vecs = eig_vecs.T
    # 4. Calculate the percentage of variance explained by each eigenvector
    tot = sum(eig_vals)
    norm_eig_vals = [(i / tot) for i in sorted(eig_vals, reverse=True)]
    # 5. Plot the cumulative sum of the eigenvalues
    plts.cumulative_eigenvalues(norm_eig_vals)
    # [task] Apply PCA to the dataset and visualize the results
    #
    n_eigenvectors = 100
    sorted_eig_vals = np.argsort(eig_vals)[::-1]
    p_matrix = eig_vecs[sorted_eig_vals[:n_eigenvectors]]
    p_data = np.matmul(p_data - mu, p_matrix.T)
    # Plot the first three principal components
    plts.scatter_pca(p_data, y_train)


