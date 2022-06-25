if __name__ == "__main__":
    # %%
    # Load data and show distribution of activities
    #
    import pandas as pd
    import numpy as np
    import plots as plts

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
    # [task] Apply PCA to the dataset
    #
    n_eigenvectors = 100
    sorted_eig_vals = np.argsort(eig_vals)[::-1]
    p_matrix = eig_vecs[sorted_eig_vals[:n_eigenvectors]]
    p_data = np.matmul(x_train-mu, p_matrix.T)
    # Plot the first three principal components
    plts.plot_pca(p_data, y_train)


