import numpy as np
import plots as plts


def PCA(x_train, n_eigenvectors):
    #
    # [task] PCA
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
    sorted_eig_vals = np.argsort(eig_vals)[::-1]
    p_matrix = eig_vecs[sorted_eig_vals[:n_eigenvectors]]
    p_data = np.matmul(x_train - mu, p_matrix.T)
    p_data.astype(np.float64)
    return p_data


def LDA(x_train, y_train, n_classes):
    #
    # [task] LDA
    #
    # 1. Group every feature by the activity
    _,n = x_train.shape
    c_data = []
    for i in range(n_classes):
        c_data.append(x_train[y_train == i])
    # 2. Compute the mean for each activity and store the cardinality of each activity
    mu_c = [np.mean(c_data[i], axis=0) for i in range(n_classes)]
    shape_c = [c_data[i].shape[0] for i in range(n_classes)]
    # 3. Calculate the between and within-class scatter matrix
    sw = np.zeros((n, n), dtype=np.complex128)
    sb = np.zeros((n, n), dtype=np.complex128)
    mu = np.mean(x_train, axis=0)
    for i in range(n_classes):
        sw += np.dot((c_data[i] - mu_c[i]).T, c_data[i] - mu_c[i]) / shape_c[i]
        sb += np.dot(mu_c[i] - mu, (mu_c[i] - mu).T)

    # 4. Project the data to the new space formed by K-1 eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(sw) * sb)
    eig_vecs = eig_vecs.T
    sorted_eig_vals = np.argsort(eig_vals)[::-1]
    p_matrix = eig_vecs[sorted_eig_vals[:n_classes - 1]]
    p_data = np.matmul(x_train - mu, p_matrix.T)
    return p_data
