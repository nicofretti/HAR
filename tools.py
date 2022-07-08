import numpy as np
import plots as plts


def PCA(x_train, n_eigenvectors):
    #
    # [task] PCA
    #
    # 1. Calculate mean vector and remove the mean from the dataset
    mu = x_train.mean(axis=0)

    # 2. Calculate the scatter matrix
    s = np.matmul((x_train - mu).T, (x_train - mu))

    # 3. Calculate eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eig(s)
    eig_vecs = eig_vecs.real;eig_vecs = eig_vecs.T
    eig_vals = eig_vals.real

    # 4. Plot the cumulative sum of the eigenvalue
    tot = sum(eig_vals)
    norm_eig_vals = [(i / tot) for i in sorted(eig_vals, reverse=True)]
    plts.cumulative_eigenvalues(norm_eig_vals)

    # 5. Take the highest n_eigenvectors
    sorted_index = np.argsort(abs(eig_vals))[::-1]
    p_matrix = eig_vecs[sorted_index[:n_eigenvectors]]

    return p_matrix


def LDA(x_train, y_train, n_classes):
    #
    # [task] LDA
    #
    # 1. Group every feature by the activity
    _, n = x_train.shape

    # 2. Calculate the mean vector
    mu = np.mean(x_train, axis=0)

    # 3. Group the data by activity and calculate the mean vector for each activity
    data_c = []
    for i in range(n_classes):
        data_c.append(x_train[y_train == i])
    # 4. Calculate Within and Between class scatter matrix
    sw = np.zeros((n, n))
    sb = np.zeros((n, n))
    for i in range(n_classes):
        mu_c = np.mean(data_c[i], axis=0)
        n_c = data_c[i].shape[0]
        sw += np.matmul((data_c[i] - mu_c).T, (data_c[i] - mu_c)) / n_c
        sb += np.matmul((mu_c - mu)[np.newaxis].T, (mu_c - mu)[np.newaxis])

    # 5. Calculate the eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(sw) @ sb)
    eig_vecs = eig_vecs.real;eig_vecs = eig_vecs.T
    eig_vals = eig_vals.real

    # 6. Take the highest eigenvectors n_classes-1
    sorted_index = np.argsort(abs(eig_vals))[::-1]
    p_matrix = eig_vecs[sorted_index[:n_classes - 1]]
    return p_matrix
