import pandas as pd
import numpy as np
import plots as plts

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    activities = train["Activity"].value_counts()
    activities = {
        "index": activities.index,
        "count": activities.values
    }
    # Draw a pie chart to see the distribution of activities
    plts.activities_distribution(activities)
    x_train = train.drop(["subject", "Activity"], axis=1)
    y_train = train.Activity
    w, h = train.shape
    # Calculate scatter matrix of the dataset
    # 1. Calculate mean vector
    mean_vect = x_train.mean(axis=0)
    # 2. Calculate covariance matrix
    cov_mat = np.cov(x_train.T)
    # 3. Calculate eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # 4. Calculate the percentage of variance explained by each eigenvector
    tot = sum(eig_vals)
    norm_eig_val = [(i / tot) for i in sorted(eig_vals, reverse=True)]
    # 5. Plot the cumulative sum of the eigenvalues
    plts.cumulative_eigenvalues(norm_eig_val)
    # Adding bar coverage to the plot
    #print(mean_vect)

