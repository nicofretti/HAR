
if __name__ == "__main__":

    # %%
    # Load data and show distribution of activities
    #
    from sklearn.neighbors import KNeighborsClassifier
    # import lda
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import pandas as pd
    import numpy as np
    import plots as plts
    import numpy as np
    import tools

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
    # Rename the activities to numbers
    labels = {
        'LAYING': 0,
        'SITTING': 1,
        'STANDING': 2,
        'WALKING': 3,
        'WALKING_DOWNSTAIRS': 4,
        'WALKING_UPSTAIRS': 5
    }
    y_train = y_train.map(labels)
    w, h = train.shape
    n_eigenvectors = 100
    x_train = x_train.astype(np.float64)
    pca_proj = tools.PCA(x_train, n_eigenvectors)
    pca_data = np.matmul(x_train, pca_proj)
    # Plot the first three principal components
    plts.scatter_with_labels(pca_data, y_train, labels.keys())
    # %%
    lda_proj = tools.LDA(pca_data, y_train, n_classes=6)
    lda_data = np.matmul(pca_data, lda_proj)
    plts.scatter_with_labels(lda_data, y_train, labels.keys())
    # p_data = p_data.astype(np.float64)
    # %%
    # [task] Use k-nearest neighbors to predict the activity of the test dataset
    #
    # 1. Create a KNN classifier
    lda_data = lda_data.astype(np.float64)
    knn = KNeighborsClassifier(n_neighbors=6)
    # 2. Fit the classifier to the training data
    # Cast p_data from complex to float
    knn.fit(lda_data, y_train)
    print(knn.score(lda_data, y_train))

