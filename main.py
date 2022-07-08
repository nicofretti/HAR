if __name__ == "__main__":
    # %%
    # Load data and show distribution of activities
    # and importing the libraries
    #
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix
    import tools
    import plots

    train = pd.read_csv("data/train.csv")
    activities = train["Activity"].value_counts()
    activities = {
        "index": activities.index,
        "count": activities.values
    }

    plots.activities_distribution(activities)
    x_train = train.drop(["subject", "Activity"], axis=1).values
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

    # %%
    # [task] Apply PCA to the data and visualize the results
    #
    n_eigenvectors = 150
    pca_proj = tools.PCA(x_train, n_eigenvectors)
    pca_data = np.matmul(x_train, pca_proj.T)
    # Plot the first three principal components
    plots.scatter_with_labels(pca_data, y_train, list(labels.keys()))

    # %%
    # [task] Apply LDA to the data and visualize the results
    #
    lda_proj = tools.LDA(pca_data, y_train, 6)
    lda_data = np.matmul(pca_data, lda_proj.T)
    plots.scatter_with_labels(lda_data, y_train, list(labels.keys()))

    # %%
    # [task] Use k-nearest neighbors to predict the activity of the test dataset
    #
    # 1. Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=6)

    # 2. Fit the classifier to the training data
    knn.fit(lda_data, y_train)
    # %%
    # [task] Use the classifier to predict the activity of the test dataset
    test = pd.read_csv("data/test.csv")
    x_test = test.drop(["subject", "Activity"], axis=1).values
    y_test = test.Activity.map(labels).values
    x_test = np.matmul(x_test, pca_proj.T)
    x_test = np.matmul(x_test, lda_proj.T)
    y_pred = knn.predict(x_test)
    # Show the confusion matrix
    plots.confusion_matrix(confusion_matrix(y_test, y_pred))
    # Show the accuracy of the classifier
    print("Accuracy:", knn.score(x_test, y_test))
