
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
    p_data = tools.PCA(x_train, n_eigenvectors)
    # Plot the first three principal components
    plts.scatter_with_labels(p_data, y_train, labels.keys())
    # %%
    import tools
    lda_data = tools.LDA(p_data, y_train, n_classes=6)
    plts.scatter_with_labels(lda_data, y_train, labels.keys())
    # p_data = p_data.astype(np.float64)
    # %%
    # [task] Use k-nearest neighbors to predict the activity of the test dataset
    #
    # 1. Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=6)
    # 2. Fit the classifier to the training data
    # Cast p_data from complex to float
    lda_data = lda_data.astype(np.float64)
    knn.fit(lda_data, y_train)
    # 3. Load the test data
    test = pd.read_csv("data/test.csv")
    x_test = test.drop(["subject", "Activity"], axis=1)

    x_test = tools.PCA(x_test, n_eigenvectors)
    x_test = tools.LDA(x_test, y_train, n_classes=6)
    y_test = test.Activity
    y_test = y_test.map(labels)
    x_test = x_test.astype(np.float64)
    # 4. Predict the activity of the test data
    y_pred = knn.predict(x_test)
    # 5. Print the coverage of the classifier
    print("Coverage:", np.mean(y_pred == y_test))


