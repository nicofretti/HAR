# %%
# [task] Use K-MEANS to cluster the data
#
from random import uniform
from math import dist

n_clusters = k;max_iter = 6;iteration = 0;
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
y_pred = np.empty((0))
for x in datas:
    dists = [dist(x, centroids[i]) for i in range(n_clusters)]
    y_pred = np.append(y_pred, dists.index(np.min(dists)))
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
    fit[i] = np.count_nonzero(np.equal((y_pred+i)%k,y_train_id))

# %%
from sklearn.metrics import confusion_matrix

y_pred = (y_pred + 2) % k
labels_name = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
y_pred = y_pred.astype(np.int64)
y_pred_labels = [labels_name[x] for x in y_pred]
p_data = p_data.astype(np.float64)
plts.scatter_with_labels(p_data, y_train)
plts.scatter_with_labels(p_data, y_pred_labels)