import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

activities_colors = sns.color_palette('pastel')[0:6]
activities = {
    "LAYING": activities_colors[0],
    "SITTING": activities_colors[1],
    "STANDING": activities_colors[2],
    "WALKING": activities_colors[3],
    "WALKING_DOWNSTAIRS": activities_colors[4],
    "WALKING_UPSTAIRS": activities_colors[5]
}

def activities_distribution(activities):
    # Draw a pie chart to see the distribution of activities
    plt.pie(activities["count"], colors=activities_colors, labels=activities["index"], autopct='%1.1f%%')
    plt.show()


def cumulative_eigenvalues(eigenvalues):
    # Draw bar and line plot to see the cumulative sum of the eigenvalues
    cum_eigenvalues = np.cumsum(eigenvalues)
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cum_eigenvalues) + 1), cum_eigenvalues, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()

def plot_pca(p_data, y_train):
    # Compute color for each activity
    colors = [activities[i] for i in y_train]
    # Draw a scatter plot to see the first three principal components
    # Draw scatter in 3d
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(p_data.iloc[:, 0], p_data.iloc[:, 1], p_data.iloc[:, 2], c=colors)
    plt.show()



