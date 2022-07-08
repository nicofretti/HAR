import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

activities_colors = sns.color_palette('pastel')[0:6]


def activities_distribution(activities):
    # Draw a pie chart to see the distribution of activities
    plt.pie(activities["count"], colors=sns.color_palette('pastel')[0:6], labels=activities["index"], autopct='%1.1f%%')
    plt.show()


def cumulative_eigenvalues(eigenvalues):
    # Draw bar and line plot to see the cumulative sum of the eigenvalues
    cum_eigenvalues = np.cumsum(eigenvalues)
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(cum_eigenvalues) + 1), cum_eigenvalues, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()


def scatter_with_labels(p_data, y_train, labels):
    # Draw scatter in 3d
    colors = [activities_colors[i] for i in y_train]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(p_data[:, 0], p_data[:, 1], p_data[:, 2], c=colors)
    # Adding legend and axis labels
    ax.set_xlabel('f_1')
    ax.set_ylabel('f_2')
    ax.set_zlabel('f_3')
    # Adding legend
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Patch(color=activities_colors[i], label=labels[i]))
    plt.legend(
        handles=handles,
        loc='lower left',
    )
    plt.show()


def confusion_matrix(matrix):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.show()
