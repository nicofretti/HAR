import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def activities_distribution(activities):
    # Draw a pie chart to see the distribution of activities
    return ;
    plt.pie(activities["count"], colors=sns.color_palette('pastel')[0:6], labels=activities["index"], autopct='%1.1f%%')
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


