import os
import sys
import numpy

from matplotlib import pyplot

from machine_learning import KMeans

if __name__ == '__main__':

    X = numpy.random.rand(10000, 2) * 100

    kmeans   = KMeans(n_clusters = 10, verbosity = 1, modality = 'Lloyd')
    kmediods = KMeans(n_clusters = 10, verbosity = 1, modality = 'k-Mediods')

    kmeans.fit(X)
    kmediods.fit(X)

    print(kmeans.cluster_centers_)
    print(kmediods.cluster_centers_)

    clusters = kmeans.cluster_centers_
    pyplot.scatter(clusters[:, 0], clusters[:, 1], s = 40, color = 'red', alpha = 0.8)
    clusters = kmediods.cluster_centers_
    pyplot.scatter(clusters[:, 0], clusters[:, 1], s = 40, color = 'orange', alpha = 0.8)
    pyplot.scatter(X[:, 0], X[:, 1], s = 10, color = 'blue', alpha = 0.2)
    pyplot.show()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, axes = pyplot.subplots(nrows = 1, ncols = 2)
    axis = axes[0]
    y = kmediods.predict(X)
    labels = numpy.unique(y)
    for l in labels:
        axis.scatter(X[y == l, 0], X[y == l, 1], s = 10, color = colors[l], alpha = 0.5)
    clusters = kmediods.cluster_centers_
    axis.scatter(clusters[:, 0], clusters[:, 1], s = 50, color = 'black', alpha = 0.8, marker = '*')
    axis.set_title('k-Mediods')
    #
    axis = axes[1]
    clusters = kmeans.cluster_centers_
    y = kmeans.predict(X)
    labels = numpy.unique(y)
    for l in labels:
        axis.scatter(X[y == l, 0], X[y == l, 1], s = 10, color = colors[l], alpha = 0.5)
    axis.scatter(clusters[:, 0], clusters[:, 1], s = 30, color = 'black', alpha = 0.8, marker = 'p')
    axis.set_title('k-Means')
    pyplot.show()
