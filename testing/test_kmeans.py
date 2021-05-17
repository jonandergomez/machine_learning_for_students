import os
import sys
import numpy

from matplotlib import pyplot

from machine_learning import KMeans

if __name__ == '__main__':

    X = numpy.random.rand(10000, 2)

    kmeans   = KMeans(n_clusters = 10, verbosity = 1, modality = 'Lloyd')
    kmediods = KMeans(n_clusters = 10, verbosity = 1, modality = 'k-Mediods')

    kmeans.fit(X)
    kmediods.fit(X)

    print(kmeans.cluster_centers_)
    print(kmediods.cluster_centers_)

    pyplot.scatter(X[:, 0], X[:, 1], s = 10, color = 'blue', alpha = 0.2)
    clusters = kmeans.cluster_centers_
    pyplot.scatter(clusters[:, 0], clusters[:, 1], s = 40, color = 'red', alpha = 0.8)
    clusters = kmediods.cluster_centers_
    pyplot.scatter(clusters[:, 0], clusters[:, 1], s = 40, color = 'orange', alpha = 0.8)
    pyplot.show()
