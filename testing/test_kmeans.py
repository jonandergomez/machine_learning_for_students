"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: May 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Testing K-Means

"""

import os
import sys
import time
import numpy

from matplotlib import pyplot

from machine_learning import KMeans

if __name__ == '__main__':

    lloyd   = KMeans(n_clusters = 10, verbosity = 1, modality = 'Lloyd')
    kmediods = KMeans(n_clusters = 10, verbosity = 1, modality = 'k-Mediods')
    original_kmeans = KMeans(n_clusters = 10, verbosity = 1, modality = 'original-k-Means')

    N = 100000
    N_cut = 10000
    X = numpy.random.rand(N, 2) * 100

    print('estimating with Lloyd')
    starting_time = time.process_time_ns()
    lloyd.fit(X[:N_cut])
    lloyd_process_time = time.process_time_ns() - starting_time
    print()

    print('estimating with Original K-Means')
    starting_time = time.process_time_ns()
    original_kmeans.fit(X[:N_cut])
    original_k_means_process_time = time.process_time_ns() - starting_time
    print()

    print('estimating with K-Mediods')
    starting_time = time.process_time_ns()
    kmediods.fit(X[:N_cut])
    kmediods_process_time = time.process_time_ns() - starting_time
    print()

    print()
    print('BENCHMARKING')
    print('    %-20s  %12.3f ms' % ('Lloyd', lloyd_process_time / 1.0e+6))
    print('    %-20s  %12.3f ms' % ('Original K-Means', original_k_means_process_time / 1.0e+6))
    print('    %-20s  %12.3f ms' % ('K-Mediods', kmediods_process_time / 1.0e+6))

    #print(lloyd.cluster_centers_)
    #print(kmediods.cluster_centers_)
    #print(original_kmeans.cluster_centers_)


    list_of_models = [('Lloyd', lloyd, 'red'), ('K-Mediods', kmediods, 'green'), ('Original K-Means', original_kmeans, 'magenta')]


    pyplot.scatter(X[:N_cut, 0], X[:N_cut, 1], s = 10, color = 'blue', alpha = 0.2)
    for model_name, model, color in list_of_models:
        clusters = model.cluster_centers_
        pyplot.scatter(clusters[:, 0], clusters[:, 1], s = 100, color = color, alpha = 1.0, label = model_name)
    pyplot.legend()
    pyplot.show()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, axes = pyplot.subplots(nrows = 1, ncols = len(list_of_models), figsize = (4 * len(list_of_models), 4))
    for i in range(len(list_of_models)):
        axis = axes[i]
        model_name, model, _ = list_of_models[i]
        y = model.predict(X)
        labels = numpy.unique(y)
        for l in labels:
            axis.scatter(X[y == l, 0], X[y == l, 1], s = 10, color = colors[l], alpha = 0.5)
        clusters = model.cluster_centers_
        axis.scatter(clusters[:, 0], clusters[:, 1], s = 50, color = 'black', alpha = 1.0, marker = '*')
        axis.set_title(model_name)
    #
    pyplot.show()
