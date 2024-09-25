"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.2
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Testing K-Means

"""

import os
import sys
import time
import numpy

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from matplotlib import pyplot

from machine_learning import KMeans

if __name__ == '__main__':

    generated_data_type = 'uniform'
    N =      100000
    N_cut =   50000
    N_paint = 20000
    K = 10
    K2 = 7 # set a smaller value to study the behaviour of the different algorithms when using make_blobs()

    for i in range(1, len(sys.argv)):
        if '--data-type' == sys.argv[i]:
            generated_data_type = sys.argv[i + 1]
        elif '--N' == sys.argv[i]:
            N = int(sys.argv[i + 1])
        elif '--N-cut' == sys.argv[i]:
            N_cut = int(sys.argv[i + 1])
        elif '--N-paint' == sys.argv[i]:
            N_paint = int(sys.argv[i + 1])
        elif '--K' == sys.argv[i]:
            K = int(sys.argv[i + 1])
        elif '--K2' == sys.argv[i]:
            K2 = int(sys.argv[i + 1])
        #else:
        #    raise Exception('Command-line argument not recognized!')

    N_cut = min(N_cut, N)
    N_paint = min(N_paint, N)
    K2 = min(K2, K)

    lloyd   = KMeans(n_clusters = K, verbosity = 1, modality = 'Lloyd', init = 'KMeans++')
    lloyd.epsilon = 1.0e-9
    kmediods = KMeans(n_clusters = K, verbosity = 1, modality = 'k-Mediods', init = 'KMeans++')
    lloyd.epsilon = 1.0e-8
    original_kmeans = KMeans(n_clusters = K, verbosity = 1, modality = 'original-k-Means')
    selective_splitting   = KMeans(n_clusters = K, verbosity = 1, modality = 'Lloyd', init = 'KMeans++')
    selective_splitting.epsilon = 1.0e-9

    if generated_data_type == 'uniform':
        X = numpy.random.rand(N, 2) * 100
    elif generated_data_type == 'clusters':
        X, y_true = make_blobs(n_samples = N, n_features = 2, centers = K2, cluster_std = 2.0, center_box = (-50.0, 50.0), shuffle = True)
    else:
        raise Exception(f'Data type for generating synthetic data not recognized as valid: {generated_data_type}')

    print('estimating with Lloyd')
    starting_time = time.process_time_ns()
    lloyd.fit(X)
    lloyd_process_time = time.process_time_ns() - starting_time
    print()

    print('estimating with Original K-Means')
    starting_time = time.process_time_ns()
    original_kmeans.fit(X)
    original_k_means_process_time = time.process_time_ns() - starting_time
    print()

    print('estimating with new version')
    starting_time = time.process_time_ns()
    selective_splitting.selective_splitting(X, K = K, verbose = 1)
    selective_splitting_process_time = time.process_time_ns() - starting_time
    print() #selective_splitting.n_clusters)

    print('estimating with K-Mediods')
    starting_time = time.process_time_ns()
    kmediods.fit(X[:N_cut])
    kmediods_process_time = time.process_time_ns() - starting_time
    print()

    print()
    print('BENCHMARKING')
    print('    %-20s  %12.3f ms' % ('Lloyd', lloyd_process_time / 1.0e+6))
    print('    %-20s  %12.3f ms' % ('Original K-Means', original_k_means_process_time / 1.0e+6))
    print('    %-20s  %12.3f ms' % ('Selective Splitting', selective_splitting_process_time / 1.0e+6))
    print('    %-20s  %12.3f ms' % ('K-Mediods', kmediods_process_time / 1.0e+6))

    #print(lloyd.cluster_centers_)
    #print(kmediods.cluster_centers_)
    #print(original_kmeans.cluster_centers_)


    list_of_models = list()
    list_of_models.append(('Lloyd', lloyd, 'red'))
    list_of_models.append(('K-Mediods', kmediods, 'green'))
    list_of_models.append(('Original K-Means', original_kmeans, 'magenta'))
    list_of_models.append(('Selective Splitting', selective_splitting, 'orange'))


    pyplot.scatter(X[:N_paint, 0], X[:N_paint, 1], s = 10, color = 'blue', alpha = 0.2)
    for model_name, model, color in list_of_models:
        clusters = model.cluster_centers_
        pyplot.scatter(clusters[:, 0], clusters[:, 1], s = 100, color = color, alpha = 1.0, label = model_name)
    pyplot.legend()
    pyplot.show()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, axes = pyplot.subplots(nrows = 1, ncols = len(list_of_models), figsize = (4 * len(list_of_models), 6))
    X_reduced = X[:N_paint]
    for i in range(len(list_of_models)):
        axis = axes[i]
        model_name, model, _ = list_of_models[i]
        y = model.predict(X)
        y_reduced = y[:N_paint]
        #
        print('computing silhoutte score for model', i)
        silh_score = silhouette_score(X, y) if len(X) <= 20000 else 0.0
        print('computing calinski-harabasz score for model', i)
        ch_score = calinski_harabasz_score(X, y)
        print('computing davies-bouldin score for model', i)
        db_score = davies_bouldin_score(X, y)
        #
        labels = numpy.unique(y)
        for l in labels:
            axis.scatter(X_reduced[y_reduced == l, 0], X_reduced[y_reduced == l, 1], s = 10, color = colors[l], alpha = 0.5)
            axis.set_xlabel('Silhouette score = %.3f \n Calinsky Harabasz score = %.3f \n Davies Bouldin score = %.3f' % (silh_score, ch_score, db_score))
        clusters = model.cluster_centers_
        axis.scatter(clusters[:, 0], clusters[:, 1], s = 50, color = 'black', alpha = 1.0, marker = '*')
        axis.set_title(model_name)
    #
    pyplot.tight_layout()
    pyplot.show()
