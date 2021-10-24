"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Testing K-Means

"""

import os
import sys
import time
import math
import numpy

from matplotlib import pyplot

from machine_learning import KMeans

if __name__ == '__main__':

    band_width = 1
    K = 100

    for i in range(len(sys.argv)):
        if   sys.argv[i] == '--band-width': band_width = float(sys.argv[i + 1])
        elif sys.argv[i] == '--k'         :          K = int(sys.argv[i + 1])

    # dataset generator
    N_1 = 5000
    N_2 = 5000

    X_1 = numpy.random.rand(N_1, 2)
    X_1[:, 0] *= 10
    X_1[:, 1] *=  3
    X_1[:, 0] +=  2
    X_1[:, 1] +=  2
    X_2 = numpy.random.rand(N_2, 2)
    X_2[:, 0] *=  3
    X_2[:, 1] *= 10
    X_2[:, 0] +=  8
    X_2[:, 1] +=  1



    # k-means codebook generators
    lloyd_1 = KMeans(n_clusters = K, verbosity = 1, modality = 'Lloyd', init = 'KMeans++')
    lloyd_1.epsilon = 1.0e-9
    lloyd_2 = KMeans(n_clusters = K, verbosity = 1, modality = 'Lloyd', init = 'KMeans++')
    lloyd_2.epsilon = 1.0e-9

    # k-means codebook estimation
    lloyd_1.fit(X_1)
    lloyd_2.fit(X_2)

    def assignment_1(p, h):
        h_2 = h ** 2
        density_1 = numpy.exp(-0.5 * (((X_1 - p) ** 2).sum(axis = 1) / h_2)).sum() / len(X_1)
        density_2 = numpy.exp(-0.5 * (((X_2 - p) ** 2).sum(axis = 1) / h_2)).sum() / len(X_2)
        prob_1 = density_1 / (1.0e-5 + density_1 + density_2)
        score = 2 * prob_1 - 1
        return score

    def assignment_2(p, h):
        h_2 = h ** 2
        density_1 = numpy.exp(-0.5 * (((lloyd_1.cluster_centers_ - p) ** 2).sum(axis = 1) / h_2)).sum() / len(lloyd_1.cluster_centers_)
        density_2 = numpy.exp(-0.5 * (((lloyd_2.cluster_centers_ - p) ** 2).sum(axis = 1) / h_2)).sum() / len(lloyd_2.cluster_centers_)
        prob_1 = density_1 / (1.0e-5 + density_1 + density_2)
        score = 2 * prob_1 - 1
        return score
        

    # graphics preparation
    _x_, _y_ = numpy.mgrid[0:15:complex(0, 100), 0:15:complex(0, 100)]
    _xy_ = numpy.zeros(_x_.shape + (2, ))
    _xy_[:, :, 0] = _x_[:, :]
    _xy_[:, :, 1] = _y_[:, :]

    _z1_ = numpy.zeros(_xy_.shape[:2])
    _z2_ = numpy.zeros(_xy_.shape[:2])
    for i in range(_xy_.shape[0]):
        for j in range(_xy_.shape[1]):
            _z1_[i, j] = assignment_1(_xy_[i, j], band_width)
            _z2_[i, j] = assignment_2(_xy_[i, j], band_width)


    fig, axes = pyplot.subplots(nrows = 2, ncols = 2, figsize = (15, 12))
    #
    axis = axes[0, 0]
    axis.scatter(X_1[:, 0], X_1[:, 1], s = 10, color = 'blue',   alpha = 0.2, label = 'class 1')
    axis.scatter(X_2[:, 0], X_2[:, 1], s = 10, color = 'orange', alpha = 0.2, label = 'class 2')
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 15)
    axis.legend()
    #
    axis = axes[0, 1]
    codebook_1 = lloyd_1.cluster_centers_
    codebook_2 = lloyd_2.cluster_centers_
    axis.scatter(codebook_1[:, 0], codebook_1[:, 1], s = 30, color = 'blue',   alpha = 0.7, label = 'class 1')
    axis.scatter(codebook_2[:, 0], codebook_2[:, 1], s = 30, color = 'orange', alpha = 0.7, label = 'class 2')
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 15)
    axis.legend()
    #
    axis = axes[1, 0]
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 15)
    axis.contourf(_x_, _y_, _z1_, levels = 50, cmap = 'RdBu')
    #
    axis = axes[1, 1]
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 15)
    axis.contourf(_x_, _y_, _z2_, levels = 50, cmap = 'RdBu')
    axis.scatter(codebook_1[:, 0], codebook_1[:, 1], s = 30, color = 'blue',   alpha = 0.7)
    axis.scatter(codebook_2[:, 0], codebook_2[:, 1], s = 30, color = 'orange', alpha = 0.7)
    #
    pyplot.suptitle(f'band width = {band_width}  K = {K}')
    pyplot.tight_layout()
    pyplot.savefig(f'/tmp/fig-kde-kmeans-{K}-{band_width}.svg', format = 'svg')
    pyplot.savefig(f'/tmp/fig-kde-kmeans-{K}-{band_width}.png', format = 'png')
    pyplot.show()
