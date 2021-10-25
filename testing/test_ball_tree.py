"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Testing BallTree for k-NN or KDE

"""

import os
import sys
import time
import math
import numpy

from sklearn.datasets import make_blobs
from matplotlib import pyplot
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from machine_learning import BallTree

if __name__ == '__main__':

    K = 10
    '''
    '''
    N = 100
    X, y = make_blobs(n_samples = N, n_features = 2, centers = K, cluster_std = 5.0, center_box = (-50.0, 50.0), shuffle = True)
    '''
    N = 50
    X = numpy.array([[x, x] for x in numpy.linspace(-50, 50, N)])
    y = numpy.array([i % K for i in range(N)])
    '''

    print('generation of a BallTree')
    starting_time = time.process_time_ns()
    #bt = BallTree(min_samples_to_split = len(X) // 5) #int(math.ceil(math.sqrt(len(X)))))
    bt = BallTree()
    bt.fit(X, y)
    bt_process_time = time.process_time_ns() - starting_time
    print()

    print()
    print('BENCHMARKING')
    print('    %-20s  %12.3f ms' % ('BallTree', bt_process_time / 1.0e+6))

    fig, axis = pyplot.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
    circles = list()
    def print_balls(split):
        print(split.center)
        circles.append(Circle((split.center[0], split.center[1]), math.sqrt(split.radius)))
        print(split.center)
        if split.left  is not None: print_balls(split.left)
        if split.right is not None: print_balls(split.right)
    print_balls(bt.root_split)

    p = PatchCollection(circles, alpha = 1.0)
    p.set_color('#1f77b4')
    p.set_alpha(0.2)
    axis.add_collection(p)
    axis.scatter(X[:, 0], X[:, 1], s = 30, color = 'black', alpha = 1.0)
    axis.set_xlim(-100, 100)
    axis.set_ylim(-100, 100)
    pyplot.tight_layout()
    pyplot.show()
