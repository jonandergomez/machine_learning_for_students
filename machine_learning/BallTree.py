"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    BallTree implementation

"""

import sys
import time
import math
import random
import numpy
import heapq

try:
    from pyspark.rdd import RDD
except:
    RDD = None

class BallTreeNode:
    def __init__(self, S_n):
        self.S_n = S_n
        if RDD is not None and isinstance(S_n, RDD):
            pass
        else:
            self.center = S_n[1].mean(axis = 0)
            #self.radius = math.sqrt(max(((S_n[1] - self.center) ** 2).sum(axis = 1)))
            self.radius = max(((S_n[1] - self.center) ** 2).sum(axis = 1))
        self.left = None
        self.right = None

class BallTree:
    """
    This class implements generates a BallTree with non-disjoint subsets at each split.
    """
    
    def __init__(self, min_samples_to_split = None):
        self.min_samples_to_split = min_samples_to_split
        self.dim = None
    # ------------------------------------------------------------------------------

    def fit(self, X, y):
        if RDD is not None and (isinstance(X, RDD) or isinstance(y, RDD)):
            # it is assumed each item in Xy is a tuple, (y, x), where y is the label or the real value to be predicted
            t = Xy.first()
            assert type(t) == tuple
            x = t[1]
            assert type(x) == numpy.ndarray
            assert len(x.shape) == 1
            self.dim = x.shape[0]
            self.root_split = self.do_split(Xy)
        else:
            if self.min_samples_to_split is None:
                self.min_samples_to_split = int(math.ceil(math.sqrt(len(X))))
            assert type(X) == numpy.ndarray
            assert len(X.shape) == 2
            self.dim = X.shape[1]
            self.root_split = self.do_split((y, X))
        #
        return self
    # ------------------------------------------------------------------------------

    def do_split(self, S_n):
        split = BallTreeNode(S_n)
        #
        if RDD is not None and isinstance(S_n, RDD):
            if S_n.count() < self.min_samples_to_split:
                split.S_n.persist()
                return split
            #
        elif type(S_n) == tuple:
            X = S_n[1]
            if len(X) < self.min_samples_to_split: return split
            #
            x_0 = X[random.randint(0, len(X) - 1)]
            x_1 = X[((X - x_0) ** 2).sum(axis = 1).argmax()]
            x_2 = X[((X - x_1) ** 2).sum(axis = 1).argmax()]
            #
            projection = numpy.dot(X, (x_1 - x_2))
            projection = [(i, projection[i]) for i in range(len(projection))]
            projection.sort(key = lambda x: x[1])
            median_pos = len(projection) // 2
            #
            indices = [t[0] for t in projection[:median_pos]]
            S_l = (S_n[0][indices], S_n[1][indices])
            indices = [t[0] for t in projection[median_pos:]]
            S_r = (S_n[0][indices], S_n[1][indices])
            #
            print(len(S_n[0]), len(S_l[0]), len(S_r[0]))
            if len(S_l[0]) != len(S_n[0]) and len(S_r[0]) != len(S_n[0]):
                split.left  = self.do_split(S_l)
                split.right = self.do_split(S_r)
                split.S_n = None # splitted nodes do no require data
            #
        return split
    # ------------------------------------------------------------------------------

    def unpersist(self, node = None):
        if node is None: node = self.root_split
        #
        if node.S_n is not None:
            if RDD is not None and isinstance(node.S_n, RDD):
                node.S_n.unpersist()
        else:
            self.unpersist(node.left)
            self.unpersist(node.right)
    # ------------------------------------------------------------------------------

    def get_knn(self, x, K):
        pq = list()
        self.explore_nodes(self.root_split, x, pq, K)
        #
        assert len(pq) > 0
        points = [(t[1], t[2]) for t in pq]
        #
        return points
    # ------------------------------------------------------------------------------

    def squared_distance(p1, p2):
        return ((p1 - p2) ** 2).sum()
        
    def explore_nodes(self, split, x, pq, K):
        #print('len(pq)', len(pq), K)
        # distance to the ball, negative when inside the ball
        d = BallTree.squared_distance(split.center, x) - split.radius
        # distance to the sample in the top of the heap
        d_pq = BallTree.squared_distance(x, pq[0][2]) if len(pq) > 0 else numpy.inf
        if d > d_pq: # the sample x is further from the ball than to the sample on top of the heap
            return None
        elif split.S_n is not None:
            distances = ((split.S_n[1] - x) ** 2).sum(axis = 1)
            if len(distances.shape) != 1:
                raise Exception(f'incorrect shape {distances.shape}')
            # chooses samples from the subset S_n whose distance to x is lower than the distance
            # to the top of the heap and updates 'd_pq', the distance to the heap, 
            # i.e., to the furthest sample in the set of K nearest neighbours so far, that is why
            # it is not possible to use a filter and we have to visit all the samples in the ball
            for i in range(len(split.S_n[0])):
                if distances[i] < d_pq:
                    y = split.S_n[0][i]
                    z = split.S_n[1][i]
                    item = (-distances[i], y, z)
                    if len(pq) >= K:
                        heapq.heapreplace(pq, item)
                    else:
                        heapq.heappush(pq, item)
                    d_pq = BallTree.squared_distance(x, pq[0][2]) if len(pq) > 0 else numpy.inf
        else:
            dl = numpy.inf
            dr = numpy.inf
            if split.left  is not None: dl = ((split.left.center  - x) ** 2).sum()
            if split.right is not None: dr = ((split.right.center - x) ** 2).sum()
            if dl <= dr:
                if split.left  is not None: self.explore_nodes(split.left,  x, pq, K)
                if split.right is not None: self.explore_nodes(split.right, x, pq, K)
            else:
                if split.right is not None: self.explore_nodes(split.right, x, pq, K)
                if split.left  is not None: self.explore_nodes(split.left,  x, pq, K)
        return None
    # ------------------------------------------------------------------------------
