"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 4.0
    Date: May 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

# My KMeans implementation

import os
import sys
import random
import numpy
import gzip
import pickle

from sklearn import metrics

class KMeans:
    """
        Implementation of KMeans. This class provides several variants:
            1. Original KMeans
            2. Lloyd's algorithm with random or KMeans++ (Katsavounidis) initializations
            3. LBG incremental clustering
            4. K-Mediods - centroids are real samples, selected within each cluster
               as the sample that minimises the distance of all the samples in the
               cluster to it


        [LBG] Linde, Y.; Buzo, A.; Gray, R. (1980).
              "An Algorithm for Vector Quantizer Design".
              IEEE Transactions on Communications. 28: 84–95.
              doi:10.1109/TCOM.1980.1094577
    """

    # --------------------------------------------------------------------------------
    def __init__(self,  n_clusters = 2,
                        number_of_initializations = 1,
                        init = 'random',
                        max_iter = 1000,
                        modality = 'Lloyd',
                        verbosity = 0):
        self.max_iter = max_iter
        self.number_of_initializations = number_of_initializations
        self.init = init
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.cluster_pred = None
        self.J = 9.9e+300
        self.oldJ = 9.9e+300
        self.modality = modality
        if modality not in ['Lloyd', 'k-Mediods', 'k-Means', 'original-k-Means', 'LBG']:
            raise Exception('Wrong modality ' + modality)
        self.verbosity = verbosity
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def fit(self, X):

        if self.modality in ['Lloyd', 'k-Mediods']:
            #
            self.cluster_centers_ = numpy.zeros([self.n_clusters, X.shape[1]])
            if self.init in ['Katsavounidis', 'KMeans++']:
                self.katsavounidis(X)
            elif self.init == 'random':
                for c in range(self.n_clusters):
                    self.cluster_centers_[c][:] = X[numpy.random.randint(len(X))]
            else:
                print('ERROR: init method not specified!')
                sys.exit(1)
            #
            if self.modality == 'Lloyd':
                self.lloyd(X)
            else:
                self.k_mediods(X)
            #
        elif self.modality == 'LBG':
            #
            self.lbg(X, K = self.n_clusters)
            #
        elif self.modality in ['k-Means', 'original-k-Means']:
            #
            self.original_k_means(X, K = self.n_clusters)
            #
        else:
            raise Exception('Wrong modality ' + self.modality)
        #
        self.cluster_pred = None
        return self
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def lloyd(self, X):
        self.oldJ = self.J
        self.cluster_pred = self.predict(X)

        iteration = 0
        changes_counter = 1
        epsilon = 1.0e-4
        if self.verbosity > 0:
            print('iteration', iteration, 'max_iter', self.max_iter,
                    'changes', changes_counter, 'relative improvement', self.improvement())
        while iteration < self.max_iter and changes_counter > 0 and self.improvement() > epsilon:
            changes_counter = self.fit_iteration(X)
            iteration += 1
            if self.verbosity > 0:
                print('iteration', iteration, 'max_iter', self.max_iter,
                        'changes', changes_counter, 'relative improvement', self.improvement())

        return self
    # --------------------------------------------------------------------------------
    def k_mediods(self, X):
        self.oldJ = self.J
        self.cluster_pred = self.predict(X)

        iteration = 0
        changes_counter = 1
        epsilon = 1.0e-4

        if self.verbosity > 0:
            print('iteration', iteration, 'max_iter', self.max_iter,
                    'changes', changes_counter, 'relative improvement', self.improvement())
        while iteration < self.max_iter and changes_counter > 0 and self.improvement() > epsilon:
            changes_counter = self.fit_iteration(X)
            iteration += 1
            if self.verbosity > 0:
                print('iteration', iteration, 'max_iter', self.max_iter,
                        'changes', changes_counter, 'relative improvement', self.improvement())

        return self
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def fit_iteration(self, X):

        for c in range(len(self.cluster_centers_)):
            samples_in_cluster = X[self.cluster_pred == c]
            if len(samples_in_cluster) > 0:
                if self.modality == 'k-Mediods':
                    self.cluster_centers_[c, :] = self.mediod_from(samples_in_cluster)
                else:
                    self.cluster_centers_[c, :] = numpy.mean(samples_in_cluster, axis = 0)
            else:
                self.cluster_centers_[c, :] = 0.0

        self.oldJ = self.J
        y_pred = self.predict(X)

        changes_counter = numpy.abs(self.cluster_pred - y_pred).sum()
        self.cluster_pred[:] = y_pred[:]
        #print self.cluster_centers_ # For looking the evolution

        return changes_counter
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def mediod_from(self, samples):
        m = 0
        min_dist = sum(metrics.pairwise.euclidean_distances(samples, [samples[0]]))
        for i in range(1, len(samples)):
            dist = sum(metrics.pairwise.euclidean_distances(samples, [samples[i]]))
            if dist < min_dist:
                min_dist = dist
                m = i
        return samples[m]
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def lbg(self, X, K = None, verbose = 0):
        #
        if K is None : raise Exception('The number of desired clusters must be specified!')
        #
        #distortion = numpy.ones( X.shape[1] )
        distortion = X.std(axis = 0)
        #
        self.cluster_centers_ = numpy.array([X.mean(axis = 0)])
        #
        while len(self.cluster_centers_) < K :
            self.split(distortion)
            if verbose > 0: print("|codebook| = %d" % len(self.cluster_centers_))
            self.lloyd(X)
            if verbose > 0: print("distortion = %f" % self.J)
            #distortion = self.cluster_centers_.std(axis=0)
            distortion = 0.5 * distortion
        #
        self.n_clusters = len(self.cluster_centers_)
    # --------------------------------------------------------------------------------
    def split(self, distortion):
        new_codebook = numpy.zeros([2 * len(self.cluster_centers_), len(distortion)])
        for i in range(len(self.cluster_centers_)):
            new_codebook[2 * i    , :] = self.cluster_centers_[i] + distortion
            new_codebook[2 * i + 1, :] = self.cluster_centers_[i] - distortion
        self.cluster_centers_ = new_codebook
    # --------------------------------------------------------------------------------
    def drop_empty_clusters(self, X):
        y_pred = self.predict(X)
        centroids=list()
        for c in range(len(self.cluster_centers_)):
            if sum(y_pred == c) > 0:
                centroids.append(self.cluster_centers_[c].copy())
        self.cluster_centers_ = numpy.array(centroids)
        self.n_clusters = len(self.cluster_centers_)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def original_k_means(self, X):
        self.original_k_means_init(X)
    # --------------------------------------------------------------------------------
    def original_k_means_init(self, X):
        if type(X) not in [numpy.ndarray, list]:
            raise Exception('X must be a numpy array or a list!')
        dim = None
        if type(X) is list:
            if type(X[0]) is not numpy.ndarray:
                raise Exception('Elements in X must numpy arrays!')
            dim = X[0].shape[0]
        else:
            dim = X.shape[1]

        if len(X) < self.n_clusters:
            raise Exception('Initialisation must be done with a minimum number of samples/vectors equal to the number of clusters!')

        K = self.n_clusters
        self.counters = numpy.ones(self.n_clusters)
        #self.cluster_centers_ = numpy.zeros([self.n_clusters, dim])
        #for i in range(K):
        #    self.cluster_centers_[i, :] = X[i, :]
        self.cluster_centers_ = numpy.copy(X[:K])

        self.original_k_means_iteration(X[K:])
    # --------------------------------------------------------------------------------
    def original_k_means_iteration(self, X):
        for n in range(len(X)):
            # Classification
            _diffs_ = self.cluster_centers_ - X[n]
            _distances_ = (_diffs_ * _diffs_).sum(axis = 1)
            k = _distances_.argmin()
            # Partition update
            self.counters[k] += 1
            # Codebook update
            self.cluster_centers_[k] = ((self.cluster_centers_[k] * (self.counters[k] - 1)) + X[n]) / self.counters[k]
            #
            if sum(numpy.isnan(self.cluster_centers_[k].ravel())) > 0:
                raise Exception(f'FATAL ERROR: update of cluster {k} by sample {X[n]} generates NaN')
            # alpha = 1.0e-6
            #self.cluster_centers_[k] = self.cluster_centers_[k] * (1 - alpha) + X[n] * alpha

    # --------------------------------------------------------------------------------
    def predict(self, X):
        distances = metrics.pairwise.euclidean_distances(X, self.cluster_centers_)
        Y = numpy.argmin(distances, axis = 1)
        self.J = numpy.min(distances, axis = 1).sum()
        '''
        self.J = 0.0
        Y = numpy.ones(len(X), dtype = int) * -1
        for n in range(len(X)):
            Y[n] = numpy.argmin(distances[n])
            self.J = self.J + distances[n][Y[n]]
        '''
        self.J = self.J / len(X)
        return Y
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def distances(self, X):
        return metrics.pairwise.euclidean_distances(X, self.cluster_centers_)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def improvement(self):
        return abs((self.oldJ - self.J) / (abs(self.J) + 10e-5))
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def katsavounidis(self, X):
        if self.init == 'Katsavounidis':
            self.cluster_centers_[0] = numpy.mean(X, axis = 0)
        elif self.init == 'KMeans++':
            self.cluster_centers_[0] = X[random.randint(0, len(X) - 1)]
        else:
            raise Exception('Non-expected initialization' + self.init)
        c = 1
        while c < self.n_clusters:
            distances = metrics.pairwise.euclidean_distances(X, self.cluster_centers_[:c])
            distances = numpy.min(distances, axis = 1)
            self.cluster_centers_[c, :] = X[numpy.argmax(distances), :]
            c += 1
        distances = metrics.pairwise.euclidean_distances(X, self.cluster_centers_)
        distances = numpy.min(distances, axis = 1)
        self.cluster_centers_[0, :] = X[ numpy.argmax(distances), :]
        #print self.cluster_centers_
    # --------------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    def save(self, filename = None):
        if filename is not None:
            with gzip.open(filename, 'wb') as f: pickle.dump(self, f)
            f.close()
    # -------------------------------------------------------------------------

def kmeans_load(filename=None):
    if filename is None : raise Exception('Impossible to load a codebook without the filename!')
    model = None
    if os.path.exists(filename) and os.path.isfile(filename):
        with gzip.open(filename, 'rb') as f: model = pickle.load(f)
        f.close()
    #
    return model
# -------------------------------------------------------------------------
