"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 5.0
    Date: Oct 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

# My KMeans implementation

import os
import sys
import time
import random
import numpy
import gzip
import pickle

from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics import pairwise_distances

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
        self.epsilon = 1.0e-4
        self.block = max(1000, min(50000, int((1024 ** 2) / n_clusters * 8)))
        if modality not in ['Lloyd', 'k-Mediods', 'k-Means', 'original-k-Means', 'LBG']:
            raise Exception('Wrong modality ' + modality)
        self.verbosity = verbosity
        self.X_norm_squared = None
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def fit(self, X):
        #
        self.X_norm_squared = (X ** 2).sum(axis = 1).reshape(-1, 1)
        #
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
            self.original_k_means(X)
            #
        else:
            raise Exception('Wrong modality ' + self.modality)
        #
        self.cluster_pred = None
        self.X_norm_squared = None
        return self
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def lloyd(self, X):
        t_start = time.time()
        self.oldJ = self.J
        self.cluster_pred = self.predict(X)

        iteration = 0
        changes_counter = 1

        if self.verbosity > 1:
            print('iteration', iteration, 'max_iter', self.max_iter,
                    'changes', changes_counter, 'relative improvement', self.improvement(), 'time lapse', time.time() - t_start)

        while iteration < self.max_iter and changes_counter > 0 and self.improvement() > self.epsilon:
            t_start = time.time()
            changes_counter = self.fit_iteration(X)
            iteration += 1
            if self.verbosity > 0:
                print('iteration', iteration, 'max_iter', self.max_iter,
                        'changes', changes_counter, 'relative improvement', self.improvement(), 'time lapse', time.time() - t_start)

        return self
    # --------------------------------------------------------------------------------
    def k_mediods(self, X):
        t_start = time.time()
        self.oldJ = self.J
        self.cluster_pred = self.predict(X)

        iteration = 0
        changes_counter = 1

        if self.verbosity > 1:
            print('iteration', iteration, 'max_iter', self.max_iter,
                    'changes', changes_counter, 'relative improvement', self.improvement(), 'time lapse', time.time() - t_start)

        while iteration < self.max_iter and changes_counter > 0 and self.improvement() > self.epsilon:
            t_start = time.time()
            changes_counter = self.fit_iteration(X)
            iteration += 1
            if self.verbosity > 0:
                print('iteration', iteration, 'max_iter', self.max_iter,
                        'changes', changes_counter, 'relative improvement', self.improvement(), 'time lapse', time.time() - t_start)

        return self
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def fit_iteration(self, X):
        #print('updating codebook', flush = True)
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

        changes_counter = numpy.abs(self.cluster_pred != y_pred).sum()
        self.cluster_pred[:] = y_pred[:]

        return changes_counter
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def mediod_from(self, samples):
        m = 0
        min_dist = numpy.inf
        i = 0
        while i < len(samples):
            distances = euclidean_distances(samples[i : i + self.block], samples[i : i + self.block], squared = True).sum(axis = 1)
            j = distances.argmin()
            if distances[j] < min_dist:
                min_dist = distances[j]
                m = i + j
            #
            i += self.block
        #
        return samples[m]
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def selective_splitting(self, X, K = None, verbose = 0):
        #
        if K is None : raise Exception('The number of desired clusters must be specified!')
        #
        self.cluster_centers_ = numpy.zeros([K, X.shape[1]])
        temp_kmeans = KMeans(n_clusters = 2, modality = 'Lloyd', init = 'KMeans++', verbosity = verbose)
        temp_kmeans.fit(X)
        self.n_clusters = temp_kmeans.n_clusters
        for c in range(self.n_clusters):
            self.cluster_centers_[c][:] = temp_kmeans.cluster_centers_[c][:]
        #
        while self.n_clusters < K:
            y_pred = self.predict(X)
            counters_and_class_index = [(sum(y_pred == c), c) for c in range(self.n_clusters)]
            counters_and_class_index.sort(key = lambda x: x[0], reverse = True)
            i = 0
            m = self.n_clusters
            #while 2 * i < m and self.n_clusters < K:
            if i < m: # to FORCE entering just one time
                c = counters_and_class_index[i][1] # get the class index to be split
                temp_kmeans = KMeans(n_clusters = 2, modality = 'Lloyd', init = 'KMeans++', verbosity = verbose)
                temp_kmeans.fit(X[y_pred == c])
                self.cluster_centers_[c              ][:] = temp_kmeans.cluster_centers_[0][:]
                self.cluster_centers_[self.n_clusters][:] = temp_kmeans.cluster_centers_[1][:]
                self.n_clusters += 1
                i += 1
            #
            self.lloyd(X)
        #
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
            k = ((self.cluster_centers_ - X[n]) ** 2).sum(axis = 1).argmin()
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
        #print('predicting', flush = True)
        Y = list()
        self.J = 0
        i = 0
        while i < len(X):
            if self.X_norm_squared is not None:
                distances = euclidean_distances(X[i : i + self.block], self.cluster_centers_, squared = True, X_norm_squared = self.X_norm_squared[i : i + self.block])
                #distances = pairwise_distances(X[i : i + self.block], self.cluster_centers_, n_jobs = -1)
            else:
                distances = euclidean_distances(X[i : i + self.block], self.cluster_centers_, squared = True, X_norm_squared = None)
                #distances = pairwise_distances(X[i : i + self.block], self.cluster_centers_, n_jobs = -1)
            #
            Y += distances.argmin(axis = 1).tolist()
            self.J += distances.min(axis = 1).sum()
            #
            i += self.block
        #
        self.J /= len(X)
        return numpy.array(Y)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def distances(self, X, Y = None, squared = False):
        distances = list()
        if Y is None:
            Y = self.cluster_centers_
        i = 0
        while i < len(X):
            if self.X_norm_squared is not None:
                distances.append(euclidean_distances(X[i : i + self.block], Y, squared = squared, X_norm_squared = self.X_norm_squared[i : i + self.block]))
                #distances.append(pairwise_distances(X[i : i + self.block], Y, n_jobs = -1))
            else:
                distances.append(euclidean_distances(X[i : i + self.block], Y, squared = squared, X_norm_squared = None))
                #distances.append(pairwise_distances(X[i : i + self.block], Y, n_jobs = -1))
            i += self.block
        distances = numpy.vstack(distances)
        return distances
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def improvement(self):
        return abs((self.oldJ - self.J) / (abs(self.J) + 10e-5))
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def katsavounidis(self, X):
        if self.X_norm_squared is None:
            self.X_norm_squared = (X ** 2).sum(axis = 1).reshape(-1, 1)
        #
        if self.init == 'Katsavounidis':
            self.cluster_centers_[0] = numpy.mean(X, axis = 0)
        elif self.init == 'KMeans++':
            self.cluster_centers_[0] = X[random.randint(0, len(X) - 1)]
        else:
            raise Exception('Non-expected initialization' + self.init)
        c = 1
        while c < self.n_clusters:
            t_start = time.time()
            distances = self.distances(X, Y = self.cluster_centers_[:c], squared = True)
            distances = numpy.min(distances, axis = 1)
            self.cluster_centers_[c, :] = X[numpy.argmax(distances), :]
            print(self.init, 'c = ', c, 'time lapse', time.time() - t_start)
            c += 1
        if self.init == 'KMeans++':
            distances = self.distances(X, Y = self.cluster_centers_, squared = True)
            distances = numpy.min(distances, axis = 1)
            self.cluster_centers_[0, :] = X[numpy.argmax(distances), :]
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
