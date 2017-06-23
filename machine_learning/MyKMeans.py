"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 3.0
    Date: November 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

# My KMeans implementation

import os
import sys
import numpy
import gzip
import pickle

from sklearn import metrics

class KMeans:
    """
    """
    
    # --------------------------------------------------------------------------------
    def __init__( self, n_clusters=2, number_of_initializations=1, init='random', max_iter=1000 ):
        self.max_iter = max_iter
        self.number_of_initializations = number_of_initializations
        self.init = init
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.cluster_pred = None
        self.J = 9.9e+300
        self.oldJ = 9.9e+300
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def fit( self, X, modality='Lloyd' ):

        if modality=='Lloyd':
            #
            self.cluster_centers_ = numpy.zeros( [self.n_clusters, X.shape[1] ] )
            if self.init == 'Katsavounidis' :
                self.katsavounidis(X)
            elif self.init == 'random':
                for c in range( self.n_clusters ):
                    self.cluster_centers_[c][:] = X[ numpy.random.randint( len(X) ) ]
            else:
                print( 'ERROR: init method not specified!' )
                sys.exit(1)
            #
            self.lloyd( X, num_iter=0 )
            #
        elif modality == 'LBG':
            #
            self.lgb( X, K=self.n_clusters )
            #
        elif modality == 'k-Means':
            #
            self.original_k_means( X, K=self.n_clusters )
            #
        else:
            raise Exception( 'Wrong modality ' + modality )
        #
        return self
    # --------------------------------------------------------------------------------
 
    # --------------------------------------------------------------------------------
    def lloyd( self, X, num_iter=0 ):

        self.oldJ = self.J
        self.cluster_pred = self.predict( X )

        if num_iter <= 0: num_iter = self.max_iter

        iteration=0
        changes_counter=1
        epsilon = 1.0e-4
        #print( '*', iteration, num_iter, changes_counter, self.improvement() )
        while iteration < num_iter and changes_counter > 0 and self.improvement() > epsilon:
            #print( '-', iteration, num_iter, changes_counter, self.improvement() )
            changes_counter = self.fit_iteration( X )
            iteration=iteration+1

        return self
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def fit_iteration( self, X ):

        for c in range( self.n_clusters ):
            samples_in_cluster = X[ self.cluster_pred == c ]
            if len(samples_in_cluster) > 0 :
                self.cluster_centers_[c,:] = numpy.mean( samples_in_cluster, axis=0 )
            else:
                self.cluster_centers_[c,:] = 0.0

        self.oldJ = self.J
        y_pred = self.predict(X)

        changes_counter = numpy.abs(self.cluster_pred - y_pred).sum()
        self.cluster_pred[:] = y_pred[:]
        #print self.cluster_centers_ # For looking the evolution

        return changes_counter
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def lbg( self, X, K=None, verbose=0 ):
        #
        if K is None : raise Exception( 'The number of desired clusters must be specified!' )
        #
        #distortion = numpy.ones( X.shape[1] )
        distortion = X.std( axis=0 )
        #
        self.cluster_centers_ = numpy.array( [ X.mean(axis=0) ] )
        #
        while len(self.cluster_centers_) < K :
            self.split( distortion )
            if verbose > 0: print( "|codebook| = %d" % len(self.cluster_centers_) )
            self.lloyd( X )
            if verbose > 0: print( "distortion = %f" % self.J )
            #distortion = self.cluster_centers_.std(axis=0)
            distortion = 0.5 * distortion
        #
        self.n_clusters = len(self.cluster_centers_)
    # --------------------------------------------------------------------------------
    def split( self, distortion ):
        new_codebook = numpy.zeros( [2*len(self.cluster_centers_), len(distortion) ] )
        for i in range(len(self.cluster_centers_)):
            new_codebook[2*i,  :] = self.cluster_centers_[i] + distortion
            new_codebook[2*i+1,:] = self.cluster_centers_[i] - distortion
        self.cluster_centers_ = new_codebook
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def original_k_means( self, X, K=None ):
        #
        if K is None : raise Exception( 'The number of desired clusters must be specified!' )
        #
        codebook = numpy.copy( X[:K] )
        counters = numpy.ones( K )
        for n in range(K,len(X)):
            # Classification
            _diffs_ = codebook - X[n]
            _distances_ = (_diffs_ * _diffs_).sum(axis=1)
            k = _distances_.argmin()
            # Partition update
            counters[k] += 1
            # Codebook update
            codebook[k] = ((codebook[k] * (counters[k]-1)) + X[n]) / counters[k]
        #
        self.n_clusters = K
        self.cluster_centers_ = codebook
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def predict( self, X ):
        Y = numpy.ones( len(X), dtype=int ) * -1
        distances = metrics.pairwise.euclidean_distances( X, self.cluster_centers_ )
        self.J = 0.0
        for n in range(len(X)):
            Y[n] = numpy.argmin( distances[n] )
            self.J = self.J + distances[n][Y[n]]
        self.J = self.J / len(X)
        return Y
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    def improvement( self ):
        return abs( (self.oldJ - self.J) / self.J )
        
    # --------------------------------------------------------------------------------


    # --------------------------------------------------------------------------------
    def katsavounidis( self, X ):
        self.cluster_centers_[0] = numpy.mean( X, axis=0 )
        c=1
        while c < self.n_clusters:
            distances = metrics.pairwise.euclidean_distances( X, self.cluster_centers_[:c] )
            distances = numpy.min( distances, axis=1 )
            self.cluster_centers_[c,:] = X[ numpy.argmax(distances), : ]
            c=c+1
        distances = metrics.pairwise.euclidean_distances( X, self.cluster_centers_ )
        distances = numpy.min( distances, axis=1 )
        self.cluster_centers_[0,:] = X[ numpy.argmax(distances), : ]
        #print self.cluster_centers_
    # --------------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    def save( self, filename=None ):
        if filename is not None:
            with gzip.open( filename, 'wb' ) as f: pickle.dump( self, f )
            f.close()
    # -------------------------------------------------------------------------

def load( filename=None ):
    if filename is None : raise Exception( 'Impossible to load a codebook without the filename!' )
    model = None
    if os.path.exists( filename ) and os.path.isfile( filename ):
        with gzip.open( filename, 'rb' ) as f: model = pickle.load( f )
        f.close()
    #
    return model
# -------------------------------------------------------------------------
