"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: October 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import sys
import numpy

class MyGaussian:
    """
    This class implements the Maximum Likelihood Estimation for obtaining
    the mean and the variance of each class given the training data.

    As the purpose is to classify instead of returning the probability of
    belonging to each class given a sample, the denominator for computing
    the a posteriori probability is not calculated. 
    Only the class that maximizes the probablity is returned as label for
    each sample.

    For prediction only logarithms are used, while it is not needed to return
    probabilities, the computations are faster and more robust.
    
    The use of numpy.outer() method is for computing the vectorial product of
    two vectors in order to obtain a 2-D matrix as a result. See the documentation
    of 'numpy' for more details.
    """

    allowed_covar_types = [ 'full', 'diag', 'tied', 'tied_diag' ]
    
    def __init__( self, covar_type='full' ):
        #
        if covar_type not in MyGaussian.allowed_covar_types:
            raise Exception( 'Wrong covar type provided: %s ' % covar_type )
        #
        self.covar_type = covar_type
        self.num_classes=0
        self.log_priori=None
        self.mu=None
        self.sigma=None
        self.L=None
        self.dim=0
        self.targets=None
        self.log_2_pi = numpy.log( 2 * numpy.pi )


    # ------------------------------------------------------------------------------
    def fit( self, X, Y ):
        self.dim=X.shape[1]
        self.targets = numpy.unique(Y)
        self.num_classes = len(self.targets)
        self.log_priori = numpy.zeros( self.num_classes )
        self.mu = numpy.zeros( [ self.num_classes, self.dim ] )
        self.sigma = numpy.zeros( [ self.num_classes, self.dim, self.dim ] )
        self.L = numpy.zeros( [ self.num_classes, self.dim, self.dim ] )
        self.log_determinants = numpy.zeros( self.num_classes )

        global_sigma = numpy.identity( self.dim )
        global_L = numpy.identity( self.dim )
        if self.covar_type in [ 'tied', 'tied_diag' ]:
            global_mu = X.mean(axis=0)
            global_sigma = numpy.cov( X.T )
            if self.covar_type == 'tied_diag':
                global_sigma = numpy.diag( numpy.diag( global_sigma ) )
                global_L = numpy.sqrt( global_sigma )
            else:
                global_L = numpy.linalg.cholesky( global_sigma )
        
        for i in range(len(self.targets)):
            target=self.targets[i]
            subset = X[ [Y==target] ]
            self.mu[i] = subset.mean(axis=0)
            #
            if self.covar_type in ['full', 'diag']:
                self.sigma[i] = numpy.cov( subset.T )
                if self.covar_type == 'diag':
                    self.sigma[i] = numpy.diag( numpy.diag( self.sigma[i] ) )
                    self.L[i] = numpy.sqrt( self.sigma[i] )
                else:
                    self.L[i] = numpy.linalg.cholesky( self.sigma[i] )
            else:
                self.sigma[i] = global_sigma
                self.L[i] = global_L
            #
            self.log_determinants[i] = 2 * numpy.log( numpy.diag( self.L[i] ) ).sum()
            #
            self.log_priori[i] = numpy.log( len(subset) ) - numpy.log( len(X) )
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    def mahalanobis( self, x ):
        dist = numpy.zeros( self.num_classes )
        for i in range(self.num_classes):
            d = x - self.mu[i]
            v = numpy.linalg.solve( self.L[i], d )
            dist[i] = numpy.dot( v, v )
        return dist
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    def predict( self, X ):
        Y = numpy.zeros( len(X) )
        for n in range(len(X)):
            dists = self.mahalanobis(X[n])
            max_log_prob=-numpy.inf
            Y[n]=0
            for i in range(self.num_classes):
                log_prob = self.log_priori[i] - 0.5*( dists[i] + self.log_determinants[i] + self.log_2_pi )
                if log_prob > max_log_prob :
                    max_log_prob = log_prob
                    Y[n] = self.targets[i]
        return Y
    # ------------------------------------------------------------------------------
