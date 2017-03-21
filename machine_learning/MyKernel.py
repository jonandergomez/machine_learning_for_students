"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: October 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import sys
import numpy
from sklearn import metrics

class MyKernel:
    """
    This class implements a Kernel Density Estimator by using Parzen Windows.
    """
    
    def __init__( self, bandwidth=1, kernel='gaussian' ):
        self.n_classes=0
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.h = 1
        self.data_ = None


    # ------------------------------------------------------------------------------
    def fit( self, X ):
        #self.data_ = numpy.clone(X)
        self.data_ = X
        self.h = self.bandwidth / numpy.sqrt( len(X) )

        return self
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    def score_samples( self, X ):
        h = self.h
        factor = 1.0 / numpy.sqrt( 2 * numpy.pi * h * h )
        #factor = 1.0 / numpy.sqrt( 2 * numpy.pi )
        log_factor = -0.5 * numpy.log( numpy.pi * h * h )

        log_dens = numpy.zeros( len(X) )
        log_len = numpy.log( len(self.data_) )
        for n in range(len(X)):
            distances = metrics.pairwise.euclidean_distances( self.data_, X[n].reshape(1,-1), squared=True )
            distances = - distances/(2*h*h)
            x = -numpy.inf
            for i in range(len(distances)): x = numpy.logaddexp( x, distances[i] )
            log_dens[n] = log_factor + x - log_len
            
        return log_dens
    # ------------------------------------------------------------------------------
