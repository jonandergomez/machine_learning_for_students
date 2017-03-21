"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: June 2014
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import sys
import numpy

from . import MyKernel

class MyKernelClassifier:
    """
    This class implements a classifier based on Kernel Density Estimator.

    The purpose is to classify each sample according to the class with higher probability density.

    """
    
    def __init__( self, h=None ):
        self.num_classes=0
        self.dim=0
        self.targets = None
        self.estimators = None # Kernel Density Estimators, one per class
        self.h = h


    # ------------------------------------------------------------------------------
    def fit( self, X, Y ):
        self.dim = X.shape[1]
        # Establish the value of 'h' if not set previously
        if self.h is None: self.h = max( 7, 2.5 * self.dim )
        self.targets = numpy.unique(Y)
        self.num_classes = len(self.targets)
        # Separate the training samples of each class in order to do the estimation
        samples_per_class = []
        for k in range(self.num_classes):
            samples_per_class.append( X[ Y==self.targets[k] ] )
        kernel='gaussian' # This could be a parameter for the constructor, but the
                          # current implementation of MyKernel.py doesn't allow a
                          # different kernel type.
        self.estimators=[]
        for k in range(self.num_classes):
            self.estimators.append( MyKernel( kernel=kernel, bandwidth=self.h ) )
            #self.estimators.append( MyKernel( kernel=kernel, bandwidth=1.0*len(samples_per_class[k])/6000.0 ) )
            self.estimators[k].fit( samples_per_class[k] )
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    def predict( self, X ):

        Y = numpy.zeros( len(X) )
        best_log_dens = numpy.zeros( len(X) )

        for k in range(self.num_classes):
            log_dens = self.estimators[k].score_samples(X)
            if 0 == k :
                best_log_dens[:] = log_dens[:]
                Y[:] = self.targets[0]
            else:
                for n in range(len(X)):
                    if log_dens[n] > best_log_dens[n]:
                        best_log_dens[n] = log_dens[n]
                        Y[n] = self.targets[k]
        return Y
    # ------------------------------------------------------------------------------
