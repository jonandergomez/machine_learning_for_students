"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: June 2015
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import sys
import numpy

class LinearDiscriminant:
    """
    This class implements Linear Discriminant Classifiers based on Least Mean Squares approach.

    The purpose is to classify each sample according to which class
    has a higher value of the linear function.

    """
    
    def __init__( self, l2_penalty=1e-5 ):
        self.num_classes=0
        self.dim=0
        self.targets = None
        self.weights = None
        self.l2_penalty = l2_penalty


    # ------------------------------------------------------------------------------
    def fit( self, X, Y, _targets_=None, ):
        self.dim=X.shape[1]
        if _targets_ is None:
            self.targets = numpy.unique(Y)
        else:
            self.targets = _targets_
        self.num_classes = len(self.targets)

        # Prepare Z as a matrix of one-hot vectors
        Z = numpy.zeros( [len(Y), self.num_classes], dtype=int )
        for n in range(len(Y)):
            Z[n][int(Y[n])] = 1

        #
        # W.T = dot( dot( inv( dot( X.T, X) - l2_penalty*I), X.T ), Z )
        # X.T is the transpose of X
        #
        temp = numpy.dot( X.T, X )
        temp = temp - self.l2_penalty * numpy.identity( temp.shape[0] )
        temp = numpy.linalg.inv( temp )
        temp = numpy.dot( temp, X.T )

        self.weights = numpy.dot( temp, Z )
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    def predict( self, X ):
        Y = numpy.zeros( len(X), dtype=int )
        Z = numpy.dot( X, self.weights )
        Y = numpy.argmax( Z, axis=1 )
        return Y
    # ------------------------------------------------------------------------------
