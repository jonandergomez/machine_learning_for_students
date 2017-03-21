"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: November 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import numpy


class StandardScaler:
    """
    """

    def __init__( self, mean=None, std=None ):

        self.mean = mean
        self.std = std

    def bitmap( self, value ):
        
        if type(value) == numpy.ndarray :
            _values_ = numpy.array( [ (v-self.mean)/self.std for v in value ] )
        else:
            _values_ = numpy.array( [ (v-self.mean)/self.std for v in [value] ] )

        return _values_

    def __len__( self ): return 1

    def convert( self, x ): return (x - self.mean) / self.std

    def revert(  self, x ): return x * self.std + self.mean
