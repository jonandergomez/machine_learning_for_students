"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: November 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import numpy
#from matplotlib import pyplot
import sys
import os
import numpy


class CategoriesToBitmap:

    def __init__( self, values=None, subset=None, to_be_excluded=None, mass=0.95, verbose=0 ):
        
        if values is not None:
            pass
        elif subset is not None:
            # Convert the data in a Numpy array
            _x_ = numpy.array(subset)
            # Get the set of possible values
            _v_ = numpy.unique(_x_)
            # Build a dictionary with all the possible values
            _d_ = dict()
            i=0
            if to_be_excluded is None: to_be_excluded = list()
            for v in _v_:
                if v not in _d_  and  v not in to_be_excluded:
                    _d_[v] = i
                    i+=1

            # Computes an array of counters for all the possible values
            _c_ = numpy.zeros(len(_d_))
            for v in subset:
                if v in _d_:
                    _c_[_d_[v]] += 1

            # Looks for the threshold for ignoring values with low probability
            _c_ = _c_ / _c_.sum()
            _temp_ = _c_.copy()
            _temp_.sort()
            i=len(_temp_)
            accum = 0.0
            while i > 0 and accum <= mass :
                i-=1
                accum += _temp_[i]
            
            threshold = _temp_[i]

            if verbose is not None and verbose > 0:
                print(threshold)
                print(_c_)

            values = list()
            for v in _v_:
                if v in _d_ and _c_[_d_[v]] >= threshold:
                    values.append(v)
        else:
            raise Exception( "Impossible to create an object of the class CategoriesToBitmap without values or a subset where to compute the values from." );

        self.values=dict()
        i=0
        for value in values:
            self.values[value] = i
            i+=1
    # -----------------------------------------------------                

    def __len__(self): return len(self.values)

    def bitmap(self,value):
        x = numpy.zeros(len(self))
        if value in self.values:
            x[ self.values[value] ] = 1.0
        return x



if __name__ == '__main__':
    
    B = [ 'AA', 'AA', 'AA', 'AA', 'BB', 'BB', 'BB', 'CC', 'CC', 'DD' ]
    X = []
    for i in range(20):
        X += [ B[ numpy.random.randint(len(B)) ] ]
    X.append( 'EE' )
    ctb = CategoriesToBitmap( subset=X, mass=0.90 )

    B.append( 'EE' )

    for x in numpy.unique(B):
        print( x , ctb.bitmap(x) )
