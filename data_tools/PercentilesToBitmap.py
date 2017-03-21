"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: November 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import numpy


class PercentilesToBitmap:

    """
    n = 40033295
    total 3359938761.126422
    percentiles: 
             0        0.0100
        400332        1.0000
        800664        1.9900
       1200996        3.1300
       1601328        4.2400
       2001660        5.0000
       2401992        5.7500
    """

    def __init__( self, filename=None, values=None, accum=False ):

        if filename is not None:
            f=open( filename, 'rt' )
            line = f.readline() # n = ...
            line = f.readline() # total ...
            line = f.readline() # percentiles:
            values = list()
            line = f.readline() # first line
            values.append( float( line.split()[1] ) ) # adds the first value
            for line in f:
                v = float( line.split()[1] )
                if v > values[-1] : # adds those new values not in the list of values yet
                    values.append(v)
            #
            f.close()

        if values is None:
            raise Exception( "No values or filename for loading the values were provided!" )

        self.accum = accum
        self.values = numpy.array( values )

    def __getitem__( self, i ): return self.values[i]
    def __len__( self ): return len(self.values)

    def bitmap( self, value ):
        
        _values_ = numpy.zeros( len(self.values) )

        if self.accum :

            i=0
            while i < len(self.values)  and  value <= self.values[i] :
                _values_[i] = 1
                i+=1

        elif value < self.values[0]:
            _values_[0] = 1.0
            return _values_

        elif value > self.values[-1]:
            _values_[-1] = 1.0
            return _values_

        else:
            left=0
            right=len(self.values)-2
            while left <= right :
                pos = (left+right)//2
                if self.values[pos] <= value <= self.values[pos+1]:
                    break
                elif value < self.values[pos]:
                    right=pos-1
                else:
                    left=pos+1
        
            #print( "pos = ", pos, " ", self.values[pos], " ", value, " ", self.values[pos+1] )
            d1 =   value - self.values[pos]
            d2 = - value + self.values[pos+1]
            _values_[pos]   = d2/(d1+d2)
            _values_[pos+1] = d1/(d1+d2)

        return _values_




if __name__ == '__main__':
    
    ptb = PercentilesToBitmap( 'src/percentiles-2.txt' )

    print( len(ptb) )
    print( ptb[0] )
    print( ptb[1] )
    print( ptb[2] )
    print( ptb[len(ptb)-2] )
    print( ptb[len(ptb)-1] )

    print( ptb.bitmap( -1.0 ) )
    print( ptb.bitmap( 1.0e+15 ) )
    """
    print( ptb.bitmap( ptb[0] ) )
    print( ptb.bitmap( ptb[1] ) )
    print( ptb.bitmap( ptb[2] ) )
    print( ptb.bitmap( ptb[-2] ) )
    print( ptb.bitmap( ptb[-1] ) )
    """
    v = numpy.random.rand()*100
    print(v)
    print( ptb.bitmap( v ) )

