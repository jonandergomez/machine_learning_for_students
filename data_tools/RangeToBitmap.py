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


class RangeToBitmap:

    def __init__( self, bounds=None, num_bits=None, h=None ):
        self.x_range = numpy.linspace( bounds[0], bounds[1], num_bits )
        if h is None:
            h = (bounds[1]-bounds[0])/(2*num_bits+1)
        self.h = h
        try:
            self.alpha = -0.5 / (h*h)
        except:
            print( bounds )
            print( num_bits )
            sys.exit(1)


    def bitmap( self, value ):
        y = value - self.x_range
        y = numpy.exp( self.alpha * y*y )
        return y / y.sum()

    
    def __len__( self ):
        return len(self.x_range)



if __name__ == '__main__':
    
    from bokeh.plotting import figure, output_file, show

    rtb = RangeToBitmap( bounds=[ numpy.log( 1.0e-5 ), 0.0 ], num_bits=10, h=None )

    output_file( '/tmp/rtb.html' )
    p = figure( title='Checking', x_range=[-16,3], x_axis_label='x', y_axis_label='bitmap value', width=900 )

    i=1
    values = numpy.linspace( -20.0, 2.0, 10 )
    for x in values:
        print( x )
        y = rtb.bitmap( x )
        print( y )
        color="#%02x%02x%02x" % ( int( (i*255) / len(values) ), 150, 150 )
        p.line( rtb.x_range, y, legend='%.4f' % x, line_width=2, line_color=color )
        i+=1

    show(p)
