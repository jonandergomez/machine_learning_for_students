"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: March 2015
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import numpy
#from matplotlib import pyplot

import numpy

def smoothing_by_kernel( y_from=None, n_from=None, n_to=None, duration=10.0, h=0.05 ):
    """
         y_from : is the array with the input signal to be smoothed and embeeded
                  into a new array with 'n_to' values
         n_from : number of points from 'y_from' to consider, if it is none all
                  the points in 'y_from' are used
           n_to : number of points for the smoothed signal
       duration : is the time in seconds the input signal corresponds to
              h : radius of Gaussian kernel used as the smoothing window

       returns 'y_to' containing the smoothed signal with 'n_to' points


       advice : 'duration' and 'h' are closely related, please use the proper values
                of these parameters according to your data
    """
    #
    if y_from is None : raise Exception( "Impossible reshaping with no data!" )
    if n_from is None : n_from = len(y_from)
    if n_to   is None : n_to = n_from
    #
    # 'x_from' contains the values of x for the input signal equally spaced
    x_from = numpy.linspace( 0.0, duration, n_from )
    # 'x_to' contains the values of x for the smoothed output signal equally spaced
    x_to   = numpy.linspace( 0.0, duration, n_to )
    # reservation of space for 'y_to'
    y_to   = numpy.zeros( n_to )
    # comuputation of the smoothed signal, each point in the input signal
    # contributes to estimated the value of all the points int the smoothed
    # output signal, but with a weight defined by the Gaussian window of
    # radius 'h'
    for t in range(n_to):
        y_filter = numpy.exp( -0.5*( (x_from - x_to[t])/h )**2 )
        y_to[t] = (y_filter * y_from[:n_from]).sum() / y_filter.sum()
    return y_to
