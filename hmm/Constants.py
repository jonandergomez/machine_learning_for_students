"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: September 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import numpy


class Constants: 
    """
        Definition of constants to be used in the computations of probabilities.
    """
    k_zero_prob            =  1.0e-300
    k_log_zero             = -1.0e+30
    k_min_allowed_prob     =  1.0e-50
    k_min_allowed_log_prob = numpy.log( k_min_allowed_prob )
