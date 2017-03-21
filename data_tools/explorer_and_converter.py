"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: November 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import numpy
#from matplotlib import pyplot

import os
import sys
import numpy
import pandas


from data_tools import FeatureExtractor

if __name__ == '__main__':

    phase=0
    data_filename=None
    metadata_input_filename=None
    metadata_output_filename=None

    for i in range(len(sys.argv)):
        
        if '--phase' == sys.argv[i] :
            phase = int(sys.argv[i+1])
        elif '--data' == sys.argv[i] :
            data_filename = sys.argv[i+1]
        elif '--input-metadata' == sys.argv[i] :
            metadata_input_filename = sys.argv[i+1]
        elif '--output-metadata' == sys.argv[i] :
            metadata_output_filename = sys.argv[i+1]
    

    data_frame = pandas.read_csv( data_filename, sep=';', delimiter=';', na_filter=True )


    if   1 == phase :
        """
            Performs the initial exploration of data in order to make a proposal
            of data type per input variable. Additionally ranges are computed.
        """

        keys, data_info = FeatureExtractor.explore_data( data_frame )

        FeatureExtractor.save_data_info( metadata_output_filename, keys, data_info )

    elif 2 == phase :
        """
            Performs some checks and recomputes intervals or percentiles distribution.
            This phase can be run several times in order to progressively refine 
            the set of values or the number or intervals or the values to be excluded.
        """

        keys, data_info = FeatureExtractor.load_data_info( filename=metadata_input_filename )

        FeatureExtractor.re_explore_data( data_frame, keys, data_info )

        FeatureExtractor.save_data_info( metadata_output_filename, keys, data_info )

    elif 3 == phase :
        """
            Converts input and output variables in order to generate the samples as
            they will be used in the classifier.
        """

        fe_input  = FeatureExtractor( filename_with_data_description=metadata_input_filename )
        fe_output = FeatureExtractor( filename_with_data_description=metadata_output_filename )

        print( len(fe_input), len(fe_output) )
        for i in range(len(data_frame)): 
            x = fe_input.convert(  data_frame, i )
            y = fe_output.convert( data_frame, i )
            #print( "%9.2f  " % x.sum(), " ".join( "{:.6f}".format(v) for v in x ) )
            if y.sum() > 0.0:
                print( " ".join( "{:.6f}".format(v) for v in x ),  "  ", " ".join( "{:.6f}".format(v) for v in y ) )

