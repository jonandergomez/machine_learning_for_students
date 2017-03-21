"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: November 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import os
import sys
import numpy
import pandas


from .StandardScaler      import StandardScaler
from .CategoriesToBitmap  import CategoriesToBitmap
from .PercentilesToBitmap import PercentilesToBitmap
from .RangeToBitmap       import RangeToBitmap


class FeatureExtractor:
    """
        { 'type' : attribute_type,
          'excluded' : excluded_values,
          'min' : min_value,
          'max' : max_value,
          'num_intervals' : num_intervals,
          'set_of_values' : set_of_values }
    """
    CATEGORICAL='categorical'
    NUMERICAL='numerical'
    RANGE_OF_VALUES='range_of_values'
    PERCENTILES='percentiles'
    PERCENTILES_ACCUM='percentiles_accum'
    ZERO_MEAN_VARIANCE_ONE='zero_mean_variance_one'
    INTEGER='integer'
    valid_data_types = [ CATEGORICAL, NUMERICAL, RANGE_OF_VALUES, PERCENTILES, PERCENTILES_ACCUM, ZERO_MEAN_VARIANCE_ONE, INTEGER ]
    
    # -----------------------------------------------------------------------------------------
    def __init__( self, filename_with_data_description=None ):

        self.columns, self.data_info = FeatureExtractor.load_data_info( filename=filename_with_data_description )

        self.extractors = dict()
        self.size=0

        for column in self.columns:

            di = self.data_info[column]
            t = self.data_info[column]['type']

            if t == FeatureExtractor.CATEGORICAL:
                extractor = CategoriesToBitmap( values=di['set_of_values'] )

            elif t == FeatureExtractor.NUMERICAL:
                extractor = PercentilesToBitmap( values=numpy.array( [ int(x) for x in di['set_of_values'] ] ) )

            elif t == FeatureExtractor.PERCENTILES:
                extractor = PercentilesToBitmap( values=numpy.array( [ float(x) for x in di['set_of_values'] ] ) )

            elif t == FeatureExtractor.RANGE_OF_VALUES:
                min_value = di['min']
                max_value = di['max']
                num_intervals = di['num_intervals']
                if num_intervals is None or num_intervals <= 0:
                    num_intervals = 10
                extractor = RangeToBitmap( bounds=[ min_value, max_value ], num_bits=num_intervals )

            elif t == FeatureExtractor.ZERO_MEAN_VARIANCE_ONE :
                
                mean = float(di['set_of_values'][0])
                std  = float(di['set_of_values'][1])
                extractor = StandardScaler( mean, std )

            elif t == FeatureExtractor.INTEGER:

                extractor = StandardScaler( 0.0, 1.0 )
                                

            self.extractors[column] = extractor
            self.size += len(extractor)
    # -------------------------------------------------------------------------------------------------

    def __len__(self): return self.size
    # -------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    def convert( self, df=None, index=-1 ):
        
        x = numpy.zeros( self.size )
        i = 0
        for column in self.columns:
            extractor = self.extractors[column]
            v = df[column][index]     
            if v != v :
                #raise Exception( "v is nan! -- %f " % v );
                if extractor is CategoriesToBitmap:
                    v = None
                else:
                    v = -1
            z = extractor.bitmap( v )
            #print( column )
            #print( type(extractor) )
            #print( type(v), v )
            #print( type(z), z )
            for k in range(len(z)):
                if z[k] is None or z[k] != z[k]: sys.stderr.write( "ERROR: ", z, '\n' )
            x[i:i+len(extractor)] = z[:]
            i+=len(extractor)
        #
        return x
    # -------------------------------------------------------------------------------------------------


    # -----------------------------------------------------------------------------------------
    def load_data_info( filename=None ):
        """
            N_ID;categorical;;;;;
            Sexo;categorical;;;;;
            Edad;rango;;0;99;;20;
            Fototipo;categorical;99;;;;
            Ojos_R;categorical;99;;;;
            Pelo R;categorical;99;;;;
        """
        keys=list()
        data_info=dict()
        f=open( filename, 'rt' )
        for line in f:
            parts=line.split(';')
            attribute_name=parts[0].strip()
            attribute_type=parts[1].strip()
            if len(parts[2]) > 0:
                excluded_values=[ x.replace( '%20', ' ' ) for x in parts[2].split() ]
            else:
                excluded_values=list()
            min_value=float(parts[3].strip()) if len( parts[3] ) > 0 else None
            max_value=float(parts[4].strip()) if len( parts[4] ) > 0 else None
            num_intervals=int(parts[5].strip()) if len( parts[5] ) > 0 else None
            if len(parts[6]) > 0:
                #set_of_values=[ float(x) for x in parts[6].split() ]
                set_of_values=[ x.replace( '%20', ' ' ) for x in parts[6].split() ]
                if   attribute_type == FeatureExtractor.NUMERICAL :
                    set_of_values=numpy.array( [int(x) for x in set_of_values], dtype=numpy.int32 )
                elif attribute_type == FeatureExtractor.PERCENTILES :
                    set_of_values=numpy.array( [float(x) for x in set_of_values] )
            else:
                set_of_values=None
            keys.append(attribute_name)
            data_info[attribute_name]={ 'type' : attribute_type,
                                        'excluded' : excluded_values,
                                        'min' : min_value,
                                        'max' : max_value,
                                        'num_intervals' : num_intervals,
                                        'set_of_values' : set_of_values };
        f.close()
        return keys,data_info
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    def save_data_info( filename=None, keys=None, data_info=None ):
        f=open( filename, 'wt' )
        for key in keys:
            d = data_info[key]

            join_format="{:s}"
            if d['type'] == FeatureExtractor.NUMERICAL: join_format="{:d}"
            elif d['type'] == FeatureExtractor.RANGE_OF_VALUES: join_format="{:f}"

            f.write( "%s;%s;%s;%s;%s;%s;%s;\n" % (
                        key,
                        d['type'],
                        #" ".join( join_format.format(x) for x in d['excluded'] ),
                        " ".join( "{:s}".format(str(x).replace( ' ', '%20' )) for x in d['excluded'] ),
                        d['min'] if d['min'] is not None else "",
                        d['max'] if d['max'] is not None else "",
                        d['num_intervals'] if d['num_intervals'] is not None else "",
                        ( " ".join( "{:s}".format(str(x).replace( ' ', '%20' )) for x in d['set_of_values'] ) if  d['set_of_values'] is not None else "" ),
                   ) )
        f.close()
    # -----------------------------------------------------------------------------------------


    # -----------------------------------------------------------------------------------------
    def explore_data( data_frame=None ):
        """
            Performs the initial exploration of data, without distinguishig between input and output variables
            in order to detect basic anomalies in data and empty columns.
        """
        if data_frame is None:
            raise Exception( 'Impossible to work with no data!' )

        df = data_frame
        keys = list()
        data_info = dict()
        for attribute_name in df.columns:
            keys.append( attribute_name )
            X = df[attribute_name].copy()
            d_type = dict()
            for x in X:
                t=type(x)
                if t in d_type:
                    d_type[type(x)] += 1.0
                else:
                    d_type[type(x)] = 1.0
            #
            type_=None
            rate_=0
            for k in d_type.keys():
                c = d_type[k] / len(X)
                if c > rate_:
                    type_=k
                    rate_=c
            #
            print( attribute_name, len(d_type), d_type, rate_, type_ )

            if rate_ < 0.90 :
                raise Exception( 'Unbalanced data type in attribute!' )

            if type_ in [ float, numpy.float32, numpy.float64 ] :
                min_value = -1.0
                for i in range(len(X)):
                    if X[i] == X[i]: min_value = min(X[i],min_value)
                empty_value = min_value - 10
                for i in range(len(X)):
                    if X[i] != X[i]: X[i] = empty_value
                type_ = FeatureExtractor.RANGE_OF_VALUES
                #
            elif type_ in [ int, numpy.int32, numpy.int64 ] :
                min_value = -1
                for i in range(len(X)):
                    if X[i] == X[i]: min_value = min(X[i],min_value)
                empty_value = min_value - 10.0
                for i in range(len(X)):
                    if X[i] != X[i]: X[i] = empty_value
                type_ = FeatureExtractor.NUMERICAL
                #
            elif type_ in [ str ] :
                empty_value = "<empty>"
                for i in range(len(X)):
                    if X[i] != X[i]:
                        X[i] = empty_value
                type_ = FeatureExtractor.CATEGORICAL
                #
            else:
                raise Exception( 'Unexpected data type in attribute!' )

            excluded_values = [ empty_value ]

            X = numpy.array( X )
                
            try:
                uniqueX = numpy.unique(X)
            except:
                for v in x : print( v, end=' ' )
                sys.exit(1)

            print( '%12s |different values| = %d' % (' ', len(uniqueX)) )
            if len(uniqueX) < 20 :
            #if len(uniqueX) < 0.1 * len(X) :
                print( '%12s ' % ' ', uniqueX )

            if type_ == FeatureExtractor.NUMERICAL: # in [ int, numpy.int32, numpy.int64 ] :
                #
                if len(uniqueX) <= 100:
                    set_of_values=list()
                    for value in uniqueX:
                        if value not in excluded_values:
                            set_of_values.append(value)
                    min_value = None
                    max_value = None
                    num_intervals=None
                    type_ = FeatureExtractor.CATEGORICAL
                else:
                    min_value = uniqueX[uniqueX > empty_value].min()
                    max_value = uniqueX.max()
                    print( '%12s Range: ' % ' ', min_value, ' .. ', max_value )
                    num_intervals=min(len(uniqueX),100)
                    set_of_values=None
                #
            elif type_ == FeatureExtractor.RANGE_OF_VALUES: #in [ float, numpy.float32, numpy.float64 ] :
                #
                min_value = uniqueX[uniqueX > empty_value].min()
                max_value = uniqueX.max()
                print( '%12s Range: ' % ' ', min_value, ' .. ', max_value )
                num_intervals=min(len(uniqueX),100)
                set_of_values=None
                #
            else:
                for i in range(len(uniqueX)):
                    uniqueX[i] = uniqueX[i].replace( ' ', '%20' )
                min_value=None
                max_value=None
                num_intervals=None
                set_of_values=list()
                for value in uniqueX:
                    if value not in excluded_values:
                        set_of_values.append(value)
            print()

            data_info[attribute_name]={ 'type' : type_,
                                        'excluded' : excluded_values,
                                        'min' : min_value,
                                        'max' : max_value,
                                        'num_intervals' : num_intervals,
                                        'set_of_values' : set_of_values };

        return keys,data_info
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    def re_explore_data( data_frame=None, keys=None, data_info=None ):
        
        for key in keys:
            d = data_info[key]
            X = data_frame[key].copy()

            type_ = d['type']
            min_ = d['min']
            max_ = d['max']
            num_intervals = d['num_intervals']
            set_of_values = d['set_of_values']
            excluded = d['excluded']

            if   type_ == FeatureExtractor.CATEGORICAL:
                pass

            elif type_ == FeatureExtractor.NUMERICAL:
                d['min'] = int(d['min'])
                d['max'] = int(d['max'])

            elif type_ == FeatureExtractor.INTEGER:
                pass
            elif type_ == FeatureExtractor.RANGE_OF_VALUES:
                pass

            elif type_ == FeatureExtractor.PERCENTILES:

                if set_of_values is None:

                    X.fillna( value = excluded[0], inplace=True )
                    a = list()
                    for x in X:
                        if x not in excluded: a.append(x)
                    X = numpy.array(a)
                    X.sort()
                    step = int( len(X) / (num_intervals-2) )
                    V = list()
                    for i in range( 0, len(X), step ):
                        if X[i] not in V:
                            V.append( X[i] )
                    V.append( X[-1] )

                    d['set_of_values'] = numpy.array(V)
                    d['num_intervals'] = len(V)
                    d['min'] = None
                    d['max'] = None

            elif type_ == FeatureExtractor.ZERO_MEAN_VARIANCE_ONE:

                a = list()
                for x in X:
                    if x not in excluded and x == x: a.append(x)
                X = numpy.array(a)
                mean = X.mean()
                std = X.std()
                #print( X, mean, std )

                d['set_of_values'] = [ mean, std ]
                d['num_intervals'] = None
                d['min'] = None
                d['max'] = None

            else:
                raise Exception( 'Unexpected data type : %s !' % type_ )
    # -----------------------------------------------------------------------------------------
