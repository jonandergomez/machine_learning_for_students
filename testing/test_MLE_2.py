"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: June 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Testing Maximum Likelihood Estimation

"""

import sys
import numpy

import machine_learning

try:
    from pyspark import SparkContext
except:
    pass


def load_samples_from_file( filename ):
    f=open( filename, "rt" )
    line = f.readline()
    dim_x = len(line.split())
    counter = 1
    for line in f: counter+=1
    f.close()
    X = numpy.zeros( [ counter, dim_x ] )
    f=open( filename, "rt" )
    n=0
    for line in f:
        X[n,:] = [ float(x) for x in line.split() ]
        n+=1
    f.close()
    return X,None

def load_samples( index_filename ):
    f=open( index_filename, "rt" )
    X = []
    Y = []
    for filename in f:
        print( "loading file " + filename.strip() )
        _x_, _y_ = load_samples_from_file( filename.strip() )                        
        X.append( _x_ )
        Y.append( _y_ )
        #print( _x_.shape )
    f.close()
    return X,Y

if __name__ == "__main__":

    """
    Usage: spark-submit --master local[4]  python/gmm-mle.py  --base-dir .   --dataset data/samples.txt.gz  --covar full      --max-components 100  2>/dev/null
           spark-submit --master local[4]  python/gmm-mle.py  --base-dir .   --dataset data/samples.txt.gz  --covar diagonal  --max-components  50  2>/dev/null
    """

    verbose=False
    covar_type='diagonal'
    max_components=300
    dataset_filename=None
    base_dir='.'
    standalone=False
    spark_context = None
    slices=8
    batch_size = 100
                                                   
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--covar":
            covar_type = sys.argv[i+1]
        elif sys.argv[i] == "--max-components":
            max_components = int(sys.argv[i+1])
        elif sys.argv[i] == "--base-dir":
            base_dir = sys.argv[i+1]
        elif sys.argv[i] == "--dataset":
            dataset_filename = sys.argv[i+1]
        elif sys.argv[i] == "--verbosity":
            verbose = int(sys.argv[i+1])
        elif sys.argv[i] == "--standalone":
            standalone = True
        elif sys.argv[i] == "--num-slices":
            slices = int(sys.argv[i+1])
        elif sys.argv[i] == "--batch-size":
            batch_size = int(sys.argv[i+1])

    if not standalone :
        spark_context = SparkContext( appName="GMM-MLE-dataset-Rodrigo" )

    #os.makedirs( base_dir+'/log',    exist_ok=True )
    #os.makedirs( base_dir+'/models', exist_ok=True )

    if spark_context is not None:

        """
            Load all the lines in a file (or files in a directory) into an RDD of text lines.

            It is assumed there is no header, each text file contains an undefined number or lines.
            - Each line represents a sample.
            - All the lines **must** contain the same number of values.
            - All the values **must** be numeric, integers or real values.
        """
        text_lines = spark_context.textFile( dataset_filename )
        print( "file(s) loaded " )
        text_lines.persist()
        num_samples = text_lines.count()
        text_lines.unpersist()
        print( "loaded %d samples " % num_samples )

        # Repartition if necessary
        #if text_lines.getNumPartitions() < slices:
        #    text_lines = text_lines.repartition( slices )
        #    print( "rdd repartitioned to %d partitions" % text_lines.getNumPartitions() )

        """
            Convert the text lines into numpy arrays.

            Taking as input the RDD text_lines, a map operation is applied to each text line in order
            to convert it into a numpy array, as a result a new RDD of numpy arrays is obtained.
            
            Nevertheless, as we need an RDD with blocks of samples instead of single samples, we 
            associate with each sample a random integer number in a specific range.

            So, instead of an RDD with of numpy arrays we get an RDD with tuples [ int, numpy.array ]
        """
        K = (num_samples + batch_size - 1) / batch_size 
        K = ((K // slices)+1)*slices
        samples = text_lines.map( lambda line: ( numpy.random.randint(K), numpy.array( [ float(x) for x in line.split() ] ) ) )

        # HAS BEEN UNPERSISTED ABOVE: text_lines.unpersist() # This RDD is no longer needed

        # Shows an example of each element in the temporary RDD of tuples [key, sample]
        print( samples.first() )
        print( type(samples.first()) )

        """
            Thanks to the random integer number used as key we can build a new RDD of blocks
            of samples, where each block contains approximately the number of samples specified
            in batch_size.
        """
        samples = samples.reduceByKey( lambda x, y: numpy.vstack( [ x, y ] ) )

        # Repartition if necessary
        if samples.getNumPartitions() < slices:
            samples = samples.repartition( slices )
            print( "rdd repartitioned to %d partitions" % samples.getNumPartitions() )

        # Shows an example of each element in the temporary RDD of tuples [key, block of samples]
        print( samples.first() )
        print( type(samples.first()) )

        """
            Convert the RDD of tuples to the definitive RDD of blocks of samples
        """
        samples = samples.map( lambda x: x[1] )

        # Shows an example of each element in the temporary RDD of blocks of samples
        print( samples.first() )
        print( type(samples.first()) )

        samples.persist()
        print( "we are working with %d blocks of approximately %d samples " % ( samples.count(), batch_size ) )

        # Shows an example of shape of the elements in the temporary RDD of blocks of samples
        print( samples.first().shape )
        # Gets the dimensionality of samples in order to create the object of the class MLE.
        dim_x = samples.first().shape[1]

        mle = machine_learning.MLE( covar_type=covar_type, dim=dim_x, log_dir=base_dir+'/log', models_dir=base_dir+'/models' )

        mle.fit_with_spark( spark_context=spark_context, samples=samples, max_components=max_components )

        samples.unpersist()
        spark_context.stop()
    else:
        X_train,Y_train = load_samples( dataset_filename )
        dim_x = 0
        if type(X_train) == list :
            dim_x = X_train[0].shape[1]
        elif type(X_train) == numpy.ndarray:
            dim_x = X_train.shape[1]
        else:
            raise Exception( 'Non accepted data structure! :: %s' % (type(X_train)) )

        mle = machine_learning.MLE( covar_type=covar_type, dim=dim_x, log_dir=base_dir+'/log', models_dir=base_dir+'/models' )
        mle.fit_standalone( samples=X_train, max_components=max_components, batch_size=500 )
