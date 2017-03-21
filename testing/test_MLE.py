"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 1.0
    Date: November 2015
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

    if not standalone :
        spark_context = SparkContext( appName="GMM-MLE-example" )
                                                                                                                                                                                           
    num_classes=50
    #X_train,Y_train,X_test,Y_test = generate_datasets.generate_multivariate_normals( 5, 2, 150, 50, 5.0, 2.0 )
    X_train,Y_train,X_test,Y_test = machine_learning.generate_datasets.generate_multivariate_normals( num_classes, 7, 25000, 5000, 15.0, 12.0 )

    #X_train,Y_train = load_samples( dataset_filename )

    #os.makedirs( base_dir+'/log',    exist_ok=True )
    #os.makedirs( base_dir+'/models', exist_ok=True )

    mle = machine_learning.MLE( covar_type=covar_type, dim=X_train.shape[1], log_dir=base_dir+'/log', models_dir=base_dir+'/models', batch_size=500 )

    if spark_context is not None:
        samples = spark_context.parallelize( X_train, slices )
        samples.persist()
        mle.fit_with_spark( spark_context=spark_context, samples=samples, max_components=max_components )
        samples.unpersist()
        spark_context.stop()
    else:
        mle.fit_standalone( samples=X_train, max_components=max_components )
