"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: June 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Python file for defining the classes contained in this directory as a package.
    This package will be referenced with the name of the directory.

    Importing this package the classes enumerated bellow are imported automatically.

"""


from . import generate_datasets
#from .KMeans import KMeans
from .GMM import GMM
from .MLE import MLE
from .MyKernel import MyKernel
from .MyKernelClassifier import MyKernelClassifier
from .MyKMeans import KMeans
from .MyGaussian import MyGaussian
from .LinearDiscriminant import LinearDiscriminant


__all__ = [ 'KMeans', 'GMM', 'MLE', 'MyKernel', 'MyKernelClassifier', 'MyGausian', 'LinearDiscriminant', 'generate_datasets' ]
