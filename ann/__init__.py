"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: October 2015

    Python file for defining the classes contained in this directory as a package.
    This package will be referenced with the name of the directory.

    Importing this package the classes enumerated bellow are imported automatically.

"""


from .Functional import Functional
from .ANN import ANN
from .ANN import load

__all__ = ['Functional', 'ANN']
