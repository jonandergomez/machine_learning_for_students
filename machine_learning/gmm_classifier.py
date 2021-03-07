'''
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: January 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

'''
import os
import sys
import numpy

from sklearn.mixture import GaussianMixture

class GMM_classifier:
    """
        According to the Bayes' rule:
            P(c) = A priori probability of classs c
            p(x|c) = Conditional probability density of observing the sample x when the state of the system corresponds to class c
            P(c|x) = P(c)*p(x|c) / p(x) -- A posteriori probability that the system is in state c when the sample x has been observed
            p(x) = Likelihood of sample x computed as the sumatory of all P(k)*p(x|k) for all the classes, not used here for classifying
    """

    def __init__(self, n_components, covar_type = 'diag', max_iter = 100, n_init = 1, verbose = 0, use_prioris = True):
        self.n_components = n_components
        self.covar_type = covar_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.verbose = verbose
        self.use_prioris = use_prioris
        #
        self._estimator_type = 'classifier'
        #
        self.num_classes = -1
        self.mixtures = None
        self.prioris = None

    def __str__(self):
        return f'GMM_classifier(n_components = {self.n_components}, covar_type = {self.covar_type}, use_prioris = {self.use_prioris})'
    
    def fit(self, X, y):
        num_classes = len(numpy.unique(y))
        if num_classes != self.num_classes:
            self.num_classes = num_classes
            self.mixtures = [GaussianMixture(n_components = self.n_components,
                                             covariance_type = self.covar_type,
                                             init_params = 'random',
                                             max_iter = self.max_iter,
                                             n_init = self.n_init,
                                             verbose = max(0, self.verbose - 1)) for c in range(self.num_classes)]
        #
        if self.use_prioris:
            self.log_prioris = numpy.array([numpy.log(sum(y == c)) for c in range(self.num_classes)]) - numpy.log(len(y))
        else:
            self.log_prioris = numpy.ones(self.num_classes) / self.num_classes
        #
        for c in range(self.num_classes):
            if self.verbose > 0:
                print("  GMM for class %2d ... " % c)
            x_train = X[y == c]
            self.mixtures[c].fit(x_train)
        #
        return self


    def log_densities(self, X):
        log_densities = numpy.zeros([len(X), self.num_classes])
        for c in range(self.num_classes):
            log_densities[:,c] = self.mixtures[c].score_samples(X)
        return log_densities

    def log_proba(self, X):
        post = self.posteriori(X)
        post = numpy.maximum(post, 1.0e-200)
        return numpy.log(post)

    def predict(self, X):
        return numpy.argmax(self.log_proba(X), axis = 1)

    def posteriori(self, X):
        log_proba = numpy.zeros([len(X), self.num_classes])
        for c in range(self.num_classes):
            log_proba[:,c] = self.log_prioris[c] + self.mixtures[c].score_samples(X)
        #
        _m_ = log_proba.max(axis = 1).reshape(-1, 1)
        log_proba -= _m_
        proba = numpy.exp(log_proba)
        _m_ = proba.sum(axis = 1).reshape(-1, 1)
        proba /= _m_
        #
        return proba

if __name__ == '__main__':
    x1 = 10 + 3 * numpy.random.randn(1000, 17)
    y1 = numpy.zeros(1000)
    x2 = -6 + 8 * numpy.random.randn(1000, 17)
    y2 = numpy.ones(1000)
    x3 = 46 + 2 * numpy.random.randn(1000, 17)
    y3 = numpy.ones(1000) * 2
    x = numpy.vstack([x1, x2, x3])
    y = numpy.hstack([y1, y2, y3]).astype(numpy.int32)
    print(x.shape)
    print(y.shape)
    gmmc = GMM_classifier(n_components = 3, covar_type = 'full', verbose = 1)
    gmmc.fit(x, y)
    z1 = 10 + 3 * numpy.random.randn(1000, 17)
    z2 = -6 + 8 * numpy.random.randn(1000, 17)
    z3 = 46 + 2 * numpy.random.randn(1000, 17)
    y1 = gmmc.predict(z1)
    y2 = gmmc.predict(z2)
    y3 = gmmc.predict(z3)
    print(sum(y1 == 0) / len(y1))
    print(sum(y2 == 1) / len(y2))
    print(sum(y3 == 2) / len(y3))
