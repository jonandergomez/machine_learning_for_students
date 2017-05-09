"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: September 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""
import numpy

from .Constants import Constants
from machine_learning import GMM

class State:
    """
    """

    valid_modalities = [ 'Discrete', 'Continuous', 'Semi-continuous' ]

    def __init__( self, identifier=None, modality='Discrete', num_symbols=None, sample_dim=1, num_mixtures=0, input_file=None, first_line=None ):
        #
        if identifier is None:
            raise Exception( 'FATAL ERROR: the state identifer cannot be empty!' )
        if modality not in State.valid_modalities:
            raise Exception( 'FATAL ERROR: the provided modality is not valid: ' + modality )
        #
        self.identifier = identifier
        self.prior = 1.0
        self.modality = modality
        self.num_symbols = num_symbols
        self.sample_dim = sample_dim
        self.num_mixtures = num_mixtures
        if self.num_symbols is None  or  self.num_symbols <= 0 :
            self.num_mixtures = max( self.num_mixtures, 1 )
        self.B = None
        self.log_B = None
        self.gmm = None
        self.gmm_accumulator = None
        if self.num_mixtures > 0 :
            self.gmm = GMM( n_components=self.num_mixtures, dim=self.sample_dim, covar_type='diagonal' )
        self.global_index=-1
        #
        if input_file is not None:
            self.load( input_file, first_line )
        elif self.num_symbols is not None and self.num_symbols > 0:
            """
            The following initialization guarantees that a small asymmetry is provided at the beginning in
            order to avoid that the updates for all the parameters are allways the same.
            The method normalize() ensures that sum(self.B) is equal to one.
            """
            self.B = numpy.random.rand( num_symbols ) * 1.0e-5
        if self.B is not None:
            self.B_accumulator = numpy.copy( self.B )
        self.normalize()
        #self.reset_accumulators()
        #
        self.prior_count = 0.0
        #
        if self.B is None and self.gmm is None:
            raise Exception( 'FATAL ERROR: with the input parameters provided was not possible to build a GMM or set B!' )
        if self.B is not None and self.gmm is not None:
            raise Exception( 'FATAL ERROR: with the input parameters provided both the GMM and B have been set!' )
    # ------------------------------------------------------------------------------------------------------------------------------------------

    
    def reset_accumulators( self ):
        if self.B is not None:
            self.B_accumulator = numpy.zeros( len(self.B) )
        else:
            self.gmm_accumulator = GMM( n_components=self.gmm.n_components, dim=self.gmm.dim, covar_type=self.gmm.covar_type, min_var=self.gmm.min_var, _for_accumulating=True )

    def accumulate_sample( self, sample, weight=1.0, numerator=None, denominator=None ):
        if self.B is not None:
            if len(self.B_accumulator.shape) > 1  and  numerator is not None  and  denominator is not None:
                self.B_accumulator[sample,0] += numerator
                self.B_accumulator[sample,1] += denominator
            else:
                self.B_accumulator[sample] += weight
        else:
            self.gmm_accumulator.accumulate_sample( sample, self.gmm, weight )

    def split_gmm( self ):
        if self.B is None and self.gmm is not None:
            self.gmm.split()
            self.num_mixtures = self.gmm.n_components
            self.reset_accumulators()

    def normalize( self ):
        if self.B is not None:
            if len(self.B_accumulator.shape) > 1 :
                _acc_ = self.B_accumulator[:,0] / self.B_accumulator[:,1]
            else:
                _acc_ = self.B_accumulator.copy()
            if _acc_.sum() <= 0.0 :
                _acc_ = numpy.random.rand( num_symbols ) * 1.0e-5
            if _acc_.min() == 0.0: # Do smoothing by setting the positions with a zero to a value which is the tenth of the minimum
                _acc_[ _acc_ == 0.0 ] = _acc_[ _acc_ > 0.0 ].min() * 0.1
            self.B = _acc_ / _acc_.sum()
            self.log_B = numpy.log( self.B + Constants.k_zero_prob )
        else:
            if self.gmm_accumulator is not None:
                self.gmm.update_parameters( self.gmm_accumulator )
    
    def b( self, sample ):
        """
            This method works logarithms, always returns the logarithm of the probability or the probability density
        """
        #
        if self.B is not None:
            return self.log_B[ sample ] # Here 'sample' **MUST** be an int
        elif self.gmm is not None: # Here sample **MUST** be a real-valued array of dim equal to self.gmm.dim
            log_densities = self.gmm.log_densities( sample )
            m = log_densities.max()
            return m + numpy.log( numpy.exp( log_densities - m ).sum() )
        else:
            return Constants.k_min_allowed_log_prob # Simply the log of a non-zero probability


    #def merge_updatings( self, other ):
    #    self.prior_count += other.prior_count

    def save( self, f ):
        f.write( '~s "%s"\n' % self.identifier )
        if self.global_index >= 0 : f.write( '<GINDEX> %d\n' % self.global_index )
        f.write( '<PRIOR> %e\n' % self.prior )
        #
        if self.B is not None:
            f.write( '<B> %d\n' % len(self.B) )
            f.write( ' '.join( '{:17.8e}'.format(value) for value in self.B ) )
            f.write( '\n' )
        #
        f.write( '<NUMMIXES> %d\n' % self.num_mixtures )
        if self.num_mixtures > 0:
            for c in range(self.gmm.n_components):
                f.write( '<MIXTURE> %d\n' % (c+1) )
                f.write( '<GCONST> %17.8e %17.8e\n' % ( self.gmm.prioris[c], self.gmm.log_determinants[c] ) )
                f.write( '<MEAN> %d\n' % self.gmm.dim )
                f.write( ' '.join( '{:17.8e}'.format(value) for value in self.gmm.mu[c] ) )
                f.write( '\n' )
                f.write( '<VARIANCE> %d\n' % self.gmm.dim )
                f.write( ' '.join( '{:17.8e}'.format(self.gmm.sigma[c][i,i]) for i in range(self.gmm.dim) ) )
                f.write( '\n' )

    def load( self, f, line ):
        if line is None: line = f.readline()
        parts = line.split()
        self.identifier = parts[1].replace( '"', '' )
        self.prior = 1.0
        self.num_mixtures = 0
        line = f.readline()
        parts = line.split()
        while parts[0] != '<NUMMIXES>':
            if parts[0] == '<GINDEX>':
                self.global_index = int(parts[1])
            elif parts[0] == '<PRIOR>':
                self.prior = float(parts[1])
            elif parts[0] == '<B>':
                self.B = numpy.zeros( int(parts[1]) )
                line = f.readline()
                parts = line.split()
                self.B[:] = [float(x) for x in parts]
            #                                
            line = f.readline()
            parts = line.split()
        #
        self.num_mixtures = int(parts[1])
        self.gmm = None
        #
        if self.num_mixtures > 0:
            self.modality = 'Continuous'
            self.gmm = GMM( n_components = self.num_mixtures, dim=self.sample_dim, covar_type='diagonal' )
            for c in range(self.num_mixtures):
                line = f.readline()
                parts = line.split()
                k=int(parts[1])-1
                if k != c: raise Exception( 'Incorrect format in file with HMM definitions' )
                # <GCONST>
                line = f.readline()
                parts = line.split()
                self.gmm.prioris[c] = float(parts[1])
                if len(parts) > 2: self.gmm.log_determinants[c] = float(parts[2])
                # <MEAN>
                line = f.readline()
                parts = line.split()
                d = int(parts[1])
                if self.gmm.dim == 1  and  d != self.gmm.dim: self.gmm.dim = d
                if self.gmm.dim != d: raise Exception( 'Different values for dimensionality of samples encountered in the file with HMM definitions' )
                line = f.readline()
                parts = line.split()
                self.gmm.mu[c] = numpy.array( [float(value) for value in parts] )
                # <VARIANCE>
                line = f.readline()
                parts = line.split()
                d = int(parts[1])
                if self.gmm.dim != d: raise Exception( 'Different values for dimensionality of samples encountered in the file with HMM definitions' )
                line = f.readline()
                parts = line.split()
                for i in range(self.gmm.dim):
                    self.gmm.sigma[c] = numpy.diag( numpy.array( [ float(value) for value in parts] ) )
            self.gmm.compute_derived_parameters()

    def __str__(self): return self.identifier
    def __lt__(self,other):  return self.identifier < other.identifier
