"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: September 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""
import sys
import numpy

from machine_learning import GMM

from .Constants   import Constants
from .State       import State
from .Transitions import Transitions

class HMM:
    """
        An HMM is characterized by the following attributes:
            
            S: the set of states

               states must be able to compute b(s,o_k), the probability of emitting
               the symbol 'o_k' by the state 's'
               in the continuous case we would say: the probabiblity of generating
               the observation or vector 'o_k' by the state 's'

            P: the set of initial probabilities of states

               P[i] is the probability that a sequence of observations starts in state 'i'

            A: the matrix of transitition probabilities from one state to another

               row 'i' contains the transition probibilities from state 'i' to each other


        An HMM can be discrete, semi-continuous or continuous.

        Depending on these modalities the computations of b(s,o_k) will be carried out in
        differerent ways.
    """

    def __init__( self, identifier=None, modality='Discrete', dict_1=None, dict_2=None, dict_3=None ):
        """
            This constructor admits the creation of an HMM in three ways:

                1) from existing states and transitions (parameters in dict_1)
                   this case is when each HMM represents a model in an application as ASR or HTR,
                   where the topology is left-to-right with two phantom states, the initial one and the final one
                   then 'P' is not needed, it is implicit in the first row of the transition matrix A

                2) from a given input file, in this case the first line is needed because it is assumed (parameters in dict_2)
                   it was read by the caller method who provides the input file
                   there are also needed transitions and states, but in this case as dictionaries, see the load() method

                3) from scratch, in this case num_states is mandatory (parameters in dict_3)
        """
        #
        if modality not in State.valid_modalities:
            raise Exception( 'Fatal error: no valid modality: ' + modality )
        #
        self.identifier = identifier
        self.S = None
        self.A = None
        self.P = None
        self.modality = modality
        self.left_to_right = True
        self.num_symbols = None
        #
        if dict_1 is not None:
            #
            self.transitions = dict_1['transitions']
            self.states      = dict_1['states']
            if 'left_to_right' in dict_1 :
                self.left_to_right = dict_1['left_to_right']
            if 'num_symbols' in dict_1 :
                self.num_symbols = dict_1['num_symbols']
            #
        elif dict_2 is not None:
            #
            if 'left_to_right' in dict_2 : self.left_to_right = dict_2['left_to_right']
            #
            self.load( dict_2['input_file'], dict_2['first_line'], dict_2['d_transitions'], dict_2['d_states'] )
            #
        elif dict_3 is not None:
            #
            num_states = dict_3['num_states']
            self.sample_dim = 1
            num_mixtures=0
            if 'left_to_right' in dict_3 :
                self.left_to_right = dict_3['left_to_right']
            if 'sample_dim' in dict_3 :
                self.sample_dim = dict_3['sample_dim']
            if 'num_mixtures' in dict_3 :
                num_mixtures = dict_3['num_mixtures']
            if self.modality == 'Discrete':
                self.num_symbols = dict_3['num_symbols']
            elif self.modality == 'Semi-continuous' :
                classifier = dict_3['classifier']
                self.num_symbols = len(classifier)
            #
            # if self.num_symbols is None the State will be created as continuous by using a GMM.
            #
            self.A = Transitions( identifier=('T_%s' % identifier), num_states=num_states, dict_args=dict_3 )
            #
            self.S = [None] * num_states
            if self.left_to_right:
                for s in range(1,len(self.S)-1):
                    self.S[s] = State( identifier=('S_%s_%d' % ( identifier, s ) ), modality=self.modality, num_symbols=self.num_symbols, sample_dim=self.sample_dim, num_mixtures=num_mixtures, input_file=None, first_line=None )
            else:
                for s in range(len(self.S)):
                    self.S[s] = State( identifier=('S_%s_%d' % ( identifier, s ) ), modality=self.modality, num_symbols=self.num_symbols, sample_dim=self.sample_dim, num_mixtures=num_mixtures, input_file=None, first_line=None )
            #
            if (not self.left_to_right) and 'pi' in dict_3 :
                self.P = dict_3['pi']
            else:
                self.P = numpy.ones( len(self.S) ) / len(self.S)
                if self.A.force_to_one_terminal_state:
                    self.P = numpy.ones( len(self.S) ) / (len(self.S)-1)
                    self.P[-1] = 0.0
            #
        #
        if self.identifier is None : raise Exception( "Cannot create an HMM without a valid identifier!" )
        if self.A          is None : raise Exception( "Cannot create an HMM without transitions!" )
        if self.S          is None : raise Exception( "Cannot create an HMM without states!" )
        if self.P is not None :
            self.log_P = numpy.log( self.P + Constants.k_zero_prob )
            self.P_accumulator = numpy.zeros( self.P.shape )
        else:
            self.P_accumulator = None
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def __len__(self): return len(self.S)
    def __str__(self): return self.identifier

    def load( self, f, line, d_transitions, d_states ):
        """
        """
        num_states=0
        if line is None: line=f.readline()
        parts = line.split()
        while parts[0] != '~h' :
            if parts[0] == '~t' :
                self.A = Transitions( input_file=f, first_line=line, dict_args=dict( left_to_right=self.left_to_right ) )
                d_transitions[ str(self.A) ] = self.A
            elif parts[0] == '~s' :
                state = State( identifier=parts[1].replace( '"', '' ), modality='Discrete', sample_dim=1, num_mixtures=0, input_file=f, first_line=line )
                d_states[str(state)] = state
            #
            line = f.readline()
            parts = line.split()
        #
        self.identifier = parts[1].replace( '"', '' )
        line = f.readline()
        while line:
            parts = line.split()
            if parts[0] == "<ENDHMM>":
                break
            elif parts[0] == "<NUMSTATES>":
                num_states=int(parts[1])
                self.S = [None] * num_states
            elif parts[0] == "<STATE>":
                i = int(parts[1])-1
                line = f.readline()
                parts = line.split()
                if parts[0] != '~s' : raise Exception( 'ERROR reading %s' % line )
                self.S[i] = d_states[ parts[1].replace( '"', '' ) ]                  
            elif parts[0] == "~t":
                self.A = d_transitions[ parts[1].replace( '"', '' ) ]
            #
            line = f.readline()
        self.P = numpy.array( [s.prior for s in self.S ] )
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def initialize_gmm_from_random_samples( self, samples ):
        for s in self.S:
            s.gmm.initialize_from( samples )
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def save( self, f, save_states_and_transitions=False ):
        """
            This method assumes that the details of transitions and states has been saved
            before and only the identifiers of each state and transition matrix should be
            stored here in the definition of the HMM
        """
        if save_states_and_transitions:
            self.A.save(f)
            for state in self.S:
                state.save(f)

        if self.P is not None:
            for i in range(len(self.S)):
                self.S[i].prior = self.P[i]

        f.write( '~h "%s"\n' % self.identifier )
        f.write( '<BEGINHMM>\n' )
        f.write( '<NUMSTATES> %d\n' % len(self.S) )
        if self.left_to_right:
            _range_ = range( 1, len(self.S)-1 )
        else:
            _range_ = range( len(self.S) )
        for i in _range_:
            s = self.S[i]
            f.write( '<STATE> %d\n' % (i+1) )
            f.write( '~s "%s"\n' % str(s) )
        f.write( '~t "%s"\n' % str(self.A) )
        f.write( '<ENDHMM>\n' )
    # --------------------------------------------------------------------------------------------------------------------------------------------


    def get_state( self, i ): return self.S[i]

    def get_states( self ): return self.S[1:-1] if self.left_to_right else self.S[:]

    def log_add( log_probs ):
        _max_ = log_probs.max()
        _sum_ = numpy.log( numpy.exp( log_probs - _max_ ).sum() ) + _max_
        return _sum_
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def forward( self, O ):
        """
            Version for training a unique HMM, not valid for the concatenation of several HMM as used in ASR or HTR

            'O' can be an array of symbols (indexes corresponding to symbols) or an array of real-valued vectors
        """

        alpha = numpy.zeros( [ len(O), len(self) ] )
        B = numpy.zeros( [ len(self), len(O) ] )

        for j in range(len(self.S)):
            B[j,0] = self.S[j].b( O[0] )
            alpha[0,j] = self.log_P[j] + B[j,0]


        for t in range( 1, alpha.shape[0] ):
            for j in range(alpha.shape[1]):
                #
                B[j,t] = self.S[j].b( O[t] )
                #
                # alpha[t,j] = sum_i( alpha[t-1,i] * A[i,j] ) * B[j, O[t] )
                #
                alpha[t,j] = HMM.log_add( alpha[t-1,:] + self.A.log_transitions[:,j] ) + B[j,t]

        return alpha, B, HMM.log_add( alpha[-1,:] )
    # --------------------------------------------------------------------------------------------------------------------------------------------
    
    def backward( self, O, B, final_probs=None, terminal_nodes=None ):
        """
            Version for training a unique HMM, not valid for the concatenation of several HMM as used in ASR or HTR
        """

        beta = numpy.zeros( [ len(O), len(self) ] )

        if final_probs is not None:
            beta[-1,:] = numpy.log( final_probs + Constants.k_zero_prob )
        elif terminal_nodes is not None:
            beta[-1,:] = Constants.k_log_zero
            for s in terminal_nodes:
                beta[-1,s] = 0.0
        else:
            beta[-1,:] = 0.0 # log(1.0)

        t=beta.shape[0]-2
        while t >= 0:
            for i in range( beta.shape[1] ):
                #
                # beta[t,i] = sum_j( A[i,j] * B[j,O[t+1]) * beta[t+1,j] )
                #
                beta[t,i] = HMM.log_add( self.A.log_transitions[i,:] + B[:,t+1] + beta[t+1,:] )
            t-=1

        return beta
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def forward_backward( self, O ):
        """
            Version for training a unique HMM, not valid for the concatenation of several HMM as used in ASR or HTR
        """
        alpha, B, _Z_ = self.forward( O )
        beta = self.backward( O, B )
        gamma = alpha + beta # This must be a sum because what is stored in 'alpha' and 'beta' are logarithms of probabilities
        # Gamma must be normalized by means of P(O|lambda) : _Z_ = log P(O|lambda)
        gamma = gamma - _Z_

        return alpha, beta, gamma, B, _Z_
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def baum_welch_reset( self ):
        self.A.reset_accumulators()
        for s in self.S: s.reset_accumulators()
        self.P_accumulator = None
        if self.P is not None:
            self.P_accumulator = numpy.zeros( self.P.shape )
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def baum_welch_update( self ):
        self.A.update_transitions()
        for s in self.S: s.normalize()
        if self.P_accumulator is not None:
            if self.A.force_to_one_terminal_state: self.P_accumulator[-1] = 0.0
            self.P = self.P_accumulator / self.P_accumulator.sum()
            self.log_P = numpy.log( self.P + Constants.k_zero_prob )
        for i in range(len(self.S)):
            self.S[i].prior = self.P[i]
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def baum_welch_from_a_list( self, list_of_observations, do_reset=True, verbose=True ):
        
        if do_reset: self.baum_welch_reset()

        if verbose: sys.stderr.write( 'Training samples %6d: \n' % len(list_of_observations) )
        counter=0
        for O in list_of_observations:
            self.baum_welch( O )
            counter+=1
            if verbose:
                sys.stderr.write( '\r %22d' % counter )
                # print( " ".join( "{}".format(x) for x in O ) )
        self.baum_welch_update()
        if verbose:
            sys.stderr.write( '\n\n' )
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def baum_welch( self, O ):
        """
            Version for training a unique HMM, not valid for the concatenation of several HMM as used in ASR or HTR
        """
        alpha, beta, gamma, B, _Z_ = self.forward_backward( O )
        #
        """
        gamma_tr = numpy.zeros( [ len(O)-1], len(self), len(self) ] )
        for t in range(gamma_tr.shape[0]):
            for i in range( gamma_tr.shape[1] ):
                for j in range( gamma_tr.shape[1] ):
                    gamma_tr[t,i,j] = alpha[t,i] + self.A.get_log_prob(i,j) + B[j,t+1] + beta[t+1,j]  -  gamma[t,i]
        #
        for i in range( gamma_tr.shape[1] ):
            for j in range( gamma_tr.shape[1] ):
                _weight_ = HMM.log_add( gamma_tr[:,i,j] )
                temp_hmm.A.accumulate_transition( i, j, numpy.exp( _weight_ ) ) # This line is candidate to be modified for accumulating logarithms
        #
        """

        # UPDATE OF THE STATE-TRANSITION PROBABILITIES
        if len(O) > 1:
            for i in range( len(self) ):
                _log_den_ = HMM.log_add( gamma[:-1,i] ) # sum(t=1..T-1, gamma[t,i] )
                for j in range( len(self) ):
                    gamma_tr = numpy.zeros( len(O)-1 )
                    for t in range(gamma_tr.shape[0]):
                        gamma_tr[t] = alpha[t,i] + self.A.get_log_prob(i,j) + B[j,t+1] + beta[t+1,j]  -  _Z_
                    _weight_ = numpy.exp( HMM.log_add( gamma_tr[:] ) - _log_den_ )
                    self.A.accumulate_transition( i, j, value=_weight_ ) # This line is candidate to be modified for accumulating logarithms
        #
        # UDPDATE OF THE STATE STARTING PROBABILITIES
        if self.P_accumulator is not None:
            self.P_accumulator[:] += gamma[0,:]

        # UDPDATE OF THE OUTPUT PROBABILITIES
        if self.modality in [ 'Discrete' ]:
            #
            for i in range( gamma.shape[1] ):
                #
                _log_den_ = HMM.log_add( gamma[:,i] ) # sum(t=1..T, gamma[t,i] )
                _den_ = numpy.exp(_log_den_)
                #
                for k in numpy.unique(O): # range(self.num_symbols)
                    _log_num_ = HMM.log_add( gamma[ O==k, i ] )
                    _weight_ = numpy.exp( _log_num_ - _log_den_ )
                    self.S[i].accumulate_sample( k, _weight_, numpy.exp(_log_num_), _den_ ) # This line is candidate to be modified for accumulating logarithms 
                #
        elif self.modality in [ 'Continuous' ]:
            #
            for j in range( len(self) ):
                #
                _log_denominator_ = HMM.log_add( gamma[:,j] ) # sum(t=1..T, gamma[t,i] )
                _denominator_ = numpy.exp( _log_denominator_ )
                #
                _log_densities_ = numpy.zeros( [ len(O), self.S[j].gmm.n_components ] )
                for t in range( len(O) ):
                    _log_densities_[t,:] = self.S[j].gmm.log_densities( O[t] )  # log( c_j_k * g_j_k( O_t ) )
                #
                log_xi = numpy.zeros( len(O) ) # A one-dimensional vector for computing _xi_t_j_k_ for fixed 'j' and 'k'
                for k in range( _log_densities_.shape[1] ):
                    log_xi[0] = self.log_P[j] + _log_densities_[0,k] + beta[0,j] # _xi_0_j_k_
                    #
                    for t in range( 1, len(O) ):
                        _temp_ = numpy.zeros( len(self) )
                        for i in range( len(self) ): # For all the states in the HMM
                            _temp_[i] = alpha[t-1,i] + self.A.get_log_prob(i,j) + _log_densities_[t,k] + beta[t,j]
                        log_xi[t] = HMM.log_add( _temp_ ) # _xi_t_j_k_  for all t > 0
                    #
                    log_xi -= _Z_ # Dividing by P(O|lambda)
                    #
                    _xi_t_j_k_ = numpy.exp( log_xi )
                    #
                    # In the following lines the code of Baum-Welch directly modifies the accumulators
                    # of the GMM of each state 'j'
                    #
                    self.S[j].gmm_accumulator.acc_posteriors[k] += _xi_t_j_k_.sum() # This value is correct because is used as the denominator for updating mean vectors and covariance matrices
                    self.S[j].gmm_accumulator.acc_sample_counter[k] += _denominator_ / self.S[j].gmm_accumulator.n_components
                    #
                    for t in range( len(O) ):
                        self.S[j].gmm_accumulator.mu[k] += _xi_t_j_k_[t] * O[t]
                        if self.S[j].gmm_accumulator.covar_type in GMM.covar_diagonal_types:
                            self.S[j].gmm_accumulator.sigma[k] += _xi_t_j_k_[t] * (O[t] * O[t]) # numpy.diagonal( O[t] * O[t] )
                        else:
                            self.S[j].gmm_accumulator.sigma[k] += _xi_t_j_k_[t] * numpy.outer( O[t], O[t] )
        else:
            raise Exception( 'Modality ' + self.modality + ' is not valid or not implemented yet!' )
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def viterbi( self, O ):
        """
            Version for testing a unique HMM, not valid for the concatenation of several HMM as used in ASR or HTR

            'O' can be an array of symbols (indexes corresponding to symbols) or an array of real-valued vectors
        """

        predecessor = numpy.ones( [ len(O), len(self) ], dtype=int ) * -1
        delta = numpy.zeros( [ len(O), len(self) ] )
        B = numpy.zeros( [ len(self), len(O) ] )

        for j in range(len(self.S)):
            delta[0,j] = self.log_P[j] + self.S[j].b( O[0] )

        for t in range( 1, delta.shape[0] ):
            for j in range(delta.shape[1]):
                #
                _temp_ = delta[t-1,:] + self.A.log_transitions[:,j]
                #
                _from_ = numpy.argmax( _temp_ )
                predecessor[t,j] = _from_
                delta[t,j] = delta[t-1,_from_] + self.S[j].b( O[t] )
                #
            #
        if self.A.force_to_one_terminal_state:        
            _best_ = len(delta[-1])-1 # According to Transitions.py the terminal state is the last one
        else:
            _best_ = numpy.argmax( delta[-1,:] )
        seq = numpy.ones( len(O) ) * -1
        t=len(O)-1
        i=_best_
        while t > 0:
            seq[t] = i
            i = predecessor[t,i]
            t = t-1
        #
        return delta[-1,_best_], seq
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def split_gmm( self ):
        for s in self.S: s.split_gmm()
    # --------------------------------------------------------------------------------------------------------------------------------------------
