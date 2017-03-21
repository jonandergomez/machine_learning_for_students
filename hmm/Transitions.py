"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: September 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""
import numpy

from .Constants import Constants

class Transitions:
    """
    """
    PROB_TRANSITION_SAME_STATE=0.5
    PROB_TRANSITION_NEXT_STATE=0.5

    def __init__( self, identifier=None, num_states=None, input_file=None, first_line=None, dict_args=None ):
        #
        self.left_to_right = True
        self.force_to_one_terminal_state = False
        self.accumulators = None
        #
        if dict_args is not None:
            if 'left_to_right' in dict_args:
                self.left_to_right = dict_args['left_to_right']
            if 'force_to_one_terminal_state' in dict_args:
                self.force_to_one_terminal_state = dict_args['force_to_one_terminal_state']
        #
        if input_file is not None:
            self.load( input_file, first_line )
        else:
            if identifier is None : raise Exception( "Cannot create an Transitions without an identifier!" )
            if num_states is None : raise Exception( "Cannot create an Transitions without knowing the number of states!" )
            #
            self.identifier = identifier
            self.num_states = num_states
            #
            if self.left_to_right:
                # In this case first and last states are phantom states for connecting models in a sequence
                self.transitions = numpy.zeros( [ num_states, num_states ] )
                self.transitions[0,1] = 1.0
                for i in range( 1, num_states-1 ):
                    self.transitions[i,i]   = Transitions.PROB_TRANSITION_SAME_STATE
                    self.transitions[i,i+1] = Transitions.PROB_TRANSITION_NEXT_STATE
            elif self.force_to_one_terminal_state:
                self.transitions = numpy.zeros( [ num_states, num_states ] )
                self.accumulators = numpy.zeros( self.transitions.shape )
                for i in range( num_states ):
                    self.accumulators[i,i:] = 1.0
            else:
                # By default all transitions are set to 1/num_states
                self.transitions = numpy.ones( [ num_states, num_states ] ) / num_states
            #                    
            self.update_transitions()
        if self.accumulators is None:
            self.accumulators = numpy.zeros( self.transitions.shape )
        #
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def accumulate_transition( self, i, j, value=1.0 ):
        self.accumulators[i,j] += value
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def reset_accumulators( self ):
        self.accumulators = numpy.zeros( self.transitions.shape )
        if self.left_to_right:
            self.accumulators[0,1] = 1.0
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def update_transitions( self ):
        #
        if self.accumulators is not None:
            self.transitions[:,:] = self.accumulators[:,:]
        #
        #self.accumulators = None
        #
        n = self.transitions.shape[0]-1 if self.left_to_right else self.transitions.shape[0]
        for s in range(n):
            sum_probs = self.transitions[s,:].sum()
            if sum_probs > 0.0:
                for i in range(self.transitions.shape[1]):
                    self.transitions[s,i] = self.transitions[s,i] / sum_probs
                    if self.transitions[s,i] < Constants.k_min_allowed_prob:
                        self.transitions[s,i] = 0.0
                #
                sum_probs = self.transitions[s,:].sum()
                self.transitions[s,:] /= sum_probs
            else:
                if self.left_to_right:
                    self.transitions[s,:]   = 0.0
                    self.transitions[s,s]   = Transitions.PROB_TRANSITION_SAME_STATE
                    self.transitions[s,s+1] = Transitions.PROB_TRANSITION_NEXT_STATE
                else:
                    self.transitions[s,:]   = 1.0/n
        #
        self.log_transitions = numpy.log( self.transitions + Constants.k_zero_prob )
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def get_prob( self, i, j ): return self.transitions[i,j]
    def get_log_prob( self, i, j ): return self.log_transitions[i,j]

    def save( self, f ):
        f.write( '~t "%s"\n' % self.identifier )
        f.write( '<TRANSP> %d\n' % self.num_states )
        for i in range( self.transitions.shape[0] ):
            f.write( '  ' )
            f.write( ' '.join( '{:15.8e}'.format(value) for value in self.transitions[i] ) )
            f.write( '\n' )
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def load( self, f, line ):
        if line is None: line=f.readline()
        parts = line.split()
        self.identifier = parts[1].replace( '"', '' ) 
        line = f.readline()
        parts = line.split()
        if parts[0] != '<TRANSP>':
            raise Exception( 'Expected <TRANSP> and found ' + parts[0] )
        self.num_states = int(parts[1])
        self.transitions = numpy.zeros( [ self.num_states, self.num_states ] )
        for i in range( self.transitions.shape[0] ):
            line = f.readline()
            parts = line.split()
            self.transitions[i,:] = [float(x) for x in parts]
        #
        self.update_transitions()
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def __len__(self): return self.num_states # self.transitions.shape[0]
    def __str__(self): return self.identifier
    def __lt__(self,other):  return self.identifier < other.identifier
