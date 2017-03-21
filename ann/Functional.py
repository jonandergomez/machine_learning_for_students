"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 3.0 
    Date: December 2015

    Class Functional (for Artificial Neural Networks) with the basic funcionality for
    computing forward (or backward) the output of a block of two layers.
    This class also contains the methods needed to update the weights and bias
    of a block.
    The "Functional" gathers the set of "basic" operations between two layers
    needed to train and to use a net as a classifier or a regressor.

"""


import sys
import numpy

class Functional:
    """
    """

    valid_activation_types=['tanh','sigmoid','linear','linear_rectified','softmax','multinomial','binary']

    # -------------------------------------------------------------------------
    def __init__(self, input_size, input_type, output_size, output_type, scale=1.0, learning_rate=None, alpha_tanh=1.7159, beta_tanh=2.0/3.0 ):
        try:
            if input_type not in Functional.valid_activation_types :
                raise TypeError( "'%s' is not a valid input type!" % input_type )
            if output_type not in Functional.valid_activation_types :
                raise TypeError( "'%s' is not a valid output type!" % output_type )
            if input_size < 1 or input_size > 10000 :
                raise TypeError( "%d is not a valid input size!" % input_size )
            if output_size < 1 or output_size > 10000 :
                raise TypeError( "%d is not a valid input size!" % output_size )

            self.input_type = input_type
            self.output_type = output_type
            #
            self.input_bias = numpy.zeros( input_size )
            self.output_bias = numpy.zeros( output_size )
            #
            alpha = scale * numpy.sqrt( 6.0 / (input_size + output_size ) )
            self.weights = (2 * alpha) * numpy.random.rand( output_size, input_size ) - alpha

            self.mean_forward  = numpy.zeros( output_size )
            self.sigma_forward = numpy.ones(  output_size )
            self.mean_backward  = numpy.zeros( input_size )
            self.sigma_backward = numpy.ones(  input_size )

            self.learning_rate = learning_rate
            self.learning_rates = None

            self.alpha_tanh = alpha_tanh
            self.beta_tanh = beta_tanh

        except TypeError as detail:
            print( 'SEVERE ERROR %s ' % detail )
            sys.exit(-1)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def get_input_size( self ): return self.weights.shape[1]
    # -------------------------------------------------------------------------
    def get_output_size( self ): return self.weights.shape[0]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_forward( self, v, drop_out_mask=None, drop_out_coeff=1.0, noise_sigma=0.0 ):
        net = self.compute_net_forward( v, drop_out_coeff )
        if noise_sigma > 0.0 : net = net + noise_sigma * numpy.random.randn( net.shape[0], net.shape[1] )
        if drop_out_mask is not None: net = net * drop_out_mask
        a = self.compute_activation_forward( net )
        if drop_out_mask is not None: a = a * drop_out_mask
        return net, a
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_backward( self, h, drop_out_mask=None, drop_out_coeff=1.0, noise_sigma=0.0 ):
        net = self.compute_net_backward( h, drop_out_coeff )
        if noise_sigma > 0.0 : net = net + noise_sigma * numpy.random.randn( net.shape[0], net.shape[1] )
        if drop_out_mask is not None: net = net * drop_out_mask
        a = self.compute_activation_backward( net )
        if drop_out_mask is not None: a = a * drop_out_mask
        return net, a
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_net_forward( self, v, drop_out_coeff=1.0 ):
        preactivation = drop_out_coeff * numpy.dot( self.weights, v ) + self.output_bias[:,numpy.newaxis]
        #if hasattr(self,'mean_forward'):
        if self.mean_forward is not None:
            preactivation = ( preactivation - self.mean_forward[:,numpy.newaxis] ) / self.sigma_forward[:,numpy.newaxis]
        return preactivation 
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_net_backward( self, h, drop_out_coeff=1.0 ):
        preactivation = drop_out_coeff * numpy.dot( self.weights.transpose(), h ) + self.input_bias[:,numpy.newaxis]
        #if hasattr(self,'mean_forward'):
        if self.mean_forward is not None:
            preactivation = ( preactivation - self.mean_backward[:,numpy.newaxis] ) / self.sigma_backward[:,numpy.newaxis]
        return preactivation 
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_activation( self, activation_type, net ):

        if activation_type == 'tanh' :

            a = self.alpha_tanh * numpy.tanh( self.beta_tanh * net )

        elif activation_type == 'sigmoid' :

            temp = numpy.maximum( -40.0, net )
            temp = numpy.minimum(  40.0, temp )
            a = 1.0/(1.0+numpy.exp( -temp ))

        elif activation_type == 'binary' :

            temp = numpy.maximum( -40.0, net )
            temp = numpy.minimum(  40.0, temp )
            a = 1.0/(1.0+numpy.exp( -temp ))

            if len(a.shape) > 1 :
                thresholds = numpy.random.rand( a.shape[0], a.shape[1] )
            else:
                thresholds = numpy.random.rand( len(a) )
            a[ a >= thresholds ] = 1.0
            a[ a <  thresholds ] = 0.0

        elif activation_type == 'linear_rectified' :

            # Sharp version of the linear rectified activation function.
            a = numpy.maximum( 0.0, net )

            # Comment the previous line and uncomment the following lines for
            # using the soft version of the linear rectified activation function.
            # In case of using the soft version do the proper changes in the method
            # compute_activation_derivative() that appears below.
            # 
            #temp = numpy.maximum( -40.0, net )
            #temp = numpy.minimum(  40.0, temp )
            #a = numpy.log( 1 + numpy.exp( temp ) )

        elif activation_type == 'linear' :

            a = net

        elif activation_type == 'softmax' :

            den = -numpy.inf
            for i in range(len(net)): den = numpy.logaddexp(den,net[i])
            a = numpy.exp( net - den )

        elif activation_type == 'multinomial' :

            den = -numpy.inf
            for i in range(len(net)): den = numpy.logaddexp(den,net[i])
            a = numpy.exp( net - den )
            # Here it remains to multiply 'a' by the sum of all the inputs.

        else:
            sys.exit(1)

        return a
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_activation_forward( self, net ):
        return self.compute_activation( self.output_type, net )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_activation_backward( self, net ):
        return self.compute_activation( self.input_type, net )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def compute_activation_derivative( self, activation_type, a, net ):

        if activation_type == 'tanh' :

            ad = self.beta_tanh * (self.alpha_tanh -(a*a)/self.alpha_tanh)

        elif activation_type == 'sigmoid' :

            ad = a * (1-a)

        elif activation_type == 'binary' :

            # Because 'a' comes binarized and the derivative should be computed before
            temp = numpy.maximum( -40.0, net )
            temp = numpy.minimum(  40.0, temp )
            temp_a = 1.0/(1.0+numpy.exp( -temp ))
            ad = temp_a * (1 - temp_a)

        elif activation_type == 'linear_rectified' :

            # Sharp version of the linear rectified activation function
            ad = numpy.ones(a.shape)
            ad[ net <= 0.0 ] = 0.0

            # Soft version of the linear rectified activation function
            #temp = numpy.maximum( -40.0, net )
            #temp = numpy.minimum(  40.0, temp )
            #ad = 1.0/(1.0 + numpy.exp( -temp ))

        elif activation_type in [ 'linear', 'softmax', 'multinomial' ] :

            ad = numpy.ones(a.shape)

        else:
            sys.exit(1)

        return ad
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def max_norm( self, max_norm_value=3.5 ):
        w_norm = numpy.linalg.norm( self.weights, axis=1 )
        for i in range( w_norm.shape[0] ):
            if w_norm[i] > max_norm_value :
                self.weights[i] = max_norm_value * self.weights[i] / w_norm[i]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def backward_propagation( self, error_output, net_output, act_output, act_input ):
        
        # delta is a matrix with 'output_size' rows and 'batch_size' columns
        # Here the elementwise product should be used
        delta = error_output * self.compute_activation_derivative( self.output_type, act_output, net_output )

        # error_input is a matrix with 'input_size' rows and 'batch_size' columns
        #
        # It is the error for the layer below.
        # Each column of this matrix is a vector with the error corresponding to
        # a given sample. Each component of the column vectors corresponds to
        # the dot/inner/scalar product of the delta corresponding to the backpropagated
        # error and the weights related with each input unit.
        #
        # for an input unit j, error_input[j] = sum(k=1,K)( delta[k] * weights[k,j] )
        #
        # So for obtaining the whole error vector
        #
        #              error_input = dot( delta, weights[:,j] )
        #
        # When using a mini-batch error_input and delta are not vectors, they are matrices
        # with one column corresponding to a different sample, so the error for a given sample
        # will be computed as follows
        #
        #              error_input[:,b] = dot( delta[:,b], weights[:,j] )
        #
        # that in matrix form is
        #
        #              error_input = dot( weights', delta )
        # 
        error_input = numpy.dot( self.weights.transpose(), delta )

        # incr_weights is a matrix of 'output_size' x 'input_size'
        # because it is the gradient for updating the weights.
        #
        # As explained before for computing the input error for the input layer
        # of a block of two consecutive layers, when working with mini-batches
        # a sample is a matrix where each column is a sample, then the number
        # of rows is the number of features (the dimensionality). For layers
        # a any level, the input dimension of the number of units (neurons) in
        # the input layer and the output dimension is the number of units (neurons)
        # in the output layer, so, the matrix of weights between two consecutive
        # layers is a matrix with a number of rows equal to the output dimension
        # and number of columns equal to the input dimension.
        #
        # The accumulated gradient for a given mini-batch of samples can be computed
        # in matrix form as follows. Then, each component of the incr_weights matrix
        # contains the accumulated gradient corresponding to the mini-batch, in such
        # a way that the weights are going to be updated once per mini-batch but with
        # the accumulated gradient.
        #
        incr_weights = numpy.dot( delta, act_input.transpose() )

        # incr_output_bias is a vector with 'output_size' elements
        #
        # Here applies the same idea, the output bias are updated with the accumulated
        # gradient for all the samples in the mini-batch. In this case it is the sum
        # of each dimension of the output bias, so the result is a column vector whose
        # components are the sum of all the components in the same row of the delta
        # matrix.
        #
        incr_output_bias = numpy.sum( delta, axis=1 )

        if self.mean_forward is not None:
            #
            incr_weights /= self.sigma_forward[:,numpy.newaxis]
            incr_output_bias /= self.sigma_forward
            #
            incr_mean_forward  = - incr_output_bias / self.sigma_forward
            incr_sigma_forward = + ( (delta * net_output) / self.sigma_forward[:,numpy.newaxis] ).sum(axis=1)
        else:
            incr_mean_forward  = None
            incr_sigma_forward = None

        return error_input, incr_weights, incr_output_bias, incr_mean_forward, incr_sigma_forward
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def forward_propagation( self, error_input, net_input, act_input, act_output ):
        
        # delta is a matrix with 'input_size' rows and 'batch_size' columns
        delta = error_input * self.compute_activation_derivative( self.input_type, act_input, net_input )

        # error_output is a matrix with 'output_size' rows and 'batch_size' columns, it is the error for the layer above
        error_output = numpy.dot( self.weights, delta )

        # incr_weights is a matrix of 'output_size' x 'input_size'
        incr_weights = numpy.dot( act_output, delta.transpose() )

        # incr_input_bias is a matrix of 'input_size'
        incr_input_bias = numpy.sum( delta, axis=1 )

        if self.mean_backward is not None:
            #
            incr_weights    /= self.sigma_backward[numpy.newaxis,:]
            incr_input_bias /= self.sigma_backward
            #
            incr_mean_backward  = - incr_input_bias / self.sigma_backward
            incr_sigma_backward = + ( (delta * net_input) / self.sigma_backward[:,numpy.newaxis] ).sum(axis=1)
        else:
            incr_mean_backward  = None
            incr_sigma_backward = None

        return error_output, incr_weights, incr_input_bias, incr_mean_backward, incr_sigma_backward
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def contrastive_divergence( self, vpos ):
        
        net1, hpos = self.compute_forward(  vpos )
        net2, vneg = self.compute_backward( hpos )
        net3, hneg = self.compute_forward(  vneg )

        if len(vpos.shape) > 1 :
            incrW = numpy.dot( hpos, vpos.transpose() ) - numpy.dot( hneg, vneg.transpose() )
            incrObias = numpy.sum( (hpos - hneg), axis=1 )
            incrIbias = numpy.sum( (vpos - vneg), axis=1 )
        else:
            incrW = numpy.outer( hpos, vpos ) - numpy.outer( hneg, vneg )
            incrObias = hpos - hneg
            incrIbias = vpos - vneg

        vbias_term = 0.0
        if self.input_type in [ 'linear' ] : # , 'linear_rectified' ]  :
            temp = vpos - self.input_bias[:,numpy.newaxis]
            vbias_term = -0.5 * (temp * temp).sum()
        else:
            vbias_term = numpy.dot( self.input_bias, vpos ).sum()

        if self.output_type in [ 'linear' ] : # , 'linear_rectified' ]  :
            temp = hpos - self.output_bias[:,numpy.newaxis]
            hidden_term = - 0.5 * (temp * temp).sum()
            #for i in range(vpos.shape[1]):
            #    hidden_term = hidden_term + numpy.dot( numpy.dot( hpos[:,i], self.weights ), vpos[:,i] )
            hidden_term = hidden_term + numpy.dot( numpy.dot( hpos.transpose(), self.weights ), vpos ).sum()
        else: 
            # For avoiding excess in the computation of log(1+exp(x))
            net1 = numpy.minimum(  40.0, net1 )
            net1 = numpy.maximum( -40.0, net1 )
            hidden_term = numpy.log( 1 + numpy.exp( net1 ) ).sum()

        free_energy = -vbias_term - hidden_term
        #print( "W  = %e  %e  %e  fe = %e" % (self.weights.sum(), self.input_bias.sum(), self.output_bias.sum(), free_energy) )
        
        return incrW, incrObias, incrIbias, free_energy
    # -------------------------------------------------------------------------
