"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 3.1 
    Date: December 2015

    Class ANN (for Artificial Neural Networks) with the funcionality for
    computing forward (or backward) the output of a net.
    This class also contains the methods needed to update the weights and bias
    of a whole net. And the methods for training the net as an autoencoder,
    which is also used for performing the unsupervised finetuning of a net.

"""


import sys
import time
import numpy
import gzip
import os.path
from sklearn.utils import shuffle

try:
    import cPickle as pickle
except:
    import pickle


from .Functional import Functional

# -------------------------------------------------------------------------
def load( modelname=None ):
    if modelname is None : raise Exception( 'Impossible to load a network without the model name!' )
    nn = None
    for suffix in [ None, 'best', '3']:
        if suffix is None:
            filename='lib/net-%s.pkl.gz' % modelname
        else:
            filename='lib/net-%s-%s.pkl.gz' % (modelname,suffix)
        if os.path.exists( filename ) and os.path.isfile( filename ):
            with gzip.open( filename, 'rb' ) as f: nn = pickle.load( f )
            f.close()
            break
    #
    return nn
# -------------------------------------------------------------------------


class ANN:
    """
    """

    # -------------------------------------------------------------------------
    def __init__( self, lib_dirname='lib', config_dirname='config', modelname=None, scale=None, max_iter=None, batch_size=None, learning_rate=None, momentum=None ):
        #
        if modelname is None: raise Exception( 'Cannot create a network!' )
        # 
        self.lib_dirname     = lib_dirname
        self.config_dirname  = config_dirname
        self.modelname       = modelname
        #
        #
        self.num_layers      =   -1
        self.learning_rate   =    1.0
        self.momentum        =    0.9
        self.l2_penalty      =    0.0
        self.c_max_norm      =    2.0
        self.max_iter        =  100
        self.batch_size      =  100
        self.scale           =    1.0
        self.olr_iterations  =   -1
        self.add_input_noise =  False
        self.add_noise       =  False
        self.batch_normalisation =  False
        self.use_updatable_learning_rate =  False
        #
        self.alpha_tanh = 1.7159
        self.beta_tanh = 2.0/3.0
        #
        filename = "%s/%s.cfg" % (config_dirname, modelname)
        with open( filename, 'rt' ) as f:
            topology = []
            for line in f:
                line=line.strip()
                if len(line) == 0  or  line[0] == '#' : continue
                parts = line.split()
                if   parts[0] == 'num_layers'           : self.num_layers          =   int(parts[1])
                elif parts[0] == 'learning_rate'        : self.learning_rate       = float(parts[1])
                elif parts[0] == 'momentum'             : self.momentum            = float(parts[1])
                elif parts[0] == 'l2_penalty'           : self.l2_penalty          = float(parts[1])
                elif parts[0] == 'c_max_norm'           : self.c_max_norm          = float(parts[1])
                elif parts[0] == 'max_iter'             : self.max_iter            =   int(parts[1])
                elif parts[0] == 'add_input_noise'      : self.add_input_noise     =  bool(parts[1])
                elif parts[0] == 'add_noise'            : self.add_noise           =  bool(parts[1])
                elif parts[0] == 'batch_size'           : self.batch_size          =   int(parts[1])
                elif parts[0] == 'batch_normalisation'  : self.batch_normalisation =  bool(parts[1])
                elif parts[0] == 'scale'                : self.scale               = float(parts[1])
                elif parts[0] == 'olr_iterations'       : self.olr_iterations      =   int(parts[1])
                elif parts[0] == 'alpha_tanh'           : self.alpha_tanh          = float(parts[1])
                elif parts[0] == 'beta_tanh'            : self.beta_tanh           = float(parts[1])
                elif parts[0] == 'layer'                :
                    if len(parts) > 5:
                        topology.append( [ int(parts[1]), str(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]) ] )
                    else:
                        topology.append( [ int(parts[1]), str(parts[2]), float(parts[3]), float(parts[4]), 0.0 ] )
        f.close()

        index_layer_size=0
        index_activation_type=1
        index_dropout_coeff=2
        index_learning_rate=3
        index_noise_sigma=4

        try:
            if len( topology ) < 2  or  len(topology) != self.num_layers : raise TypeError( "invalid number of layers!" )

            if learning_rate is not None: self.learning_rate = learning_rate
            if momentum      is not None: self.momentum      = momentum
            if batch_size    is not None: self.batch_size    = batch_size
            if max_iter      is not None: self.max_iter      = max_iter
            if scale         is not None: self.scale         = scale

            self.functionals          = [None] * self.num_layers
            self.delta_weights        = [None] * self.num_layers
            self.delta_input_bias     = [None] * self.num_layers
            self.delta_output_bias    = [None] * self.num_layers 
            self.delta_mean_forward   = [None] * self.num_layers 
            self.delta_sigma_forward  = [None] * self.num_layers 
            self.delta_mean_backward  = [None] * self.num_layers 
            self.delta_sigma_backward = [None] * self.num_layers 
            self.update_layer         = [True] * self.num_layers
            self.incr_weights         = [None] * self.num_layers
            self.drop_out_mask        = [None] * self.num_layers

            self.drop_out_coeff = [1.0] * self.num_layers
            self.noise_sigma    = [0.0] * self.num_layers

            self.drop_out_coeff[0] = topology[0][ index_dropout_coeff ]
            self.noise_sigma[0] = topology[0][ index_noise_sigma ]
            layer=1
            while layer < self.num_layers:
                #
                self.drop_out_coeff[layer] = topology[layer][ index_dropout_coeff ]
                self.noise_sigma[layer] = topology[layer][ index_noise_sigma ]
                #
                self.functionals[layer] = Functional(  input_size = topology[layer-1][ index_layer_size ],
                                                       input_type = topology[layer-1][ index_activation_type ],
                                                      output_size = topology[layer  ][ index_layer_size ],
                                                      output_type = topology[layer  ][ index_activation_type ],
                                                            scale = self.scale,
                                                    learning_rate = topology[layer][ index_learning_rate ],
                                                       alpha_tanh = self.alpha_tanh,
                                                        beta_tanh = self.beta_tanh )
                self.delta_weights[layer]        = numpy.zeros( [ topology[layer][ index_layer_size ], topology[layer-1][ index_layer_size ] ] )
                self.incr_weights[layer]         = numpy.zeros( [ topology[layer][ index_layer_size ], topology[layer-1][ index_layer_size ] ] )
                self.delta_input_bias[layer]     = numpy.zeros( topology[layer-1][ index_layer_size ] )
                self.delta_output_bias[layer]    = numpy.zeros( topology[layer  ][ index_layer_size ] )
                self.delta_mean_forward[layer]   = numpy.zeros( topology[layer  ][ index_layer_size ] )
                self.delta_sigma_forward[layer]  = numpy.zeros( topology[layer  ][ index_layer_size ] )
                self.delta_mean_backward[layer]  = numpy.zeros( topology[layer-1][ index_layer_size ] )
                self.delta_sigma_backward[layer] = numpy.zeros( topology[layer-1][ index_layer_size ] )
                layer = layer+1

        except TypeError as detail:
            print( 'SEVERE ERROR %s ' % detail )
            sys.exit(-1)
    # -------------------------------------------------------------------------

    def get_output_size( self ):
        return self.functionals[-1].output_size

    # -------------------------------------------------------------------------
    def add_layer(self, output_type, output_size ):
        #
        input_type = self.functionals[-1].output_type
        input_size = self.functionals[-1].weights.shape[0]
        #
        self.functionals.append( Functional( input_size, input_type, output_size, output_type, self.scale, alpha_tanh=self.alpha_tanh, beta_tanh=self.beta_tanh ) )
        #
        self.delta_weights.append( numpy.zeros( [ output_size, input_size ] ) )
        self.delta_input_bias.append( numpy.zeros( input_size ) )
        self.delta_output_bias.append( numpy.zeros( output_size ) )
        self.incr_weights.append( numpy.zeros( [ output_size, input_size ] ) )
        #
        self.delta_mean_forward.append( numpy.zeros( output_size ) )
        self.delta_sigma_forward.append( numpy.zeros( output_size ) )
        self.delta_mean_backward.append( numpy.zeros( input_size ) )
        self.delta_sigma_backward.append( numpy.zeros( input_size ) )
        #
        self.drop_out_mask.append( None )
        self.drop_out_coeff.append( 1.0 )
        #
        self.update_layer.append( True )
        #
        self.num_layers = self.num_layers + 1
        #
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def save( self, suffix=None ):
        if suffix is not None:
            filename = "%s/net-%s-%s.pkl.gz" % ( self.lib_dirname, self.modelname, suffix )
        else:
            filename = "%s/net-%s.pkl.gz" % ( self.lib_dirname, self.modelname )
        with gzip.open( filename, 'wb' ) as f: pickle.dump( self, f )
        f.close()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def show_topology(self):
        print( "# --------------------------------------------------------------------------------------------" )
        print( "# Layer %3d with %5d units of type %-20s Dropout %.3f   Noise Sigma %.3f                 -- Input  Layer" % (0,
            self.functionals[1].get_input_size(), self.functionals[1].input_type, self.drop_out_coeff[0], self.noise_sigma[0] ) )

        layer=1
        while layer < self.num_layers-1:
            print( "# Layer %3d with %5d units of type %-20s Dropout %.3f   Noise Sigma %.3f   mu %.8f -- Hidden Layer" % ( layer,
                self.functionals[layer].get_output_size(), self.functionals[layer].output_type, self.drop_out_coeff[layer], self.noise_sigma[layer], self.functionals[layer].learning_rate ) )
            layer=layer+1

        print( "# Layer %3d with %5d units of type %-20s Dropout %.3f   Noise Sigma %.3f   mu %.8f -- Output Layer" % ( layer,
            self.functionals[layer].get_output_size(), self.functionals[layer].output_type, self.drop_out_coeff[layer], self.noise_sigma[layer], self.functionals[layer].learning_rate ) )
        print( "# --------------------------------------------------------------------------------------------" )
        print( "# C_max_norm  %e" % self.c_max_norm )
        print( "# momentum    %e" % self.momentum )
        print( "# alpha tanh  %e" % self.alpha_tanh )
        print( "# beta  tanh  %e" % self.beta_tanh )
        print( "# --------------------------------------------------------------------------------------------" )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def reset_deltas(self):
        for layer in range(1,self.num_layers):
            self.delta_weights[layer][:][:] = 0.0
            self.delta_output_bias[layer][:] = 0.0
            self.delta_input_bias[layer][:] = 0.0
            self.incr_weights[layer][:][:] = 0.0
            self.delta_mean_forward[layer][:] = 0.0
            self.delta_sigma_forward[layer][:] = 0.0
            self.delta_mean_backward[layer][:] = 0.0
            self.delta_sigma_backward[layer][:] = 0.0
        
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def add_white_noise( self, sample ):
        if not self.add_input_noise : return sample
        #
        if self.functionals[1].input_type == 'linear' :
            sample = sample + self.noise_sigma[0] * numpy.random.randn( sample.shape[0], sample.shape[1] )
        #
        elif self.functionals[1].input_type in ['sigmoid', 'binary'] :
            noise = numpy.random.rand( sample.shape[0], sample.shape[1] )
            threshold = self.noise_sigma[0]
            noise[ noise >= threshold ] = 1.0
            noise[ noise <= threshold ] = 0.0
            sample = sample * noise + (1-sample)*(1-noise)
        #
        return sample
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def generate_drop_out_mask_for_batch( self, batch_size ):
        self.drop_out_mask[0] = self.generate_drop_out_mask( self.functionals[1].get_input_size(),
                                                             batch_size,
                                                             self.drop_out_coeff[0] )
        layer=1
        while layer < self.num_layers:
            self.drop_out_mask[layer] = self.generate_drop_out_mask( self.functionals[layer].get_output_size(),
                                                                     batch_size,
                                                                     self.drop_out_coeff[layer] )
            layer=layer+1
    # -------------------------------------------------------------------------
    def generate_drop_out_mask( self, rows, cols, coeff ):
        mask = numpy.ones( [rows, cols] )
        if coeff < 1.0 : mask[ numpy.random.rand( rows, cols ) > coeff ] = 0.0
        return mask
    # -------------------------------------------------------------------------
    def do_not_use_drop_out_for_batch( self, batch_size ):
        self.drop_out_mask[0] = self.generate_drop_out_mask( self.functionals[1].get_input_size(),
                                                             batch_size,
                                                             1 )
        layer=1
        while layer < self.num_layers:
            self.drop_out_mask[layer] = self.generate_drop_out_mask( self.functionals[layer].get_output_size(),
                                                                     batch_size,
                                                                     1 )
            layer=layer+1
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # In this method we have to use the dropout COEFFICIENT for the INPUT layer
    # because it is the PROBABILITY used for maintaining units in the INPUT layer
    # of each block of two consecutive layers. So drop_out_coeff[layer-1] is correct.
    #
    def forward_predict( self, v ):
        h = v
        layers = [ [None, h] ]
        for layer in range(1,self.num_layers):
            net, h = self.functionals[layer].compute_forward( h, drop_out_coeff=self.drop_out_coeff[layer-1], noise_sigma=0.0 )
            layers.append( [ net, h ] )
            
        return layers
    # -------------------------------------------------------------------------
            
    # -------------------------------------------------------------------------
    # In this method we have to use the dropout MASK for the OUTPUT layer
    # because it is the MASK used for maintaining units in the OUTPUT layer
    # of each block of two consecutive layers. So drop_out_mask[layer] is correct.
    #
    def forward( self, v, _num_layers_ = 0 ):
        if _num_layers_ <= 0: _num_layers_ = self.num_layers
        #
        h = v * self.drop_out_mask[0]
        layers = [ [None, h] ]
        for layer in range(1,_num_layers_):
            net, h = self.functionals[layer].compute_forward( h, drop_out_mask=self.drop_out_mask[layer], noise_sigma=self.noise_sigma[layer] )
            layers.append( [ net, h ] )
            
        return layers
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Similar to the forward() method, just above, we use here 'layer-1' as index
    # for the dropout mask because when the network is computing backward the
    # OUTPUT layer of a block of two consecutive layers is the layer below.
    # So drop_out_mask[layer-1] is correct.
    #
    def backward( self, h, _num_layers_ = 0 ):
        if _num_layers_ <= 0: _num_layers_ = self.num_layers
        #
        layers = [ [None, None] ] * _num_layers_
        layer = _num_layers_-1
        v = h * self.drop_out_mask[layer]
        layers[layer] = [None, v]
        while layer >= 1 :
            net, v = self.functionals[layer].compute_backward( v, drop_out_mask=self.drop_out_mask[layer-1], noise_sigma=self.noise_sigma[layer] )
            layer=layer-1
            layers[layer] = [ net, v ]
            
        return layers
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def back_propagation( self, layers, output, error_from_forward_propagation=None, _num_layers_=0, list_of_gradients=None, _momentum=None, _updatable_learning_rate=None ):
        if _num_layers_ <= 0: _num_layers_=self.num_layers
        #
        l=_num_layers_-1
        #
        if _momentum is None : _momentum=self.momentum
        #
        if error_from_forward_propagation is None :
            error = layers[l][1] - output
        else:
            error = error_from_forward_propagation

        mse_training = (error*error).sum() / error.shape[0]

        batch_size = error.shape[1]

        while l > 0:
            #
            error = error * self.drop_out_mask[l]
            #
            error, incr_weights, incr_output_bias, incr_mean_forward, incr_sigma_forward = \
                self.functionals[l].backward_propagation( error_output=error, 
                                                          net_output=layers[l][0],
                                                          act_output=layers[l][1],
                                                          act_input=layers[l-1][1] )
            #
            incr_weights = incr_weights + self.incr_weights[l] # to add the gradient computed during the forward propagation
            #
            if list_of_gradients is not None:
                #list_of_gradients[l] = [ incr_weights, incr_output_bias ]
                list_of_gradients[l] = incr_weights
            else:
                if _updatable_learning_rate is not None:
                    self.delta_weights[l]       = _momentum * self.delta_weights[l]       - (    _updatable_learning_rate/batch_size) * incr_weights
                    self.delta_output_bias[l]   = _momentum * self.delta_output_bias[l]   - (2.0*_updatable_learning_rate/batch_size) * incr_output_bias
                    if  incr_mean_forward is not None:
                        self.delta_mean_forward[l]  = _momentum * self.delta_mean_forward[l]  - (    _updatable_learning_rate/batch_size) * incr_mean_forward
                        #self.delta_mean_forward[l] = (_updatable_learning_rate/batch_size) * incr_mean_forward
                        self.delta_sigma_forward[l] = _momentum * self.delta_sigma_forward[l] - (    _updatable_learning_rate/batch_size) * incr_sigma_forward
                        #self.delta_sigma_forward[l] = (_updatable_learning_rate/batch_size) * incr_sigma_forward
                elif self.functionals[l].learning_rates is not None:
                    self.delta_weights[l]       = _momentum * self.delta_weights[l]       - (    self.functionals[l].learning_rates[:,numpy.newaxis]/batch_size) * incr_weights
                    self.delta_output_bias[l]   = _momentum * self.delta_output_bias[l]   - (2.0*self.functionals[l].learning_rates[:              ]/batch_size) * incr_output_bias
                    if  incr_mean_forward is not None:
                        self.delta_mean_forward[l]  = _momentum * self.delta_mean_forward[l]  - (    self.functionals[l].learning_rates[:              ]/batch_size) * incr_mean_forward
                        #self.delta_mean_forward[l]  = (self.functionals[l].learning_rates[:]/batch_size) * incr_mean_forward
                        self.delta_sigma_forward[l] = _momentum * self.delta_sigma_forward[l] - (    self.functionals[l].learning_rates[:              ]/batch_size) * incr_sigma_forward
                        #self.delta_sigma_forward[l] = (self.functionals[l].learning_rates[:]/batch_size) * incr_sigma_forward
                elif self.functionals[l].learning_rate is not None:
                    self.delta_weights[l]       = _momentum * self.delta_weights[l]       - (    self.functionals[l].learning_rate/batch_size) * incr_weights
                    self.delta_output_bias[l]   = _momentum * self.delta_output_bias[l]   - (2.0*self.functionals[l].learning_rate/batch_size) * incr_output_bias
                    if  incr_mean_forward is not None:
                        self.delta_mean_forward[l]  = _momentum * self.delta_mean_forward[l]  - (    self.functionals[l].learning_rate/batch_size) * incr_mean_forward
                        #self.delta_mean_forward[l]  = (self.functionals[l].learning_rate/batch_size) * incr_mean_forward
                        self.delta_sigma_forward[l] = _momentum * self.delta_sigma_forward[l] - (    self.functionals[l].learning_rate/batch_size) * incr_sigma_forward
                        #self.delta_sigma_forward[l] = (self.functionals[l].learning_rate/batch_size) * incr_sigma_forward
                else:
                    self.delta_weights[l]       = _momentum * self.delta_weights[l]       - (    self.learning_rate/batch_size) * incr_weights
                    self.delta_output_bias[l]   = _momentum * self.delta_output_bias[l]   - (2.0*self.learning_rate/batch_size) * incr_output_bias
                    if  incr_mean_forward is not None:
                        self.delta_mean_forward[l]  = _momentum * self.delta_mean_forward[l]  - (    self.learning_rate/batch_size) * incr_mean_forward
                        #self.delta_mean_forward[l] = (self.learning_rate/batch_size) * incr_mean_forward
                        self.delta_sigma_forward[l] = _momentum * self.delta_sigma_forward[l] - (    self.learning_rate/batch_size) * incr_sigma_forward
                        #self.delta_sigma_forward[l] = (self.learning_rate/batch_size) * incr_sigma_forward
                # end if
            # end if
            l = l-1
            # end while
        return mse_training
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def forward_propagation( self, layers, input_sample, _num_layers_=0, _momentum=None ):
        if _num_layers_ <= 0: _num_layers_=self.num_layers
        #
        #
        if _momentum is None : _momentum=self.momentum
        #
        #error = (layers[0][1] - input_sample) * self.drop_out_mask[0]
        error = (layers[0][1] - input_sample)
        #
        mse_training = (error*error).sum() / error.shape[0]

        batch_size = error.shape[1]

        l=1
        while l < _num_layers_:
            #
            error = error * self.drop_out_mask[l-1]
            #
            error, incr_weights, incr_input_bias, incr_mean_backward, incr_sigma_backward = \
                                self.functionals[l].forward_propagation( error_input=error,
                                                                         net_input=layers[l-1][0],
                                                                         act_input=layers[l-1][1],
                                                                         act_output=layers[l][1] )
            #
            #self.delta_weights[l]     = _momentum * self.delta_weights[l]    + self.learning_rate * incr_weights
            self.incr_weights[l] = incr_weights
            if self.functionals[l].learning_rates is not None:
                self.delta_input_bias[l]  = _momentum * self.delta_input_bias[l]  - (2.0*self.functionals[l].learning_rates[:] / batch_size) * incr_input_bias
                if  incr_mean_backward is not None:
                    self.delta_mean_backward[l]  = _momentum * self.delta_mean_backward[l]  - (    self.functionals[l].learning_rates[:              ]/batch_size) * incr_mean_backward
                    #self.delta_mean_backward[l]  = (self.functionals[l].learning_rates[:]/batch_size) * incr_mean_backward
                    self.delta_sigma_backward[l] = _momentum * self.delta_sigma_backward[l] - (    self.functionals[l].learning_rates[:              ]/batch_size) * incr_sigma_backward
                    #self.delta_sigma_backward[l] = (self.functionals[l].learning_rates[:]/batch_size) * incr_sigma_backward
            elif self.functionals[l].learning_rate is not None:
                self.delta_input_bias[l]  = _momentum * self.delta_input_bias[l]  - (2.0*self.functionals[l].learning_rate / batch_size) * incr_input_bias
                if  incr_mean_backward is not None:
                    self.delta_mean_backward[l]  = _momentum * self.delta_mean_backward[l]  - (    self.functionals[l].learning_rate/batch_size) * incr_mean_backward
                    #self.delta_mean_backward[l]  = (self.functionals[l].learning_rate/batch_size) * incr_mean_backward
                    self.delta_sigma_backward[l] = _momentum * self.delta_sigma_backward[l] - (    self.functionals[l].learning_rate/batch_size) * incr_sigma_backward
                    #self.delta_sigma_backward[l] = (self.functionals[l].learning_rate/batch_size) * incr_sigma_backward
            else:
                self.delta_input_bias[l]  = _momentum * self.delta_input_bias[l]  - (2.0*self.learning_rate / batch_size) * incr_input_bias
                if  incr_mean_backward is not None:
                    self.delta_mean_backward[l]  = _momentum * self.delta_mean_backward[l]  - (    self.learning_rate/batch_size) * incr_mean_backward
                    #self.delta_mean_backward[l]  = (self.learning_rate/batch_size) * incr_mean_backward
                    self.delta_sigma_backward[l] = _momentum * self.delta_sigma_backward[l] - (    self.learning_rate/batch_size) * incr_sigma_backward
                    #self.delta_sigma_backward[l] = (self.learning_rate/batch_size) * incr_sigma_backward
            #
            l = l+1
            # end while
        return mse_training, error
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def set_update_all(self):
        for i in range(len(self.update_layer)): self.update_layer[i] = True
    # -------------------------------------------------------------------------
    def unset_update_all(self):
        for i in range(len(self.update_layer)): self.update_layer[i] = False
    # -------------------------------------------------------------------------
    def unset_update_layer(self,layer):
        self.update_layer[layer] = False
    # -------------------------------------------------------------------------
    def set_update_layer(self,layer):
        self.update_layer[layer] = True
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def update_weights_and_bias(self,batch_size,_num_layers_=0):
        if _num_layers_ <= 0: _num_layers_ = self.num_layers
        #lmbda = self.learning_rate * self.l2_penalty # * batch_size
        lmbda = self.l2_penalty # * batch_size
        for layer in range(1,_num_layers_):
            if self.update_layer[layer] :
                #print "Layer = %3d %.20e %e %e" % (layer, (1.0-lmbda), numpy.linalg.norm( (1-lmbda)*self.functionals[layer].weights ), numpy.linalg.norm( self.delta_weights[layer] ) )
                self.functionals[layer].weights     = (1-lmbda)*self.functionals[layer].weights + self.delta_weights[layer]
                self.functionals[layer].output_bias = self.functionals[layer].output_bias + self.delta_output_bias[layer]
                self.functionals[layer].input_bias  = self.functionals[layer].input_bias  + self.delta_input_bias[layer]
                self.functionals[layer].max_norm( self.c_max_norm )
                if self.batch_normalisation:
                    self.functionals[layer].mean_forward  += self.delta_mean_forward[layer]
                    self.functionals[layer].sigma_forward += self.delta_sigma_forward[layer]
                    self.functionals[layer].sigma_forward = numpy.maximum( 1.0e-1, self.functionals[layer].sigma_forward )
                    self.functionals[layer].mean_backward  += self.delta_mean_backward[layer]
                    self.functionals[layer].sigma_backward += self.delta_sigma_backward[layer]
                    self.functionals[layer].sigma_backward = numpy.maximum( 1.0e-1, self.functionals[layer].sigma_backward )
    # -------------------------------------------------------------------------
    def update_weights_and_bias_from_cd(self,layer,batch_size):
        #lmbda = self.learning_rate * self.l2_penalty # * batch_size
        lmbda = self.l2_penalty # * batch_size
        if self.update_layer[layer] :
            self.functionals[layer].weights     = (1-lmbda)*self.functionals[layer].weights + self.delta_weights[layer]
            self.functionals[layer].output_bias = self.functionals[layer].output_bias + self.delta_output_bias[layer]
            self.functionals[layer].input_bias  = self.functionals[layer].input_bias  + self.delta_input_bias[layer]
            self.functionals[layer].max_norm( self.c_max_norm )

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def to_one_hot( self, Y, t ):
        output_type = self.functionals[self.num_layers-1].output_type

        if output_type in ['sigmoid','softmax','binary'] :
            min_value = 0.0
            max_value = 1.0
        elif output_type in ['tanh']:
            min_value = -self.alpha_tanh
            max_value =  self.alpha_tanh
        else:
            raise TypeError( "invalid output type for creating one-hot-vectors: %s " % output_type )

        if len(Y.shape) > 1  and  t.shape[0] == Y.shape[1] :
            for i in range(t.shape[1]): t[:,i] = Y[i]
        else:
            t[:,:] = min_value
            for i in range(t.shape[1]): t[Y[i],i] = max_value

        return t
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def fit( self, X, Y, dev_X=None, dev_Y=None, min_iter=10, max_iter=None, saving_during_training=True ):
        #
        N = len(X)
        if len(Y.shape) == 1:
            K = len(numpy.unique(Y))
            output_probabilities = False
        elif len(Y.shape) > 1  and  Y.shape[1] == self.functionals[-1].get_output_size():
            K = self.functionals[-1].get_output_size()
            output_probabilities = True
        else:
            print( Y.shape )
            print( self.functionals[-1].get_output_size() )
            raise Exception( 'Incompatible output with the output size of the network!' )
        #
        if max_iter is None : max_iter = self.max_iter
        #
        # if max_samples > 0 : max_samples = min( max_samples, len(train_X) )
        #
        epsilon = 1.0e-4
        improvement=1.0
        previous_mse_development=9.0e+90
        best_mse_development = 9.0e+90
        iteration=0
        learning_rate = self.learning_rate
        _momentum = 0.0
        _ulr = None
        self.reset_deltas()
        #
        regression = (self.functionals[-1].output_type == 'linear')
        #
        while iteration < min_iter or (iteration < max_iter and improvement > epsilon) :
            #
            #self.reset_deltas() # THIS IS PROVISIONAL
            #
            if iteration >= 5 :
                _momentum = self.momentum
            elif iteration >= 2 :
                _momentum = 0.5 * self.momentum
            #
            (X,Y) = shuffle(X,Y)
            #
            mse_training=0.0
            mse_development=0.0
            n=0
            while n < N:

                if self.olr_iterations > 0 and iteration > self.olr_iterations:
                    self.computing_optimal_learning_rate( alpha=0.01, gamma=0.01, X=X, Y=Y, K=K )

                sample = X[n:n+self.batch_size].transpose()
                dirty_sample = self.add_white_noise( sample )
                self.generate_drop_out_mask_for_batch( sample.shape[1] )
                output = Y[n:n+self.batch_size]
                n = n + self.batch_size
                t = numpy.zeros( [ K, len(output) ] )

                layers = self.forward( dirty_sample )
                if regression or output_probabilities:
                    mse_t_ = self.back_propagation( layers, output.transpose(), _momentum=_momentum )
                else:
                    mse_t_ = self.back_propagation( layers, self.to_one_hot( output, t ), _momentum=_momentum, _updatable_learning_rate=_ulr )

                mse_training = mse_training + mse_t_

                self.update_weights_and_bias( sample.shape[1] )
            # end while
            mse_training = mse_training / N

            dev_entropy = 0.0
            if dev_X is not None:
                n=0
                while n < len(dev_X):
                    sample = dev_X[n:n+self.batch_size].transpose()
                    output = dev_Y[n:n+self.batch_size]
                    n = n + self.batch_size
                    t = numpy.zeros( [ K, len(output) ] )

                    layers = self.forward_predict( sample )
                    if regression or output_probabilities:
                        t=output.transpose()
                    else:
                        self.to_one_hot( output, t )
                    diff = t - layers[self.num_layers-1][1]
                    mse_t_ = (diff * diff).sum() / diff.shape[0]
                    mse_development = mse_development + mse_t_

                    probs = numpy.copy( layers[-1][1] )
                    probs = numpy.maximum( 1.0e-15, numpy.minimum( 1.0 - 1.0e-15, probs ) )
                    #dev_entropy = dev_entropy - ( probs * numpy.log( probs ) ).sum()
                    if self.functionals[-1].output_type == 'softmax':
                        dev_entropy = dev_entropy - ( t * numpy.log( probs ) ).sum()
                # end while
                mse_development = mse_development / len(dev_X)
                dev_entropy /= len(dev_X)

                y_pred = self.predict( dev_X )
                dev_error = 100.0
                if len(dev_Y.shape) == 1:
                    dev_error = 100.0 - ( 100.0 * (dev_Y == y_pred).sum() ) / len(dev_Y)

                if mse_development < best_mse_development:
                    best_mse_development = mse_development
                    if saving_during_training: self.save( suffix='best' )

                improvement = (previous_mse_development - mse_development)/mse_development
                previous_mse_development = mse_development
            # end if

            if saving_during_training: self.save( suffix='3' )

            iteration = iteration+1
            #print( " _momentum = " + str(_momentum) )
            print( " iteration %5d   training mse %15.8e  development mse %15.8e %7.2f %10.7f improvement %15.8e  %12.0f" % (iteration, mse_training, mse_development, dev_error, dev_entropy, improvement, time.time() ) )
            sys.stdout.flush()
            if self.use_updatable_learning_rate:
                _ulr = min( mse_training, self.learning_rate )

            #self.learning_rate = (learning_rate * 100) / (100.0 + iteration)
            self.learning_rate = learning_rate * ( 0.01 + (1+numpy.exp(-0.02*300)) / (1+numpy.exp(0.02*(iteration-300))) )
        # end while
        self.learning_rate = learning_rate
        return dev_error
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def predict( self, X, _classification=True ):
        #
        N = len(X)
        top_layer=self.num_layers-1
        #
        if _classification:
            y_pred=numpy.zeros( N, dtype=int )
        else:
            y_pred=numpy.zeros( [ N, self.functionals[top_layer].get_output_size() ] )
        #
        for n in range(N):
            #
            layers = self.forward_predict( X[n:n+1].transpose() )
            y=layers[top_layer][1]
            #
            if _classification:
                y_pred[n] = numpy.argmax( y )
            else:
                y_pred[n,:] = y[:,0]
        # end for
        return y_pred
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def layerwise_pretraining( self, X, from_layer=1, to_layer=1, max_epochs=100 ):
        layer = from_layer
        while layer <= to_layer :
            self.contrastive_divergence( X, layer, max_epochs )
            layer = layer+1
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def contrastive_divergence( self, X, layer, max_epochs ):
        N = len(X)
        self.reset_deltas()
        epoch = 0
        previous_free_energy = 0.0
        while epoch < max_epochs :
            self.reset_deltas() # THIS IS PROVISIONAL
            X = shuffle(X)
            _momentum = 0.5 * self.momentum  if epoch <= 5 else self.momentum
            free_energy=0.0
            n=0
            while n < N:
                sample = X[n:n+self.batch_size].transpose()
                n = n + self.batch_size
                batch_size = sample.shape[1]
                self.generate_drop_out_mask_for_batch( sample.shape[1] )
                sample = sample * self.drop_out_mask[0]
                if layer > 1 :
                    for l in range(1,layer):
                        net, sample = self.functionals[l].compute_forward( sample, drop_out_mask=self.drop_out_mask[l], noise_sigma=0.0 )

                incr_weights, incr_output_bias, incr_input_bias, fe = self.functionals[layer].contrastive_divergence( sample )
                #
                self.delta_weights[layer]     = _momentum * self.delta_weights[layer]     + (self.learning_rate/batch_size) * incr_weights
                self.delta_output_bias[layer] = _momentum * self.delta_output_bias[layer] + (self.learning_rate/batch_size) * incr_output_bias
                self.delta_input_bias[layer]  = _momentum * self.delta_input_bias[layer]  + (self.learning_rate/batch_size) * incr_input_bias
                #
                free_energy = free_energy + fe
                #
                self.update_weights_and_bias_from_cd( layer, sample.shape[1] )

            free_energy = free_energy / N
            epoch = epoch + 1
            #print( "momentum = %e  mu = %e  batch_size = %d" % ( _momentum, self.learning_rate, sample.shape[1] ) )
            improvement = (free_energy - previous_free_energy) / numpy.abs(free_energy)
            print( "layerwise pretraining layer %d epoch = %03d  free energy = %16.8e  improvement %16.8e %12.0f " % (layer, epoch, free_energy, improvement, time.time()) )
            sys.stdout.flush()
            previous_free_energy = free_energy
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def unsupervised_finetuning( self, X, max_epochs=100, _num_layers_=0 ):
        if _num_layers_ <= 0: _num_layers_ = self.num_layers-1
        if max_epochs <= 0 : return None
        #
        N = len(X)
        if len(X) > self.batch_size: self.reset_deltas()
        epoch = 0
        _momentum = 0.0
        previous_mse = 9.0e+90
        while epoch < max_epochs :
            #
            if epoch >= 5 :
                _momentum = self.momentum
            elif epoch >= 2 :
                _momentum = 0.5 * self.momentum
            #
            self.reset_deltas() # THIS IS PROVISIONAL
            X = shuffle(X)
            mse_training = 0.0
            n=0
            while n < N:
                sample = X[n:n+self.batch_size].transpose()
                dirty_sample = self.add_white_noise( sample )
                self.generate_drop_out_mask_for_batch( sample.shape[1] )
                n = n + self.batch_size
                #
                forward_layers  = self.forward( dirty_sample, _num_layers_=_num_layers_ )
                backward_layers = self.backward( forward_layers[-1][1], _num_layers_=_num_layers_ )
                mse_t_, error   = self.forward_propagation( backward_layers, sample, _num_layers_=_num_layers_, _momentum=_momentum )
                temp_           = self.back_propagation( forward_layers, None, error, _num_layers_=_num_layers_, _momentum=_momentum )
                #
                mse_training = mse_training + mse_t_
                #
                self.update_weights_and_bias( sample.shape[1] )
                #
            # end for
            mse_training = mse_training / N
            #
            improvement = (previous_mse - mse_training)/mse_training
            previous_mse = mse_training
            #
            epoch = epoch+1
            print( "unsupervised pretraining for %3d layers at epoch %5d   training mse %15.8e  improvement %15.8e  %12.0f"
                        % (_num_layers_, epoch, mse_training, improvement, time.time() ) )
            sys.stdout.flush()
            # end while
        return mse_training
    # -------------------------------------------------------------------------

    
    # -------------------------------------------------------------------------
    def change_momentum( self, momentum ):
        self.momentum = momentum
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def change_learning_rates( self, coeff ):
        self.learning_rate *= coeff
        for l in range(1,self.num_layers):
            if self.functionals[l].learning_rate is not None:
                self.functionals[l].learning_rate *= coeff
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def computing_optimal_learning_rate( self, alpha=0.001, gamma=0.001, X=None, Y=None, K=0 ):
        # Initialize PSI
        psi = []
        psi.append( None ) # Not needed for the first layer
        for l in range(1,self.num_layers):
            psi_mat = numpy.random.randn( self.functionals[l].weights.shape[0], self.functionals[l].weights.shape[1] )
            psi_mat_norm = numpy.linalg.norm( psi_mat, axis=1 )
            for i in range(len(psi_mat_norm)): psi_mat[i] = psi_mat[i] / psi_mat_norm[i]
            psi.append( psi_mat )
        # end for
        #
        # Save the current weights of all layers
        saved_weights = [None] * self.num_layers
        for l in range(1,self.num_layers):
            saved_weights[l] = numpy.copy( self.functionals[l].weights )
        # end for
        #
        repetitions=0
        while True :
            #
            n = numpy.random.randint( len(X) ) # choose a random sample
            x = X[n:n+1].transpose() # prepare the input
            y = Y[n:n+1]             # prepare the output
            self.do_not_use_drop_out_for_batch( x.shape[1] ) 
            t = numpy.zeros( [ K, len(y) ] ) # allocate memory for the output target of the chosen sample
            #
            # COMPUTATION OF THE GRADIENT OF E(W): G1
            layers = self.forward( x ) # compute the forward
            G1 = [None] * self.num_layers # prepare gradient G1
            mse_t_ = self.back_propagation( layers=layers, output=self.to_one_hot( y, t ), list_of_gradients=G1 )
            #
            # UPDATE OF WEIGHTS
            for l in range(1,self.num_layers):
                psi_mat = psi[l].copy() # save a copy of the psi vector for a given layer
                psi_mat_norm = numpy.linalg.norm( psi_mat, axis=1 ) # computes the norm for each psi vector independently
                for i in range(len(psi_mat_norm)): psi_mat[i] = psi_mat[i] / psi_mat_norm[i] # normalize the psi vectors
                # W' = W + alpha*psi_mat
                self.functionals[l].weights[:,:] = self.functionals[l].weights[:,:] + alpha * psi_mat
                #
            # COMPUTATION OF THE GRADIENT OF E(W'): G2
            layers = self.forward( x ) # compute the forward
            G2 = [None] * self.num_layers # prepare the gradient G2
            mse_t_ = self.back_propagation( layers=layers, output=self.to_one_hot( y, t ), list_of_gradients=G2 )
            #
            # UPDATE OF PSI
            N1 = 0.0 # for storing the total norm of PSI before update
            N2 = 0.0 # for storing the total difference of norms of PSI before and after update
            norms = numpy.zeros( self.num_layers )
            # psi_ nueva=(1-gamma)psi.mat +  (alpha/gamma)*(G2-G1)
            for l in range(1,self.num_layers):
                norms[l] = numpy.linalg.norm( psi[l] )
                N1 += norms[l]
                # psi' = (1-gamma)*psi + (gamma/alpha) * (G2-G1)
                psi[l][:,:] = (1.0 - gamma) * psi[l][:,:]  + (gamma/alpha) * (G2[l][:,:] - G1[l][:,:])
                norms[l] -= numpy.linalg.norm( psi[l] )
                N2 += abs( norms[l] )
            # end for
            #
            # Restore weights
            for l in range(1,self.num_layers):
                self.functionals[l].weights[:,:] = saved_weights[l][:,:]
            # end for
            #
            #print( " %12d   N1 = %f   N2 = %f " % (repetitions,N1,N2) )
            #if repetitions > 10  and  N2 < 0.01 * N1: break
            if N2 < 0.1 * N1: break # stop optimal learning rate estimation when the difference of |norm(psi)-norm(psi')| < 10% of norm(psi)
            repetitions += 1
        # end while
        #
        # UPDATE LEARNING RATES FOR EACH LAYER
        for l in range(1,self.num_layers):
            if self.functionals[l].learning_rate is not None:
                self.functionals[l].learning_rates = self.functionals[l].learning_rate / numpy.linalg.norm( psi[l], axis=1 )
            else:
                self.functionals[l].learning_rates = self.learning_rate / numpy.linalg.norm( psi[l], axis=1 )
        # end for
    # -------------------------------------------------------------------------
