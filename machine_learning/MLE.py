"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://www.dsic.upv.es/~jon)
    Version: 2.0
    Date: October 2016
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC


    Maximum Likelihood Estimation

"""

import sys
import time
import numpy



from . import GMM

try:
    from pyspark import SparkContext
except:
    pass

class MLE:
    """
    """

# ---------------------------------------------------------------------------------            
    def __init__( self, covar_type='diagonal', dim=1, log_dir='log', models_dir='models', min_var=1.0e-5, max_iterations=200 ):

        self.max_iterations=max_iterations
        self.log_dir = log_dir
        self.models_dir = models_dir
        self.min_var = min_var
        self.gmm = GMM( n_components=1, dim=dim, covar_type=covar_type, min_var=self.min_var, _for_accumulating=False )
# ---------------------------------------------------------------------------------            

    """
        - samples must be an RDD
    """
# ---------------------------------------------------------------------------------            


# ---------------------------------------------------------------------------------            
    def standalone_epoch( self, samples, batch_size=1 ):
        _temp_gmm = GMM( self.gmm.n_components, self.gmm.dim, self.gmm.covar_type, min_var=self.min_var, _for_accumulating=True )

        if type( samples ) == numpy.ndarray:
            counter=0
            while counter < len(samples):
                _temp_gmm.accumulate_sample_batch( samples[counter:counter+batch_size].transpose(), self.gmm )
                counter += batch_size
                #sys.stdout.write( "accumulated sample %d/%d at iteration %d at %s\n" % (counter, len(samples), iteration, time.asctime() ) )
                #sys.stdout.flush()
            #
        elif type( samples ) == list:
            counter=0
            for sample in samples:
                _temp_gmm.accumulate_sample_batch( sample.transpose(), self.gmm )
                counter += sample.shape[0]
                #sys.stdout.write( "accumulated %d samples at iteration %d at %s\n" % (counter, iteration, time.asctime() ) )
                #sys.stdout.flush()
        else:
            raise Exception( "This line of code cannot be reached in normal conditions!" )
                    
        self.gmm.update_parameters( _temp_gmm )
        logL = _temp_gmm.log_likelihood / len(samples)
        return logL
# ---------------------------------------------------------------------------------            

# ---------------------------------------------------------------------------------            
    def fit( self, samples=None, epsilon=1.0e-5, batch_size=100, log_file=None ):
        if samples is None :
            raise Exception( "Maximum Likelihood Estimation cannot be done without samples!" )
        if type(samples) not in [ list, numpy.ndarray ]:
            raise Exception( "Maximum Likelihood Estimation. Non recognized data structure:: %s " % (type(samples)) )

        close_log_file=False
        if log_file is None:
            log_file = open( self.log_dir+"/OUT", 'a' )
            close_log_file=True
        #
        log_file.write( 'Starting MLE for %d components\n' %  self.gmm.n_components )
        log_file.flush()
        #
        iterations_after_epsilon_reached=10
        iteration=1
        relative_improvement = 1.0
        logL=0.0
        #
        while iterations_after_epsilon_reached > 0  and  iteration <= self.max_iterations:
            #
            old_logL=logL
            #
            logL = self.standalone_epoch( samples, batch_size=batch_size )
            aic,bic = self.gmm.compute_AIC_and_BIC( logL * len(samples) )
            #
            relative_improvement = abs( (logL-old_logL) / logL )
            log_file.write( "iteration %5d  logL = %e  delta_logL = %e  %e  %e\n" % (iteration, logL, relative_improvement, aic, bic) )
            log_file.flush()
            #
            iteration=iteration+1
            if relative_improvement < epsilon: iterations_after_epsilon_reached -= 1
        #
        if close_log_file:
            log_file.close()
        #
        self.gmm.save_to_text( self.models_dir+'/gmm' )
        #
        return logL
# ---------------------------------------------------------------------------------            

# ---------------------------------------------------------------------------------            
    def fit_standalone( self, samples=None, max_components=None, epsilon=1.0e-5, batch_size=100 ):
        if samples is None :
            raise Exception( "Maximum Likelihood Estimation cannot be done without samples!" )
        if type(samples) not in [ list, numpy.ndarray ]:
            raise Exception( "Maximum Likelihood Estimation. Non recognized data structure:: %s " % (type(samples)) )
        if max_components is None :
            raise Exception( "Maximum Likelihood Estimation cannot be done without a limit in the number of components of the GMM!" )

        log_file = open( self.log_dir+"/OUT", 'a' )

        old_gmm = self.gmm.clone()

        logL=0.0
        while self.gmm.n_components <= max_components:
            logL = self.fit( samples=samples, epsilon=epsilon, batch_size=batch_size, log_file=log_file )
            aic,bic = self.gmm.compute_AIC_and_BIC( logL * len(samples) )
            log_file.write( "n_components %5d  logL = %e  aic %e  bic %e \n" % (self.gmm.n_components, logL, aic, bic) )
            log_file.flush()
            old_gmm = self.gmm.clone()
            #
            """
            self.gmm.save_to_text( self.models_dir+'/gmm' ) NO NEEDED IF USING self.fit() 
            """
            self.gmm.split( log_file )
            self.gmm.save_to_text( self.models_dir+'/gmm-s' )
        #
        log_file.write( "MLE task completed when %d components where execeeded with %d\n" % (max_components, self.gmm.n_components) )
        log_file.flush()
        log_file.close()
        self.gmm = old_gmm
        return old_gmm
# ---------------------------------------------------------------------------------            

# ---------------------------------------------------------------------------------            
    def fit_with_spark( self, spark_context=None, samples=None, max_components=None, epsilon=1.0e-5 ):
    
        if spark_context is None :
            raise Exception( "Maximum Likelihood Estimation cannot be done without a SparkContext!" )
        if samples is None :
            raise Exception( "Maximum Likelihood Estimation cannot be done without samples!" )
        #if type(samples) not in [ list, numpy.ndarray ]:
        #    raise Exception( "Maximum Likelihood Estimation. Non recognized data structure:: %s " % (type(samples)) )
        if max_components is None :
            raise Exception( "Maximum Likelihood Estimation cannot be done without a limit in the number of components of the GMM!" )

        log_file = open( self.log_dir+"/OUT", 'w' )

        num_samples = samples.count()

        logL=0.0
        while self.gmm.n_components <= max_components:
            iteration=1
            iterations_after_epsilon_reached=10
            relative_improvement = 1.0
            while iterations_after_epsilon_reached > 0  and  iteration <= self.max_iterations:
                #
                nc=self.gmm.n_components
                dim=self.gmm.dim
                ct=self.gmm.covar_type
                mv=self.gmm.min_var
                """
                _b_gmm = spark_context.broadcast( self.gmm )
                _temp_gmm = samples.aggregate( GMM( nc, dim, ct, min_var=mv, _for_accumulating=True ), add_sample, combine_gmms )
                """
                _temp_gmm = samples.aggregate( GMM( nc, dim, ct, min_var=mv, _for_accumulating=True ), self.add_sample, combine_gmms )
                #
                old_logL=logL
                logL = _temp_gmm.log_likelihood / num_samples
                #
                relative_improvement = (old_logL-logL) / logL
                self.gmm.update_parameters( _temp_gmm )
                aic,bic = self.gmm.compute_AIC_and_BIC( logL * num_samples )
                log_file.write( "iteration %5d  logL = %e  delta_logL = %e   %e  %e\n" % (iteration, logL, relative_improvement, aic, bic) )
                log_file.flush()
                iteration=iteration+1
                if abs(relative_improvement) < epsilon: iterations_after_epsilon_reached -= 1
            #
            self.gmm.purge( log_file=log_file )
            #
            aic,bic = self.gmm.compute_AIC_and_BIC( logL * num_samples )
            log_file.write( "n_components %5d  logL = %e  aic %e  bic %e \n" % (self.gmm.n_components, logL, aic, bic) )
            log_file.flush()
            self.gmm.save_to_text( self.models_dir+'/gmm' )
            self.gmm.split( log_file )
            self.gmm.save_to_text( self.models_dir+'/gmm' )
        # ---------------------------------------------------------------------------
        log_file.write( "MLE task completed when %d components where execeeded with %d\n" % (max_components, self.gmm.n_components) )
        log_file.flush()

        log_file.close()
# ---------------------------------------------------------------------------------            

# ---------------------------------------------------------------------------------            
    def add_sample( self, gmm, sample ):
        #global _b_gmm
        #stable_gmm = _b_gmm.value
        if type(sample) == numpy.ndarray:
            if len(sample.shape) == 1:
                gmm.accumulate_sample( sample, self.gmm )
            elif len(sample.shape) == 2:
                gmm.accumulate_sample_batch( sample.transpose(), self.gmm )
            else:
                pass # An exception should be thrown
        else:
            pass # An exception should be thrown
        return gmm
# ---------------------------------------------------------------------------------            

# ---------------------------------------------------------------------------------            
"""
def add_sample( gmm, sample ):
    global _b_gmm
    stable_gmm = _b_gmm.value
    gmm.accumulate_sample( sample, stable_gmm )
    return gmm
"""
# ---------------------------------------------------------------------------------            

# ---------------------------------------------------------------------------------            
def combine_gmms( gmm1, gmm2 ):
    gmm1.add(gmm2)
    return gmm1
# ---------------------------------------------------------------------------------            
