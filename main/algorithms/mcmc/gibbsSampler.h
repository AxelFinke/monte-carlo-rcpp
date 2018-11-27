/// \file
/// \brief Implements a (particle or Metropolis-within-) Gibbs sampler.
///
/// This file contains the functions for implementing a 
/// (potentially non-centred) Gibbs sampler which updates the 
/// latent variables using
/// (a) Metropolis--Hastings updates,
////(b) a standard conditional sequential Monte Carlo algorithm or
/// (c) the embedded hidden Markov model method (Shestopaloff & Neal, 2016).

#ifndef __GIBBSSAMPLER_H
#define __GIBBSSAMPLER_H

#include "main/algorithms/smc/Smc.h"
// #include "main/ehmm/Ehmm.h" // TODO: this file should not depend on Ehmm.h!
#include "main/algorithms/mwg/Mwg.h"

/// Type of Monte Carlo algorithm used to sample the latent states
enum SamplerType 
{   
  SAMPLER_MWG = 0, // using Metropolis-within-Gibbs updates
  SAMPLER_SMC,     // using conditional SMC updates
  SAMPLER_EHMM,    // using conditional EHMM updates
  SAMPLER_EXACT    // sampling the latent variables from their full conditional posterior distribution (if available)
};
/// Type of marginalisation performed for the parameters 
/// of the observation equation.
enum MarginalisationType 
{   
  MARGINALISATION_NONE = 0, // do not analytically integrate any parameters
  MARGINALISATION_PARTIAL, // only analytically integrate out parameters at certain steps of the algorithm
  MARGINALISATION_FULL // always integrate out certain parameters
};

/// Class for running the Gibbs samplers or 
/// Metropolis-within-Gibbs algorithms.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters, class EhmmParameters> GibbsSampler
{
public:
  
  /// Initialises the class.
  GibbsSampler
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    Mwg<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, MwgParameters>& mwg,
    Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>& smc,
    Ehmm<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EhmmParameters>& ehmm
  ) : 
    rng_(rng), 
    model_(model),
    mwg_(mwg), 
    smc_(smc),
    ehmm_(ehmm)
  {
    estimateTheta_ = true;
    samplerType_ = SAMPLER_MWG;
    marginalisationType_ = MARGINALISATION_NONE;
    nonCentringProbability_ = 0.0;
    nCores_ = 1;
  }
  
  /// Specifies the type of algorithm to be used to update
  /// the latent variables.
  void setSamplerType(const SamplerType& samplerType) {samplerType_ = samplerType;}
  /// Specifies the type of analytical marginalisation of some 
  /// model parameters to be performed.
  void setMarginalisationType(const MarginalisationType& marginalisationType) {marginalisationType_ = marginalisationType;}
  /// Specifies the number of cores to be used (currently not used).
  void setNCores(const unsigned int nCores) {nCores_ = nCores;}
  /// Specifies the number of model-parameter updates in between
  /// each set of latent-variable updates.
  void setNParameterUpdates(const unsigned int nParameterUpdates) {nParameterUpdates_ = nParameterUpdates;}
  /// Specifies if the model parameters should be estimates
  /// (otherwise, they are kept fixed at their initial values).
  void setEstimateTheta(const bool estimateTheta) {estimateTheta_ = estimateTheta;}
  /// Specifies the total number of iterations (i.e. sweeps); includes burn-in.
  void setNIterations(const unsigned int nIterations) {nIterations_ = nIterations;}
  /// Specifies the probability of switching to the non-centred parametrisation
  /// (if implemented) when updating the model parameters.
  void setNonCentringProbability(const double nonCentringProbability) {nonCentringProbability_ = nonCentringProbability;}
  /// Specifies the vector of standard deviations of the uncorrelated Gaussian
  /// random-walk proposals for the full set of model parameters.
  void setProposalScales(const arma::colvec& proposalScales) {proposalScales_ = proposalScales;}
  /// Specifies the vector of standard deviations of the uncorrelated Gaussian
  /// random-walk proposals for the full set of model parameters.
  void setProposalScalesMarginalised(const arma::colvec& proposalScalesMarginalised) {proposalScalesMarginalised_ = proposalScalesMarginalised;}
  /// Runs the Gibbs sampler for some vector of initial parameter values
  /// "thetaInit" and returns the vector "output", each of which stores
  /// parameter values and potentially some other quantities generated at
  /// a single iteration.
  void runSampler(std::vector<arma::colvec>& output, const arma::colvec& thetaInit)
  {
    runSamplerBase(output, thetaInit);
  }

private:
  
  /// Runs the Gibbs sampler.
  void runSamplerBase(std::vector<arma::colvec>& output); 
  /// Evaluates the log of the complete likelihood, i.e. including
  /// the log-conditional prior density of the latent variables.
  double evaluateLogCompleteLikelihood(const arma::colvec& theta, const LatentPath& latentPath, const bool includeLatentPriorDensity);
  /// Initialises the latent-variable path object.
  void initialiseLatentPath(const arma::colvec& theta, LatentPath& latentPath);
  /// Initialises the latent-variable path.
  void initialiseLatentVariables(const arma::colvec& theta, LatentPath& latentPath)
  {
    initialiseLatentPath(theta, latentPath);
    smc_.runSampler(theta, latentPath);
  }
  /// Samples parameters which had been analytically integrated out during the 
  /// parameter-update steps from their full conditional posterior distribution.
  void sampleMarginalisedParameters(arma::colvec& theta, const LatentPath& latentPath);
  /// Proposes a new vector of parameter values
  void proposeTheta(arma::colvec& thetaProp, const arma::colvec& theta);
  /// Computes part of the log of the Metropolis--Hastings acceptance probability
  /// (the ratio of the priors and proposal densities) needed for the parameter updates.
  double computeLogAlpha(const arma::colvec& thetaProp, const arma::colvec& theta)
  {
    // NOTE: we are using a multivariate Gaussian random-walk (i.e. symmetric) proposal here!
    model_.evaluateLogPriorDensity(thetaProp) - model_.evaluateLogPriorDensity(theta);
  }
  /// Proposes a new vector of parameter values.
  void proposeTheta(arma::colvec& thetaProp, const arma::colvec& theta)
  {
//     thetaProp.set_size(theta.n_rows);
    if (model_.getMarginaliseParameters())
    {
      thetaProp = theta + proposalScalesMarginalised_ % arma::randn<arma::colvec>(theta.size());
    }
    else
    {
      thetaProp = theta + proposalScales_ % arma::randn<arma::colvec>(theta.size());
    }  
  }
  /// Proposes new set of latent variables (using the NCP).
  void proposeLatentPath(LatentPath& latentPathProp, const LatentPath& latentPath, const arma::colvec& thetaProp, const arma::colvec& theta);
  /// Updates the model parameters.
  void updateParameters(arma::colvec& theta, LatentPath& latentPath)
  {

    double logCompleteLikelihood, logCompleteLikelihoodProp; // log of the complete likelihoods
    double logAlpha; // (part of the) log-acceptance probability for the Metropolis--Hastings parameter updates
    arma::colvec thetaProp(theta.size); // vector of newly-proposed model parameters
    
    if (arma::randu() < nonCentringProbability_) // i.e. update the model parameters using the NCP
    {
      LatentPath latentPathProp;
      initialiseLatentPath(theta, latentPathProp);
      
      logCompleteLikelihood = evaluateLogCompleteLikelihood(theta, latentPath, false);
      for (unsigned int n=0; n<nParameterUpdates_; n++)
      { 
        proposeTheta(thetaProp, theta); // propose parameters   
        logAlpha = computeLogAlpha(thetaProp, theta);
        
        if (std::isfinite(logAlpha))
        {
          proposeLatentPath(latentPathProp, latentPath, thetaProp, theta); // Potentially sample the additional points needed for the NCP
          logCompleteLikelihoodProp = evaluateLogCompleteLikelihood(thetaProp, latentPathProp, false);
          logAlpha += logCompleteLikelihoodProp - logCompleteLikelihood;
        }
        else
        {
          std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
        }  
        
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        { 
          std::cout << "######################### acceptance #########################" << std::endl;
          theta = thetaProp;
          logCompleteLikelihood = logCompleteLikelihoodProp;
          latentPath = latentPathProp;
        }
      }

    }
    else // i.e. update the model parameters using the CP
    {
      logCompleteLikelihood = evaluateLogCompleteLikelihood(theta, latentPath, true);
  
      for (unsigned int n=0; n<nParameterUpdates_; n++)
      { 
        proposeTheta(thetaProp, theta); // propose parameters    
        logAlpha = computeLogAlpha(thetaProp, theta);
        
        if (std::isfinite(logAlpha))
        {
          logCompleteLikelihoodProp = evaluateLogCompleteLikelihood(thetaProp, latentPath, true);
          logAlpha += logCompleteLikelihoodProp - logCompleteLikelihood;
        }
        else
        {
          std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
        }  
        
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        { 
          std::cout << "######################### acceptance #########################" << std::endl;
          theta = thetaProp;
          logCompleteLikelihood = logCompleteLikelihoodProp;
        }
      }
    }   
  }
  /// Initialises the vector in which the output is to be stored.
  void initialiseOutput(std::vector<arma::colvec>& output);
  /// Stores output.
  void storeOutput(const unsigned int g, std::vector<arma::colvec>& output, const arma::colvec& theta, const LatentPath& latentPath);
  /// Updates the latent variables.
  void upateLatentVariables(const arma::colvec& theta, LatentPath& latentPath)
  {
    if (samplerType_ == SAMPLER_MWG)
    {    
      mwg_.runSampler(theta, latentPath);
    }
    else if (samplerType_ == SAMPLER_SMC)
    {
      smc_.runConditionalSampler(theta, latentPath);
    }
    else if (samplerType_ == SAMPLER_EHMM)
    {
      ehmm_.runConditionalSampler(theta, latentPath); // TODO
    }
  }
  
  Rng& rng_; // random number generation.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // the targeted model.
  Mwg<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, MwgParameters>& mwg_, // TODO: the Metropolis-within-Gibbs updates
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>& smc_, // the SMC updates
  Ehmm<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EhmmParameters>& ehmm_, // TODO: the embedded HMM updates
  SamplerType samplerType_; // type of algorithm used for updating the latent variables
  MarginalisationType marginalisationType_; // should certain model parameters (for which this is possible) be integrated out analytically when updating the latent variables?
  
  unsigned int nIterations_; // number of iterations (including burn-in)
  bool estimateTheta_; // should we update the model parameters?
  unsigned int nParameterUpdates_; // number of parameter updates in between each (set of) latent-variable update(s)
  arma::colvec proposalScales_, proposalScalesMarginalised_; // standard deviations of the uncorrelated Gaussian random-walk proposals for the parameter updates (for the case that we sample the full parameter vector and for the case that some parameters have been integrated out). TODO: set these and sort out whether we want matrix or vector
  double nonCentringProbability_; // probability of using the non-centred parameterisation when updating the model parameters
  unsigned int nCores_; // number of cores (this parameter is currently not used)
  
}

/// Runs the Gibbs sampler.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters,  class EhmmParameters> void GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, EhmmParameters, MwgParameters>::runSamplerBase
(
  std::vector<arma::colvec>& output, // static parameters and other output to be stored
  const arma::colvec& thetaInit // initial parameter values
)
{
 
  LatentPath latentPath; // latent variables under the standard parametrisation
  arma::colvec theta = thetaInit; // vector of model parameters

  // Initialisation
  /////////////////////////////////////////////////////////////////////////////
  
  if (marginalisationType_ == MARGINALISATION_FULL)
  {
    model_.setMarginaliseParameters(true);
  }
  else
  {
    model_.setMarginaliseParameters(false);
  }
  
  initialiseLatentVariables(theta, latentPath);
  
  // Initialise and store output:
  initialiseOutput(output);
  storeOutput(0, output, theta, latentPath);
  
  // Recursion
  /////////////////////////////////////////////////////////////////////////////
  
  for (unsigned int g=1; g<nIterations; g++)
  {
    
    if (marginalisationType_ == MARGINALISATION_FULL)
    {
      model_.setMarginaliseParameters(true);
    }
    else
    {
      model_.setMarginaliseParameters(false);
    }
    
    /// Update the latent variables
    updateLatentVariables(theta, latentPath);
    
    
    if (estimateTheta_) // i.e. unless we fix the model parameters
    { 
      if (marginalisationType_ == MARGINALISATION_FULL || marginalisationType_ == MARGINALISATION_PARTIAL)
      {
        model_.setMarginaliseParameters(true);
      }
      else
      {
        model_.setMarginaliseParameters(false);
      }
      // Update the model parameters
      updateParameters(theta, latentPath)
      
      // Sample the parameters from their full conditional posterior distribution
      // for which this full conditional distribution is available.
      sampleMarginalisedParameters(theta, latentPath);
    }
   
    // Store output:
    storeOutput(g, output, theta, latentPath);

  }
}

#endif
