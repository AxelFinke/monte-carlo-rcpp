/// \file
/// \brief Implements algorithms for comparing particle and ensemble MCMC methods.
///
/// This file contains the functions for comparing particle MCMC methods with 
/// the original and modified embedded HMM/ensemble MCMC methods.

#ifndef __COMPARISON_H
#define __COMPARISON_H

#include "model/Model.h"
#include "mcmc/Mcmc.h"
#include "smc/Smc.h"
#include "ensemble/ensembleOld.h"
#include "ensemble/ensembleNew.h"


// this file will contain two functions: one for running a PMCMC/EMCMC algorithms and the other for comparing normalising-constant estimates.

// TODO: at the moments, these functions are defined directly in linear.cpp


/// Type of MCMC algorithm to use.
enum McmcType 
{  
  MCMC_MARGINAL = 0, // exact or "pseudo-"marginal algorithms
  MCMC_GIBBS, // exact or "pseudo-" Gibbs samplers
  MCMC_ASYMMETRIC // the asymmetric Metropolis--Hastings algorithm based around (subsampled) CSMC algorithms from Yildirim et al. (2017)
};

/// Type of Monte Carlo algorithm used to sample the latent states
enum SamplerType 
{  
  SAMPLER_SMC = 0, 
  SAMPLER_ORIGINAL_EHMM, 
  SAMPLER_ALTERNATIVE_EHMM,
  SAMPLER_EXACT
};

///////////////////////////////////////////////////////////////////////////////
/// Approximating the marginal likelihood.
///////////////////////////////////////////////////////////////////////////////

/// Approximates the marginal likelihood via an SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class EnsembleOldParameters, class EnsembleNewParameters>
double approximateMarginalLikelihood
(
  const SamplerType& samplerType,
  Rng& rngDerived, 
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model, 
  const arma::colvec& theta,
  const unsigned int prop, 
  const unsigned int nSteps, 
  const unsigned int nParticles, 
  const EnsembleNewInitialisationType& ensembleNewInitialisationType, // type of initialisation of the MCMC chains at each time step of the sequential MCMC method
  const unsigned int nBurninSamples, // number of additional samples discarded as burnin at each time step in the sequential MCMC method
  const double essResamplingThreshold, 
  const arma::colvec& smcParameters,
  const arma::colvec& ensembleOldParameters, 
  const arma::colvec& ensembleNewParameters,
  const EnsembleNewLocalType& local,
  const unsigned int nCores
)
{
    // Class for running SMC algorithms.
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(prop), 
    essResamplingThreshold,
    static_cast<SmcBackwardSamplingType>(0),
    false, 0,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  
  // Class for running orginal EHMM algorithms.
  EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters> ensembleOld(
    rngDerived, model, nSteps,
    static_cast<EnsembleOldProposalType>(prop), 
    ensembleOldParameters,                                                                                                     
    nCores
  );
  
  // Class for running orginal EHMM algorithms.
  EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters> ensembleNew(
    rngDerived, model, nSteps,
    static_cast<EnsembleNewProposalType>(prop),                                                                                                                                              local, ensembleNewParameters, static_cast<SmcBackwardSamplingType>(0), nCores
  );
  
  ensembleNew.setNBurninSamples(nBurninSamples);
  ensembleNew.setEnsembleNewInitialisationType(ensembleNewInitialisationType);
  
  LatentPath latentPath;
  AuxFull<Aux> aux;
  double logLike = 0;
  
  if (samplerType == SAMPLER_SMC)
  {
    logLike = smc.runSmc(nParticles, theta, latentPath, aux, 1.0);
  }
  else if (samplerType == SAMPLER_ORIGINAL_EHMM)
  {
    logLike = ensembleOld.runSmc(nParticles, theta, latentPath,  1.0);
  }
  else if (samplerType == SAMPLER_ALTERNATIVE_EHMM)
  {
    logLike = ensembleNew.runSmc(nParticles, theta, latentPath,  1.0);
  }
  else if (samplerType == SAMPLER_EXACT)
  {
    logLike = model.evaluateLogMarginalLikelihood(theta);
  }
  
  return logLike;
}


///////////////////////////////////////////////////////////////////////////////
/// Running (pseudo-)marginal algorithms
///////////////////////////////////////////////////////////////////////////////

/// Run a PMMH algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class EnsembleOldParameters, class EnsembleNewParameters, class McmcParameters>
void runParticleMetropolis
(
  std::vector<arma::colvec>& thetaFull,
  arma::colvec& ess,            // storing the ess
  arma::colvec& acceptanceRates,  // storing the acceptance rates
  const SamplerType& samplerType,
  Rng& rngDerived, 
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model, 
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc, 
  const unsigned int nIterations,
  const unsigned int burnin,
  const bool useGradients, 
  const unsigned int fixedLagSmoothingOrder, 
  const unsigned int prop, 
  const unsigned int nSteps, 
  const unsigned int nParticles,
  const double essResamplingThreshold,
  const arma::colvec& smcParameters, 
  const arma::colvec& ensembleOldParameters, 
  const arma::colvec& ensembleNewParameters,
  const EnsembleNewLocalType& local,
  const unsigned int nCores
)
{
  
  // Class for running SMC algorithms.
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(prop), 
    essResamplingThreshold,
    static_cast<SmcBackwardSamplingType>(1),
    useGradients,
    fixedLagSmoothingOrder,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.getRefSmcParameters().setParameters(smcParameters);
  
  // Class for running orginal EHMM algorithms.
  EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle,  EnsembleOldParameters> ensembleOld(
    rngDerived, model, nSteps,
    static_cast<EnsembleOldProposalType>(prop), 
    ensembleOldParameters,                                                                                                     
    nCores
  );
  
  // Class for running orginal EHMM algorithms.
  EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters> ensembleNew(
    rngDerived, model, nSteps,
    static_cast<EnsembleNewProposalType>(prop),                                                                                                                                              local, ensembleNewParameters, static_cast<SmcBackwardSamplingType>(0), nCores
  );
  
  
  
  
  // TODO: implement support for use of gradient information
  LatentPath latentPath;
  AuxFull<Aux> aux; // TODO: implement support for correlated psuedo-marginal approaches
  arma::colvec thetaProp(model.getDimTheta());
  arma::colvec theta(model.getDimTheta());
  double logLikeProp = 0;
  double logLike = 0;
  double logAlpha = 0;
  
  model.sampleFromPrior(theta);
  thetaFull.resize(nIterations);
  thetaFull[0] = theta;
  
  if (samplerType == SAMPLER_SMC)
  {
    logLike = smc.runSmc(nParticles, theta, latentPath, aux, 1.0);
  }
  else if (samplerType == SAMPLER_ORIGINAL_EHMM)
  {
    logLike = ensembleOld.runSmc(nParticles, theta, latentPath, 1.0);
  }
  else if (samplerType == SAMPLER_ALTERNATIVE_EHMM)
  {
    logLike = ensembleNew.runSmc(nParticles, theta, latentPath, 1.0);
//     ess = ess + ensembleNew.getEss()/nIterations;
//     acceptanceRates = acceptanceRates + ensembleNew.getAcceptanceRates()/nIterations;
  }
  else if (samplerType == SAMPLER_EXACT)
  {
    logLike = model.evaluateLogMarginalLikelihood(theta);
  }
  
  for (unsigned int g=1; g<nIterations; g++)
  {
    
    std::cout << "Iteration " << g << " of MH algorithm with lower-level algorithnm " << samplerType << " with Proposal " << prop << " and local type " << static_cast<unsigned int>(local) << std::endl;
    
    mcmc.proposeTheta(thetaProp, theta);
    logAlpha  = - mcmc.evaluateLogProposalDensity(thetaProp, theta);
    logAlpha += model.evaluateLogPriorDensity(thetaProp) - model.evaluateLogPriorDensity(theta);

    if (std::isfinite(logAlpha))
    {
      if (samplerType == SAMPLER_SMC)
      {
        logLikeProp = smc.runSmc(nParticles, thetaProp, latentPath, aux, 1.0);
      }
      else if (samplerType == SAMPLER_ORIGINAL_EHMM)
      {
        logLikeProp = ensembleOld.runSmc(nParticles, thetaProp, latentPath, 1.0);
      }
      else if (samplerType == SAMPLER_ALTERNATIVE_EHMM)
      {
        logLikeProp = ensembleNew.runSmc(nParticles, thetaProp, latentPath, 1.0);
        if (g > burnin) 
        {
          ess = ess + ensembleNew.getEss()/(nIterations-burnin);
          acceptanceRates = acceptanceRates + ensembleNew.getAcceptanceRates()/(nIterations-burnin);
        }
      }
      else if (samplerType == SAMPLER_EXACT)
      {
        logLikeProp = model.evaluateLogMarginalLikelihood(thetaProp);
      }
      logAlpha += logLikeProp - logLike;
      logAlpha += mcmc.evaluateLogProposalDensity(theta, thetaProp);
    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }
    std::cout << "logAlpha: " << logAlpha << std::endl;
              
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      std::cout << "################### ACCEPTANCE ###################" << std::endl;
      theta = thetaProp;
      logLike = logLikeProp;
    }
    thetaFull[g] = theta;
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Running (pseudo-)Gibbs samplers
///////////////////////////////////////////////////////////////////////////////

/// Run a Particle Gibbs sampler.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters,  class EnsembleOldParameters, class EnsembleNewParameters, class McmcParameters>
void runParticleGibbs
(
  std::vector<arma::colvec>& thetaFull,      // static parameters to be stored
  arma::mat& someStateComponents,            // (L, nIterations)-matrix of traces of some components of some states to be stored
  std::vector<arma::umat>& parentIndicesFull, // TODO
  std::vector<arma::uvec>& particleIndicesInFull, // TODO
  std::vector<arma::uvec>& particleIndicesOutFull, // TODO
  const bool storeParentIndices, // TODO
  const bool storeParticleIndices, // TODO
  const bool initialiseStatesFromStationarity, // TODO
  arma::colvec& ess,            // storing the ess
  arma::colvec& acceptanceRates,             // storing the acceptance rates
  const arma::uvec& times,                   // vector of length L which specifies the time steps of which state components are to be stored
  const arma::uvec& components,              // vector of length L which specifies the components which are to be stored
  const SamplerType& samplerType,
  Rng& rngDerived, 
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model, 
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc, 
  const unsigned int nIterations, 
  const unsigned int burnin,
  const unsigned int prop, 
  const unsigned int nThetaUpdates,          // number of parameter updates per iteration of Gibbs samplers of CSMC-based algorithms
  const unsigned int nSteps, 
  const unsigned int nParticles,
  const double essResamplingThreshold, 
  const unsigned int backwardSamplingType,   // type of backward-sampling scheme to use with the lower-level conditional SMC kernels
  const bool useNonCentredParametrisation,   // should Gibbs-sampling type algorithms use an NCP?
  const double nonCentringProbability,       // probability of using an NCP (if useNonCentredParametrisation == true)
  const arma::colvec& smcParameters, 
  const arma::colvec& ensembleOldParameters, 
  const arma::colvec& ensembleNewParameters,
  const EnsembleNewLocalType& local,
  const ResampleType& resampleType,          // type of resampling scheme to use
  const bool estimateTheta,                  // should the static parameters be estimated?
  const arma::colvec thetaInit,              // initial value for theta (if we keep theta fixed throughout)
  const unsigned int nCores                  // number of cores
)
{
  
//   std::cout << "set up Smc class" << std::endl;
  // Class for running SMC algorithms.
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(prop), 
    essResamplingThreshold, static_cast<SmcBackwardSamplingType>(backwardSamplingType),
    false, 0, nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.getRefSmcParameters().setParameters(smcParameters);
  smc.setResampleType(resampleType);
  
//     std::cout << "set up EnsembleOld class" << std::endl;
  
  // Class for running orginal EHMM algorithms.
  EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle,  EnsembleOldParameters> ensembleOld(
    rngDerived, model, nSteps,
    static_cast<EnsembleOldProposalType>(prop), 
    ensembleOldParameters,                                                                                                     
    nCores
  );
  
    // Class for running orginal EHMM algorithms.
  EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters> ensembleNew(
    rngDerived, model, nSteps,
    static_cast<EnsembleNewProposalType>(prop),
    local, ensembleNewParameters, static_cast<SmcBackwardSamplingType>(backwardSamplingType), nCores
  );
  
//       std::cout << "set up EnsembleNew class" << std::endl;
  
  LatentPath latentPath;
  AuxFull<Aux> aux; 
  arma::colvec thetaProp(model.getDimTheta());
  arma::colvec theta(model.getDimTheta());
  double logLike = 0;
  double logCompleteLikeProp = 0;
  double logCompleteLike = 0;
  double logAlpha = 0;
  
  
  if (estimateTheta)
  {
    model.sampleFromPrior(theta);
  }
  else 
  {
    theta = thetaInit;
  }
  
  thetaFull.resize(nIterations);
  someStateComponents.set_size(components.size(), nIterations);
  thetaFull[0] = theta;
  
  if (storeParentIndices)
  {
    parentIndicesFull.resize(nIterations);
  }
  if (storeParticleIndices)
  {
    particleIndicesInFull.resize(nIterations);
    particleIndicesOutFull.resize(nIterations);
  }
    
  
  if (samplerType == SAMPLER_SMC)
  {
    if (initialiseStatesFromStationarity)
    {
      model.runGibbs(theta, latentPath, 1.0);
      logLike = smc.runCsmc(nParticles, theta, latentPath, aux, 1.0);
    }
    else
    {
      logLike = smc.runSmc(nParticles, theta, latentPath, aux, 1.0);
    }

    if (storeParentIndices)
    {
      smc.getParentIndicesFull(parentIndicesFull[0]);
    }
    if (storeParticleIndices)
    {
      smc.getParticleIndicesIn(particleIndicesInFull[0]);
      smc.getParticleIndicesOut(particleIndicesOutFull[0]); 
    }
  }
  else if (samplerType == SAMPLER_ORIGINAL_EHMM)
  {
    
    if (initialiseStatesFromStationarity)
    {
      model.runGibbs(theta, latentPath, 1.0);
      logLike = ensembleOld.runCsmc(nParticles, theta, latentPath, 1.0);
    }
    else
    {
      logLike = ensembleOld.runSmc(nParticles, theta, latentPath, 1.0);
    }

//     if (storeParentIndices)
//     {
//       ensembleOld.getParentIndicesFull(parentIndicesFull[0]);
//     }
    if (storeParticleIndices)
    {
      ensembleOld.getParticleIndicesIn(particleIndicesInFull[0]);
      ensembleOld.getParticleIndicesOut(particleIndicesOutFull[0]); 
    }
  }
  else if (samplerType == SAMPLER_ALTERNATIVE_EHMM)
  {
    
    if (initialiseStatesFromStationarity)
    {
      model.runGibbs(theta, latentPath, 1.0);
      logLike = ensembleNew.runCsmc(nParticles, theta, latentPath, 1.0);
    }
    else
    {
      logLike = ensembleNew.runSmc(nParticles, theta, latentPath, 1.0);
    }
//     ess = ess + ensembleNew.getEss()/nIterations;
//     acceptanceRates = acceptanceRates + ensembleNew.getAcceptanceRates()/nIterations;
    
    if (storeParentIndices)
    {
      ensembleNew.getParentIndicesFull(parentIndicesFull[0]);
    }
    if (storeParticleIndices)
    {
      ensembleNew.getParticleIndicesIn(particleIndicesInFull[0]);
      ensembleNew.getParticleIndicesOut(particleIndicesOutFull[0]); 
    }
  }
  else if (samplerType == SAMPLER_EXACT)
  {
    model.runGibbs(theta, latentPath, 1.0);
  }
  
//   std::cout << "finished initial iteration"  << std::endl;
  
  
  // Storing some state components // TODO: this only currently works if latentPath is of type arma::mat
  for (unsigned int l=0; l<components.size(); l++)
  {  
    
//     std::cout << "components: " << components(l) << "; times: " << times(l)  << std::endl;
    someStateComponents(l,0) = latentPath(components(l),times(l));
  }
  
//   std::cout << logLike << std::endl;
  
  for (unsigned int g=1; g<nIterations; g++)
  {
    
//     std::cout << "Iteration " << g << " of particle Gibbs sampler with lower-level algorithnm " << samplerType << " with Proposal " << prop << " and local type " << static_cast<unsigned int>(local) << std::endl;

    if (estimateTheta)
    {
//       std::cout << "start evaluate logCompleteLike()" << std::endl;
      logCompleteLike = model.evaluateLogCompleteLikelihood(theta, latentPath, 1.0);
//             std::cout << "finished evaluate logCompleteLike()" << std::endl;
      
      
      for (unsigned int n=0; n<nThetaUpdates; n++)
      {
        // Propose parameters.        
        mcmc.proposeTheta(thetaProp, theta);
        logAlpha = mcmc.computeLogAlpha(thetaProp, theta);
        
        if (std::isfinite(logAlpha))
        {
//                   std::cout << "start evaluate logCompleteLikeProp()" << std::endl;
          logCompleteLikeProp = model.evaluateLogCompleteLikelihood(thetaProp, latentPath, 1.0);
          logAlpha += logCompleteLikeProp - logCompleteLike;
          
//                   std::cout << "finished evaluate logCompleteLikeProp()" << std::endl;
        }
        else
        {
          std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
        }  
        
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        { 
          std::cout << "######################### acceptance #########################" << std::endl;
          theta = thetaProp;
          logCompleteLike = logCompleteLikeProp;
        }
      }
    }
    

      
    if (samplerType == SAMPLER_SMC)
    {
      logLike = smc.runCsmc(nParticles, theta, latentPath, aux, 1.0);
      
      if (storeParentIndices)
      {
        smc.getParentIndicesFull(parentIndicesFull[g]);
      }
      if (storeParticleIndices)
      {
        smc.getParticleIndicesIn(particleIndicesInFull[g]);
        smc.getParticleIndicesOut(particleIndicesOutFull[g]); 
      }
    }
    else if (samplerType == SAMPLER_ORIGINAL_EHMM)
    {
      logLike = ensembleOld.runCsmc(nParticles, theta, latentPath, 1.0);
      
//       if (storeParentIndices)
//       {
//         ensembleOld.getParentIndicesFull(parentIndicesFull[g]);
//       }
      if (storeParticleIndices)
      {
        ensembleOld.getParticleIndicesIn(particleIndicesInFull[g]);
        ensembleOld.getParticleIndicesOut(particleIndicesOutFull[g]); 
      }
    }
    else if (samplerType == SAMPLER_ALTERNATIVE_EHMM)
    {
              
//           std::cout << "start runCSMC" << std::endl;
      logLike = ensembleNew.runCsmc(nParticles, theta, latentPath, 1.0);
      
      if (storeParentIndices)
      {
        ensembleNew.getParentIndicesFull(parentIndicesFull[g]);
      }
      if (storeParticleIndices)
      {
        ensembleNew.getParticleIndicesIn(particleIndicesInFull[g]);
        ensembleNew.getParticleIndicesOut(particleIndicesOutFull[g]); 
      }
      
//         std::cout << "finished runCSMC" << std::endl;
      
      if (g > burnin) 
      {
        ess = ess + ensembleNew.getEss()/(nIterations-burnin);
        acceptanceRates = acceptanceRates + ensembleNew.getAcceptanceRates()/(nIterations-burnin);
      }
            

    }
    else if (samplerType == SAMPLER_EXACT)
    {
      model.runGibbs(theta, latentPath, 1.0);
    }
    
    thetaFull[g] = theta;
    // Storing some state components // TODO: this only currently works if latentPath is of type arma::mat
    for (unsigned int l=0; l<components.size(); l++)
    {
      someStateComponents(l,g) = latentPath(components(l),times(l));
    }
  }
}




///////////////////////////////////////////////////////////////////////////////
/// Running the MH algorithm with asymmetric acceptace ratio
/// from Yildirim, Andrieu, Doucet and Chopin (2017)
///////////////////////////////////////////////////////////////////////////////

/// Runs the MH algorithm with asymmetric acceptance ratio
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters,  class EnsembleOldParameters, class EnsembleNewParameters, class McmcParameters>
void runAsymmetricMh
(
  std::vector<arma::colvec>& thetaFull,      // static parameters to be stored
  arma::mat& someStateComponents,            // (L, nIterations)-matrix of traces of some components of some states to be stored
  const bool initialiseStatesFromStationarity, 
  const arma::uvec& times,                   // vector of length L which specifies the time steps of which state components are to be stored
  const arma::uvec& components,              // vector of length L which specifies the components which are to be stored
  const SamplerType& samplerType,
  Rng& rngDerived, 
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model, 
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc, 
  const unsigned int nIterations, 
  const unsigned int burnin,
  const unsigned int prop, 
  const unsigned int nSteps, 
  const unsigned int nParticles,
  const unsigned int nSubsampledPaths,
  const double essResamplingThreshold, 
  const arma::colvec& smcParameters, 
  const arma::colvec& ensembleOldParameters, 
  const arma::colvec& ensembleNewParameters,
  const EnsembleNewLocalType& local,
  const ResampleType& resampleType,          // type of resampling scheme to use
  const arma::colvec thetaInit,              // initial value for theta (if we keep theta fixed throughout)
  const unsigned int nCores                  // number of cores
)
{
  
  unsigned int backwardSamplingType = 1;
  bool useNonCentredParametrisation = false;
  double nonCentringProbability = 0.0;
  
//   std::cout << "set up Smc class" << std::endl;
  // Class for running SMC algorithms.
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(prop), 
    essResamplingThreshold, static_cast<SmcBackwardSamplingType>(backwardSamplingType),
    false, 0, nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.getRefSmcParameters().setParameters(smcParameters);
  smc.setResampleType(resampleType);
  
//     std::cout << "set up EnsembleOld class" << std::endl;
  
  // Class for running orginal EHMM algorithms.
  EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle,  EnsembleOldParameters> ensembleOld(
    rngDerived, model, nSteps,
    static_cast<EnsembleOldProposalType>(prop), 
    ensembleOldParameters,                                                                                                     
    nCores
  );
  
    // Class for running orginal EHMM algorithms.
  EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters> ensembleNew(
    rngDerived, model, nSteps,
    static_cast<EnsembleNewProposalType>(prop),
    local, ensembleNewParameters, static_cast<SmcBackwardSamplingType>(backwardSamplingType), nCores
  );
  
//       std::cout << "set up EnsembleNew class" << std::endl;
  
  LatentPath latentPath;
  LatentPath latentPathProp;
  std::vector<LatentPath> subsampledLatentPaths(nSubsampledPaths); 
  unsigned int pathIndexProp; // index of proposed path (among all the subsampled paths)
  AuxFull<Aux> aux; 
  arma::colvec thetaProp(model.getDimTheta());
  arma::colvec theta(model.getDimTheta());
  arma::colvec thetaTilde(model.getDimTheta()); // the "intermediate" parameter values at which the CSMC algorithm is run
  double logLike = 0; // values stored in this variable are not really used by the algorithm
//   double logCompleteLikeProp = 0;
  double logCompleteLike = 0;
  double logAlpha = 0;
  double v; // uniform random variable needed for deciding between Proposal 1 and 2
  arma::colvec logUnnormalisedWeights(nSubsampledPaths);
  arma::colvec selfNormalisedWeights(nSubsampledPaths);
  
  
  
  
//   model.sampleFromPrior(theta);
  
//   theta = thetaInit;
  thetaFull.resize(nIterations);
  someStateComponents.set_size(components.size(), nIterations);

  
  std::cout << "started initial iteration"  << std::endl;
    
  
  logLike = - std::numeric_limits<double>::infinity();
  logCompleteLike = - std::numeric_limits<double>::infinity();
  
  if (initialiseStatesFromStationarity)
  {
    while (!std::isfinite(logLike) || !std::isfinite(logCompleteLike))
    {
      model.sampleFromPrior(theta);
      model.runGibbs(theta, latentPath, 1.0);
      logLike = smc.runCsmc(nParticles, theta, latentPath, aux, 1.0);
      logCompleteLike = model.evaluateLogCompleteLikelihood(theta, latentPath, 1.0);
    }
  }
  else 
  {
    if (samplerType == SAMPLER_SMC)
    {
      if (smc.getSmcProposalType() == SMC_PROPOSAL_GAUSSIAN_RANDOM_WALK)
      {
        smc.setSmcProposalType(SMC_PROPOSAL_PRIOR);
        while (!std::isfinite(logLike) || !std::isfinite(logCompleteLike))
        {
          model.sampleFromPrior(theta);
          logLike = smc.runSmc(nParticles, theta, latentPath, aux, 1.0); // NOTE: there is no "unconditional" version of the random-walk CSMC algorithm
          logCompleteLike = model.evaluateLogCompleteLikelihood(theta, latentPath, 1.0);
        }
        smc.setSmcProposalType(SMC_PROPOSAL_GAUSSIAN_RANDOM_WALK);
      }
      else 
      {
        while (!std::isfinite(logLike) || !std::isfinite(logCompleteLike))
        {
          model.sampleFromPrior(theta);
          logLike = smc.runSmc(nParticles, theta, latentPath, aux, 1.0); 
          logCompleteLike = model.evaluateLogCompleteLikelihood(theta, latentPath, 1.0);
        }
      }
    }
    else if (samplerType == SAMPLER_ORIGINAL_EHMM)
    {
      while (!std::isfinite(logLike) || !std::isfinite(logCompleteLike))
      {
        model.sampleFromPrior(theta);
        logLike = ensembleOld.runSmc(nParticles, theta, latentPath, 1.0);
        logCompleteLike = model.evaluateLogCompleteLikelihood(theta, latentPath, 1.0);
      }
    }
    else if (samplerType == SAMPLER_ALTERNATIVE_EHMM)
    { 
      while (!std::isfinite(logLike) || !std::isfinite(logCompleteLike))
      {
        model.sampleFromPrior(theta);
        logLike = ensembleNew.runSmc(nParticles, theta, latentPath, 1.0);
        logCompleteLike = model.evaluateLogCompleteLikelihood(theta, latentPath, 1.0);
      }
    }
    else if (samplerType == SAMPLER_EXACT)
    {
      std::cout << "WARNING: MH algorithm with asymmetric acceptance ratio does not work with exact Gibbs updates for the latent states!" << std::endl;
    }
  }
  
  
  thetaFull[0] = theta;


  // Storing some state components // TODO: this only currently works if latentPath is of type arma::mat
  for (unsigned int l=0; l<components.size(); l++)
  {  
    someStateComponents(l,0) = latentPath(components(l),times(l));
  }
  
    
  std::cout << "finished initial iteration"  << std::endl;
  
//   std::cout << logLike << std::endl;
  
  for (unsigned int g=1; g<nIterations; g++)
  {
    std::cout << "theta at iteration " << g << ": " << theta.t() << std::endl;
//     std::cout << "Iteration " << g << " of particle Gibbs sampler with lower-level algorithnm " << samplerType << " with Proposal " << prop << " and local type " << static_cast<unsigned int>(local) << std::endl;

    // Proposes parameters.        
    mcmc.proposeTheta(thetaProp, theta);
    thetaTilde = (thetaProp + theta) / 2.0; // the parameter values at which the CSMC algorithm is run
   
    // Runs the CSMC algorithm at intermediate parameter values.
    if (samplerType == SAMPLER_SMC)
    {
      logLike = smc.runCsmcWithoutPathSampling(nParticles, thetaTilde, latentPath, aux, 1.0); 
      for (unsigned int m=0; m<nSubsampledPaths; m++)
      {
        smc.samplePath(subsampledLatentPaths[m]);
      }
    }
    else if (samplerType == SAMPLER_ORIGINAL_EHMM)
    {
      logLike = ensembleOld.runCsmcWithoutPathSampling(nParticles, thetaTilde, latentPath, 1.0);
      for (unsigned int m=0; m<nSubsampledPaths; m++)
      {
        ensembleOld.samplePath(subsampledLatentPaths[m]);
      }
    }
    else if (samplerType == SAMPLER_ALTERNATIVE_EHMM)
    {
      logLike = ensembleNew.runCsmcWithoutPathSampling(nParticles, thetaTilde, latentPath, 1.0);
      for (unsigned int m=0; m<nSubsampledPaths; m++)
      {
        ensembleNew.samplePath(subsampledLatentPaths[m]);
      }
    }

    v = arma::randu();
    
    if (v < 1/2)
    {
       logUnnormalisedWeights.fill(
         mcmc.computeLogAlpha(thetaProp, theta) - std::log(nSubsampledPaths) 
         + model.evaluateLogCompleteLikelihood(thetaTilde, latentPath, 1.0)
         - model.evaluateLogCompleteLikelihood(theta, latentPath, 1.0)
       );
       for (unsigned int m=0; m<nSubsampledPaths; m++)
       {
         logUnnormalisedWeights(m) = logUnnormalisedWeights(m)
           + model.evaluateLogCompleteLikelihood(thetaProp, subsampledLatentPaths[m], 1.0)               
           - model.evaluateLogCompleteLikelihood(thetaTilde, subsampledLatentPaths[m], 1.0);
       }
       
       selfNormalisedWeights = arma::exp(normaliseExp(logUnnormalisedWeights, logAlpha));
       pathIndexProp = sampleInt(selfNormalisedWeights);
       
    }
    else 
    {
       pathIndexProp = arma::as_scalar(arma::randi(1, arma::distr_param(0,nSubsampledPaths-1)));
       logUnnormalisedWeights.fill(
         mcmc.computeLogAlpha(theta, thetaProp) - std::log(nSubsampledPaths) 
         + model.evaluateLogCompleteLikelihood(thetaTilde, subsampledLatentPaths[pathIndexProp], 1.0)
         - model.evaluateLogCompleteLikelihood(thetaProp, subsampledLatentPaths[pathIndexProp], 1.0)
       );      

       for (unsigned int m=0; m<nSubsampledPaths; m++)
       {
         logUnnormalisedWeights(m) = logUnnormalisedWeights(m)
           + model.evaluateLogCompleteLikelihood(theta, subsampledLatentPaths[m], 1.0)               
           - model.evaluateLogCompleteLikelihood(thetaTilde, subsampledLatentPaths[m], 1.0);
       }
       
       logAlpha = - std::log(arma::accu(arma::exp(logUnnormalisedWeights)));
    }
    
    if (std::isfinite(logAlpha))
    {
      // Empty
    }
    else
    {
      std::cout << "--------- WARNING: logAlpha is not finite! --------- " << logAlpha << std::endl;  
    }  
    
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    { 
      std::cout << "######################### acceptance #########################" << std::endl;
      theta = thetaProp;
      latentPath = subsampledLatentPaths[pathIndexProp];
//       logCompleteLike = logCompleteLikeProp;
    }
    

   
    
    thetaFull[g] = theta;
    // Storing some state components // TODO: this only currently works if latentPath is of type arma::mat
    for (unsigned int l=0; l<components.size(); l++)
    {
      someStateComponents(l,g) = latentPath(components(l),times(l));
    }
  }
}

#endif
