/// \file
/// \brief Implements PMMH algorithms (potentially with a delayed-acceptance step).
///
/// This file contains the functions for implementing adaptive particle-marginal
/// Metropolis--Hastings algorithms in which part of the marginal likelihood can 
/// potentially be evaluated analytically so that it can be beneficial to use
/// a delayed-aceptance step.

#ifndef __PMMH_H
#define __PMMH_H

#include "model/Model.h"
#include "mcmc/Mcmc.h"
#include "smc/Smc.h"
#include "smc/default/single.h"
#include "time.h"

///////////////////////////////////////////////////////////////////////////////
/// PMMH algorithm potentially with delayed acceptance
///////////////////////////////////////////////////////////////////////////////

/// Run a PMMH algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void runPmmh
(
  std::vector<arma::colvec>& thetaFull,
  std::vector<LatentPath>& latentPathFull,
  double& cpuTime, // storing the total time needed to run the algorithm
  double& acceptanceRateStage1, // storing the acceptance rate at stage 1 (if we use a delayed-acceptance step)
  double& acceptanceRateStage2, // storing the acceptance rate at stage 2
  Rng& rngDerived, 
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model, 
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>& smc, 
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc, 
  const arma::colvec& thetaInit,
  const bool samplePath, // should one trajectory of the latent variables/particles be stored at each iteration?
  const unsigned int nCores
)
{
    
  // Starting the timer:
  clock_t t1,t2; // timing
  t1 = clock(); // start timer
  
  // TODO: implement support for use of gradient information
  LatentPath latentPathProp, latentPath;
  AuxFull<Aux> aux; // TODO: implement support for correlated psuedo-marginal approaches
  arma::colvec thetaProp(model.getDimTheta());
  arma::colvec theta(model.getDimTheta());
  double logLikeStage1Prop = 0.0;
  double logLikeStage1 = 0.0;
  double logLikeStage2Prop = 0.0;
  double logLikeStage2 = 0.0;
  
  // Empirical covariance matrix of previously sampled parameter vectors to 
  // used for adaptive proposals a la Peters et al. (2010)
  arma::mat sampleMoment2;
  arma::colvec sampleMoment1;
  if (mcmc.getUseAdaptiveProposal())
  {
    sampleMoment2.zeros(model.getDimTheta(), model.getDimTheta());
    sampleMoment1.zeros(model.getDimTheta());
  }

  double logAlpha = 0.0;
  acceptanceRateStage1 = 0.0;
  acceptanceRateStage2 = 0.0;

  // Initial iteration:
  theta = thetaInit;
  thetaFull.resize(mcmc.getNIterations());
  if (samplePath)
  {
    latentPathFull.resize(mcmc.getNIterations());
  }
  thetaFull[0] = theta;
  
          std::cout << "start evaluate partial log-like" << std::endl;
  
  logLikeStage1 = model.evaluateLogMarginalLikelihoodFirst(theta, latentPath);

          std::cout << "started smc algorithm" << std::endl;
          
  logLikeStage2 = smc.runSmc(smc.getNParticles(), theta, latentPath, aux, 1.0);
  
  if (samplePath)
  {
    latentPathFull[0]  = latentPath;
  }
            std::cout << "finished smc algorithm" << std::endl;
  
  
  
  // TODO: problem: the prior density is numerically too low!

  if (mcmc.getUseDelayedAcceptance())
  {
    for (unsigned int g=1; g<mcmc.getNIterations(); g++)
    {
      std::cout << "theta: " << theta.t() << std::endl;
      
      if (mcmc.getUseAdaptiveProposal()) // NOTE: check this!
      {
        sampleMoment1 = (theta + sampleMoment1 * (g-1)) / g;
        sampleMoment2 = (theta * arma::trans(theta) + sampleMoment2 * (g-1)) / g;
        // NOTE: we add some small constant to the diagonal to ensure invertibility.
        mcmc.setSampleCovarianceMatrix(sampleMoment2 - sampleMoment1 * arma::trans(sampleMoment1) + 0.0001 * arma::eye(sampleMoment2.n_rows, sampleMoment2.n_cols)); // NOTE: check this!
      }
      
      std::cout << "Iteration " << g << " of the PMMH algorithm with delayed acceptance" << std::endl;
      mcmc.proposeTheta(g, thetaProp, theta);
      logLikeStage1Prop = model.evaluateLogMarginalLikelihoodFirst(thetaProp, latentPathProp);
      logAlpha = mcmc.evaluateLogProposalDensity(g, theta, thetaProp) -
        mcmc.evaluateLogProposalDensity(g, thetaProp, theta) +
        model.evaluateLogPriorDensity(thetaProp) - 
        model.evaluateLogPriorDensity(theta) + 
        logLikeStage1Prop - 
        logLikeStage1;
        
            std::cout << "logProposalDensityNum: " << mcmc.evaluateLogProposalDensity(g, theta, thetaProp) << std::endl;
      std::cout << "logProposalDensityDen: " << mcmc.evaluateLogProposalDensity(g, thetaProp, theta) << std::endl;
      std::cout << "logPriorDensityNum: " << model.evaluateLogPriorDensity(thetaProp) << std::endl;
      std::cout << "logPriorDensityDen: " << model.evaluateLogPriorDensity(theta) << std::endl;
      std::cout << "logLikeStage1Prop: " << logLikeStage1Prop << std::endl;
      std::cout << "logLikeStage1: " << logLikeStage1 << std::endl;
        
      std::cout << "logAlpha at Stage 1: " << logAlpha << std::endl;
        
      if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
      {
        std::cout << "################### ACCEPTANCE AT STAGE 1 ###################" << std::endl;
        if (g > mcmc.getNBurninSamples()) { acceptanceRateStage1++; };
        logLikeStage2Prop = smc.runSmc(smc.getNParticles(), thetaProp, latentPathProp, aux, 1.0);
        logAlpha = logLikeStage2Prop - logLikeStage2;
        
              std::cout << "logLikeStage2Prop: " << logLikeStage2Prop << std::endl;
      std::cout << "logLikeStage2: " << logLikeStage2 << std::endl;
        
        std::cout << "logAlpha at Stage 2: " << logAlpha << std::endl;
                  
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
          std::cout << "################### ACCEPTANCE AT STAGE 2 ###################" << std::endl;
          theta = thetaProp;
          logLikeStage2 = logLikeStage2Prop;
          logLikeStage1 = logLikeStage1Prop;
          if (samplePath)
          {
            latentPath = latentPathProp;
          }
          if (g > mcmc.getNBurninSamples()) { acceptanceRateStage2++; };
        }
      }
      thetaFull[g] = theta;
      if (samplePath)
      {
        latentPathFull[g]  = latentPath;
      }
    }
  }
  else // i.e. if we do not use delayed acceptance
  {
    for (unsigned int g=1; g<mcmc.getNIterations(); g++)
    {
      
            std::cout << "theta: " << theta.t() << std::endl;
      
      if (mcmc.getUseAdaptiveProposal()) // NOTE: check this!
      {
        sampleMoment1 = (theta + sampleMoment1 * (g-1)) / g;
        sampleMoment2 = (theta * arma::trans(theta) + sampleMoment2 * (g-1)) / g;
        // NOTE: we add some small constant to the diagonal to ensure invertibility.
        mcmc.setSampleCovarianceMatrix(sampleMoment2 - sampleMoment1 * arma::trans(sampleMoment1) + 0.0001 * arma::eye(sampleMoment2.n_rows, sampleMoment2.n_cols)); // NOTE: check this!
      }
      
      std::cout << "Iteration " << g << " of the standard PMMH algorithm" << std::endl;
      mcmc.proposeTheta(g, thetaProp, theta);
      logLikeStage1Prop = model.evaluateLogMarginalLikelihoodFirst(thetaProp, latentPathProp);
      logAlpha = mcmc.evaluateLogProposalDensity(g, theta, thetaProp) -
        mcmc.evaluateLogProposalDensity(g, thetaProp, theta) +
        model.evaluateLogPriorDensity(thetaProp) - 
        model.evaluateLogPriorDensity(theta) + 
        logLikeStage1Prop - 
        logLikeStage1;
        
      std::cout << "logProposalDensityNum: " << mcmc.evaluateLogProposalDensity(g, theta, thetaProp) << std::endl;
      std::cout << "logProposalDensityDen: " << mcmc.evaluateLogProposalDensity(g, thetaProp, theta) << std::endl;
      std::cout << "logPriorDensityNum: " << model.evaluateLogPriorDensity(thetaProp) << std::endl;
      std::cout << "logPriorDensityDen: " << model.evaluateLogPriorDensity(theta) << std::endl;
      std::cout << "logLikeStage1Prop: " << logLikeStage1Prop << std::endl;
      std::cout << "logLikeStage1: " << logLikeStage1 << std::endl;
        
      if (std::isfinite(logAlpha))
      {
        logLikeStage2Prop = smc.runSmc(smc.getNParticles(), thetaProp, latentPathProp, aux, 1.0);
        logAlpha += logLikeStage2Prop - logLikeStage2;
        
                      std::cout << "logLikeStage2Prop: " << logLikeStage2Prop << std::endl;
      std::cout << "logLikeStage2: " << logLikeStage2 << std::endl;
        std::cout << "logAlpha: " << logAlpha << std::endl;       
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
          std::cout << "################### ACCEPTANCE ###################" << std::endl;
          theta = thetaProp;
          logLikeStage2 = logLikeStage2Prop;
          logLikeStage1 = logLikeStage1Prop;
          if (samplePath)
          {
            latentPath = latentPathProp;
          }
          if (g > mcmc.getNBurninSamples()) { acceptanceRateStage2++; };
        }
      }
      else
      {
        std::cout << "--------- WARNING: skipped due to non-finite acceptance probability  --------- " << logAlpha << std::endl;  
      }
      thetaFull[g] = theta;
      
      if (samplePath)
      {
        latentPathFull[g]  = latentPath;
      }
    }
  }

  if (mcmc.getUseDelayedAcceptance())
  {
    acceptanceRateStage2 = acceptanceRateStage2 / std::max(acceptanceRateStage1, 1.0); 
    acceptanceRateStage1 = acceptanceRateStage1 / mcmc.getNKeptSamples();
  }
  else
  {
    acceptanceRateStage2 = acceptanceRateStage2 / mcmc.getNKeptSamples(); 
  }
  
  t2 = clock(); // stop timer 
  cpuTime = (static_cast<double>(t2)-static_cast<double>(t1)) / CLOCKS_PER_SEC; // elapsed time in seconds

}






///////////////////////////////////////////////////////////////////////////////
/// Particle Gibbs sampler
///////////////////////////////////////////////////////////////////////////////

/// Run a Particle Gibbs sampler.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void runPg
(
  std::vector<arma::colvec>& thetaFull,
  double& cpuTime, // storing the total time needed to run the algorithm
  double& acceptanceRate, // storing the acceptance rate for the MH updates
  Rng& rngDerived, 
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model, 
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>& smc, 
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc, 
  const arma::colvec& thetaInit,
  const bool estimateTheta,
  const unsigned int nThetaUpdates,
  const unsigned int csmc, // type of backward or ancestor sampling used in the Gibbs samplers
  const unsigned int nCores
)
{
    
  // Starting the timer:
  clock_t t1,t2; // timing
  t1 = clock(); // start timer
  
  // TODO: implement support for use of gradient information
  LatentPath latentPath;
  AuxFull<Aux> aux; // TODO: implement support for correlated psuedo-marginal approaches
  arma::colvec thetaProp(model.getDimTheta());
  arma::colvec theta(model.getDimTheta());
  double logLike = 0;
  double logCompleteLikeProp = 0;
  double logCompleteLike = 0;
  double logAlpha = 0;
  acceptanceRate = 0.0;
  

  // Initial iteration:
  theta = thetaInit;
  thetaFull.resize(mcmc.getNIterations());
  thetaFull[0] = theta;
  

  
                    std::cout << "started smc algorithm" << std::endl;
  logLike = smc.runSmc(smc.getNParticles(), theta, latentPath, aux, 1.0);
          
              std::cout << "finished smc algorithm" << std::endl;
              
                        std::cout << "start evaluate complete log-like" << std::endl;
//   logLike = model.evaluateLogCompleteLikelihood(theta, latentPath);

 

  
  for (unsigned int g=1; g<mcmc.getNIterations(); g++)
  {
    std::cout << "Iteration " << g << " of the PG sampler" << std::endl;
    
    if (estimateTheta)
    {
//       std::cout << "start evaluate logCompleteLike()" << std::endl;
      logCompleteLike = model.evaluateLogCompleteLikelihood(theta, latentPath);
//             std::cout << "finished evaluate logCompleteLike()" << std::endl;
      
              std::cout << "logCompleteLike: " << logCompleteLike << std::endl;
      
      
      for (unsigned int n=0; n<nThetaUpdates; n++)
      {
        // Propose parameters.        
        mcmc.proposeTheta(thetaProp, theta);
        
        
        logAlpha = mcmc.computeLogAlpha(thetaProp, theta);
        
//         std::cout << "logAlpha before logCompleteLike: " << logAlpha << std::endl;
        
        if (std::isfinite(logAlpha))
        {
//                   std::cout << "start evaluate logCompleteLikeProp()" << std::endl;
          logCompleteLikeProp = model.evaluateLogCompleteLikelihood(thetaProp, latentPath);
          logAlpha += logCompleteLikeProp - logCompleteLike;
          
//                   std::cout << "finished evaluate logCompleteLikeProp()" << std::endl;
        }
        else
        {
          std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
        }  
        
//         std::cout << "logCompleteLike: " << logCompleteLike << "; logCompleteLikeProp: " << logCompleteLikeProp << std::endl;
        
//         std::cout << "logAlpha after logCompleteLike: " << logAlpha << std::endl;
        
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        { 
          std::cout << "######################### acceptance #########################" << std::endl;
          theta = thetaProp;
          logCompleteLike = logCompleteLikeProp;
          if (g > mcmc.getNBurninSamples()) { acceptanceRate++; };
        }
      }
    }
    
        std::cout << "theta: " << theta.t() << std::endl;
        
    thetaFull[g] = theta;
    logLike = smc.runCsmc(smc.getNParticles(), theta, latentPath, aux, 1.0);
  }
  
  t2 = clock(); // stop timer 
  cpuTime = (static_cast<double>(t2)-static_cast<double>(t1)) / CLOCKS_PER_SEC; // elapsed time in seconds

}

#endif
