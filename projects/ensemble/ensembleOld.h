/// \file
/// \brief Implements the original embedded HMM/ensemble MCMC methods.
///
/// This file contains the functions associated with the EnsembleOld class which
/// implements the original (pseudo-marginal and pseudo-Gibbs)
/// embedded HMM/ensemble MCMC methods from various
/// papers by Radford M. Neal and Alexander Y. Shestopaloff.

#ifndef __ENSEMBLEOLD_H
#define __ENSEMBLEOLD_H

#include "main/model/Model.h"

// [[Rcpp::depends("RcppArmadillo")]]

/// Type of proposal for the latent variables.
enum EnsembleOldProposalType 
{ 
  ENSEMBLE_OLD_PROPOSAL_DEFAULT = 0, // the ensemble measure is some Gaussian distribution; the MCMC kernel draws IID samples from it
  ENSEMBLE_OLD_PROPOSAL_ALTERNATE, // the ensemble measure is some Gaussian distribution; the MCMC kernel draws IID samples from it
  ENSEMBLE_OLD_PROPOSAL_STATIONARY, // the ensemble measure is the stationary distribution; the MCMC kernel draws IID samples from it
  ENSEMBLE_OLD_PROPOSAL_STATIONARY_MH // the ensemble measure is the stationary distribution; the MCMC kernel is a Gaussian random-walk MH kernel
};

/// Class template for running (conditional) embedded HMM/ensemble MCMC algorithms.
template<class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters> class EnsembleOld
{
public:
  
  /// Initialises the class.
  EnsembleOld
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const unsigned int nSteps,
    const EnsembleOldProposalType prop,
    const arma::colvec& algorithmParameters, 
    const unsigned int nCores
  ) : 
    rng_(rng), 
    model_(model),
    nSteps_(nSteps),
    prop_(prop),
    nCores_(nCores)
  {
    samplePath_ = true; // TODO: make this accessible from the outside
    ensembleOldParameters_.setParameters(algorithmParameters); // determines additional parameters of the algorithm
  }
  
  /// Returns the estimate of the normalising constant.
  double getLoglikelihoodEstimate() const {return logLikelihoodEstimate_;}
  /// Returns the number of SMC steps.
  unsigned int getNSteps() const {return nSteps_;}
  /// Returns the number of particles.
  unsigned int getNParticles() const {return nParticles_;}
  /// Returns the complete set of all parent indices.
//   void getParentIndicesFull(arma::umat& parentIndicesFull) {parentIndicesFull = parentIndicesFull_;}
  /// Returns the complete set of the particle indices associated with the input reference path
  void getParticleIndicesIn(arma::uvec& particleIndicesIn) {particleIndicesIn = particleIndicesIn_;}
  /// Returns the complete set of the particle indices associated with the output reference path
  void getParticleIndicesOut(arma::uvec& particleIndicesOut) {particleIndicesOut = particleIndicesOut_;}
  /// Converts a particle path into the set of all latent variables in the model.
  void convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath);
  /// Converts the set of all latent variables in the model into a particle path.
  void convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath);
  
  /// Runs an SMC algorithm.
  double runSmc
  (
    const unsigned int nParticles,
    const arma::colvec& theta,
    LatentPath& latentPath, 
    const double inverseTemperature
  )
  {
    model_.setUnknownParameters(theta);
    model_.setInverseTemperature(inverseTemperature);
    nParticles_ = nParticles;
    isConditional_ = false;
    
    runEnsembleOldBase();
    if (samplePath_)
    {
      samplePath(latentPath);
    }
    return getLoglikelihoodEstimate();
  }
  /// Runs a conditional SMC algorithm.
  double runCsmc
  (
    const unsigned int nParticles, 
    const arma::colvec& theta,
    LatentPath& latentPath,
    const double inverseTemperature
  )
  {
    model_.setUnknownParameters(theta);
    model_.setInverseTemperature(inverseTemperature);
    nParticles_ = nParticles;
    isConditional_ = true;
    convertLatentPathToParticlePath(latentPath, particlePath_);
    runEnsembleOldBase();
    samplePath(latentPath);
    return getLoglikelihoodEstimate();
  }
  /// Runs a conditional SMC algorithm.
  double runCsmcWithoutPathSampling
  (
    const unsigned int nParticles, 
    const arma::colvec& theta,
    LatentPath& latentPath,
    const double inverseTemperature
  )
  {
    model_.setUnknownParameters(theta);
    model_.setInverseTemperature(inverseTemperature);
    nParticles_ = nParticles;
    isConditional_ = true;
    convertLatentPathToParticlePath(latentPath, particlePath_);
    runEnsembleOldBase();
//     samplePathBase();
//     this->convertParticlePathToLatentPath(particlePath_, latentPath);
    return getLoglikelihoodEstimate();
  }
  /// Selects one particle path.
  void samplePath(LatentPath& latentPath)
  {
    samplePathBase();
    this->convertParticlePathToLatentPath(particlePath_, latentPath);
  }

  
private:
  
  /// Samples a single particle from the ensemble measure rho.
  Particle sampleFromProposal(const unsigned int t);
  /// Applies a Markov kernel which is invariant with respect to the ensemble measure rho.
  Particle applyKernel(const unsigned int t, const Particle& particleOld);
  /// Evaluates the log-proposal density.
  double evaluateLogProposalDensity(const unsigned int t, const Particle& particle);
  /// Runs the algorithm.
  void runEnsembleOldBase();
  /// Samples one particle path from the particle system.
  void samplePathBase();
  
  Rng& rng_; // random number generation.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // the targeted model.
  unsigned int nParticles_; // number of particles.
  unsigned int nSteps_; // number of steps.
  EnsembleOldProposalType prop_; // type of proposal for the latent variables
  bool samplePath_; // sample and store a single particle.
  bool isConditional_; // are we using a conditional algorithm?
  double logLikelihoodEstimate_; // estimate of the normalising constant.
  std::vector<std::vector<Particle>> particlesFull_; // (nSteps_, nParticles_)-dimensional: holds all particles
  arma::mat unnormalisedWeightsFull_; // (nParticles_, nSteps_)-dimensional: holds all unnormalised weight
  std::vector<Particle> particlePath_; // single particle path needed for conditional SMC algorithms
  arma::uvec particleIndicesIn_; // particle indices associated with the single input particle path
  arma::uvec particleIndicesOut_; // particle indices associated with the single output particle path
  EnsembleOldParameters ensembleOldParameters_; // holds some additional auxiliary parameters for the algorithm.
  unsigned int nCores_; // number of cores to use (not currently used)
  
};

/// Runs the SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters>
void EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters>::runEnsembleOldBase()
{
  /////////////////////////////////////////////////////////////////////////////
//   std::cout << "setting up runEnsembleBase()" <<std::endl; /////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  // Step 0 of the algorithm
  ///////////////////////////////////////////////////////////////////////////

  particlesFull_.resize(nSteps_);
  unnormalisedWeightsFull_.zeros(nParticles_, nSteps_);
  
  if (isConditional_) {samplePath_ = true;}
 
  if (samplePath_) // i.e. if we run a conditional SMC algorithm 
  {
    particleIndicesIn_.set_size(nSteps_);
  }
  
  /////////////////////////////////////////////////////////////////////////////
//   std::cout << "Step 0" <<std::endl; /////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  particlesFull_[0].resize(nParticles_);
  
  if (isConditional_) 
  { 
    particleIndicesIn_(0) = arma::as_scalar(arma::randi(1, arma::distr_param(0,nParticles_-1)));
    particlesFull_[0][particleIndicesIn_(0)] = particlePath_[0];
    if (particleIndicesIn_(0) > 0)
    {
      for (unsigned int n=particleIndicesIn_(0)-1; n != static_cast<unsigned>(-1); n--)
      {
        particlesFull_[0][n] = this->applyKernel(0, particlesFull_[0][n+1]);
      }
    }
  }
  else
  {
    particleIndicesIn_(0) = 0;
    particlesFull_[0][particleIndicesIn_(0)] = this->sampleFromProposal(0);
  }
  
  for (unsigned int n=particleIndicesIn_(0)+1; n<nParticles_; n++)
  {
    particlesFull_[0][n] = this->applyKernel(0, particlesFull_[0][n-1]);
  }
  
  // Evaluate the "log-weights"
  for (unsigned int n=0; n<nParticles_; n++)
  {
    unnormalisedWeightsFull_(n,0) = 
      std::exp(model_.evaluateLogInitialDensity(particlesFull_[0][n])
      + model_.evaluateLogObservationDensity(0, particlesFull_[0][n])
      - this->evaluateLogProposalDensity(0, particlesFull_[0][n]) 
      - std::log(nParticles_));
  }

  
  /////////////////////////////////////////////////////////////////////////////
  // Step t, t>0, of the algorithm
  /////////////////////////////////////////////////////////////////////////////
  
  /////////////////////////////////////////////////////////////////////////////
//   std::cout << "step t>0" <<std::endl; /////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  for (unsigned int t=1; t<nSteps_; t++)
  {
    particlesFull_[t].resize(nParticles_);
      
    if (isConditional_) 
    { 
      particleIndicesIn_(t) = arma::as_scalar(arma::randi(1, arma::distr_param(0,nParticles_-1)));
      particlesFull_[t][particleIndicesIn_(t)] = particlePath_[t];
      if (particleIndicesIn_(t) > 0)
      {
        for (unsigned int n=particleIndicesIn_(t)-1; n != static_cast<unsigned>(-1); n--)
        {
          particlesFull_[t][n] = this->applyKernel(t, particlesFull_[t][n+1]);
        }
      }
    }
    else
    {
      particleIndicesIn_(t) = 0;
      particlesFull_[t][particleIndicesIn_(t)] = this->sampleFromProposal(t);
    }
    for (unsigned int n=particleIndicesIn_(t)+1; n<nParticles_; n++)
    {
      particlesFull_[t][n] = this->applyKernel(t, particlesFull_[t][n-1]);
    }
    
    // Evaluate the "log-weights"
    for (unsigned int n=0; n<nParticles_; n++)
    {
      for (unsigned int m=0; m<nParticles_; m++)
      {
        unnormalisedWeightsFull_(n,t) += unnormalisedWeightsFull_(m,t-1) *
          std::exp(model_.evaluateLogTransitionDensity(t, particlesFull_[t][n], particlesFull_[t-1][m]) 
          + model_.evaluateLogObservationDensity(t, particlesFull_[t][n])
          - this->evaluateLogProposalDensity(t, particlesFull_[t][n]) 
          - std::log(nParticles_));
      }
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////
//   std::cout << "calculating logLike" <<std::endl; /////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  // Updating the estimate of the normalising constant:
  logLikelihoodEstimate_ = std::log(arma::sum(unnormalisedWeightsFull_.col(nSteps_-1)));

  
  /////////////////////////////////////////////////////////////////////////////
//   std::cout << "sampling path" <<std::endl; /////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////////////////////////////
  // Sample a single particle path from existing particles
  ///////////////////////////////////////////////////////////////////////////
  
//   arma::colvec logWeightsAux(nParticles_);
//    
//   if (samplePath_)
//   {
//     
// //     std::cout << "ensembleOld started sampling new path" << std::endl;
//     // Sampling a single particle path:
//     particlePath_.resize(nSteps_);
//     particleIndicesOut_.set_size(nSteps_);
//     
// //     std::cout << "final-time weights: " << arma::trans(arma::log(unnormalisedWeightsFull_.col(nSteps_-1))) << std::endl;
//     
//     // Final-time particle:
//     particleIndicesOut_(nSteps_-1) = sampleInt(normaliseWeights(arma::log(unnormalisedWeightsFull_.col(nSteps_-1))));
//     particlePath_[nSteps_-1]       = particlesFull_[nSteps_-1][particleIndicesOut_(nSteps_-1)];
//     
//     // Recursion for the particles at previous time steps:
//     for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
//     { 
// //       std::cout << "ensembleOld backward sampling at time " << t << std::endl;
//       
//       for (unsigned int n=0; n<nParticles_; n++)
//       {
//         logWeightsAux(n)  = std::log(unnormalisedWeightsFull_(n,t)) + 
//                             model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particlesFull_[t][n]);
//                             /// NOTE: for more general models than state-space models this needs to be modified!
//       }
//       particleIndicesOut_(t) = sampleInt(normaliseWeights(logWeightsAux));
//       particlePath_[t]    = particlesFull_[t][particleIndicesOut_(t)];
//     }
//     
// //     std::cout << "ensembleOld finished sampling new path" << std::endl;
//   }
  
}




/// Samples one particle path from the particle system.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters>
void EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters>::samplePathBase()
{
  arma::colvec logWeightsAux(nParticles_);

//     std::cout << "ensembleOld started sampling new path" << std::endl;
  // Sampling a single particle path:
  particlePath_.resize(nSteps_);
  particleIndicesOut_.set_size(nSteps_);
  
//     std::cout << "final-time weights: " << arma::trans(arma::log(unnormalisedWeightsFull_.col(nSteps_-1))) << std::endl;
  
  // Final-time particle:
  particleIndicesOut_(nSteps_-1) = sampleInt(normaliseWeights(arma::log(unnormalisedWeightsFull_.col(nSteps_-1))));
  particlePath_[nSteps_-1]       = particlesFull_[nSteps_-1][particleIndicesOut_(nSteps_-1)];
  
  // Recursion for the particles at previous time steps:
  for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
  { 
//       std::cout << "ensembleOld backward sampling at time " << t << std::endl;
    
    for (unsigned int n=0; n<nParticles_; n++)
    {
      logWeightsAux(n)  = std::log(unnormalisedWeightsFull_(n,t)) + 
                          model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particlesFull_[t][n]);
                          /// NOTE: for more general models than state-space models this needs to be modified!
    }
    particleIndicesOut_(t) = sampleInt(normaliseWeights(logWeightsAux));
    particlePath_[t]    = particlesFull_[t][particleIndicesOut_(t)];
  }
  
//     std::cout << "ensembleOld finished sampling new path" << std::endl;

}
#endif
