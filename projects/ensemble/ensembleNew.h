/// \file
/// \brief Implements the original embedded HMM/ensemble MCMC methods.
///
/// This file contains the functions associated with the EnsembleOld class which
/// implements the newer type of (pseudo-Gibbs)
/// embedded HMM/ensemble MCMC method from 
/// Alexander Y. Shestopaloff & Radford M. Neal (2016)
/// as well as novel auxiliary-particle filter-type extensions
/// and pseudo-marginal versions.

#ifndef __ENSEMBLENEW_H
#define __ENSEMBLENEW_H

#include "main/model/Model.h"
#include "main/algorithms/smc/Smc.h"

/// Type of proposal for the latent variables.
enum EnsembleNewProposalType 
{ 
  ENSEMBLE_NEW_PROPOSAL_PRIOR = 0, 
  ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL,
  ENSEMBLE_NEW_PROPOSAL_FA_APF,
  ENSEMBLE_NEW_PROPOSAL_APF
};


/// Type of proposal for the latent variables.
enum EnsembleNewInitialisationType 
{ 
  ENSEMBLE_NEW_INITIALISATION_STATIONARY = 0, 
  ENSEMBLE_NEW_INITIALISATION_BURNIN
};

/// Type of proposal for the parent indices.
enum EnsembleNewLocalType 
{ 
  ENSEMBLE_NEW_LOCAL_NONE = 0, // samples (x_t^n, a_{t-1}^n) conditionally IID from rho_t (this coincides with a simple SMC algorithm with multinomial resampling)
  ENSEMBLE_NEW_LOCAL_RANDOM_WALK, // global proposal for the parent indices, i.e. sampled from marginal under rho; shift update for the state
  ENSEMBLE_NEW_LOCAL_AUTOREGRESSIVE, // global proposal for the parent indices, i.e. sampled from marginal under rho; autoregressive update for the state
  ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK, // autoregressive  proposal for parent index after Hilbert sort
  ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE // random-walk proposal for parent index after Hilbert sort
};

/// Class template for running (conditional) embedded HMM/ensemble MCMC algorithms.
template<class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> class EnsembleNew
{
public:
  
  /// Initialises the class.
  EnsembleNew
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const unsigned int nSteps,
    const EnsembleNewProposalType prop,
    const EnsembleNewLocalType local,
    const arma::colvec& algorithmParameters, 
    const SmcBackwardSamplingType csmc,
    const unsigned int nCores
  ) : 
    rng_(rng), 
    model_(model),
    nSteps_(nSteps),
    prop_(prop), 
    local_(local),
    csmc_(csmc),
    nCores_(nCores)
  {
    samplePath_ = true; // TODO: make this accessible from the outside
    ensembleNewParameters_.setParameters(algorithmParameters); // determines additional parameters of the algorithm 
    init_ = ENSEMBLE_NEW_INITIALISATION_STATIONARY;
    nBurninSamples_ = 0;
  }
  
  /// Returns the estimate of the normalising constant.
  double getLoglikelihoodEstimate() const {return logLikelihoodEstimate_;}
  /// Returns the number of SMC steps.
  unsigned int getNSteps() const {return nSteps_;}
  /// Returns the number of particles.
  unsigned int getNParticles() const {return nParticles_;}
  /// Returns the effective sample sizes for each time step.
  arma::colvec getEss() const {return ess_;}
  /// Returns the acceptance rates for each time step.
  arma::colvec getAcceptanceRates() const {return acceptanceRates_;}
  /// Returns the complete set of all parent indices.
  void getParentIndicesFull(arma::umat& parentIndicesFull) {parentIndicesFull = parentIndicesFull_;}
  /// Returns the complete set of the particle indices associated with the input reference path
  void getParticleIndicesIn(arma::uvec& particleIndicesIn) {particleIndicesIn = particleIndicesIn_;}
  /// Returns the complete set of the particle indices associated with the output reference path
  void getParticleIndicesOut(arma::uvec& particleIndicesOut) {particleIndicesOut = particleIndicesOut_;}
  /// Specifies the number of burn-in samples.
  void setNBurninSamples(const unsigned int nBurninSamples) {nBurninSamples_ = nBurninSamples;}
  /// Specifies the method for initialising the MCMC chains at each time step.
  void setEnsembleNewInitialisationType(const EnsembleNewInitialisationType& init) {init_ = init;}
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
    
    runEnsembleNewBase();
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
    runEnsembleNewBase();
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
    runEnsembleNewBase();
//     samplePath(latentPath);
    return getLoglikelihoodEstimate();
  }
  /// Selects one particle path.
  void samplePath(LatentPath& latentPath)
  {
    samplePathBase();
    this->convertParticlePathToLatentPath(particlePath_, latentPath);
  }

  
private:
  
  /// Samples a discrete random variable needed for proposing a parent index.
  unsigned int proposeIndex(const unsigned int indexOld)
  {
    unsigned int newIndex;
    if (indexOld +1 < nParticles_ && indexOld > 0)
    {
      newIndex = arma::as_scalar(arma::randi(1, arma::distr_param(static_cast<int>(indexOld-1),static_cast<int>(indexOld+1))));
    }
    else if (indexOld == nParticles_ - 1)
    {
      newIndex = arma::as_scalar(arma::randi(1, arma::distr_param(static_cast<int>(indexOld-1),static_cast<int>(indexOld))));
    }
    else // i.e. if indexOld == 0
    {
      newIndex = arma::as_scalar(arma::randi(1, arma::distr_param(static_cast<int>(indexOld),static_cast<int>(indexOld+1))));
    }
    return newIndex;
  }
  /// Evaluate log-density of proposal kernel for a discrete random variable needed for proposing a parent index.
  double evaluateLogProposalDensityIndex(const unsigned int indexNew, const unsigned int indexOld)
  { 
    double logDensity;
    if (indexOld +1 < nParticles_ && indexOld > 0)
    {
      logDensity = - std::log(3.0);
    }
    else // i.e. if indexOld == 0
    {
      logDensity = - std::log(2.0);
    }
    return logDensity;
  }
  /// Samples a particle from some proposal kernel.
  Particle proposeParticle(const Particle particleOld);
  /// Evaluate log-density of proposal kernel for particle.
  double evaluateLogProposalDensityParticle(const Particle particleNew, const Particle particleOld);
  
  /// Samples from rho_t.
  void sampleFromRho(const unsigned int t, const unsigned int n, std::vector<Particle>& particlesNew, arma::uvec& particleIndices, const arma::colvec& selfNormalisedWeights);
  /// Samples from rho_0.
  void sampleFromInitialRho(const unsigned int n, std::vector<Particle>& particles);
  
  /// Samples from approximation of rho_t.
  void sampleFromRhoApproximation(const unsigned int t, const unsigned int n, std::vector<Particle>& particlesNew, arma::uvec& particleIndices, const arma::colvec& potentialProposalValues);
  /// Samples from approximation of rho_0.
  void sampleFromInitialRhoApproximation(const unsigned int n, std::vector<Particle>& particles);
  
  // Evaluates the unnormalised(!) log-density of rho_t.
  double evaluateLogDensityRho(const unsigned int t, const Particle& particle, const unsigned int parentIndex, const arma::colvec& logUnnormalisedWeights);
  /// Evaluates the unnormalised(!) log-density of rho_1.
  double evaluateLogDensityInitialRho(const Particle& particle);
  
  /// Applies rho_t-invariant kernel
  void applyKernel(const unsigned int t, const unsigned int n, const unsigned int m, std::vector<Particle>& particles, arma::uvec& parentIndices, const arma::colvec& logUnnormalisedWeights, const arma::colvec& selfNormalisedWeights, const arma::colvec& potentialProposalValues);
  /// Applies rho_0-invariant kernel
  void applyInitialKernel(const unsigned int n, const unsigned int m, std::vector<Particle>& particles);

  /// Calculates log-unnormalised particle weights at the first SMC Step.
  void computeLogInitialParticleWeights(const std::vector<Particle>& particles, arma::colvec& logUnnormalisedWeights);
  /// Calculates the log-unnormalised particle weights at later SMC steps.
  void computeLogParticleWeights
  (
    const unsigned int t, 
    const std::vector<Particle>& particlesNew, 
    arma::colvec& logUnnormalisedWeights
  );
  /// Calculates Gaussian realisation from a single particle at the first SMC step.
  void determineGaussiansFromInitialParticles(const std::vector<Particle>& particles, std::vector<Aux>& aux1);
  /// Calculates Gaussian realisation from a single particle at some later SMC step.
  void determineGaussiansFromParticles
  (
    const unsigned int t, 
    const std::vector<Particle>& particlesNew, 
    std::vector<Aux>& aux1
  );
  
  // Samples a single particle index via backward sampling.
  unsigned int backwardSampling
  (
    const unsigned int t,
    const arma::colvec& logUnnormalisedWeights,
    const std::vector<Particle>& particles
  );
  /// Computes (part of the) unnormalised "future" target density needed for backward
  /// or ancestor sampling.
  double logDensityUnnormalisedTarget(const unsigned int t, const Particle& particle);
  /// Runs the SMC algorithm.
  void runEnsembleNewBase();
  /// Samples one particle path from the particle system.
  void samplePathBase();

  Rng& rng_; // random number generation.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // the targeted model.
  unsigned int nParticles_; // number of particles.
  unsigned int nBurninSamples_; // number of samples discarded as burn-in if MCMC chains are not initialised from stationarity at each time step.
  unsigned int nSteps_; // number of SMC steps.
  EnsembleNewInitialisationType init_; // proposal (kernel) for the latent states.
  EnsembleNewProposalType prop_; // proposal (kernel) for the latent states.
  EnsembleNewLocalType local_; // proposal (kernel) for the latent states.
  bool samplePath_; // sample and store a single particle.
  SmcBackwardSamplingType csmc_; // type of backward sampling to use within Csmc.
  bool isConditional_; // are we using a conditional SMC algorithm?
  double logLikelihoodEstimate_; // estimate of the normalising constant.
  std::vector<std::vector<Particle>> particlesFull_; // (nSteps_, nParticles_)-dimensional: holds all particles
  std::vector<Particle> particlePath_; // single particle path needed for conditional SMC algorithms
  arma::uvec particleIndicesIn_; // particle indices associated with the single input particle path
  arma::uvec particleIndicesOut_; // particle indices associated with the single input particle path
  arma::umat parentIndicesFull_; // (nParticles_, nSteps_)-dimensional: holds all parent indices
  arma::mat logUnnormalisedWeightsFull_; // (nParticles_, nSteps_)-dimensional: holds all log-unnormalised weight
  EnsembleNewParameters ensembleNewParameters_; // holds some additional auxiliary parameters for the SMC algorithm.
  arma::colvec selfNormalisedWeights_; // self-normalised weights 
  arma::uvec sortedIndices_; // sorted particle indices generated by Hilbert sort
  unsigned int sortedParentIndex_, sortedParentIndexTemp_; // post sorted index of the particle of the distinguished path
  arma::colvec ess_; // effective sample size at each time step.
  arma::colvec acceptanceRates_; // acceptanceRates at each time step.
  unsigned int nCores_; // number of cores to use (not currently used)
  
};

/// Runs the SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters>
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::runEnsembleNewBase()
{

  
//       std::cout << "start runEnsembleNewBase" << std::endl;
  ess_.zeros(nSteps_);
  acceptanceRates_.zeros(nSteps_);
    
  arma::uvec parentIndices(nParticles_); // parent indices associated with a single SMC step
  std::vector<Particle> particlesNew(nParticles_); // particles from previous step
  
  std::vector<Aux> aux(nParticles_); // holds auxiliary multivariate Gaussian random variables
  
//   unsigned int singleParentIndex = 0; // parent index for a single particle
//   unsigned int singleParticleIndex = 0; // particle index for a single particle 
  
  arma::colvec logUnnormalisedWeights(nParticles_); // unnormalised log-weights associated with a single SMC step
  logUnnormalisedWeights.fill(-std::log(nParticles_)); // start with uniform weights
  
  arma::colvec logWeightsAux(nParticles_);
  
  arma::colvec potentialProposalValues(nParticles_);
    
  
  sortedIndices_.set_size(nParticles_); // sorted particle indices generated by Hilbert sort
  
  logLikelihoodEstimate_ = 0; // log of the estimated marginal likelihood   
  
  ///////////////////////////////////////////////////////////////////////////
  // Step 0 of the SMC algorithm
  ///////////////////////////////////////////////////////////////////////////
  
  if (isConditional_) {samplePath_ = true;}
  
  if (samplePath_) // i.e. if we run a conditional SMC algorithm 
  {
    particleIndicesIn_.set_size(nSteps_);
    particleIndicesIn_(0) = arma::as_scalar(arma::randi(1, arma::distr_param(0,static_cast<int>(nParticles_-1))));
  }
  
  
//     std::cout << "apply initial kernel" << std::endl;
  
  if (isConditional_) 
  { 
//     std::cout << "sampling b_1" << std::endl;
    particleIndicesIn_(0) = arma::as_scalar(arma::randi(1, arma::distr_param(0,static_cast<int>(nParticles_-1))));
//     std::cout << particleIndicesIn_(0) << std::endl;
    particlesNew[particleIndicesIn_(0)] = particlePath_[0];
    
//     std::cout << "applying initial kernel" << std::endl;
    if (particleIndicesIn_(0) > 0)
    {
      for (unsigned int n=particleIndicesIn_(0)-1; n != static_cast<unsigned>(-1); n--)
      {
        this->applyInitialKernel(n, n+1, particlesNew);
      }
    }
  }
  else
  {
    particleIndicesIn_(0) = 0;
    
    if (init_ == ENSEMBLE_NEW_INITIALISATION_STATIONARY)
    {
      this->sampleFromInitialRho(0, particlesNew);
    }
    else if (init_ == ENSEMBLE_NEW_INITIALISATION_BURNIN)
    {
      std::vector<Particle> particlesBurnin(nBurninSamples_);
      this->sampleFromInitialRhoApproximation(0, particlesBurnin);
      for (unsigned int n=1; n<nBurninSamples_; n++)
      {
        this->applyInitialKernel(n, n-1, particlesBurnin);
      }
      particlesNew[0]  = particlesBurnin[nBurninSamples_-1]; 
    }
  
  }
  for (unsigned int n=particleIndicesIn_(0)+1; n<nParticles_; n++)
  {
    this->applyInitialKernel(n, n-1, particlesNew);
  }
  
  
  particlesFull_.resize(nSteps_);
  particlesFull_[0] = particlesNew;
  parentIndicesFull_.set_size(nParticles_, nSteps_-1);
  
//   std::cout << "compute initial weights" << std::endl;
  
  computeLogInitialParticleWeights(particlesNew, logUnnormalisedWeights);
  
//      std::cout << "======================" << (local_ == ENSEMBLE_NEW_LOCAL_HILBERT) << std::endl;
  if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK)
  {
    determineGaussiansFromInitialParticles(particlesNew, aux);
//     /////////////////////////////////////////// only for debugging
//     std::cout << "stated outputting aux:" << std::endl;
//     for (unsigned int n=0; n<nParticles_; n++)
//     {
//       std::cout << aux[n].t() << " ";
//     }
//     std::cout << std::endl;
//      std::cout << "finished outputting aux" << std::endl;
    ///////////////////////////////////////////
  }
//     std::cout << "======================" << (local_ == ENSEMBLE_NEW_LOCAL_HILBERT) << std::endl;
  

  logUnnormalisedWeightsFull_.set_size(nParticles_, nSteps_);
  logUnnormalisedWeightsFull_.col(0) = logUnnormalisedWeights;
  
  /////////////////////////////////////////////////////////////////////////////
  // Step t, t>0, of the SMC algorithm
  /////////////////////////////////////////////////////////////////////////////
  
  for (unsigned int t=1; t<nSteps_; t++)
  {
//     std::cout << "ENSEMBLE NEW STEP " << t << std::endl; 
    /////////////////////
    parentIndices.zeros();
    //////////////////////
    logLikelihoodEstimate_ += std::log(arma::sum(arma::exp(logUnnormalisedWeights))); 
    
    ///////////////////////////////////////////////////////////////////////////
    // Hilbert sort of parent particles for more efficient local moves
    ///////////////////////////////////////////////////////////////////////////
    if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK)
    {
      // TODO: we need to use the logistic transform here:
      //resample::hilbertSortBase(convertGaussianToUniform(aux), sortedIndices_);
//       resample::hilbertSort(particlesFull_[t-1], sortedIndices_, -3.0, 3.0); // WARNING: hilbert resampling is currently disabled!
//       std::cout << "sorted Indices_: " << sortedIndices_.t() << std::endl;
    }
    selfNormalisedWeights_ = normaliseWeights(logUnnormalisedWeights);

    if (prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
    {
      potentialProposalValues.fill(1.0/nParticles_);
    }
    else
    {
      potentialProposalValues = selfNormalisedWeights_;
    }
    
//     std::cout << "ESS at time " << t << std::endl;
    
    ess_(t-1) = 1.0 / arma::dot(selfNormalisedWeights_, selfNormalisedWeights_) / nParticles_;
    
        
    if (!arma::is_finite(selfNormalisedWeights_))
    {
      std::cout << "WARNING: W contains NaNs! at step " << t << std::endl;
    }
    
//     std::cout << "start apply kernel" << std::endl; 
    
    if (isConditional_) 
    { 
      particleIndicesIn_(t) = arma::as_scalar(arma::randi(1, arma::distr_param(0,static_cast<int>(nParticles_-1))));
      
      /////////////////////////////////////////////////////////////////////////
      // Ancestor sampling
      /////////////////////////////////////////////////////////////////////////
      
      // Determining the parent index of the current input particle:
      if (csmc_ == SMC_BACKWARD_SAMPLING_ANCESTOR) // via ancestor sampling
      {
        /*
        for (unsigned int n=0; n<nParticles_; n++)
        {
          logWeightsAux(n)  = logUnnormalisedWeights(n) + 
                              model_.evaluateLogTransitionDensity(t, particlePath_[t], particlesFull_[t-1][n]);
                              /// NOTE: for more general models than state-space models this needs to be modified!
        }
        parentIndices(particleIndicesIn_(t)) = sampleInt(normaliseWeights(logWeightsAux));
        */
        
//                std::cout << "start AS" << std::endl;
        parentIndices(particleIndicesIn_(t)) = backwardSampling(t-1, logUnnormalisedWeights, particlesFull_[t-1]);
//          std::cout << "finished AS" << std::endl;
        
//         std::cout << "a_{t-1}^{b_t}: " << parentIndices(particleIndicesIn_(t)) << "; b_{t-1}: " << particleIndicesIn_(t-1) <<  std::endl;
                
      }
      else // not via ancestor sampling
      {
        parentIndices(particleIndicesIn_(t)) = particleIndicesIn_(t-1);
      }
      
      if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK)
      {
        sortedParentIndex_ = arma::as_scalar(arma::find(sortedIndices_ == parentIndices(particleIndicesIn_(t)), 1, "first"));
        sortedParentIndexTemp_ = sortedParentIndex_;
      }

      particlesNew[particleIndicesIn_(t)] = particlePath_[t];
      
      if (particleIndicesIn_(t) > 0)
      {
        for (unsigned int n=particleIndicesIn_(t)-1; n != static_cast<unsigned>(-1); n--)
        {
          this->applyKernel(t, n, n+1, particlesNew, parentIndices, logUnnormalisedWeights, selfNormalisedWeights_, potentialProposalValues);
//           sampleFromRho(t, n, particlesNew, parentIndices, selfNormalisedWeights_);
        }
      }
      if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK) {sortedParentIndexTemp_ = sortedParentIndex_;}
    }
    else
    {
      particleIndicesIn_(t) = 0;
      
      if (init_ == ENSEMBLE_NEW_INITIALISATION_STATIONARY)
      {
        this->sampleFromRho(t, 0, particlesNew, parentIndices, selfNormalisedWeights_);
      }
      else if (init_ == ENSEMBLE_NEW_INITIALISATION_BURNIN)
      {
        arma::uvec parentIndicesBurnin(nBurninSamples_);
        std::vector<Particle> particlesBurnin(nBurninSamples_);
        this->sampleFromRhoApproximation(t, 0, particlesBurnin, parentIndicesBurnin, potentialProposalValues);
        for (unsigned int n=1; n<nBurninSamples_; n++)
        {
          this->applyKernel(t, n, n-1, particlesBurnin, parentIndicesBurnin, logUnnormalisedWeights, selfNormalisedWeights_,  potentialProposalValues);
        }
        particlesNew[0]  = particlesBurnin[nBurninSamples_-1];
        parentIndices(0) = parentIndicesBurnin(nBurninSamples_-1);
        
      }
      if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK)
      {
        sortedParentIndex_ = arma::as_scalar(arma::find(sortedIndices_ == parentIndices(particleIndicesIn_(t)), 1, "first"));
        sortedParentIndexTemp_ = sortedParentIndex_;
      }
      
    }
    
//         std::cout << "finished apply kernel 1" << std::endl; 
    
    if (particleIndicesIn_(t)+1 < nParticles_)
    {
      for (unsigned int n=particleIndicesIn_(t)+1; n<nParticles_; n++)
      {
        this->applyKernel(t, n, n-1, particlesNew, parentIndices, logUnnormalisedWeights, selfNormalisedWeights_, potentialProposalValues);
      }
    }
    
//        std::cout << "finished apply kernel 2" << std::endl; 
       
//      std::cout << "parent indices after SECOND kernel" << std::endl;
//       std::cout << parentIndices.t() << std::endl;
//            std::cout << "finished applyKernel()" << std::endl;
    
//         std::cout << "finished applyKernel" << std::endl;
    
    
    // Storing the entire particle system:
    particlesFull_[t] = particlesNew; 
    parentIndicesFull_.col(t-1) = parentIndices;
    
//     std::cout << parentIndices.t() << std::endl;
    
//       std::cout << "======================" << (local_ == ENSEMBLE_NEW_LOCAL_HILBERT) << std::endl;
//       std::cout << "started determine Gaussian from particles" << std::endl;
    if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK)
    {
      determineGaussiansFromParticles(t, particlesNew, aux);
    }
    
//           std::cout << "finished determine Gaussian from particles" << std::endl;
//             std::cout << "======================" << (local_ == ENSEMBLE_NEW_LOCAL_HILBERT) << std::endl;

//    std::cout << "start compute weights" << std::endl; 
    logUnnormalisedWeights.fill(-std::log(nParticles_)); // resetting the weights
    computeLogParticleWeights(t, particlesNew, logUnnormalisedWeights);
    logUnnormalisedWeightsFull_.col(t) = logUnnormalisedWeights;
    
//        std::cout << "finished compute weights" << std::endl; 
    
//     std::cout << "logUnnormalisedWeights: " << logUnnormalisedWeights.t() << std::endl;

  }
  
  // Updating the estimate of the normalising constant:
  logLikelihoodEstimate_ += std::log(arma::sum(arma::exp(logUnnormalisedWeights)));
  selfNormalisedWeights_ = normaliseWeights(logUnnormalisedWeights);
  

  
  ess_(nSteps_-1) = 1.0 / arma::dot(selfNormalisedWeights_, selfNormalisedWeights_) / nParticles_;

  
  ///////////////////////////////////////////////////////////////////////////
  // Sample a single particle path from existing particles
  ///////////////////////////////////////////////////////////////////////////
  
//         std::cout << "sample new path backwards" << std::endl;
        
//   if (samplePath_)
//   {
// //     std::cout << "ensembleNew started sampling new path" << std::endl;
//         
//     // Sampling a single particle path:
//     particlePath_.resize(nSteps_);
//     particleIndicesOut_.set_size(nSteps_);
//     
// //         std::cout << "final-time weights: " << arma::trans(logUnnormalisedWeights) << std::endl;
//     
//     // Final-time particle:
//     particleIndicesOut_(nSteps_-1) = sampleInt(normaliseWeights(logUnnormalisedWeights));
//     
// //      std::cout << "final-time output particle index: " << particleIndicesOut_(nSteps_-1) << std::endl;
//     
//     
//     particlePath_[nSteps_-1] = particlesFull_[nSteps_-1][particleIndicesOut_(nSteps_-1)];
//     
// //        std::cout << "final-time output particle: " << particlePath_[nSteps_-1] << std::endl;
//     
//     
//     // Recursion for the particles at previous time steps:
//     if (isConditional_ && csmc_ == SMC_BACKWARD_SAMPLING_STANDARD)
//     { // i.e. we employ the usual backward-sampling recursion
//       for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
//       { 
// //         std::cout << "ensembleNew backward sampling at time " << t << std::endl;
//             
//         particleIndicesOut_(t) = backwardSampling(t, logUnnormalisedWeightsFull_.col(t), particlesFull_[t]);
//         /*
//         for (unsigned int n=0; n<nParticles_; n++)
//         {
//           logWeightsAux(n)  = logUnnormalisedWeightsFull_(n,t) + 
//                               model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particlesFull_[t][n]);
//                               /// NOTE: for more general models than state-space models this needs to be modified!
//         }
//         particleIndicesOut_(t) = sampleInt(normaliseWeights(logWeightsAux));
//         */
//         particlePath_[t] = particlesFull_[t][particleIndicesOut_(t)];
//       }
//     }
//     else // i.e we just trace back the ancestral lineage
//     {             
//       for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
//       { 
//         particleIndicesOut_(t) = parentIndicesFull_(particleIndicesOut_(t+1), t);
//         particlePath_[t] = particlesFull_[t][particleIndicesOut_(t)];
//       }
//     }
//     
// //     std::cout << "ensembleNew finished sampling new path" << std::endl;
//   }
}

/// Samples a single particle index via backward sampling.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
unsigned int EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::backwardSampling
(
  const unsigned int t,
  const arma::colvec& logUnnormalisedWeights,
  const std::vector<Particle>& particles
)
{
  arma::colvec WAux(nParticles_);
  for (unsigned int n=0; n<nParticles_; n++)
  {
    WAux(n) = logUnnormalisedWeights(n) + logDensityUnnormalisedTarget(t, particles[n]);
  }
  normaliseWeightsInplace(WAux);
  if (!arma::is_finite(WAux))
  {
    std::cout << "WARNING: WAux contains NaNs!" << std::endl;
  }
  return sampleInt(WAux);
}


/// Samples one particle path from the particle system.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::samplePathBase()
{
//     std::cout << "ensembleNew started sampling new path" << std::endl;
      
  // Sampling a single particle path:
  particlePath_.resize(nSteps_);
  particleIndicesOut_.set_size(nSteps_);
  
//         std::cout << "final-time weights: " << arma::trans(logUnnormalisedWeights) << std::endl;
  
  // Final-time particle:
  particleIndicesOut_(nSteps_-1) = sampleInt(normaliseWeights(logUnnormalisedWeightsFull_.col(nSteps_-1)));
  
//      std::cout << "final-time output particle index: " << particleIndicesOut_(nSteps_-1) << std::endl;
  
  
  particlePath_[nSteps_-1] = particlesFull_[nSteps_-1][particleIndicesOut_(nSteps_-1)];
  
//        std::cout << "final-time output particle: " << particlePath_[nSteps_-1] << std::endl;
  
  
  // Recursion for the particles at previous time steps:
  if (isConditional_ && csmc_ == SMC_BACKWARD_SAMPLING_STANDARD)
  { // i.e. we employ the usual backward-sampling recursion
    for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
    { 
//         std::cout << "ensembleNew backward sampling at time " << t << std::endl;
          
      particleIndicesOut_(t) = backwardSampling(t, logUnnormalisedWeightsFull_.col(t), particlesFull_[t]);
      /*
      for (unsigned int n=0; n<nParticles_; n++)
      {
        logWeightsAux(n)  = logUnnormalisedWeightsFull_(n,t) + 
                            model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particlesFull_[t][n]);
                            /// NOTE: for more general models than state-space models this needs to be modified!
      }
      particleIndicesOut_(t) = sampleInt(normaliseWeights(logWeightsAux));
      */
      particlePath_[t] = particlesFull_[t][particleIndicesOut_(t)];
    }
  }
  else // i.e we just trace back the ancestral lineage
  {             
    for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
    { 
      particleIndicesOut_(t) = parentIndicesFull_(particleIndicesOut_(t+1), t);
      particlePath_[t] = particlesFull_[t][particleIndicesOut_(t)];
    }
  }
  
//     std::cout << "ensembleNew finished sampling new path" << std::endl;
}
#endif
