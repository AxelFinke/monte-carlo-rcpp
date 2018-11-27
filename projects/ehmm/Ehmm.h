/// \file
/// \brief Implements embedded HMM methods.
///
/// This file contains the functions associated with the Ehmm class which
/// implements the newer type of (pseudo-Gibbs)
/// embedded HMM/ensemble MCMC method from 
/// Alexander Y. Shestopaloff & Radford M. Neal (2016)
/// as well as novel auxiliary-particle filter-type extensions
/// and pseudo-marginal versions.

#ifndef __EHMM_H
#define __EHMM_H

#include "main/model/Model.h"
#include "main/algorithms/smc/Smc.h"

/// Type of proposal for the latent variables.
enum EhmmProposalType 
{ 
  EHMM_PROPOSAL_PRIOR = 0, 
  EHMM_PROPOSAL_FA_APF
};

/// Type of proposal for the parent indices.
enum EhmmLocalType 
{ 
  EHMM_LOCAL_NONE = 0, // samples (x_t^n, a_{t-1}^n) conditionally IID from rho_t (this coincides with a simple SMC algorithm with multinomial resampling)
  EHMM_LOCAL_RANDOM_WALK, // global proposal for the parent indices, i.e. sampled from marginal under rho; shift update for the state
  EHMM_LOCAL_AUTOREGRESSIVE, // global proposal for the parent indices, i.e. sampled from marginal under rho; autoregressive update for the state
  EHMM_LOCAL_HILBERT_RANDOM_WALK, // autoregressive  proposal for parent index after Hilbert sort
  EHMM_LOCAL_HILBERT_AUTOREGRESSIVE // random-walk proposal for parent index after Hilbert sort
};

/// Class template for running (conditional) embedded HMM/ensemble MCMC algorithms.
template<class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EhmmParameters> class Ehmm
{
public:
  
  /// Initialises the class.
  Ehmm
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const unsigned int nSteps,
    const EhmmProposalType prop,
    const EhmmLocalType local,
    const arma::colvec& algorithmParameters, 
    const CsmcType csmc,
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
    ehmmParameters_.setParameters(algorithmParameters); // determines additional parameters of the algorithm 
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
    
    runEhmmBase();
    if (samplePath_)
    {
      this->convertParticlePathToLatentPath(particlePath_, latentPath);
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
    runEhmmBase();
    this->convertParticlePathToLatentPath(particlePath_, latentPath);
    return getLoglikelihoodEstimate();
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
  void sampleFromRho(const unsigned int t, const unsigned int n, std::vector<Particle>& particlesNew, arma::uvec& particleIndices, const arma::colvec& logUnnormalisedWeights);
  /// Samples from rho_0.
  void sampleFromInitialRho(const unsigned int n, std::vector<Particle>& particles);
  
  // Evaluates the unnormalised(!) log-density of rho_t.
  double evaluateLogDensityRho(const unsigned int t, const Particle& particle, const unsigned int parentIndex, const arma::colvec& logUnnormalisedWeights);
  /// Evaluates the unnormalised(!) log-density of rho_1.
  double evaluateLogDensityInitialRho(const Particle& particle);
  
  /// Applies rho_t-invariant kernel
  void applyKernel(const unsigned int t, const unsigned int n, const unsigned int m, std::vector<Particle>& particles, arma::uvec& parentIndices, const arma::colvec& logUnnormalisedWeights, const arma::colvec& selfNormalisedWeights);
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
  void runEhmmBase();

  Rng& rng_; // random number generation.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // the targeted model.
  unsigned int nParticles_; // number of particles.
  unsigned int nSteps_; // number of SMC steps.
  EhmmProposalType prop_; // proposal (kernel) for the latent states.
  EhmmLocalType local_; // proposal (kernel) for the latent states.
  bool samplePath_; // sample and store a single particle.
  CsmcType csmc_; // type of backward sampling to use within Csmc.
  bool isConditional_; // are we using a conditional SMC algorithm?
  double logLikelihoodEstimate_; // estimate of the normalising constant.
  std::vector<std::vector<Particle>> particlesFull_; // (nSteps_, nParticles_)-dimensional: holds all particles
  std::vector<Particle> particlePath_; // single particle path needed for conditional SMC algorithms
  arma::uvec particleIndices_; // particle indices associated with the single particle path
  arma::umat parentIndicesFull_; // (nParticles_, nSteps_)-dimensional: holds all parent indices
  arma::mat logUnnormalisedWeightsFull_; // (nParticles_, nSteps_)-dimensional: holds all log-unnormalised weight
  EhmmParameters ehmmParameters_; // holds some additional auxiliary parameters for the SMC algorithm.
  arma::colvec selfNormalisedWeights_; // self-normalised weights 
  arma::uvec sortedIndices_; // sorted particle indices generated by Hilbert sort
  unsigned int sortedParentIndex_, sortedParentIndexTemp_; // post sorted index of the particle of the distinguished path
  arma::colvec ess_; // effective sample size at each time step.
  arma::colvec acceptanceRates_; // acceptanceRates at each time step.
  unsigned int nCores_; // number of cores to use (not currently used)
  
};

/// Runs the SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EhmmParameters>
void Ehmm<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EhmmParameters>::runEhmmBase()
{

  
//       std::cout << "start runEhmmBase" << std::endl;
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
  
  sortedIndices_.set_size(nParticles_); // sorted particle indices generated by Hilbert sort
  
  logLikelihoodEstimate_ = 0; // log of the estimated marginal likelihood   
  
  ///////////////////////////////////////////////////////////////////////////
  // Step 0 of the SMC algorithm
  ///////////////////////////////////////////////////////////////////////////
  
  if (isConditional_) {samplePath_ = true;}
  
  if (samplePath_) // i.e. if we run a conditional SMC algorithm 
  {
    particleIndices_.set_size(nSteps_);
    particleIndices_(0) = arma::as_scalar(arma::randi(1, arma::distr_param(0,static_cast<int>(nParticles_-1))));
  }
  
  
//     std::cout << "apply initial kernel" << std::endl;
  
  if (isConditional_) 
  { 
//     std::cout << "sampling b_1" << std::endl;
    particleIndices_(0) = arma::as_scalar(arma::randi(1, arma::distr_param(0,static_cast<int>(nParticles_-1))));
//     std::cout << particleIndices_(0) << std::endl;
    particlesNew[particleIndices_(0)] = particlePath_[0];
    
//     std::cout << "applying initial kernel" << std::endl;
    if (particleIndices_(0) > 0)
    {
      for (unsigned int n=particleIndices_(0)-1; n != static_cast<unsigned>(-1); n--)
      {
        this->applyInitialKernel(n, n+1, particlesNew);
      }
    }
  }
  else
  {
    particleIndices_(0) = 0;
    this->sampleFromInitialRho(0, particlesNew);
  }
  for (unsigned int n=particleIndices_(0)+1; n<nParticles_; n++)
  {
    this->applyInitialKernel(n, n-1, particlesNew);
  }
  
  
  particlesFull_.resize(nSteps_);
  particlesFull_[0] = particlesNew;
  parentIndicesFull_.set_size(nParticles_, nSteps_);
  
//   std::cout << "compute initial weights" << std::endl;
  
  computeLogInitialParticleWeights(particlesNew, logUnnormalisedWeights);
  
//      std::cout << "======================" << (local_ == EHMM_LOCAL_HILBERT) << std::endl;
  if (local_ == EHMM_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == EHMM_LOCAL_HILBERT_RANDOM_WALK)
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
//     std::cout << "======================" << (local_ == EHMM_LOCAL_HILBERT) << std::endl;
  

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
    if (local_ == EHMM_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == EHMM_LOCAL_HILBERT_RANDOM_WALK)
    {
      // TODO: we need to use the logistic transform here:
      //resample::hilbertSortBase(convertGaussianToUniform(aux), sortedIndices_);
//       resample::hilbertSort(particlesFull_[t-1], sortedIndices_, -3.0, 3.0); // WARNING: hilbert resampling is currently disabled!
//       std::cout << "sorted Indices_: " << sortedIndices_.t() << std::endl;
    }
    selfNormalisedWeights_ = normaliseWeights(logUnnormalisedWeights);
    
//     std::cout << "ESS at time " << t << std::endl;
    
    ess_(t-1) = 1.0 / arma::dot(selfNormalisedWeights_, selfNormalisedWeights_) / nParticles_;
    
        
    if (!arma::is_finite(selfNormalisedWeights_))
    {
      std::cout << "WARNING: W contains NaNs! at step " << t << std::endl;
    }
    
//     std::cout << "start apply kernel" << std::endl; 
    
    if (isConditional_) 
    { 
      particleIndices_(t) = arma::as_scalar(arma::randi(1, arma::distr_param(0,static_cast<int>(nParticles_-1))));
      
      /////////////////////////////////////////////////////////////////////////
      // Ancestor sampling
      /////////////////////////////////////////////////////////////////////////
      
      // Determining the parent index of the current input particle:
      if (csmc_ == CSMC_BACKWARD_SAMPLING_ANCESTOR) // via ancestor sampling
      {
        /*
        for (unsigned int n=0; n<nParticles_; n++)
        {
          logWeightsAux(n)  = logUnnormalisedWeights(n) + 
                              model_.evaluateLogTransitionDensity(t, particlePath_[t], particlesFull_[t-1][n]);
                              /// NOTE: for more general models than state-space models this needs to be modified!
        }
        parentIndices(particleIndices_(t)) = sampleInt(normaliseWeights(logWeightsAux));
        */
        
//                std::cout << "start AS" << std::endl;
        parentIndices(particleIndices_(t)) = backwardSampling(t-1, logUnnormalisedWeights, particlesFull_[t-1]);
//          std::cout << "finished AS" << std::endl;
        
//         std::cout << "a_{t-1}^{b_t}: " << parentIndices(particleIndices_(t)) << "; b_{t-1}: " << particleIndices_(t-1) <<  std::endl;
                
      }
      else // not via ancestor sampling
      {
        parentIndices(particleIndices_(t)) = particleIndices_(t-1);
      }
      
      if (local_ == EHMM_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == EHMM_LOCAL_HILBERT_RANDOM_WALK)
      {
        sortedParentIndex_ = arma::as_scalar(arma::find(sortedIndices_ == parentIndices(particleIndices_(t)), 1, "first"));
        sortedParentIndexTemp_ = sortedParentIndex_;
      }

      particlesNew[particleIndices_(t)] = particlePath_[t];
      
      if (particleIndices_(t) > 0)
      {
        for (unsigned int n=particleIndices_(t)-1; n != static_cast<unsigned>(-1); n--)
        {
          this->applyKernel(t, n, n+1, particlesNew, parentIndices, logUnnormalisedWeights, selfNormalisedWeights_);
//           sampleFromRho(t, n, particlesNew, parentIndices, selfNormalisedWeights_);
        }
      }
      if (local_ == EHMM_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == EHMM_LOCAL_HILBERT_RANDOM_WALK) {sortedParentIndexTemp_ = sortedParentIndex_;}
    }
    else
    {
      particleIndices_(t) = 0;
      
      this->sampleFromRho(t, 0, particlesNew, parentIndices, selfNormalisedWeights_);
      
      if (local_ == EHMM_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == EHMM_LOCAL_HILBERT_RANDOM_WALK)
      {
        sortedParentIndex_ = arma::as_scalar(arma::find(sortedIndices_ == parentIndices(particleIndices_(t)), 1, "first"));
        sortedParentIndexTemp_ = sortedParentIndex_;
      }
      
    }
    
//         std::cout << "finished apply kernel 1" << std::endl; 
    
    if (particleIndices_(t)+1 < nParticles_)
    {
      for (unsigned int n=particleIndices_(t)+1; n<nParticles_; n++)
      {
        this->applyKernel(t, n, n-1, particlesNew, parentIndices, logUnnormalisedWeights, selfNormalisedWeights_);
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
    
//       std::cout << "======================" << (local_ == EHMM_LOCAL_HILBERT) << std::endl;
//       std::cout << "started determine Gaussian from particles" << std::endl;
    if (local_ == EHMM_LOCAL_HILBERT_AUTOREGRESSIVE || local_ == EHMM_LOCAL_HILBERT_RANDOM_WALK)
    {
      determineGaussiansFromParticles(t, particlesNew, aux);
    }
    
//           std::cout << "finished determine Gaussian from particles" << std::endl;
//             std::cout << "======================" << (local_ == EHMM_LOCAL_HILBERT) << std::endl;

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
        
  if (samplePath_)
  {
    // Sampling a single particle path:
    particlePath_.resize(nSteps_);
    
    // Final-time particle:
    particleIndices_(nSteps_-1) = sampleInt(normaliseWeights(logUnnormalisedWeights));
    particlePath_[nSteps_-1] = particlesFull_[nSteps_-1][particleIndices_(nSteps_-1)];
    
    // Recursion for the particles at previous time steps:
    if (isConditional_ && csmc_ == CSMC_BACKWARD_SAMPLING_STANDARD)
    { // i.e. we employ the usual backward-sampling recursion
      for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
      { 
        particleIndices_(t) = backwardSampling(t, logUnnormalisedWeightsFull_.col(t), particlesFull_[t]);
        /*
        for (unsigned int n=0; n<nParticles_; n++)
        {
          logWeightsAux(n)  = logUnnormalisedWeightsFull_(n,t) + 
                              model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particlesFull_[t][n]);
                              /// NOTE: for more general models than state-space models this needs to be modified!
        }
        particleIndices_(t) = sampleInt(normaliseWeights(logWeightsAux));
        */
        particlePath_[t] = particlesFull_[t][particleIndices_(t)];
      }
    }
    else // i.e we just trace back the ancestral lineage
    {             
      for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
      { 
        particleIndices_(t) = parentIndicesFull_(particleIndices_(t+1), t);
        particlePath_[t] = particlesFull_[t][particleIndices_(t)];
      }
    }
  }
}

/// Samples a single particle index via backward sampling.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EhmmParameters> 
unsigned int Ehmm<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EhmmParameters>::backwardSampling
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

#endif
