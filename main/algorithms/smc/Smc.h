/// \file
/// \brief Performing (C)SMC-based inference in some model. 
///
/// This file contains the functions associated with the Smc class.

#ifndef __SMC_H
#define __SMC_H

#include <time.h> 
#include "main/model/Model.h"
#include "main/algorithms/smc/resample.h"
// #include "smc/hilbertResample.h"

// [[Rcpp::depends("RcppArmadillo")]]

/// Specifiers for various resampling algorithms (currently only multinomial and systematic resampling are implemented!):
enum ResampleType
{ 
  SMC_RESAMPLE_MULTINOMIAL = 0, 
  SMC_RESAMPLE_SYSTEMATIC,
  SMC_RESAMPLE_RESIDUAL, 
  SMC_RESAMPLE_STRATIFIED
};

/// Backward sampling types.
enum SmcBackwardSamplingType 
{  
  SMC_BACKWARD_SAMPLING_NONE = 0,
  SMC_BACKWARD_SAMPLING_STANDARD,
  SMC_BACKWARD_SAMPLING_ANCESTOR
};
/// Type of proposal (kernel) for the latent states/latent variable
/// within importance sampling or SMC.
enum SmcProposalType 
{ 
  SMC_PROPOSAL_PRIOR = 0, 
  SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL,
  SMC_PROPOSAL_FA_APF,
  SMC_PROPOSAL_ALTERNATE_0,
  SMC_PROPOSAL_ALTERNATE_1,
  SMC_PROPOSAL_LOOKAHEAD,
  SMC_PROPOSAL_GAUSSIAN_RANDOM_WALK // this uses the generalised Gaussian random-walk proposal from Tjelmeland (2004); Note that this can only be used for conditional SMC methods, i.e. if isConditional_ == true. Also, this option requires storeHistory_ to be set to true
};
/// Converts a vector of standard Gaussian values to the corresponding
/// uniform random variables.
std::vector<arma::colvec> convertGaussianToUniform(const std::vector<arma::colvec>& aux1)
{
  std::vector<arma::colvec> uniforms(aux1.size());
  for (unsigned int n=0; n<aux1.size(); n++)
  {
    uniforms[n].set_size(aux1[n].n_rows);
    for (unsigned int v=0; v<uniforms[n].n_rows; v++)
    {
      uniforms[n](v) = R::pnorm(aux1[n](v), 0.0, 1.0, true, false);
    }
  }
  return uniforms;
}
/// Converts a vector of standard Gaussian values to the corresponding
/// uniform random variables.
std::vector<double> convertGaussianToUniform(const std::vector<double>& aux1)
{
  std::vector<double> uniforms(aux1.size());
  for (unsigned int n=0; n<aux1.size(); n++)
  {
    uniforms[n] = R::pnorm(aux1[n], 0.0, 1.0, true, false);
  }
  return uniforms;
}
/// Holds all the Gaussian auxiliary random variables needed for 
/// using correlated pseudo-marginal approaches.
template<class Aux> class AuxFull
{
public:
  
  /// Initialises the class.
  AuxFull() {}
  /// Destructor.
  ~AuxFull() {}
  
  /// Adds correlated Gaussian noise to the elements of this class
  void addCorrelatedGaussianNoise(const double correlationParameter)
  {
    // TODO: make this more efficient
    for (unsigned int t=0; t<aux2_.size(); t++)
    {
       for (unsigned int n=0; n<aux1_[t].size(); n++)
       {
         addCorrelatedGaussianNoise(correlationParameter, aux1_[t][n]);
       }
       aux2_[t] = correlationParameter * aux2_[t] + sqrt(1 - pow(correlationParameter, 2.0)) * arma::randn();
    }
  }
  /// Resizes the components of aux1_ to be consistent
  /// with a specific number of particles.
  void resizeInner(const unsigned int nParticles)
  {
    for (unsigned int t=0; t<aux1_.size(); t++)
    {
      aux1_[t].resize(nParticles);
    }
  }
  /// Changes the number of components of aux1_ and aux2_ to be consistent
  /// with a specific number of SMC steps.
  void resizeOuter(const unsigned int nSteps)
  {
    aux2_.resize(nSteps);
    aux1_.resize(nSteps);
  }
  /// Resizes aux1_ and aux2_.
  void resize(const unsigned int nSteps, const unsigned int nParticles)
  {
    resizeOuter(nSteps);
    resizeInner(nParticles);
  }
  /// Gaussian random variables which are transformations of the particles.
  std::vector< std::vector<Aux> > aux1_;
  /// Gaussian random variables associated with resampling steps.
  std::vector<double> aux2_; 
  
private:
  
  /// Proposes new values for the Gaussian auxiliary variables
  /// using a Crank--Nicolson proposal (needs to be 
  /// implemented by the user).
  void addCorrelatedGaussianNoise(const double correlationParameter, Aux& aux);
  
};

/// Class template for running (conditional) SMC algorithms or other forms 
/// of importance sampling.
template<class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> class Smc
{
public:
  
  /// Initialises the class.
  Smc
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const unsigned int nSteps,
    const SmcProposalType smcProposalType,
    const double essResamplingThreshold,
    const SmcBackwardSamplingType smcBackwardSamplingType,
    const bool approximateGradient,
    unsigned int fixedLagSmoothingOrder,
    const unsigned int nCores
  ) : 
    rng_(rng), 
    model_(model),
    nSteps_(nSteps),
    smcProposalType_(smcProposalType), 
    essResamplingThreshold_(essResamplingThreshold),
    smcBackwardSamplingType_(smcBackwardSamplingType),
    approximateGradient_(approximateGradient),
    fixedLagSmoothingOrder_(fixedLagSmoothingOrder),
    nCores_(nCores)
  {
    storeHistory_ = true; // TODO: make this accessible from the outside 
    samplePath_   = true; // TODO: make this accessible from the outside
    resampleType_ = SMC_RESAMPLE_SYSTEMATIC;
//     weightsContainNans_ = false;
  }
  
  /// Initialises the class without specifying many of the parameters.
  Smc
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const unsigned int nCores
  ) : 
    rng_(rng), 
    model_(model),
    nCores_(nCores)
  {
    smcProposalType_ = 0;
    smcBackwardSamplingType_ = 0;
    useGaussianParametrisation_ = false;
    determineParticlesFromGaussians_ = false;
    nSteps_ = model_.getNObservations();
    fixedLagSmoothingOrder_ = 0;
    approximateGradient_ = false; 
    storeHistory_ = true;
    samplePath_   = true;
//     weightsContainNans_ = false;
    resampleType_ = SMC_RESAMPLE_SYSTEMATIC;
  }
  
  /// Returns the SMC parameters.
  const SmcParameters& getSmcParameters() const {return smcParameters_;}
  /// Returns the SMC parameters.
  SmcParameters& getRefSmcParameters() {return smcParameters_;}
  /// Returns the estimate of the normalising constant.
  double getLoglikelihoodEstimate() const {return logLikelihoodEstimate_;}
  /// Returns the number of SMC steps.
  unsigned int getNSteps() const {return nSteps_;}
  /// Returns the number of particles.
  unsigned int getNParticles() const {return nParticles_;}
  /// Returns whether we should calculate particles from the Gaussian auxiliary variables.
  /// Only relevant when useGaussianParametrisation is TRUE.
  bool getDetermineParticlesFromGaussians() const {return determineParticlesFromGaussians_;}
  /// Returns the complete set of all parent indices.
  void getParentIndicesFull(arma::umat& parentIndicesFull) {parentIndicesFull = parentIndicesFull_;}
  /// Returns the complete set of the particle indices associated with the input reference path
  void getParticleIndicesIn(arma::uvec& particleIndicesIn) {particleIndicesIn = particleIndicesIn_;}
  /// Returns the complete set of the particle indices associated with the output reference path
  void getParticleIndicesOut(arma::uvec& particleIndicesOut) {particleIndicesOut = particleIndicesOut_;}
  /// Specifies the effective-sample-size threshold for adaptive resampling.
  void setEssResamplingThreshold(const double essResamplingThreshold) {essResamplingThreshold_ = essResamplingThreshold;}
  /// Specifies the number of particles.
  void setNParticles(const unsigned int nParticles) {nParticles_ = nParticles;}
  /// Specifies resampling scheme.
  void setResampleType(const ResampleType resampleType) {resampleType_ = resampleType;}
  /// Specifies the proposal kernel for the states.
  void setSmcProposalType(const SmcProposalType& smcProposalType) {smcProposalType_ = smcProposalType;}
  /// Returns the proposal kernel for the states.
  SmcProposalType getSmcProposalType() const {return smcProposalType_;}
  /// Returns the number of lookahead steps.
  unsigned int getNLookaheadSteps() const {return nLookaheadSteps_;}
  /// Specifies the number of lookahead steps.
  void setNLookaheadSteps(const unsigned int nLookaheadSteps) {nLookaheadSteps_ = nLookaheadSteps;}
  /// Specify whether we should use Gaussian auxiliary variables as required
  /// for using correlated pseudo-marginal kernels.
  void setUseGaussianParametrisation(bool useGaussianParametrisation)
  {
    useGaussianParametrisation_ = useGaussianParametrisation;
  }
  /// Returns whether we should use Gaussian auxiliary variables as required
  /// for using correlated pseudo-marginal kernels.
  bool getUseGaussianParametrisation() const {return useGaussianParametrisation_;}
  /// Specify whether we should calculate particles from the Gaussian auxiliary variables.
  /// Only relevant when useGaussianParametrisation is TRUE.
  void setDetermineParticlesFromGaussians(const bool determineParticlesFromGaussians)
  {
    determineParticlesFromGaussians_ = determineParticlesFromGaussians;
  }
  /// Should we generate one sample path at the end of the algorithm?
  void setSamplePath(const bool samplePath) {samplePath_ = samplePath;}
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
    AuxFull<Aux>& auxFull,
    const double inverseTemperature
  );
  /// Runs a conditional SMC algorithm.
  double runCsmc
  (
    const unsigned int nParticles, 
    const arma::colvec& theta,
    LatentPath& latentPath,
    AuxFull<Aux>& auxFull,
    const double inverseTemperature
  );
  /// Runs a conditional SMC algorithm but without selecting a new path.
  double runCsmcWithoutPathSampling
  (
    const unsigned int nParticles, 
    const arma::colvec& theta,
    LatentPath& latentPath,
    AuxFull<Aux>& auxFull,
    const double inverseTemperature
  );
  /// Runs an SMC algorithm and approximates the gradient.
  double runSmc
  (
    const unsigned int nParticles,
    const arma::colvec& theta,
    LatentPath& latentPath, 
    AuxFull<Aux>& auxFull,
    arma::colvec& gradientEstimate,
    const double inverseTemperature
  );
  /// Runs an SMC algorithm and approximates the gradient.
  double runSmc
  (
    const arma::colvec& theta,
    LatentPath& latentPath, 
    AuxFull<Aux>& auxFull,
    arma::colvec& gradientEstimate
  );
  /// Runs a conditional SMC algorithm and approximates the gradient.
  double runCsmc
  (
    const unsigned int nParticles, 
    const arma::colvec& theta,
    LatentPath& latentPath,
    AuxFull<Aux>& auxFull,
    arma::colvec& gradientEstimate,
    const double inverseTemperature
  );
  /// Runs the conditional SMC algorithm without returning the log-likelihood.
  void runSampler
  (
    const arma::colvec& theta,
    LatentPath& latentPath
  )
  {
    model_.setUnknownParameters(theta);
    isConditional_ = false;
    AuxFull<Aux> auxFull;
    runSmcBase(auxFull);
    if (samplePath_)
    {
      samplePath(latentPath);
    }
  };
  /// Runs the conditional SMC algorithm without returning the log-likelihood.
  void runConditionalSampler
  (
    const arma::colvec& theta,
    LatentPath& latentPath
  )
  {
    model_.setUnknownParameters(theta);
    isConditional_ = true;
    AuxFull<Aux> auxFull;
    runSmcBase(auxFull);
    if (samplePath_)
    {
      samplePath(latentPath);
    }
  };
  /// Selects one particle path.
  void samplePath(LatentPath& latentPath)
  {
    samplePathBase();
    this->convertParticlePathToLatentPath(particlePath_, latentPath);
  }
    
  
  
private:
  
  /// Samples particles at the first SMC step.
  void sampleInitialParticles(std::vector<Particle>& particles);
  /// Samples particles at later SMC steps.
  void sampleParticles(const unsigned int t, std::vector<Particle>& particlesNew, const std::vector<Particle>& particlesOld);
  /// Samples particles in static models.
  void sampleParticles(const unsigned int t, std::vector<Particle>& particles);
  /// Calculates log-unnormalised particle weights at the first SMC Step.
  void computeLogInitialParticleWeights(const std::vector<Particle>& particles, arma::colvec& logUnnormalisedWeights);
  /// Calculates the log-unnormalised particle weights at later SMC steps.
  void computeLogParticleWeights
  (
    const unsigned int t, 
    const std::vector<Particle>& particlesNew, 
    const std::vector<Particle>& particlesOld, 
    arma::colvec& logUnnormalisedWeights
  );
  /// Calculates the log-unnormalised particle weights
  /// in static models.
  void computeLogParticleWeights
  (
    const unsigned int t, 
    const std::vector<Particle>& particlesNew, 
    arma::colvec& logUnnormalisedWeights
  );
  /// Calculates Gaussian realisation from a single particle at the first SMC step.
  void determineGaussiansFromInitialParticles(const std::vector<Particle>& particles, std::vector<Aux>& aux1);
  /// Calculates a single particle from the Gaussian random variables at the first SMC step.
  void determineInitialParticlesFromGaussians(std::vector<Particle>& particles, const std::vector<Aux>& aux1);
  /// Calculates Gaussian realisation from a single particle at some later SMC step.
  void determineGaussiansFromParticles
  (
    const unsigned int t, 
    const std::vector<Particle>& particlesNew, 
    const std::vector<Particle>& particlesOld, 
    std::vector<Aux>& aux1
  );
  /// Calculates a single particle from the Gaussian random variables at some later SMC step.
  void determineParticlesFromGaussians
  (
    const unsigned int t,  
    std::vector<Particle>& particlesNew, 
    const std::vector<Particle>& particlesOld, 
    const std::vector<Aux>& aux1
  );
  /// Calculates Gaussian realisation from a single particle
  /// in static models.
  void determineGaussiansFromParticles
  (
    const unsigned int t, 
    const std::vector<Particle>& particlesNew,
    std::vector<Aux>& aux1
  );
  /// Calculates a single particle from the Gaussian random variables
  /// in static models.
  void determineParticlesFromGaussians
  (
    const unsigned int t,  
    std::vector<Particle>& particlesNew,
    const std::vector<Aux>& aux1
  );
  /// Samples a single particle index via backward sampling.
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
  void runSmcBase(AuxFull<Aux>& aux);
  /// Samples one particle path from the particle system.
  void samplePathBase();
  /// Calculates smoothing estimate (at the moment: the gradient) via fixed-lag smoothing.
  void runFixedLagSmoothing(arma::colvec& gradientEstimate);
  /// Updates the gradient estimate for a particular SMC step.
  /// We need to supply the Step-t component of the gradient of the 
  /// log-unnormalised target density here, i.e. the sum of the gradients 
  /// of the transition density and observation density, in the case of 
  /// state-space models.
  void updateGradientEstimate(const unsigned int t, const unsigned int n, arma::colvec& gradientEstimate);

  Rng& rng_; // random number generation.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // the targeted model.
  unsigned int nParticles_; // number of particles.
  unsigned int nSteps_; // number of SMC steps.
  unsigned int nLookaheadSteps_; // number of lookahead steps to use (value $0$ indicates that only the current observation is taken into account).
  arma::colvec stepTimes_; // the times at which SMC updates are performed (in continuous-time models) 
  ResampleType resampleType_; // the kind of resampling scheme to use
  SmcProposalType smcProposalType_; // proposal (kernel) for the latent states.
  double essResamplingThreshold_; // proportion of ESS/nParticles below which resampling is triggered.
  bool useGaussianParametrisation_; // for use with the Correlated Pseudo-Marginal approach.
  bool determineParticlesFromGaussians_; // when using Gaussian reparametrisation, should we just transform Gaussians into particles?
  bool storeHistory_; // store history of the particle system?
  bool samplePath_; // sample and store a single particle.
//   bool weightsContainNans_; // do the particle weights contain NaNs?
  SmcBackwardSamplingType smcBackwardSamplingType_; // type of backward sampling to use within Csmc.
  bool approximateGradient_; // should we approximate the gradient as part of the SMC algorithm?
  unsigned int fixedLagSmoothingOrder_; // lag order used in fixed-lag smoothing.
  bool isConditional_; // are we using a conditional SMC algorithm?
  double logLikelihoodEstimate_; // estimate of the normalising constant.
  std::vector<std::vector<Particle>> particlesFull_; // (nSteps_, nParticles_)-dimensional: holds all particles
  std::vector<Particle> particlePath_; // single particle path needed for conditional SMC algorithms
  arma::uvec particleIndicesIn_; // particle indices associated with the single input particle path
  arma::uvec particleIndicesOut_; // particle indices associated with the single output particle path
  arma::umat parentIndicesFull_; // (nParticles_, nSteps_)-dimensional: holds all parent indices
  arma::mat logUnnormalisedWeightsFull_; // (nParticles_, nSteps_)-dimensional: holds all log-unnormalised weight
  SmcParameters smcParameters_; // holds some additional auxiliary parameters for the SMC algorithm.
  unsigned int nCores_; // number of cores to use (not currently used)
  
};

/// Runs the SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runSmcBase
(
  AuxFull<Aux>& auxFull
)
{
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   clock_t t1,t2; // timing
//   double seconds1;
//   t1 = clock(); // start timer
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////

  double ess; // effective sample size
  double u; // single uniform random variable used for systematic resampling
  
  arma::uvec parentIndices(nParticles_); // parent indices associated with a single SMC step
  std::vector<Particle> particlesOld(nParticles_); // particles from current step
  std::vector<Particle> particlesNew(nParticles_); // particles from previous step
  
  unsigned int singleParentIndex   = 0; // parent index for a single particle
  unsigned int singleParticleIndex = 0; // particle index for a single particle 
  
  arma::colvec logUnnormalisedWeights(nParticles_);    // unnormalised log-weights associated with a single SMC step
  logUnnormalisedWeights.fill(-std::log(nParticles_)); // start with uniform weights
  arma::colvec selfNormalisedWeights(nParticles_);     // normalised weights associated with a single SMC step
  
  logLikelihoodEstimate_ = 0; // log of the estimated marginal likelihood   
  
  ///////////////////////////////////////////////////////////////////////////
  // Step 0 of the SMC algorithm
  ///////////////////////////////////////////////////////////////////////////
  
//    std::cout << "################## SMC Step 0 ###################" << std::endl;
//    if (isConditional_) {std::cout << particlePath_[0] << std::endl;}

  if (isConditional_) {samplePath_ = true;}
  if (approximateGradient_) {storeHistory_ = true;}
  
  if (samplePath_) // i.e. if we run a conditional SMC algorithm 
  {
    storeHistory_ = true;
    particleIndicesIn_.set_size(nSteps_);
    if (resampleType_ == SMC_RESAMPLE_SYSTEMATIC)
    {
      particleIndicesIn_(0) = arma::as_scalar(arma::randi(1, arma::distr_param(0,nParticles_-1)));
    }
    else if (resampleType_ == SMC_RESAMPLE_MULTINOMIAL) 
    {
      particleIndicesIn_(0) = 0;
    }
  }
  
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   t2 = clock(); // stop timer 
//   seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//   std::cout << "setting up SMC filter took: " << seconds1 << " seconds." << std::endl;
//   t1 = clock(); // start timer
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  
  
//   std::cout << "sampling at Time 0" << std::endl;
  if (useGaussianParametrisation_)
  {
    if (determineParticlesFromGaussians_)
    {
      determineInitialParticlesFromGaussians(particlesNew, auxFull.aux1_[0]);
    }
    else
    {
      auxFull.resizeOuter(nSteps_);
      auxFull.resizeInner(nParticles_);
      sampleInitialParticles(particlesNew);
      determineGaussiansFromInitialParticles(particlesNew, auxFull.aux1_[0]);
    }
  }
  else
  {
    sampleInitialParticles(particlesNew);
  }
  // NOTE: the following line must be implemented within sampleInitialParticles()
//   if (isConditional_) {particlesNew[particleIndicesIn_(0)] = particlePath_[0];}
  
  
    ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   t2 = clock(); // stop timer 
//   seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//   std::cout << "initial sampling step took: " << seconds1 << " seconds." << std::endl;
//   t1 = clock(); // start timer
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  
//   std::cout << "weighting at Time 0" << std::endl;
  computeLogInitialParticleWeights(particlesNew, logUnnormalisedWeights);
  
  ////////////////////////////////////////////////////////////////////
//   std::cout << "logUnnormalisedWeights at time 0" << std::endl;
//   std::cout << logUnnormalisedWeights.t() << std::endl;
  ////////////////////////////////////////////////////////////////////
  
  if (storeHistory_)
  {
    particlesFull_.resize(nSteps_);
    particlesFull_[0] = particlesNew;
    parentIndicesFull_.set_size(nParticles_, nSteps_-1);
    logUnnormalisedWeightsFull_.set_size(nParticles_, nSteps_);
    logUnnormalisedWeightsFull_.col(0) = logUnnormalisedWeights;
  } 
  
    ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   t2 = clock(); // stop timer 
//   seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//   std::cout << "computing initial weights took: " << seconds1 << " seconds." << std::endl;
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  
  
  
  /////////////////////////////////////////////////////////////////////////////
  // Step t, t>0, of the SMC algorithm
  /////////////////////////////////////////////////////////////////////////////
  
  for (unsigned int t=1; t<nSteps_; t++)
  {
//     std::cout << "################# SMC, Step " << t << " #########################" <<std::endl; 
// if (isConditional_) {std::cout << particlePath_[t] << std::endl;}
    
    ///////////////////////////////////////////////////////////////////////////
    // Ancestor sampling
    ///////////////////////////////////////////////////////////////////////////
    
    if (isConditional_) // i.e. if we run a conditional SMC algorithm 
    {
      // Determining the parent index of the current input particle:
      if (smcBackwardSamplingType_ == SMC_BACKWARD_SAMPLING_ANCESTOR) // via ancestor sampling
      {
        singleParentIndex = backwardSampling(t-1, logUnnormalisedWeights, particlesFull_[t-1]);
      }
      else // not via ancestor sampling
      {
        singleParentIndex = particleIndicesIn_(t-1);
      }
    }
  
    ///////////////////////////////////////////////////////////////////////////
    // Adaptive (conditional) systematic resampling.
    // NOTE: When used with correlated pseudo-marginal kernels,
    // resampling is currently enforced at each time step.
    ///////////////////////////////////////////////////////////////////////////
    
      ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   t1 = clock(); // start timer
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
    
    
    // self-normalised weights:
    selfNormalisedWeights = normaliseWeights(logUnnormalisedWeights);
    // effective sample size:
    ess = 1.0 / arma::dot(selfNormalisedWeights, selfNormalisedWeights); 
    
//     std::cout << "started resampling" << std::endl;
    
//     std::cout << "ESS/N: " << (ess / nParticles_) << std::endl;
        
        
    if (!arma::is_finite(selfNormalisedWeights))
    {
//       std::cout << "WARNING: W (in the filter) contains NaNs at time " << t << "!" << std::endl;
      
      
      ///////////////////////////////////
//       for (unsigned int n=0; n<nParticles_; n++)
//       {
//         std::cout << particlesNew[n].t() << " ----- ";
//       }
     
//       weightsContainNans_ = true;

//       if (t <= 1) 
//       {
//       std::cout << "logUnnormalisedWeights: " << logUnnormalisedWeights.t() << std::endl;
//       }
      /////////////////////////////////////
    }
    
//     std::cout << "logUnnormalisedWeights: " << logUnnormalisedWeights.t() << std::endl;
//     std::cout << selfNormalisedWeights.t() << std::endl;
    
//        std::cout << "incremental logZ estimate at Time " << t-1 << " after re-weighting: " << std::log(arma::sum(arma::exp(logUnnormalisedWeights))) << std::endl;

    ////////////////////
//     std::cout << "loglike. est. at " << t << ": " << logLikelihoodEstimate_ << " ";
    ///////////////////
    
    if (ess < nParticles_ * essResamplingThreshold_ || 
        useGaussianParametrisation_) 
    {
      
//       std::cout << "resampling in the filter at time " << t << std::endl;
      // update estimate of the normalising constant:
      logLikelihoodEstimate_ += std::log(arma::sum(arma::exp(logUnnormalisedWeights))); 
      
     
      
      if (useGaussianParametrisation_)
      {
        if (determineParticlesFromGaussians_)
        {
          u = R::pnorm(auxFull.aux2_[t-1], 0.0, 1.0, true, false);
        }
        else 
        {
          u = arma::randu();
          auxFull.aux2_[t-1] = R::qnorm(u, 0.0, 1.0, true, false);
          
        }
      }
      else 
      {
        u = arma::randu();
      }
      
      // Obtaining the parent indices via adaptive systamatic resampling:           
      if (isConditional_) // "conditional" resampling
      {
        if (resampleType_ == SMC_RESAMPLE_SYSTEMATIC)
        {
          resample::conditionalSystematicBase(u, parentIndices, 
                                              singleParticleIndex, 
                                              selfNormalisedWeights, 
                                              nParticles_, singleParentIndex);
        }
        else if (resampleType_ == SMC_RESAMPLE_MULTINOMIAL)
        {
          resample::conditionalMultinomialBase(parentIndices, 
                                              singleParticleIndex, 
                                              selfNormalisedWeights, 
                                              nParticles_, singleParentIndex);
          
        }
        
        particleIndicesIn_(t) = singleParticleIndex;
      }
      else // "unconditional" resampling
      {
        // Hilbert-curve sorting and systematic resampling
        if (useGaussianParametrisation_) 
        {     
          // TODO: we need to use the logistic transform here, right?
         std::cout << "WARNING: the resample::hilbertBase() function has been deactivated because it causes compilation problems; this means that the correlated pseudo-marginal approach is corrently not working!" << std::endl;
        // resample::hilbertBase(u, parentIndices, particlesNew, selfNormalisedWeights, nParticles_, -3.0, 3.0);
        
        }
        else // standard systematic resampling (without sorting)
        {
          if (resampleType_ == SMC_RESAMPLE_SYSTEMATIC)
          {
            resample::systematicBase(u, parentIndices, selfNormalisedWeights, nParticles_);
          }
          else if (resampleType_ == SMC_RESAMPLE_MULTINOMIAL)
          {
            resample::multinomialBase(parentIndices, selfNormalisedWeights, nParticles_);
          }
        }
      }
      logUnnormalisedWeights.fill(-std::log(nParticles_)); // resetting the weights
    } 
    else // i.e. no resampling:
    {
      
//                 std::cout << "no resampling at step " << t << std::endl;
                
      if (isConditional_) 
      {
        particleIndicesIn_(t) = particleIndicesIn_(t-1);
      }
      parentIndices = arma::linspace<arma::uvec>(0, nParticles_-1, nParticles_);
    }
    // Determining the parent particles based on the parent indices: 
    for (unsigned int n=0; n<nParticles_; n++)
    {
      particlesOld[n] = particlesNew[parentIndices(n)]; 
    }
    
      ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   t2 = clock(); // stop timer 
//   seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//   std::cout << "resampling at time " << t << " took: " << seconds1 << " seconds." << std::endl;
//   t1 = clock(); // start timer
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
    
//             std::cout << "finished resampling" << std::endl;
    
//      std::cout << "incremental logZ estimate at Time " << t << " before re-weighting: " << std::log(arma::sum(arma::exp(logUnnormalisedWeights))) << std::endl;

     
    ///////////////////////////////////////////////////////////////////////////
    // Sampling and reweighting:
    ///////////////////////////////////////////////////////////////////////////
    
    if (useGaussianParametrisation_)
    {
      if (determineParticlesFromGaussians_)
      {
        determineParticlesFromGaussians(t, particlesNew, particlesOld, auxFull.aux1_[t]);
      }
      else
      {
        sampleParticles(t, particlesNew, particlesOld);
        determineGaussiansFromParticles(t, particlesNew, particlesOld, auxFull.aux1_[t]);
      }
    }
    else
    {
//       std::cout << "started sampleParticles()" << std::endl;
      sampleParticles(t, particlesNew, particlesOld);
//        std::cout << "finished sampleParticles()" << std::endl;
      
      /*
       std::cout << "particlesNew" << std::endl;
       for (unsigned int n=0; n<nParticles_; n++)
       {
         std::cout << particlesNew[n];
       }
       std::cout << std::endl;
      */
       
    }
    // NOTE: the following line of code must be incorporated into sampleParticles();
//     if (isConditional_) {particlesNew[particleIndicesIn_(t)] = particlePath_[t];}
    
      ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   t2 = clock(); // stop timer 
//   seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//   std::cout << "sampling at time " << t << " took: " << seconds1 << " seconds." << std::endl;
//   t1 = clock(); // start timer
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////

//      std::cout << "started computeLogParticleWeights" << std::endl;
    computeLogParticleWeights(t, particlesNew, particlesOld, 
                              logUnnormalisedWeights);
    
     ////////////////////////////////////////////////////////////////////
//   std::cout << "logUnnormalisedWeights at time t" << std::endl;
//   std::cout << logUnnormalisedWeights.t() << std::endl;
  ////////////////////////////////////////////////////////////////////
    
//      std::cout << "finished computeLogParticleWeights" << std::endl;
    if (storeHistory_)
    {
//           std::cout << "started store history" << std::endl;
      // Storing the entire particle system:
      particlesFull_[t] = particlesNew; 
      parentIndicesFull_.col(t-1) = parentIndices;
      logUnnormalisedWeightsFull_.col(t) = logUnnormalisedWeights;
//           std::cout << "finished store history" << std::endl;
    }
    
      ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
//   t2 = clock(); // stop timer 
//   seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//   std::cout << "computing weights at time " << t << " took: " << seconds1 << " seconds." << std::endl;
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  ///////////////////////////////
  }
  
//   if (!arma::is_finite(selfNormalisedWeights))
//   {
//     std::cout << "WARNING: W (in the filter) contains NaNs at time " << t << "!" << std::endl;
//     weightsContainNans_ = true;
//   }
  
  // Updating the estimate of the normalising constant:
  logLikelihoodEstimate_ += std::log(arma::sum(arma::exp(logUnnormalisedWeights)));
  
  
//   std::cout << "logLikelihoodEstimate:" << logLikelihoodEstimate_ << std::endl;

  
//   if (weightsContainNans_)
//   {
//     std::cout << "WARNING: SMC filter weights contained NaNs! Likelihood estimate set to zero!" << std::endl;
//     logLikelihoodEstimate_ = - std::numeric_limits<double>::infinity();
//   }
   
//   std::cout << "final incremental logZ" << nSteps_-1 << ": " << std::log(arma::sum(arma::exp(logUnnormalisedWeights))) << std::endl;
  
//       std::cout << "end of the SMC loop" << std::endl;
      
  ///////////////////////////////////////////////////////////////////////////
  // Sample a single particle path from existing particles
  ///////////////////////////////////////////////////////////////////////////
  
//   if (samplePath_)
//   {
//     // Sampling a single particle path:
//     particlePath_.resize(nSteps_);
//     
//     particleIndicesOut_.set_size(nSteps_);
//     
//     // Final-time particle:
//     particleIndicesOut_(nSteps_-1) = sampleInt(normaliseWeights(logUnnormalisedWeightsFull_.col(nSteps_-1)));
//     particlePath_[nSteps_-1]       = particlesFull_[nSteps_-1][particleIndicesOut_(nSteps_-1)];
//     
//     // Recursion for the particles at previous time steps:
//     if (isConditional_ && smcBackwardSamplingType_ == SMC_BACKWARD_SAMPLING_STANDARD)
//     { // i.e. we employ the usual backward-sampling recursion
//       for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
//       { 
//         particleIndicesOut_(t) = backwardSampling(t, logUnnormalisedWeightsFull_.col(t), particlesFull_[t]);
//         particlePath_[t]       = particlesFull_[t][particleIndicesOut_(t)];
//       }
//     }
//     else // i.e we just trace back the ancestral lineage
//     {  
//       for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
//       { 
//         particleIndicesOut_(t) = parentIndicesFull_(particleIndicesOut_(t+1), t);
//         particlePath_[t]       = particlesFull_[t][particleIndicesOut_(t)];
//       }
//     }
//   }
  
//   std::cout << "end of the SMC function" << std::endl;
}


/// Samples one particle path from the particle system.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::samplePathBase()
{
  // Sampling a single particle path:
  particlePath_.resize(nSteps_);
  
  particleIndicesOut_.set_size(nSteps_);
  
  // Final-time particle:
  particleIndicesOut_(nSteps_-1) = sampleInt(normaliseWeights(logUnnormalisedWeightsFull_.col(nSteps_-1)));
  particlePath_[nSteps_-1]       = particlesFull_[nSteps_-1][particleIndicesOut_(nSteps_-1)];
  
  // Recursion for the particles at previous time steps:
  if (isConditional_ && smcBackwardSamplingType_ == SMC_BACKWARD_SAMPLING_STANDARD)
  { // i.e. we employ the usual backward-sampling recursion
    for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
    { 
      particleIndicesOut_(t) = backwardSampling(t, logUnnormalisedWeightsFull_.col(t), particlesFull_[t]);
      particlePath_[t]       = particlesFull_[t][particleIndicesOut_(t)];
    }
  }
  else // i.e we just trace back the ancestral lineage
  {  
    for (unsigned int t=nSteps_-2; t != static_cast<unsigned>(-1); t--)
    { 
      particleIndicesOut_(t) = parentIndicesFull_(particleIndicesOut_(t+1), t);
      particlePath_[t]       = particlesFull_[t][particleIndicesOut_(t)];
    }
  }
}

/// Runs an SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runSmc
(
  const unsigned int nParticles,
  const arma::colvec& theta,
  LatentPath& latentPath, 
  AuxFull<Aux>& auxFull,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = false;
  
  /// We need to loop this over however many SMC runs we need for the model
  runSmcBase(auxFull);
  if (samplePath_)
  {
    samplePath(latentPath);
  }
  
  return getLoglikelihoodEstimate();
}
/// Runs a conditional SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runCsmc
(
  const unsigned int nParticles, 
  const arma::colvec& theta,
  LatentPath& latentPath,
  AuxFull<Aux>& auxFull,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = true;
  convertLatentPathToParticlePath(latentPath, particlePath_);
  runSmcBase(auxFull);
  samplePath(latentPath);
  return getLoglikelihoodEstimate();
}
/// Runs a conditional SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runCsmcWithoutPathSampling
(
  const unsigned int nParticles, 
  const arma::colvec& theta,
  LatentPath& latentPath,
  AuxFull<Aux>& auxFull,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = true;
  convertLatentPathToParticlePath(latentPath, particlePath_);
  runSmcBase(auxFull);
//   samplePath(latentPath);
  return getLoglikelihoodEstimate();
}
/// Runs an SMC algorithm and approximates the gradient.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runSmc
(
  const unsigned int nParticles,
  const arma::colvec& theta,
  LatentPath& latentPath, 
  AuxFull<Aux>& auxFull,
  arma::colvec& gradientEstimate,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = false;
  
  if (approximateGradient_)
  {
    if (gradientEstimate.size() != model_.getDimTheta())
    {
      gradientEstimate.set_size(model_.getDimTheta());
    }
    gradientEstimate.zeros();
    model_.addGradLogPriorDensity(gradientEstimate);
  }
  
  runSmcBase(auxFull);
  if (samplePath_)
  {
    samplePath(latentPath);
  }
  if (approximateGradient_)
  {
    runFixedLagSmoothing(gradientEstimate);
  }
  return getLoglikelihoodEstimate();
}
/// Runs an SMC algorithm and approximates the gradient.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runSmc
(
  const arma::colvec& theta,
  LatentPath& latentPath, 
  AuxFull<Aux>& auxFull,
  arma::colvec& gradientEstimate
)
{
  model_.setUnknownParameters(theta);
  isConditional_ = false;
  
  if (approximateGradient_)
  {
    if (gradientEstimate.size() != model_.getDimTheta())
    {
      gradientEstimate.set_size(model_.getDimTheta());
    }
    gradientEstimate.zeros();
    model_.addGradLogPriorDensity(gradientEstimate);
  }
  
  runSmcBase(auxFull);
  if (samplePath_)
  {
    samplePath(latentPath);
  }
  if (approximateGradient_)
  {
    runFixedLagSmoothing(gradientEstimate);
  }
  return getLoglikelihoodEstimate();
}
/// Runs a conditional SMC algorithm and approximates the gradient.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runCsmc
(
  const unsigned int nParticles, 
  const arma::colvec& theta,
  LatentPath& latentPath,
  AuxFull<Aux>& auxFull,
  arma::colvec& gradientEstimate,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = true;
  
  if (approximateGradient_)
  {
    if (gradientEstimate.size() != model_.getDimTheta())
    {
      gradientEstimate.set_size(model_.getDimTheta());
    }
    gradientEstimate.zeros();
    model_.addGradLogPriorDensity(gradientEstimate);
  }
  
  
  convertLatentPathToParticlePath(latentPath, particlePath_);
  runSmcBase(auxFull);
  samplePath(latentPath);
  if (approximateGradient_)
  {
    runFixedLagSmoothing(gradientEstimate);
  }
  return getLoglikelihoodEstimate();
}

/*
/// Calculates a fixed-lag smoothing approximation of the gradient
// TODO: implement this differently for static models
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runFixedLagSmoothing(arma::colvec& gradientEstimate)
{
  model_.addGradLogPriorDensity(gradientEstimate);
  unsigned int singleParticleIndex;
  
  for (unsigned int n=0; n<nParticles_; n++)
  {
    for (unsigned int t=nSteps_-1; t != static_cast<unsigned>(-1); t--)
    { 
      singleParticleIndex = n;
      for (unsigned int s=std::min(t+fixedLagSmoothingOrder_, nSteps_-1); s>t; s--)
      {
        singleParticleIndex = parentIndicesFull_(singleParticleIndex, s-1);
      }
      updateGradientEstimate(t, singleParticleIndex, gradientEstimate);
    }
  }
}
/// Samples a single particle index via backward sampling.
// TODO: implement this differently for static models
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
unsigned int Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::backwardSampling
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
*/
#endif
