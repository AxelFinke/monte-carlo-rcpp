/// \file
/// \brief Performing MCMC-based inference in some model. 
///
/// This file contains the functions associated with an the Mcmc class.

#ifndef __MCMC_H
#define __MCMC_H

#include "model/Model.h"

// [[Rcpp::depends("RcppArmadillo")]]

/// Type of proposal (kernel) for the parameters
/// within a Metropolis--Hastings update.
enum McmcKernelType 
{ 
  MCMC_KERNEL_TRUNCATED_GAUSSIAN_RANDOM_WALK_METROPOLIS_HASTINGS = 0,
  MCMC_KERNEL_GAUSSIAN_RANDOM_WALK_METROPOLIS_HASTINGS,
  MCMC_KERNEL_TRUNCATED_GAUSSIAN_INDEPENDENT_METROPOLIS,
  MCMC_KERNEL_GAUSSIAN_INDEPENDENT_METROPOLIS
};

/// Generic class for performing MCMC updates.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class McmcParameters> class Mcmc
{
  
public:
  
  /// Initialises the class.
  Mcmc
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const unsigned int nCores
    
  ) : rng_(rng), model_(model), nCores_(nCores)
  { 
    kern_ = static_cast<McmcKernelType>(0);
    proposalScale_ = 1.0;
    proposalDownscaleProbability_ = 0.0;
    useGradients_ = false;
    useAdaptiveProposal_ = false;
    useAdaptiveProposalScaleFactor1_ = false;
    useDelayedAcceptance_ = false;
    isWithinSmcSampler_ = false;
  }
  /// Constructor.
  Mcmc
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const McmcKernelType kern,
    const bool useGradients,
    const double proposalDownscaleProbability,
    const arma::colvec rwmhSd,
    const double crankNicolsonScaleParameter,
    const unsigned int nCores
    
  ) : 
    rng_(rng), 
    model_(model), 
    kern_(kern),
    useGradients_(useGradients),
    proposalDownscaleProbability_(proposalDownscaleProbability),
    rwmhSd_(rwmhSd),
    crankNicolsonScaleParameter_(crankNicolsonScaleParameter),
    nCores_(nCores)
  { 
    proposalScale_ = 1.0;
    useAdaptiveProposalScaleFactor1_ = false;
    isWithinSmcSampler_ = false;
  }
  
  /// Specifies the number of iterations
  void setNIterations(const unsigned int nIterations, const double burninPercentage) 
  {
    nIterations_ = nIterations;
    nBurninSamples_ = std::floor(nIterations * burninPercentage); // total number of burnin iterations
    nKeptSamples_ = nIterations - nBurninSamples_; // total number of samples that are kept after discarding burn-in samples
  }
  /// Returns the total number of iterations
  unsigned int getNIterations() {return nIterations_;}
  /// Returns the number of samples kept after discarding burnin-samples.
  unsigned int getNKeptSamples() {return nKeptSamples_;}
  /// Returns the number of samples discarded as burnin.
  unsigned int getNBurninSamples() {return nBurninSamples_;}
  /// Specifies whether the proposals use gradient information.
  void setUseGradients(const bool useGradients) {useGradients_ = useGradients;}
  /// Specifies whether we should use the mixture proposal from Peters et al. (2010).
  void setUseAdaptiveProposal(const bool useAdaptiveProposal) {useAdaptiveProposal_ = useAdaptiveProposal;}
  /// Specifies the sample covariance matrix needed for the adaptive mixture proposal from Peters et al. (2010).
  void setSampleCovarianceMatrix(const arma::mat& sampleCovarianceMatrix) {sampleCovarianceMatrix_ = sampleCovarianceMatrix;}
  /// Specifies the sample mean needed for adaptive independence Metropolis updates.
  void setSampleMean(const arma::colvec& sampleMean) {sampleMean_ = sampleMean;}
  /// Specifies whether we should use the adaptive mixture proposal from Peters et al. (2010).
  void setAdaptiveProposalParameters(const arma::colvec& adaptiveProposalParameters)
  {
    mixtureProposalWeight1_   = adaptiveProposalParameters(0);
    proposalScaleFactor1_     = adaptiveProposalParameters(1);
    proposalScaleFactor1Original_ = adaptiveProposalParameters(1);
    proposalScaleFactor2_     = adaptiveProposalParameters(2);
    nonAdaptPercentage_       = adaptiveProposalParameters(3);
    proposalScaleFactor1DecreaseFactor_  = adaptiveProposalParameters(4);
    proposalScaleFactor1IncreaseFactor_  = adaptiveProposalParameters(5);
    acceptanceRateLowerBound_ = adaptiveProposalParameters(6);
    acceptanceRateUpperBound_ = adaptiveProposalParameters(7);
  }
  /// Specifies whether we should use the adaptive mixture proposal from Peters et al. (2010).
  void setAdaptiveProposalParameters(const arma::colvec& adaptiveProposalParameters, const unsigned int nIterations) 
  {
    mixtureProposalWeight1_   = adaptiveProposalParameters(0);
    proposalScaleFactor1_     = adaptiveProposalParameters(1);
    proposalScaleFactor1Original_ = adaptiveProposalParameters(1);
    proposalScaleFactor2_     = adaptiveProposalParameters(2);
    nonAdaptPercentage_       = adaptiveProposalParameters(3);
    proposalScaleFactor1DecreaseFactor_ = adaptiveProposalParameters(4);
    proposalScaleFactor1IncreaseFactor_ = adaptiveProposalParameters(5);
    acceptanceRateLowerBound_ = adaptiveProposalParameters(6);
    acceptanceRateUpperBound_ = adaptiveProposalParameters(7);
    
    nNonAdaptSamples_         = std::floor(nIterations * nonAdaptPercentage_); // total number of burnin iterations
  }
  /// Returns whether use the mixture proposal from Peters et al. (2010).
  bool getUseAdaptiveProposal() {return useAdaptiveProposal_;}
  /// Specifies whether the MCMC kernels are used within an SMC sampler.
  void setIsWithinSmcSampler(const bool isWithinSmcSampler) {isWithinSmcSampler_ = isWithinSmcSampler;}
  /// Specifies whether to use a delayed-acceptance proposal.
  void setUseDelayedAcceptance(const bool useDelayedAcceptance) {useDelayedAcceptance_ = useDelayedAcceptance;}
  /// Returns whether we use a delayed-acceptance proposal.
  bool getUseDelayedAcceptance() {return useDelayedAcceptance_;}
  /// Returns whether or not the proposals use gradient information.
  bool getUseGradients() {return useGradients_;}
  /// Specifies the vector of RWMH proposal scales.
  void setRwmhSd(const arma::colvec& rwmhSd) {rwmhSd_ = rwmhSd;}
  /// Specifies whether the proposal scale (i.e. the scalar by which the sample covariance matrix is multiplied)
  /// should be addapted according the acceptance rate when the proposal scale is determined adaptively.
  void setUseAdaptiveProposalScaleFactor1(const bool useAdaptiveProposalScaleFactor1) {useAdaptiveProposalScaleFactor1_ = useAdaptiveProposalScaleFactor1;}
  /// Returns whether the proposal scale (i.e. the scalar by which the sample covariance matrix is multiplied)
  /// is be addapted according the acceptance rate when the proposal scale is determined adaptively.
  bool getUseAdaptiveProposalScaleFactor1() {return useAdaptiveProposalScaleFactor1_;}
  /// Specifies proposal scale (i.e. the scalar by which the sample covariance matrix is multiplied) 
  /// when the proposal scale is determined adaptively.
  void setProposalScaleFactor1(const double proposalScaleFactor1) {proposalScaleFactor1_ = proposalScaleFactor1;}
  /// Returns proposal scale (i.e. the scalar by which the sample covariance matrix is multiplied) when the proposal scale is determined adaptively.
  double getProposalScaleFactor1() {return proposalScaleFactor1_;}
  /// Adapts the proposal scale (i.e. the scalar by which the sample covariance matrix is multiplied) 
  /// as a function of some observed acceptance rate.
  void adaptProposalScaleFactor1(const double acceptanceRate) 
  {
    if (acceptanceRate > acceptanceRateUpperBound_) 
    {
      proposalScaleFactor1_ = proposalScaleFactor1_ * proposalScaleFactor1IncreaseFactor_;
      std::cout << "proposalScaleFactor1_ increased to " << proposalScaleFactor1_ << " because the acceptance rate was above " << acceptanceRateUpperBound_ << std::endl;
      std::cout << " proposalScaleFactor1_ now at " << proposalScaleFactor1_ / proposalScaleFactor1Original_ * 100 << " percent of the original value " << std::endl;
    }
    else if (acceptanceRate < acceptanceRateLowerBound_)
    {
      proposalScaleFactor1_ = proposalScaleFactor1_ * proposalScaleFactor1DecreaseFactor_;
      std::cout << "proposalScaleFactor1_ decreased to " << proposalScaleFactor1_ << " because the acceptance rate was below " << acceptanceRateLowerBound_ << std::endl;
      std::cout << " proposalScaleFactor1_ now at " << proposalScaleFactor1_ / proposalScaleFactor1Original_ * 100 << " percent of the original value " << std::endl;
    }
  }
  /// Specifies the type of RWMH proposal kernel.
  void setKern(const McmcKernelType& kern) {kern_ = kern;}
  /// Proposes parameters for the RWMH kernel.
  void setProposalScale(double proposalScale) {proposalScale_ = proposalScale;}
  /// Returns the Crank--Nicolson correlation parameter needed for correlated
  /// pseudo-marginal kernels
  double getCrankNicolsonScale(const unsigned int nParticles, const unsigned int nObservations) const {return std::exp(- crankNicolsonScaleParameter_ * static_cast<double>(nParticles) / nObservations);} 
  /// Returns the Crank--Nicolson correlation parameter needed for correlated
  /// pseudo-marginal kernels
  double getCrankNicolsonScale(const unsigned int nParticles, const double inverseTemperature) const {return std::exp(- crankNicolsonScaleParameter_ * static_cast<double>(nParticles) / inverseTemperature);} 
  /// Samples the set of parameters from the proposal kernel
  /// if we use a Gaussian Random-Walk Metropolis--Hastings kernel
  /// (truncated to the support of the parameters).
  void proposeTheta(arma::colvec& thetaNew, const arma::colvec& thetaOld);
  /// Samples the set of parameters from the proposal kernel
  /// if we use a Gaussian Random-Walk Metropolis--Hastings kernel
  /// (truncated to the support of the parameters). If useAdaptiveProposal == true and if 
  /// g > nBurninSamples
  /// then this function samples from the adaptive mixture proposal from Peters et al. (2010).
  void proposeTheta(const unsigned int g, arma::colvec& thetaNew, const arma::colvec& thetaOld);
  /// Samples the set of parameters from the proposal kernel 
  /// if we use a MALA kernel (truncated to the support of the parameters).
  void proposeTheta(arma::colvec& thetaNew, const arma::colvec& thetaOld, const arma::colvec& gradient);
  /// Evaluates the log-density of the proposal kernel for the parameters
  /// if we use a Gaussian Random-Walk Metropolis--Hastings kernel
  /// (truncated to the support of the parameters).
  double evaluateLogProposalDensity(const arma::colvec& thetaProp, const arma::colvec& theta);
  /// Evaluates the log-density of the proposal kernel for the parameters
  /// if we use a Gaussian Random-Walk Metropolis--Hastings kernel
  /// (truncated to the support of the parameters). If useAdaptiveProposal == true and if 
  /// g > nBurninSamples
  /// then this function returns zero because the adaptive mixture proposal from
  /// Peters et al. (2010) is symmetric.
  double evaluateLogProposalDensity(const unsigned int g, const arma::colvec& thetaProp, const arma::colvec& theta);
  /// Evaluates the log-density of the proposal kernel for the parameters 
  /// if we use a MALA kernel (truncated to the support of the parameters).
  double evaluateLogProposalDensity(const arma::colvec& thetaProp, const arma::colvec& theta, const arma::colvec& gradient);
  /// Computes (part of the) log-acceptance ratio.
  double computeLogAlpha(const arma::colvec& thetaProp, const arma::colvec& theta)
  {
    double x = model_.evaluateLogPriorDensity(thetaProp) -
               model_.evaluateLogPriorDensity(theta) +
               evaluateLogProposalDensity(theta, thetaProp) -
               evaluateLogProposalDensity(thetaProp, theta);
               
    return(x);
  }
  /// Computes (part of the) log-acceptance ratio
  /// (in the case that we use gradient information)
  double computeLogAlpha(const arma::colvec& thetaProp, const arma::colvec& theta, const arma::colvec gradientProp, const arma::colvec& gradient)
  {
    double x = model_.evaluateLogPriorDensity(thetaProp) -
               model_.evaluateLogPriorDensity(theta) +
               evaluateLogProposalDensity(theta, thetaProp, gradientProp) -
               evaluateLogProposalDensity(thetaProp, theta, gradient);
               
    return(x);
  }
  /// Determines whether or not the standard deviation of the random-walk 
  /// Metropolis--Hastings proposal kernel should be scaled 
  /// by a particular factor at a particular iteration.
  void sampleProposalScale(double factor)
  {
    if (arma::randu() < proposalDownscaleProbability_)
    {
      proposalScale_ = factor;
    }
    else
    {
      proposalScale_ = 1.0;
    }
  }

private:
  
  Rng& rng_; // Random number generation
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // the model
  McmcKernelType kern_; // type of Metropolis--Hastings proposal kernel
  
  unsigned int nIterations_; // total number of iterations
  unsigned int nKeptSamples_; // number of samples kept after burn-in
  unsigned int nBurninSamples_; // total number of samples discarded (note that nIterations = nKeptSamples + nBurninSamples)
  unsigned int nNonAdaptSamples_; // total number of samples before adaptation takes place
  
  bool useGradients_; // are we using gradient information in the proposals?
  bool useAdaptiveProposal_; // should we use the mixture proposal from Peters et al. (2010)?
  bool useAdaptiveProposalScaleFactor1_; // should we adapt proposalScaleFactor1_ if the accaptance rate is too high/low?
  bool useDelayedAcceptance_; // should we use delayed acceptance kernels?
  bool isWithinSmcSampler_; // are the MCMC kernels used within an SMC sampler (so that we do not wait for a burnin period before using adaptive kernels)?
  
  double mixtureProposalWeight1_; // weight of the first component in the mixture proposal from Peters et al. (2010).
  double proposalScaleFactor1_; // scaling factor for the first component in the mixture proposal from Peters et al. (2010).
  double proposalScaleFactor1Original_; // same as proposalScaleFactor1_ unless this factor is adapted according to the acceptance rate.
  double proposalScaleFactor2_; // scaling factor for the second component in the mixture proposal from Peters et al. (2010).
  double proposalScaleFactor1DecreaseFactor_; // factor by which proposalScaleFactor1_ is multiplied if the acceptance rate is too low.
  double proposalScaleFactor1IncreaseFactor_; // factor by which proposalScaleFactor1_ is multiplied if the acceptance rate is too high.
  double nonAdaptPercentage_; // percentage of iterations before adaptation takes place
  double acceptanceRateLowerBound_, acceptanceRateUpperBound_; // adapt proposalScaleFactor1_ if the acceptance rate falls outside this interval.
  
  double proposalScale_; // factor for downscaling the proposal density
  double proposalDownscaleProbability_; // probability of downscaling the RWMH proposal kernel
  arma::colvec rwmhSd_; // proposal scales for the random-walk Metropolis--Hastings kernel
  double crankNicolsonScaleParameter_; // correlation parameter for Crank--Nicolson proposals
  arma::mat sampleCovarianceMatrix_; // empirical covariance matrix of the mean of the previously generated parameter vectors
  arma::colvec sampleMean_; // empirical mean of the previously generated parameter vectors
  unsigned int nCores_; // number of cores to use (not currently used)
  
};

/// Samples the set of parameters from the proposal kernel.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class McmcParameters>
void Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>::proposeTheta
(
  arma::colvec& thetaNew, 
  const arma::colvec& thetaOld,
  const arma::colvec& gradient
)
{
  thetaNew.set_size(thetaOld.n_rows);
  double gradientL2Norm = sqrt(arma::as_scalar(arma::sum(arma::pow(gradient, 2.0))));
  for (unsigned int k=0; k<thetaNew.n_rows; k++)
  {
    thetaNew(k) = gaussian::rtnorm(model_.getSupportMin(k), model_.getSupportMax(k), 
                         thetaOld(k) + pow(proposalScale_ * rwmhSd_(k), 2.0) * gradient(k)/gradientL2Norm / 2.0, 
                         proposalScale_ * rwmhSd_(k));
  }
}
/// Samples the set of parameters from the proposal kernel.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class McmcParameters>
void Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>::proposeTheta
(
  arma::colvec& thetaNew, 
  const arma::colvec& thetaOld
)
{
  //std::cout << "thetaOld: " << thetaOld.t() << std::endl;
  thetaNew.set_size(thetaOld.n_rows);
  for (unsigned int k=0; k<thetaNew.n_rows; k++)
  {
//     std::cout << "supportMin: " << model_.getSupportMin(k) << " supportMax: " << model_.getSupportMax(k) << " sd: " << proposalScale_ * rwmhSd_(k) << std::endl;
    thetaNew(k) = gaussian::rtnorm(model_.getSupportMin(k), model_.getSupportMax(k), 
                         thetaOld(k), proposalScale_ * rwmhSd_(k));
  
  }
}
/// Evaluates the log-density of the proposal kernel for the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class McmcParameters>
double Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>::evaluateLogProposalDensity
(
  const arma::colvec& thetaNew, 
  const arma::colvec& thetaOld,
  const arma::colvec& gradient
)
{
  double logDensity = 0.0;
  double gradientL2Norm = sqrt(arma::as_scalar(arma::sum(arma::pow(gradient, 2.0))));
  for (unsigned int k=0; k<thetaNew.size(); k++)
  {
    logDensity += gaussian::dtnorm(thetaNew(k), model_.getSupportMin(k), model_.getSupportMax(k), 
                         thetaOld(k) + pow(proposalScale_ * rwmhSd_(k), 2.0) * gradient(k)/gradientL2Norm / 2.0, 
                         proposalScale_ * rwmhSd_(k), true, true);
  }
  return logDensity;
}
/// Evaluates the log-density of the proposal kernel for the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class McmcParameters>
double Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>::evaluateLogProposalDensity
(
  const arma::colvec& thetaNew, 
  const arma::colvec& thetaOld
)
{
  double logDensity = 0.0;
  for (unsigned int k=0; k<thetaNew.size(); k++)
  {
    logDensity += gaussian::dtnorm(thetaNew(k), model_.getSupportMin(k), model_.getSupportMax(k), 
                         thetaOld(k), proposalScale_ * rwmhSd_(k), 
                         true, true);
  }
  return logDensity;
}

/// Samples the set of parameters from the proposal kernel.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class McmcParameters>
void Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>::proposeTheta
(
  const unsigned int g,
  arma::colvec& thetaNew, 
  const arma::colvec& thetaOld
)
{
  if (useAdaptiveProposal_ && (isWithinSmcSampler_ || g >= nNonAdaptSamples_))
  {
//     std::cout << "proposing theta using the adaptive Gaussian proposal!" << std::endl;
    if (arma::randu() < mixtureProposalWeight1_)
    {
      thetaNew = thetaOld + arma::chol(proposalScaleFactor1_ * sampleCovarianceMatrix_) * arma::randn<arma::colvec>(thetaOld.size());
    }
    else
    {
      thetaNew = thetaOld + std::sqrt(proposalScaleFactor2_) * arma::randn<arma::colvec>(thetaOld.size());
    }
  }  
  else
  {
//     std::cout << "proposing theta using independent truncated Gaussians!" << std::endl;
    proposeTheta(thetaNew, thetaOld);
  }
}
/// Evaluates the log-density of the proposal kernel for the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class McmcParameters>
double Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>::evaluateLogProposalDensity
(
  const unsigned int g,
  const arma::colvec& thetaNew, 
  const arma::colvec& thetaOld
)
{
  if (useAdaptiveProposal_ && (isWithinSmcSampler_ || g >= nNonAdaptSamples_))
  {
    return 0.0; // the adaptive mixture proposal is symmetric!
  }  
  else
  {
    return evaluateLogProposalDensity(thetaNew, thetaOld);
  }
}

#endif
