/// \file
/// \brief Performing inference in a multivariate linear Gaussian state-space model via EHMM methods.
///
/// This file contains the functions for performing inference in 
/// a multivariate linear Gaussian state-space model via 
/// embedded HMM/ensemble MCMC methods.

#ifndef __LINEARENSEMBLE_H
#define __LINEARENSEMBLE_H

#include "main/applications/linear/linear.h"
#include "projects/ensemble/ensembleOld.h" // TODO! this file shouldn't depend on these
#include "projects/ensemble/ensembleNew.h" // TODO! this file shouldn't depend on these!

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the algorithm
////////////////////////////////////////////////////////////////////////////////

/// Holds some additional auxiliary parameters for the original EHMM algorithms.
class EnsembleOldParameters
{
public:
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& par)
  {
    baseMean_ = par(arma::span(0,par.size()-3));
    baseVar_ = par(par.size()-2);
    dispersionFactor_ = par(par.size()-1);
  }
  /// Determines the stationary mean and covariance matrix
  void setStationaryParameters(const unsigned int dimX, const arma::mat A, const double b)
  {
    arma::mat diagMat1;
    arma::mat diagMat2;
    
    diagMat1.eye(dimX, dimX);
    diagMat2.eye(dimX*dimX, dimX*dimX);
    stationaryMean_.zeros(dimX);
    stationaryCovarianceMatrix_= convertArmaVecToArmaMat(arma::inv(diagMat2 - arma::kron(A, A)) * b * b * arma::vectorise(diagMat1), dimX, dimX);
  }
  
  arma::colvec baseMean_; // mean of the the "ensemble measure" rho
  double baseVar_; // variance of each independent component in the "ensemble measure" rho
  arma::colvec stationaryMean_; // stationary mean of the latent chain
  arma::mat stationaryCovarianceMatrix_; // stationary covariance matrix of the latent chain
  double dispersionFactor_; // factor multiplying the variance of rho if prop_ == ENSEMBLE_OLD_PROPOSAL_ALTERNATE.
};

/// Holds some additional auxiliary parameters for the alternative EHMM algorithms.
class EnsembleNewParameters
{
public:
  // Empty
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& par)
  {
    particleProposalScale_ = par(0);
    autoregressiveCorrelationParameter_ = par(1);
    additionalScaleFactorRW_ = par(2);
    additionalScaleFactorAR_ = par(3);
  }
  
  double particleProposalScale_; // scale (i.e. std. deviation) of the random-walk proposal kernels used within the MH kernels $R_{\theta,t}$
  double autoregressiveCorrelationParameter_; // correlation parameter used for the autoregressive updates
  double additionalScaleFactorRW_, additionalScaleFactorAR_; // additionally scales the standard deviation of the AR and RW proposal kernels
};

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the EnsembleOld class
///////////////////////////////////////////////////////////////////////////////

/// Samples a single particle from the ensemble measure rho.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters> 
Particle EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters>::sampleFromProposal(const unsigned int t)
{
  if (prop_ == ENSEMBLE_OLD_PROPOSAL_DEFAULT) 
  {
    return ensembleOldParameters_.baseMean_ + std::sqrt(ensembleOldParameters_.baseVar_) *   arma::randn<arma::colvec>(model_.getModelParameters().getDimLatentVariable());
  }
  else if (prop_ == ENSEMBLE_OLD_PROPOSAL_ALTERNATE) 
  {
    return ensembleOldParameters_.baseMean_ + std::sqrt(ensembleOldParameters_.baseVar_ * ensembleOldParameters_.dispersionFactor_) *   arma::randn<arma::colvec>(model_.getModelParameters().getDimLatentVariable());  
  }
  else // i.e. if prop_ == ENSEMBLE_OLD_PROPOSAL_STATIONARY
  {
    if (t == 0)
    {
      ensembleOldParameters_.setStationaryParameters(model_.getModelParameters().getDimLatentVariable(), model_.getModelParameters().getA(), model_.getModelParameters().getB(0,0));
    }
    return ensembleOldParameters_.stationaryMean_ + arma::chol(ensembleOldParameters_.stationaryCovarianceMatrix_) *   arma::randn<arma::colvec>(model_.getModelParameters().getDimLatentVariable());
  }
}
/// Applies a Markov kernel which is invariant with respect to the ensemble measure rho.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters> 
Particle EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters>::applyKernel(const unsigned int t, const Particle& particleOld)
{
  // Here, we apply the trivial, perfectly mixing kernel which just generates IID samples from the target.
  if (prop_ == ENSEMBLE_OLD_PROPOSAL_DEFAULT) 
  {
    return ensembleOldParameters_.baseMean_ + std::sqrt(ensembleOldParameters_.baseVar_) *   arma::randn<arma::colvec>(model_.getModelParameters().getDimLatentVariable());
  }
  else if (prop_ == ENSEMBLE_OLD_PROPOSAL_ALTERNATE)
  {
    return ensembleOldParameters_.baseMean_ + std::sqrt(ensembleOldParameters_.baseVar_ * ensembleOldParameters_.dispersionFactor_) *   arma::randn<arma::colvec>(model_.getModelParameters().getDimLatentVariable());  
  }
  else if (prop_ == ENSEMBLE_OLD_PROPOSAL_STATIONARY) // i.e. if prop_ == ENSEMBLE_OLD_PROPOSAL_STATIONARY
  {
    if (t == 0)
    {
      ensembleOldParameters_.setStationaryParameters(model_.getModelParameters().getDimLatentVariable(), model_.getModelParameters().getA(), model_.getModelParameters().getB(0,0));
    }
    return sampleFromProposal(t);
  } 
  else 
  {
    if (t == 0)
    {
      ensembleOldParameters_.setStationaryParameters(model_.getModelParameters().getDimLatentVariable(), model_.getModelParameters().getA(), model_.getModelParameters().getB(0,0));
    }
    arma::colvec particleProp = particleOld + std::sqrt(ensembleOldParameters_.baseVar_) * arma::randn<arma::colvec>(particleOld.size());
    
    double logAlpha = evaluateLogProposalDensity(t, particleProp) - evaluateLogProposalDensity(t, particleOld);
    
    
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      return particleProp;
    }
    else
    {
      return particleOld;
    }
  }
}
/// Evaluates the log-proposal density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters> 
double EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters>::evaluateLogProposalDensity
(
  const unsigned int t,
  const Particle& particle
)
{
  if (prop_ == ENSEMBLE_OLD_PROPOSAL_DEFAULT) 
  {
    return arma::as_scalar(gaussian::evaluateDensityMultivariate(particle, ensembleOldParameters_.baseMean_, ensembleOldParameters_.baseVar_, false, true));
  }
  else if (prop_ == ENSEMBLE_OLD_PROPOSAL_ALTERNATE) 
  {
    return arma::as_scalar(gaussian::evaluateDensityMultivariate(particle, ensembleOldParameters_.baseMean_, ensembleOldParameters_.baseVar_* ensembleOldParameters_.dispersionFactor_, false, true));
  }
  else 
  {
    return arma::as_scalar(gaussian::evaluateDensityMultivariate(particle, ensembleOldParameters_.stationaryMean_, ensembleOldParameters_.stationaryCovarianceMatrix_, false, true));
  }
}
/// Converts a particle path into the set of all latent variables in the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters> 
void EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters>::convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath)
{
  convertStdVecToArmaMat(particlePath, latentPath);
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class EnsembleOldParameters> 
void EnsembleOld<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, EnsembleOldParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  convertArmaMatToStdVec(latentPath, particlePath);
}




///////////////////////////////////////////////////////////////////////////////
/// Member functions of the EnsembleNew class
///////////////////////////////////////////////////////////////////////////////
/*
/// Samples particles at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::sampleFromRho
(
  const unsigned int t, 
  const unsigned int n,
  std::vector<Particle>& particlesNew,
  arma::uvec& parentIndices
)
{
  double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
  double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
  unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  arma::colvec mu(dimLatentVariable);
  
  parentIndices(n) = sampleInt(selfNormalisedWeights_);
  
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR) 
  {
    mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(n)];
    particlesNew[n] = gaussian::sampleMultivariate(1, mu, b2, false);
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {      
    double sigma = 1.0/(1.0/d2 + 1.0/b2);          
    mu = sigma*(1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(n)]);                   
    particlesNew[n] = gaussian::sampleMultivariate(1, mu, sigma, false);
  }
}*/

/// Computes a particle weight at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::computeLogParticleWeights
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t), model_.getModelParameters().getC() * particlesNew[n], model_.getModelParameters().getD(0,0), true, true));
    }
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL)
  {     
//     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t), model_.getModelParameters().getCA() * particlesFull_[t-1][parentIndicesFull_(n,t-1)], model_.getModelParameters().getOptWeightVar(), false, true));
    }
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF) // TODO
  {     
    if (t+1 < nSteps_)
    {
//       double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//       double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
      for (unsigned int n=0; n<getNParticles(); n++)
      {
        logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t+1), model_.getModelParameters().getCA() * particlesNew[n], model_.getModelParameters().getOptWeightVar(), false, true));
      }
    }
    else
    {
      // Do nothing here because the final-time incremental weights are unity.
    }
  }
}
/// Reparametrises particles at Step 0 to obtain the values of Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::determineGaussiansFromParticles
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew, 
  std::vector<Aux>& aux1
)
{
  arma::colvec mu(model_.getModelParameters().getDimLatentVariable());
//   double sigma;
  
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
  {
    double sigma = model_.getModelParameters().getB(0,0);
    for (unsigned int n=0; n<nParticles_; n++)
    {
      mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndicesFull_(n,t-1)];
      aux1[n] = (particlesNew[n] - mu) / sigma;
    }
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {     
//     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//     sigma = std::sqrt(1.0/(1.0/d2 + 1.0/b2));
    
    for (unsigned int n=0; n<nParticles_; n++)
    {
      mu = model_.getModelParameters().getOptPropVar() * (model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(t) + model_.getModelParameters().getInvBBTA()  * particlesFull_[t-1][parentIndicesFull_(n,t-1)]);
      aux1[n] = arma::inv(arma::chol(model_.getModelParameters().getOptPropVar())) * (particlesNew[n] - mu);
    }
  }
}
/*
/// Samples particles at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::sampleFromInitialRho
(
  const unsigned int n,
  std::vector<Particle>& particlesNew
)
{
  double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
  unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  arma::colvec mu(dimLatentVariable);
  
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR) 
  {
    particlesNew[n] = gaussian::sampleMultivariate(1, model_.getModelParameters().getM0(), model_.getModelParameters().getC0(), true);
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {      
    arma::mat sigma = arma::inv(1.0/d2 * arma::eye(dimLatentVariable, dimLatentVariable) + arma::inv(model_.getModelParameters().getC0()));        
    
    mu = sigma*(1.0/d2 * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());        
    particlesNew[n] = gaussian::sampleMultivariate(1, mu, sigma, false);
  }
}*/

/// Computes the incremental particle weights at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::computeLogInitialParticleWeights
(
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
  {
    for (unsigned int n=0; n<nParticles_; n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getC() * particlesNew[n], model_.getModelParameters().getD(0,0), true, true));
    }
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL)
  {   
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
/*    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getM0(), model_.getModelParameters().getC0(0,0) + d2, false, true));
    }  */ 
    logWeights += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getOptWeightInitialMean(), model_.getModelParameters().getOptWeightInitialVar(), false, true));
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    if (nSteps_ > 0)
    {
//       double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//       double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
      for (unsigned int n=0; n<nParticles_; n++)
      {
        logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(1), model_.getModelParameters().getCA() * particlesNew[n], model_.getModelParameters().getOptWeightVar(), false, true));
      }
    }
    logWeights += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getOptWeightInitialMean(), model_.getModelParameters().getOptWeightInitialVar(), false, true));
  }
}
/// Reparametrises the particles at Step t to obtain the value of 
/// Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::determineGaussiansFromInitialParticles
(
  const std::vector<Particle>& particlesNew, 
  std::vector<Aux>& aux1
)
{ 
//   std::cout << "converting to initial gaussians" << std::endl;
//   std::cout << "number of auxiliary variables before: " << aux1.size() << std::endl;
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
  {
    for (unsigned int n=0; n<nParticles_; n++)
    {
      aux1[n] = (particlesNew[n] - model_.getModelParameters().getM0()) / std::sqrt(model_.getModelParameters().getC0(0,0));
    }
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {     
    unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
//     double d2         = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//     double sigma      = d2 + 1.0 / (model_.getModelParameters().getC0(0,0));    
    arma::colvec mu(dimLatentVariable);
    
    for (unsigned int n=0; n<nParticles_; n++)
    {
      mu = model_.getModelParameters().getOptPropInitialVar() *(model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());   
      aux1[n] = arma::inv(arma::chol(model_.getModelParameters().getOptPropInitialVar())) * (particlesNew[n] - mu); // / std::sqrt(sigma);
    }
  }
}


/// Samples a particle from some proposal kernel.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
Particle EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::proposeParticle(const Particle particleOld)
{
  return particleOld + ensembleNewParameters_.particleProposalScale_ * ensembleNewParameters_.additionalScaleFactorRW_ * arma::randn<arma::colvec>(particleOld.size()); 
}
/// Evaluate log-density of proposal kernel for particle.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
double EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::evaluateLogProposalDensityParticle(const Particle particleNew, const Particle particleOld)
{
  return arma::as_scalar(gaussian::evaluateDensityMultivariate(particleNew, particleOld, ensembleNewParameters_.particleProposalScale_ * ensembleNewParameters_.additionalScaleFactorRW_, true, true));
}
/// Samples from rho_t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::sampleFromRho(const unsigned int t, const unsigned int n, std::vector<Particle>& particlesNew, arma::uvec& parentIndices, const arma::colvec& selfNormalisedWeights)
{
  parentIndices(n) = sampleInt(selfNormalisedWeights);
  
  double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//   double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
  unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  arma::colvec mu(dimLatentVariable);
  
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR) 
  {
    mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(n)];
    particlesNew[n] = gaussian::sampleMultivariate(1, mu, b2, false);
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {      
//     double sigma = 1.0/(1.0/d2 + 1.0/b2);          
    mu = model_.getModelParameters().getOptPropVar()*(model_.getModelParameters().getCTinvDDT()  * model_.getObservations().col(t) + model_.getModelParameters().getInvBBTA() * particlesFull_[t-1][parentIndices(n)]);                   
    particlesNew[n] = gaussian::sampleMultivariate(1, mu, model_.getModelParameters().getOptPropVar(), false);
  }
}
/// Samples from approximation of rho_t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::sampleFromRhoApproximation(const unsigned int t, const unsigned int n, std::vector<Particle>& particlesNew, arma::uvec& parentIndices, const arma::colvec& potentialProposalValues)
{
  parentIndices(n) = sampleInt(potentialProposalValues);
  
  double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//   double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
  unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  arma::colvec mu(dimLatentVariable);
  
//   if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR) 
//   {
    mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(n)];
    particlesNew[n] = gaussian::sampleMultivariate(1, mu, b2, false);
//   }
//   else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
//   {      
// //     double sigma = 1.0/(1.0/d2 + 1.0/b2);          
//     mu = model_.getModelParameters().getOptPropVar()*(model_.getModelParameters().getCTinvDDT()  * model_.getObservations().col(t) + model_.getModelParameters().getInvBBTA() * particlesFull_[t-1][parentIndices(n)]);                   
//     particlesNew[n] = gaussian::sampleMultivariate(1, mu, model_.getModelParameters().getOptPropVar(), false);
//   }
}
/// Samples from rho_0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::sampleFromInitialRho(const unsigned int n, std::vector<Particle>& particles)
{
//   double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//   unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR) 
  {
    particles[n] = gaussian::sampleMultivariate(1, model_.getModelParameters().getM0(), model_.getModelParameters().getC0(), true);
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {      
//     arma::mat sigma = arma::inv(1.0/d2 * arma::eye(dimLatentVariable, dimLatentVariable) + arma::inv(model_.getModelParameters().getC0()));        
    
    arma::colvec mu = model_.getModelParameters().getOptPropInitialVar()*(model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());        
    particles[n] = gaussian::sampleMultivariate(1, mu, model_.getModelParameters().getOptPropInitialVar(), false);
  }
}
/// Samples from approximation of rho_0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::sampleFromInitialRhoApproximation(const unsigned int n, std::vector<Particle>& particles)
{
//   double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//   unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  
//   if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR) 
//   {
    particles[n] = gaussian::sampleMultivariate(1, model_.getModelParameters().getM0(), model_.getModelParameters().getC0(), true);
//   }
//   else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
//   {      
// //     arma::mat sigma = arma::inv(1.0/d2 * arma::eye(dimLatentVariable, dimLatentVariable) + arma::inv(model_.getModelParameters().getC0()));        
//     
//     arma::colvec mu = model_.getModelParameters().getOptPropInitialVar()*(model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());        
//     particles[n] = gaussian::sampleMultivariate(1, mu, model_.getModelParameters().getOptPropInitialVar(), false);
//   }
}

/// Evaluates the unnormalised(!) log-density of rho_t
/// Actually, this is not rho_t but rather just rho divided by the weight.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
double EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::evaluateLogDensityRho(const unsigned int t, const Particle& particle, const unsigned int parentIndex, const arma::colvec& logUnnormalisedWeights)
{
  
  unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  arma::colvec mu(dimLatentVariable);
  double logDensity = logUnnormalisedWeights(parentIndex);
  
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
  {
    double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
    mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndex];
    logDensity += arma::as_scalar(gaussian::evaluateDensityMultivariate(particle, mu, b2, false, true));
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {   
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//     double sigma = 1.0/(1.0/d2 + 1.0/b2);
    mu = model_.getModelParameters().getOptPropVar()*(model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(t) + model_.getModelParameters().getInvBBTA() * particlesFull_[t-1][parentIndex]);                      
    logDensity += arma::as_scalar(gaussian::evaluateDensityMultivariate(particle, mu, model_.getModelParameters().getOptPropVar(), false, true));
  }
  return logDensity;
}
/// Evaluates the unnormalised(!) log-density of rho_1.
/// Actually, this is not rho_1 but rather just rho divided by the weight.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
double EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::evaluateLogDensityInitialRho
(const Particle& particle)
{
  double logDensity = 0;
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
  {
    logDensity = arma::as_scalar(gaussian::evaluateDensityMultivariate(particle, model_.getModelParameters().getM0(), model_.getModelParameters().getC0(), false, true));
  }
  else if (prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF)
  {   
//     unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    
//     arma::mat sigma = arma::inv(1.0/d2 * arma::eye(dimLatentVariable, dimLatentVariable) + arma::inv(model_.getModelParameters().getC0()));        
    arma::colvec mu = model_.getModelParameters().getOptPropInitialVar()*(model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());        
    
    logDensity = arma::as_scalar(gaussian::evaluateDensityMultivariate(particle, mu,  model_.getModelParameters().getOptPropInitialVar(), false, true));
  }
  return logDensity;
}

/// Applies rho_t-invariant kernel.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::applyKernel(const unsigned int t, const unsigned int n, const unsigned int m, std::vector<Particle>& particles, arma::uvec& parentIndices, const arma::colvec& logUnnormalisedWeights, const arma::colvec& selfNormalisedWeights, const arma::colvec& potentialProposalValues)
{
  if (local_ == ENSEMBLE_NEW_LOCAL_NONE) // note that this leads to a standard SMC algorithm
  {
    sampleFromRho(t, n, particles, parentIndices, selfNormalisedWeights);
    acceptanceRates_(t) += 1.0/nParticles_;
  }
  else if (local_ == ENSEMBLE_NEW_LOCAL_RANDOM_WALK)
  { 
    unsigned int parentIndexProp = sampleInt(potentialProposalValues);
    Particle particleProp = proposeParticle(particles[m]); // propose particle from a Gaussian random-walk kernel
//     Particle particleProp = particles[m] + model_.getModelParameters().getA() * (particlesFull_[t-1][parentIndexProp] - particlesFull_[t-1][parentIndices(m)]);
    
    double logAlpha = 
      evaluateLogDensityRho(t, particleProp, parentIndexProp, logUnnormalisedWeights)
      - evaluateLogDensityRho(t, particles[m], parentIndices(m), logUnnormalisedWeights)
      + std::log(potentialProposalValues(parentIndices(m)))
      - std::log(potentialProposalValues(parentIndexProp)); // note that the proposal kernel for the particles is symmetric so that its density cancels out
    
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      parentIndices(n) = parentIndexProp;
      particles[n] = particleProp;
      acceptanceRates_(t) += 1.0/nParticles_;
    }
    else
    {
      parentIndices(n) = parentIndices(m);
      particles[n] = particles[m];
    }
  }
//   else if (local_ == ENSEMBLE_NEW_LOCAL_AUTOREGRESSIVE)
//   { 
//     unsigned int dimX = model_.getModelParameters().getDimLatentVariable();
//     double rho = ensembleNewParameters_.autoregressiveCorrelationParameter_;
//     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
// //     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//       
//     // Propose new parent index
//     unsigned int parentIndexProp = sampleInt(selfNormalisedWeights);
//     
//     // Propose new particle
//     
//     // NEW: here, we are always using the mean and covariance matrix of $f_\theta$:
//     double sigma = b2; // variance of the proposal;
//     arma::colvec muProp = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndexProp];
//     arma::colvec mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(m)];
//     
//     // OLD: here, we were using the mean and covariance matrix of $q_{\theta,t}$:
//     /*
//     double sigma;
//     arma::colvec mu, muProp;
//     if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
//     {
//       sigma = b2; // variance of the proposal;
//       muProp = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndexProp];
//       mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(m)];
//     } 
//     else // i.e. if prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF or prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL
//     {
//       sigma = 1.0/(1.0/d2 + 1.0/b2); // variance of the proposal;
//       muProp = sigma*(1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesFull_[t-1][parentIndexProp]);
//       mu = sigma*(1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(m)]);
//     }
//     */
//     Particle particleProp = muProp + rho * (particles[m] - muProp) + std::sqrt(1-rho*rho) * std::sqrt(sigma) * ensembleNewParameters_.additionalScaleFactorAR_ * arma::randn<arma::colvec>(dimX);
//  
//     // Evaluate log-acceptance probability
//     double logAlpha = 
//       evaluateLogDensityRho(t, particleProp, parentIndexProp, logUnnormalisedWeights)
//       - evaluateLogDensityRho(t, particles[m], parentIndices(m), logUnnormalisedWeights)
//       + std::log(selfNormalisedWeights(parentIndices(m)))
//       - std::log(selfNormalisedWeights(parentIndexProp))
//       + arma::as_scalar(gaussian::evaluateDensityMultivariate(particles[m], mu + rho * (particleProp - mu), (1-rho*rho)*ensembleNewParameters_.additionalScaleFactorAR_*ensembleNewParameters_.additionalScaleFactorAR_*sigma, false, true))
//       - arma::as_scalar(gaussian::evaluateDensityMultivariate(particleProp, muProp + rho * (particles[m] - muProp), (1-rho*rho)*ensembleNewParameters_.additionalScaleFactorAR_*ensembleNewParameters_.additionalScaleFactorAR_*sigma, false, true));
//     
//     if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
//     {
//       parentIndices(n) = parentIndexProp;
//       particles[n] = particleProp;
//       acceptanceRates_(t) += 1.0/nParticles_;
//     }
//     else
//     {
//       parentIndices(n) = parentIndices(m);
//       particles[n] = particles[m];
//     }
//   }
//   else if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK)
//   {
//     unsigned int k = sortedParentIndexTemp_;
//     unsigned int kProp = proposeIndex(k);
//     
//     unsigned int parentIndexProp = sortedIndices_(kProp);
//     Particle particleProp = proposeParticle(particles[m]); // propose particle from a Gaussian random-walk kernel
//     
//     double logAlpha = 
//       evaluateLogDensityRho(t, particleProp, parentIndexProp, logUnnormalisedWeights)
//       - evaluateLogDensityRho(t, particles[m], parentIndices(m), logUnnormalisedWeights)
//       + evaluateLogProposalDensityIndex(k, kProp)
//       - evaluateLogProposalDensityIndex(kProp, k); // note that the proposal kernel for the particles is symmetric so that its density cancels out
//     
//     if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
//     {
//       parentIndices(n) = parentIndexProp;
//       particles[n] = particleProp;
//       sortedParentIndexTemp_ = kProp;
//       acceptanceRates_(t) += 1.0/nParticles_;
//     }
//     else
//     {
//       parentIndices(n) = parentIndices(m);
//       particles[n] = particles[m];
//     }
//   }
//   else if (local_ == ENSEMBLE_NEW_LOCAL_HILBERT_AUTOREGRESSIVE)
//   { 
//     unsigned int dimX = model_.getModelParameters().getDimLatentVariable();
//     double rho = ensembleNewParameters_.autoregressiveCorrelationParameter_;
//     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
// //     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//       
//     // Propose new parent index
//     unsigned int k = sortedParentIndexTemp_;
//     unsigned int kProp = proposeIndex(k);
//     unsigned int parentIndexProp = sortedIndices_(kProp);
//    
//     // Propose new particle
//     
//     // NEW: here, we are always using the mean and covariance matrix of $f_\theta$:
//     double sigma = b2; // variance of the proposal;
//     arma::colvec muProp = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndexProp];
//     arma::colvec mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(m)];
//     
//     // OLD: here, we were using the mean and covariance matrix of $q_{\theta,t}$:
//     /*
//     double sigma;
//     arma::colvec mu, muProp;
//     if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
//     {
//       sigma = b2; // variance of the proposal;
//       muProp = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndexProp];
//       mu = model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(m)];
//     } 
//     else // i.e. if prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF or prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL
//     {
//       sigma = 1.0/(1.0/d2 + 1.0/b2); // variance of the proposal;
//       muProp = sigma*(1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesFull_[t-1][parentIndexProp]);
//       mu = sigma*(1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesFull_[t-1][parentIndices(m)]);
//     }
//     */
//     Particle particleProp = muProp + rho * (particles[m] - muProp) + std::sqrt(1-rho*rho) *ensembleNewParameters_.additionalScaleFactorAR_* std::sqrt(sigma) * arma::randn<arma::colvec>(dimX);
//     
//     // Evaluate log-acceptance probability
//     double logAlpha = 
//       evaluateLogDensityRho(t, particleProp, parentIndexProp, logUnnormalisedWeights)
//       - evaluateLogDensityRho(t, particles[m], parentIndices(m), logUnnormalisedWeights)
//       + evaluateLogProposalDensityIndex(k, kProp)
//       - evaluateLogProposalDensityIndex(kProp, k)
//       + arma::as_scalar(gaussian::evaluateDensityMultivariate(particles[m], mu + rho * (particleProp - mu), (1-rho*rho)*ensembleNewParameters_.additionalScaleFactorAR_*ensembleNewParameters_.additionalScaleFactorAR_*sigma, false, true))
//       - arma::as_scalar(gaussian::evaluateDensityMultivariate(particleProp, muProp + rho * (particles[m] - muProp), (1-rho*rho)*ensembleNewParameters_.additionalScaleFactorAR_*ensembleNewParameters_.additionalScaleFactorAR_*sigma, false, true));
//     
//     if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
//     {
//       parentIndices(n) = parentIndexProp;
//       particles[n] = particleProp;
//       sortedParentIndexTemp_ = kProp;
//       acceptanceRates_(t) += 1.0/nParticles_;
//     }
//     else
//     {
//       parentIndices(n) = parentIndices(m);
//       particles[n] = particles[m];
//     }
//   }
}
/// Applies rho_0-invariant kernel.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::applyInitialKernel
(
  const unsigned int n, 
  const unsigned int m, 
  std::vector<Particle>& particles
)
{
  if (local_ == ENSEMBLE_NEW_LOCAL_NONE)
  {
    sampleFromInitialRho(n, particles);
    acceptanceRates_(0) += 1.0/nParticles_;
  }
  else if (local_ == ENSEMBLE_NEW_LOCAL_RANDOM_WALK || local_ == ENSEMBLE_NEW_LOCAL_HILBERT_RANDOM_WALK)
  {
    Particle particleProp = proposeParticle(particles[m]); // propose particle from a Gaussian random-walk kernel
    
    double logAlpha = evaluateLogDensityInitialRho(particleProp) - evaluateLogDensityInitialRho(particles[m]);
    // note that the proposal kernel for the particles is symmetric so that its density cancels out
    
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      particles[n] = particleProp;
      acceptanceRates_(0) += 1.0/nParticles_;
    }
    else
    {
      particles[n] = particles[m];
    }
  }
//   else // i.e. if we use autoregressive proposals for the states
//   {
//     unsigned int dimX = model_.getModelParameters().getDimLatentVariable();
//     double rho = ensembleNewParameters_.autoregressiveCorrelationParameter_;
// //     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
// //     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
//       
//     // Propose new particle
//     
//     // NEW: here, we are always using the mean and covariance matrix of $f_\theta$:
//     arma::mat sigma = model_.getModelParameters().getC0(); // variance of the proposal
//     arma::colvec muProp = model_.getModelParameters().getM0();
//     
//     // OLD: here, we were using the mean and covariance matrix of $q_{\theta,t}$:
//     /*
//     arma::mat sigma;
//     arma::colvec muProp;
//     if (prop_ == ENSEMBLE_NEW_PROPOSAL_PRIOR)
//     {
//       sigma = model_.getModelParameters().getC0(); // variance of the proposal
//       muProp = model_.getModelParameters().getM0();
//     } 
//     else // i.e. if prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF or prop_ == ENSEMBLE_NEW_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL
//     {
//       sigma = arma::inv(1.0/d2 * arma::eye(dimX, dimX) + arma::inv(model_.getModelParameters().getC0())); // variance of the proposal;
//       muProp = sigma*(1.0/d2 * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());    
//     }
//     */
//     
//     Particle particleProp = muProp + rho * (particles[m] - muProp) + std::sqrt(1-rho*rho) *ensembleNewParameters_.additionalScaleFactorAR_* arma::chol(sigma) * arma::randn<arma::colvec>(dimX);
//  
//     // Evaluate log-acceptance probability
//     double logAlpha = evaluateLogDensityInitialRho(particleProp) - evaluateLogDensityInitialRho(particles[m])
//       + arma::as_scalar(gaussian::evaluateDensityMultivariate(particles[m], muProp + rho * (particleProp - muProp), (1-rho*rho)*ensembleNewParameters_.additionalScaleFactorAR_*ensembleNewParameters_.additionalScaleFactorAR_*sigma, false, true))
//       - arma::as_scalar(gaussian::evaluateDensityMultivariate(particleProp, muProp + rho * (particles[m] - muProp), (1-rho*rho)*ensembleNewParameters_.additionalScaleFactorAR_*ensembleNewParameters_.additionalScaleFactorAR_*sigma, false, true));
//     
//     if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
//     {
//       particles[n] = particleProp;
//       acceptanceRates_(0) += 1.0/nParticles_;
//     }
//     else
//     {
//       particles[n] = particles[m];
//     }
//   }
}
/// Computes (part of the) unnormalised "future" target density needed for 
/// backward or ancestor sampling.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters>
double EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::logDensityUnnormalisedTarget
(
  const unsigned int t,
  const Particle& particle
)
{
  // For conditionally IID models, we actually only need to return 0;
  // for state-space models, we need to evaluate the log-transition density;
  // for kth-order Markov models, we need to evaluate the log-unnormalised
  // target density k steps in the future.
  
  if (prop_ == ENSEMBLE_NEW_PROPOSAL_FA_APF) 
  {
//     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    
    return model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particle) - arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t+1), model_.getModelParameters().getCA() * particle, model_.getModelParameters().getOptWeightVar(), false, true));
    
  }
  else 
  {
    return model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particle);
  }
}

/// Converts a particle path into the set of all latent variables in the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath)
{
  convertStdVecToArmaMat(particlePath, latentPath);
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class EnsembleNewParameters> 
void EnsembleNew<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EnsembleNewParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  convertArmaMatToStdVec(latentPath, particlePath);
}

#endif
