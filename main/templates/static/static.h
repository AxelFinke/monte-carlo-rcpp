/// \file
/// \brief Some definitions of members of the Model and Smc class common all static models.
///
/// This file contains the functions for implementing the Model and Smc
/// class for any kind of static model, i.e. for any model in which 
/// the latent variables are conditionally IID given the static parameters.

#ifndef __STATIC_H
#define __STATIC_H

#include "main/model/Model.h"
#include "main/algorithms/smc/Smc.h"


///////////////////////////////////////////////////////////////////////////////
/// Some types
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Aux class
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Model class
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Smc class
///////////////////////////////////////////////////////////////////////////////

/// Computes (part of the) unnormalised "future" target density needed for 
/// backward or ancestor sampling.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::logDensityUnnormalisedTarget
(
  const unsigned int t,
  const Particle& particle
)
{
  // For conditionally IID models, we actually only need to return 0;
  return 0;
}
/// Updates the gradient estimate for a particular SMC step.
/// We need to supply the Step-t component of the gradient of the 
/// log-unnormalised target density here, i.e. the sum of the gradients 
/// of the transition density and observation density, in the case of 
/// state-space models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::updateGradientEstimate
(
  const unsigned int t, 
  const unsigned int singleParticleIndex,
  arma::colvec& gradientEstimate
)
{
  model_.addGradLogLatentPriorDensity(t, particlesFull_[t][singleParticleIndex], gradientEstimate);
  model_.addGradLogObservationDensity(t, particlesFull_[t][singleParticleIndex], gradientEstimate);
}
/// Samples particles at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleInitialParticles
(
  std::vector<Particle>& particlesNew
)
{
  sampleParticles(0, particlesNew);
}
/// Samples particles at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleParticles
(
  const unsigned int t,
  std::vector<Particle>& particlesNew,
  const std::vector<Particle>& particlesOld
)
{
  sampleParticles(t, particlesNew);
}
/// Computes the incremental particle weights at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogInitialParticleWeights
(
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{
  computeLogParticleWeights(0, particlesNew, logWeights);
}
/// Computes the incremental particle weights at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogParticleWeights
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew,
  const std::vector<Particle>& particlesOld,
  arma::colvec& logWeights
)
{
  computeLogParticleWeights(t, particlesNew, logWeights);
}
/// Reparametrises the particles at Step 0 to obtain the value of Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromInitialParticles
(
  const std::vector<Particle>& particlesNew, 
  std::vector<Aux>& aux1
)
{
  determineGaussiansFromParticles(0, particlesNew, aux1);
}
/// Reparametrises the particles at Step t to obtain the value of Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromParticles
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew,  
  const std::vector<Particle>& particlesOld,  
  std::vector<Aux>& aux1
)
{
  determineGaussiansFromParticles(t, particlesNew, aux1);
}
/// Reparametrises Gaussians at Step 0 to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineInitialParticlesFromGaussians
(
  std::vector<Particle>& particlesNew,  
  const std::vector<Aux>& aux1
)
{
  determineParticlesFromGaussians(0, particlesNew, aux1);
}
/// Reparametrises Gaussians at Step t to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineParticlesFromGaussians
(
  const unsigned int t,
  std::vector<Particle>& particlesNew,  
  const std::vector<Particle>& particlesOld,  
  const std::vector<Aux>& aux1
)
{
  determineParticlesFromGaussians(t, particlesNew, aux1);
}
/// Calculates a fixed-lag smoothing approximation of the gradient
// TODO: simplify this!
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runFixedLagSmoothing(arma::colvec& gradientEstimate)
{
//   if (gradientEstimate.size() != model_.getDimTheta())
//   {
//     gradientEstimate.set_size(model_.getDimTheta());
//   }
//   gradientEstimate.zeros();
//   model_.addGradLogPriorDensity(gradientEstimate);
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
// TODO: simplify this!
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

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Mcmc class
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Optim class
///////////////////////////////////////////////////////////////////////////////

#endif
