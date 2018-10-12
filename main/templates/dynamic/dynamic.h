/// \file
/// \brief Some definitions of members of various classes, common for all state-space models.
///
/// This file contains the functions for implementing various classes
/// for any kind of state-space model

#ifndef __DYNAMIC_H
#define __DYNAMIC_H

#include "model/Model.h"
#include "smc/Smc.h"

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

/// Calculates a fixed-lag smoothing approximation of the gradient
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runFixedLagSmoothing(arma::colvec& gradientEstimate)
{
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
