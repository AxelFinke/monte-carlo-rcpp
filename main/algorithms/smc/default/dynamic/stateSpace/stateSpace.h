/// \file
/// \brief Some definitions of members of the Smc class common for all state-space models.
///
/// This file contains the functions for implementing the Smc
/// class for any kind of state-space model

#ifndef __STATESPACE_H
#define __STATESPACE_H

#include "smc/default/dynamic/dynamic.h"

/// Computes (part of the) unnormalised "future" target density needed for 
/// backward or ancestor sampling.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::logDensityUnnormalisedTarget
(
  const unsigned int t,
  const Lat& particle,
  const std::vector<Lat>& particlePath
)
{
  // For conditionally IID models, we actually only need to return 0;
  // for state-space models, we need to evaluate the log-transition density;
  // for kth-order Markov models, we need to evaluate the log-unnormalised
  // target density k steps in the future.
  
  return model_.evaluateLogTransitionDensity(t+1, particlePath[t+1], particle);
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
  if (t > 0)
  { 
    unsigned int singleParentIndex = parentIndicesFull_(singleParticleIndex, t-1);
    
    model_.addGradLogTransitionDensity(t, particlesFull_[t][singleParticleIndex], 
                                       particlesFull_[t-1][singleParentIndex], gradientEstimate);
  } 
  else
  {
    model_.addGradLogInitialDensity(t, particlesFull_[t][singleParticleIndex], gradientEstimate);
  }
  model_.addGradLogObservationDensity(t, particlesFull_[t][singleParticleIndex], gradientEstimate);
}
#endif
