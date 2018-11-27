/// \file
/// \brief Some definitions of members of various classes, common for all state-space models.
///
/// This file contains the functions for implementing various classes
/// for any kind of state-space model

#ifndef __STATESPACE_H
#define __STATESPACE_H

#include "main/templates/dynamic/dynamic.h"

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

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Mcmc class
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Optim class
///////////////////////////////////////////////////////////////////////////////

#endif
