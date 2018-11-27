/// \file
/// \brief Some definitions of members of the Smc class common all euclidean state-space models.
///
/// This file contains the functions for implementing the smc
/// class for any kind of state-space model in which both the latent states
/// and the observations can be represented as arma::colvec's.

#ifndef __MULTIVARIATE_H
#define __MULTIVARIATE_H

#include "main/algorithms/smc/default/dynamic/stateSpace/stateSpace.h"
#include "main/algorithms/smc/default/single.h"


/// Holds a single particle
typedef arma::colvec Particle;

/// Holds (some of the) Gaussian auxiliary variables generated as part of 
/// the SMC algorithm.
/// Can be used to hold the normal random variables used for implementing
/// correlated pseudo-marginal kernels. Otherwise, this may not need to 
/// be used.
typedef arma::colvec Aux;

/// Proposes new values for the Gaussian auxiliary variables
/// using a Crank--Nicolson proposal.
template <class Aux> 
void AuxFull<Aux>::addCorrelatedGaussianNoise(const double correlationParameter, Aux& aux)
{
  aux = correlationParameter * aux + sqrt(1 - std::pow(correlationParameter, 2.0)) * arma::randn<arma::colvec>(aux.n_rows);
}

/// Converts a particle path into the set of all latent variables in the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath)
{
  latentPath = arma::conv_to<arma::mat>::from(particlePath);
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  particlePath = arma::conv_to<std::vector<arma::colvec>>::from(latentPath);
}
#endif
