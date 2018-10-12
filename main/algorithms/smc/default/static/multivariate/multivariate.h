/// \file
/// \brief Some definitions of members of the Model class common all euclidean static models.
///
/// This file contains the functions for implementing the Model
/// class for any kind of static model in which both the latent states
/// and the observations can be represented as arma::colvec's.

#ifndef __MULTIVARIATE_H
#define __MULTIVARIATE_H

#include "model/default/static/static.h"

/// Holds a single particle
typedef arma::colvec Particle;

/// Holds a single particle
typedef arma::colvec LatentVariable;

/// Holds all latent variables:
typedef arma::mat LatentPath;

/// Holds all reparametrised latent variables:
typedef arma::mat LatentPathRepar;

/// Holds (some of the) Gaussian auxiliary variables generated as part of 
/// the SMC algorithm.
/// Can be used to hold the normal random variables used for implementing
/// correlated pseudo-marginal kernels. Otherwise, this may not need to 
/// be used.
typedef arma::colvec Aux;

/// Holds all observations.
typedef arma::mat Observations;

/// Proposes new values for the Gaussian auxiliary variables
/// using a Crank--Nicolson proposal.
template <class Aux> 
void AuxFull<Aux>::addCorrelatedGaussianNoise(const double correlationParameter, Aux& aux)
{
  aux = correlationParameter * aux + sqrt(1 - std::pow(correlationParameter, 2.0)) * arma::randn<arma::colvec>(aux.n_rows);
}

/// Simulates observations from the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::simulateData()
{
  observations_.set_size(dimLatentVariable_, nObservations_);
  latentPath_.set_size(dimObservation_, nObservations_);
  for (unsigned int t=0; t<nObservations_; ++t)
  {
    sampleFromLatentPrior(t, latentPath_.col(t));
    sampleFromObservationEquation(t, observation_.col(t), latentPath_.col(t));
  }
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihood
(
  const LatentPath& x
)
{
  double logLike = 0.0;
  for (unsigned int t=0; t<nObservations_; ++t)
  {
    logLike += evaluateLogLatentPriorDensity(t, x.col(t)) + 
               evaluateLogObservationDensity(t, x.col(t));
  }
  return logLike;
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
