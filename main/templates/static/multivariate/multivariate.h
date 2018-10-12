/// \file
/// \brief Some definitions of member functions common all euclidean static models.
///
/// This file contains the functions for implementing various classes 
/// for any kind of static model in which both the latent states
/// and the observations can be represented as arma::colvec's.

#ifndef __MULTIVARIATE_H
#define __MULTIVARIATE_H

#include "templates/static/static.h"
#include "smc/default/single.h"

///////////////////////////////////////////////////////////////////////////////
/// Some types
///////////////////////////////////////////////////////////////////////////////

/// Holds a single particle
typedef arma::colvec LatentVariable;

/// Holds all latent variables:
typedef arma::mat LatentPath;

/// Holds all reparametrised latent variables:
typedef arma::mat LatentPathRepar;

/// Holds all observations.
typedef arma::mat Observations;

/// Holds a single particle
typedef arma::colvec Particle;

/// Holds (some of the) Gaussian auxiliary variables generated as part of 
/// the SMC algorithm.
/// Can be used to hold the normal random variables used for implementing
/// correlated pseudo-marginal kernels. Otherwise, this may not need to 
/// be used.
typedef arma::colvec Aux;

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Aux class
///////////////////////////////////////////////////////////////////////////////

/// Proposes new values for the Gaussian auxiliary variables
/// using a Crank--Nicolson proposal.
template <class Aux> 
void AuxFull<Aux>::addCorrelatedGaussianNoise(const double correlationParameter, Aux& aux)
{
  aux = correlationParameter * aux + sqrt(1 - std::pow(correlationParameter, 2.0)) * arma::randn<arma::colvec>(aux.n_rows);
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Model class
///////////////////////////////////////////////////////////////////////////////

/// Simulates observations from the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::simulateData(const arma::colvec& extraParameters)
{
  observations_.set_size(modelParameters.getDimLatentVariable(), nObservations_);
  latentPath_.set_size(modelParameters.getDimLatentObservation(), nObservations_);
  for (unsigned int t=0; t<observations_.n_cols; ++t)
  {
    latentPath_.col(t) = sampleFromLatentPrior(t);
    sampleFromObservationEquation(t, observation_, latentPath_.col(t));
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
  for (unsigned int t=0; t<observations_.n_cols; ++t)
  {
    logLike += evaluateLogLatentPriorDensity(t, x.col(t)) + 
               evaluateLogObservationDensity(t, x.col(t));
  }
  return logLike;
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Smc class
///////////////////////////////////////////////////////////////////////////////

/// Converts a particle path into the set of all latent variables in the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath)
{
   convertStdVecToArmaMat(particlePath, latentPath);
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  convertArmaMatToStdVec(latentPath, particlePath);
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Mcmc class
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Optim class
///////////////////////////////////////////////////////////////////////////////
#endif
