/// \file
/// \brief Some definitions of members of the Model class common all univariate static models.
///
/// This file contains the functions for implementing the Model
/// class for any kind of static model in which both the latent states
/// and the observations can be represented as double's.

#ifndef __UNIVARIATE_H
#define __UNIVARIATE_H

#include "main/templates/static/static.h"
#include "main/algorithms/smc/default/single.h"

///////////////////////////////////////////////////////////////////////////////
/// Some types
///////////////////////////////////////////////////////////////////////////////

/// Holds a single set of latent variables (e.g. those associated with a single
/// time step or single observation) in a state-space model.
typedef double LatentVariable;

/// Holds all latent variables:
typedef arma::colvec LatentPath;

/// Holds all reparametrised latent variables:
typedef arma::colvec LatentPathRepar;

/// Holds all observations.
typedef arma::colvec Observations;

/// Holds a single particle
typedef double Particle;

/// Holds (some of the) Gaussian auxiliary variables generated as part of 
/// the SMC algorithm.
/// Can be used to hold the normal random variables used for implementing
/// correlated pseudo-marginal kernels. Otherwise, this may not need to 
/// be used.
typedef double Aux;

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Aux class
///////////////////////////////////////////////////////////////////////////////

/// Proposes new values for the Gaussian auxiliary variables
/// using a Crank--Nicolson proposal.
template <class Aux> 
void AuxFull<Aux>::addCorrelatedGaussianNoise(const double correlationParameter, Aux& aux)
{
  aux = correlationParameter * aux + std::sqrt(1 - std::pow(correlationParameter, 2.0)) * arma::randn();
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Model class
///////////////////////////////////////////////////////////////////////////////

/// Simulates observations from the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::simulateData(const arma::colvec& extraParameters)
{
  observations_.set_size(nObservations_);
  latentPath_.set_size(nObservations_);
  for (unsigned int t=0; t<observations_.n_rows; ++t)
  {
    latentPath_(t) = sampleFromLatentPrior(t);
    sampleFromObservationEquation(t, observations_, latentPath_(t));
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
  for (unsigned int t=0; t<observations_.n_rows; ++t)
  {
    logLike += evaluateLogLatentPriorDensity(t, x(t)) + 
               evaluateLogObservationDensity(t, x(t));
               
    if (!std::isfinite(logLike))
    {
      break;
    }
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
  latentPath = arma::conv_to<arma::colvec>::from(particlePath);
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  particlePath = arma::conv_to<std::vector<double>>::from(latentPath);
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Mcmc class
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Optim class
///////////////////////////////////////////////////////////////////////////////

#endif
