/// \file
/// \brief Some definitions of members common all euclidean state-space models.
///
/// This file contains the functions for implementing various
/// classes for any kind of state-space model in which both the latent states
/// and the observations can be represented as arma::colvec's.

#ifndef __MULTIVARIATE_H
#define __MULTIVARIATE_H

#include "main/templates/dynamic/stateSpace/stateSpace.h"
#include "main/algorithms/smc/default/single.h"

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
 
//     std::cout << "dimX: " << modelParameters_.getDimLatentVariable() << std::endl;
//     std::cout << "dimY: " << modelParameters_.getDimObservation() << std::endl;
//       std::cout << "nObservations_: " << std::endl;
//     std::cout << nObservations_ << std::endl;
  
  observations_.set_size(modelParameters_.getDimObservation(), nObservations_);
  latentPath_.set_size(modelParameters_.getDimLatentVariable(), nObservations_);


  latentPath_.col(0) = sampleFromInitialDistribution();
  sampleFromObservationEquation(0, observations_, latentPath_.col(0));
  for (unsigned int t=1; t<observations_.n_cols; ++t)
  {
    latentPath_.col(t) = sampleFromTransitionEquation(t, latentPath_.col(t-1));
    sampleFromObservationEquation(t, observations_, latentPath_.col(t));
  }
  
  /*
  arma::mat x, y;
  
  y.set_size(modelParameters_.getDimObservation(), nObservations_);
  x.set_size(modelParameters_.getDimLatentVariable(), nObservations_);
  
  arma::colvec XNew, XOld;
  
  sampleFromInitialDistribution(XNew);
  sampleFromObservationEquation(0, y, XNew);
  XOld = XNew;
  x.col(0) = XNew;
  for (unsigned int t=1; t<observations_.n_cols; ++t)
  {
    sampleFromTransitionEquation(t, XNew, XOld);
    sampleFromObservationEquation(t, y, XNew);
    XOld = XNew;
    x.col(t) = XNew;
  }
  observations_ = y;
  latentPath_ = x;
  */
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihood(const LatentPath& x)
{
  double logLike = evaluateLogInitialDensity(x.col(0)) + 
                   evaluateLogObservationDensity(0, x.col(0));
                   
  for (unsigned int t=1; t<observations_.n_cols; ++t)
  {
    logLike += evaluateLogTransitionDensity(t, x.col(t), x.col(t-1)) + 
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
  // latentPath = arma::conv_to<arma::mat>::from(particlePath);
  
  convertStdVecToArmaMat(particlePath, latentPath);
  

//   unsigned int dimX = particlePath[0].n_rows;
//   unsigned int T = particlePath.size();
//   
//   latentPath.set_size(dimX, T);
//   for (unsigned int t=0; t<T; t++)
//   {
//     latentPath.col(t) = particlePath[t];
//   }
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  
  //particlePath = arma::conv_to<std::vector<arma::colvec>>::from(latentPath);
    
  convertArmaMatToStdVec(latentPath, particlePath);
  /*
  
  unsigned int T = latentPath.n_cols;

  particlePath.resize(T);
  for (unsigned int t=0; t<T; t++)
  {
    particlePath[t] = latentPath.col(t);
  }*/

}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Mcmc class
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Optim class
///////////////////////////////////////////////////////////////////////////////
#endif
