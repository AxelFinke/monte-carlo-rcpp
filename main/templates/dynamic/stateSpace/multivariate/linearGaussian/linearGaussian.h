/// \file
/// \brief Some definitions of members for multivariate linear-Gaussian state-space models.
///
/// This file contains the functions for implementing the various
/// classes for any kind of state-space model in which both the latent states
/// and the observations can be represented as arma::colvec's.

#ifndef __LINEARGAUSSIAN_H
#define __LINEARGAUSSIAN_H

#include "main/templates/dynamic/stateSpace/multivariate/multivariate.h"
#include "main/rng/gaussian.h"

/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromInitialDistribution()
{
  return modelParameters_.getM0() + std::sqrt(getInverseTemperatureLat()) * arma::chol(modelParameters_.getC0()) * arma::randn<arma::colvec>(modelParameters_.getDimLatentVariable());
}
/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromTransitionEquation(const unsigned int t, const LatentVariable& latentVariableOld)
{
  return modelParameters_.getA() * latentVariableOld + std::sqrt(getInverseTemperatureLat()) * modelParameters_.getB() * arma::randn<arma::colvec>(modelParameters_.getDimLatentVariable());
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogInitialDensity(const LatentVariable& latentVariable)
{
  return arma::as_scalar(gaussian::evaluateDensityMultivariate(latentVariable, modelParameters_.getM0(), std::sqrt(getInverseTemperatureLat()) * modelParameters_.getC0(), false, true));
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld)
{
  return arma::as_scalar(gaussian::evaluateDensityMultivariate(latentVariableNew, modelParameters_.getA() * latentVariableOld, std::sqrt(getInverseTemperatureLat()) * modelParameters_.getB(), true, true));
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable)
{
  return arma::as_scalar(gaussian::evaluateDensityMultivariate(getObservations().col(t), modelParameters_.getC() * latentVariable, std::sqrt(getInverseTemperatureObs()) * modelParameters_.getD(), true, true));
}
/// Samples a single observation according to the observation equation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromObservationEquation(const unsigned int t, Observations& observations, const LatentVariable& latentVariable)
{
  observations.col(t) = modelParameters_.getC() * latentVariable + std::sqrt(getInverseTemperatureObs()) * modelParameters_.getD() * arma::randn<arma::colvec>(modelParameters_.getDimObservation());
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables using a (partially) non-centred parametrisation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihoodRepar(const LatentPathRepar& latentPathRepar)
{
  return evaluateLogCompleteLikelihood(latentPathRepar);
}
#endif
