/// \file
/// \brief Some definitions of members of Smc if we use a single SMC algorithm for all latent variables.
///
/// This file contains some definitions of members of Smc class 
/// if we use a single SMC algorithm for all latent variable

#ifndef __SINGLE_H
#define __SINGLE_H

#include "smc/Smc.h"
/*
/// Runs an SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runSmc
(
  const unsigned int nParticles,
  const arma::colvec& theta,
  LatentPath& latentPath, 
  AuxFull<Aux>& auxFull,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = false;
  
  /// We need to loop this over however many SMC runs we need for the model
  runSmcBase(auxFull);
  if (samplePath_)
  {
    convertParticlePathToLatentPath(particlePath_, latentPath);
  }
  
  return getLoglikelihoodEstimate();
}
/// Runs a conditional SMC algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runCsmc
(
  const unsigned int nParticles, 
  const arma::colvec& theta,
  LatentPath& latentPath,
  AuxFull<Aux>& auxFull,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = true;
  convertLatentPathToParticlePath(latentPath, particlePath_);
  runSmcBase(auxFull);
  convertParticlePathToLatentPath(particlePath_, latentPath);
  return getLoglikelihoodEstimate();
}
/// Runs an SMC algorithm and approximates the gradient.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runSmc
(
  const unsigned int nParticles,
  const arma::colvec& theta,
  LatentPath& latentPath, 
  AuxFull<Aux>& auxFull,
  arma::colvec& gradientEstimate,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = false;
  
  if (approximateGradient_)
  {
    if (gradientEstimate.size() != model_.getDimTheta())
    {
      gradientEstimate.set_size(model_.getDimTheta());
    }
    gradientEstimate.zeros();
    model_.addGradLogPriorDensity(gradientEstimate);
  }
  
  runSmcBase(auxFull);
  if (samplePath_)
  {
    convertParticlePathToLatentPath(particlePath_, latentPath);
  }
  if (approximateGradient_)
  {
    runFixedLagSmoothing(gradientEstimate);
  }
  return getLoglikelihoodEstimate();
}
/// Runs an SMC algorithm and approximates the gradient.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runSmc
(
  const arma::colvec& theta,
  LatentPath& latentPath, 
  AuxFull<Aux>& auxFull,
  arma::colvec& gradientEstimate
)
{
  model_.setUnknownParameters(theta);
  isConditional_ = false;
  
  if (approximateGradient_)
  {
    if (gradientEstimate.size() != model_.getDimTheta())
    {
      gradientEstimate.set_size(model_.getDimTheta());
    }
    gradientEstimate.zeros();
    model_.addGradLogPriorDensity(gradientEstimate);
  }
  
  runSmcBase(auxFull);
  if (samplePath_)
  {
    convertParticlePathToLatentPath(particlePath_, latentPath);
  }
  if (approximateGradient_)
  {
    runFixedLagSmoothing(gradientEstimate);
  }
  return getLoglikelihoodEstimate();
}
/// Runs a conditional SMC algorithm and approximates the gradient.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::runCsmc
(
  const unsigned int nParticles, 
  const arma::colvec& theta,
  LatentPath& latentPath,
  AuxFull<Aux>& auxFull,
  arma::colvec& gradientEstimate,
  const double inverseTemperature
)
{
  model_.setUnknownParameters(theta);
  model_.setInverseTemperature(inverseTemperature);
  nParticles_ = nParticles;
  isConditional_ = true;
  
  if (approximateGradient_)
  {
    if (gradientEstimate.size() != model_.getDimTheta())
    {
      gradientEstimate.set_size(model_.getDimTheta());
    }
    gradientEstimate.zeros();
    model_.addGradLogPriorDensity(gradientEstimate);
  }
  
  
  convertLatentPathToParticlePath(latentPath, particlePath_);
  runSmcBase(auxFull);
  convertParticlePathToLatentPath(particlePath_, latentPath);
  if (approximateGradient_)
  {
    runFixedLagSmoothing(gradientEstimate);
  }
  return getLoglikelihoodEstimate();
}*/

#endif