/// \file
/// \brief Generating data and performing inference in the Student-t toy model. 
///
/// This file contains the functions for implementing the abstract model
/// class for the Student-t toy model.

#ifndef __STUDENT_H
#define __STUDENT_H

#include "base/templates/static/univariate/univariate.h"
#include "base/mcmc/Mcmc.h"
// #include "optim/default/default.h" // TODO: do we really need this dependence?
#include "base/rng/gaussian.h"

// [[Rcpp::depends("RcppArmadillo")]]

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the model
////////////////////////////////////////////////////////////////////////////////

/// Holds all the static model parameters.
class ModelParameters
{
  
public:
  
  /// Returns the known degrees of freedom.
  double getDf() const {return df_;};
  /// Returns the unknown location parameter.
  double getLocation() const {return location_;}; 
  /// Specifies the known degrees of freedom.
  void setDf(const double df) {df_ = df;};
  /// Specifies the unknown location parameter.
  void setLocation(const double location) {location_ = location;};
  /// Determines the model parameters from arma::colvec theta.
  void setUnknownParameters(const arma::colvec& theta)
  {
    setLocation(theta(0));
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyperParameters)
  {
    setDf(hyperParameters(0));
  }
  
private:
  
  double df_; // (known) degrees of freedom
  double location_; // (unknown) location parameter
  
};

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the algorithm
////////////////////////////////////////////////////////////////////////////////

/// Holds some additional auxiliary parameters for the SMC algorithm.
class SmcParameters
{
  
public:
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& algorithmParameters)
  {
    // Empty;
  }
  
private:
};

/// Holds some additional auxiliary parameters for the SMC algorithm.
class McmcParameters
{
  
public:
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& algorithmParameters)
  {
    // Empty;
  }
  
private:
};

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Model class
///////////////////////////////////////////////////////////////////////////////

/// Evaluates the log-prior density of the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogPriorDensity()
{
  return -std::log(getSupportMax(0) - getSupportMin(0));
}
/// Samples the set of parameters from the prior.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromPrior(arma::colvec& theta)
{
  theta(0) = getSupportMin(0) + arma::randu() * (getSupportMax(0) - getSupportMin(0));
}
/// Samples a single latent variable its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromLatentPrior(const unsigned int t)
{
  return rng_.randomGamma(modelParameters_.getDf()/2.0, 2.0/modelParameters_.getDf());
}
/// Samples a single observation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromObservationEquation(
  const unsigned int t, 
  Observations& observations,
  const LatentVariable& latentVariable
)
{
  observations_(t) = modelParameters_.getLocation() + 1.0 / std::sqrt(latentVariable * getInverseTemperatureObs()) * arma::randn();
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogLatentPriorDensity(const unsigned int t, const LatentVariable& latentVariable)
{
  return R::dgamma(latentVariable, ((modelParameters_.getDf()-2) * getInverseTemperatureLat() + 2)/2.0, 2.0/(modelParameters_.getDf() * getInverseTemperatureLat()), true);
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(
  const unsigned int t, 
  const LatentVariable& latentVariable
)
{
  return R::dnorm(getObservations()(t), modelParameters_.getLocation(), 1.0/std::sqrt(latentVariable*getInverseTemperatureObs()), true);
}
/// Increases the gradient by the gradient of the log-prior density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogPriorDensity(arma::colvec& gradient)
{
  // Do nothing for this model because the the gradient of the log-prior
  // density is 0.
}
/// Increases the gradient by the gradient of the log-transition density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogLatentPriorDensity(
  const unsigned int t, 
  const LatentVariable& latentVariableNew,
  arma::colvec& gradient
)
{
  // Do nothing for this model because the conditional
  // prior of the latent variables does not depend on the 
  // unkown parameter.
}
/// Increases the gradient by the gradient of the log-observation density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogObservationDensity(
  const unsigned int t, 
  const LatentVariable& latentVariable, 
  arma::colvec& gradient
)
{
  gradient(0) += - inverseTemperatureObs_ * latentVariable * (modelParameters_.getLocation() - getObservations()(t));
}
/// Evaluates the marginal likelihood of the parameters (with the latent 
/// variables integrated out). Note that analytical expressions for this 
/// will not be available for most models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihood()
{
  double logLike = 0;
  for (unsigned int t=0; t<observations_.n_rows; ++t)
  {
    logLike += R::dt(getObservations()(t) - modelParameters_.getLocation(), modelParameters_.getDf(), true);
  }
  return logLike;
}
/// Evaluates the score. Note that analytical expressions for this 
/// will not be available for most models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateScore(arma::colvec& score)
{
  score.set_size(dimTheta_); // NOTE: we probably don't need this line
  // TODO
}
/// Generates latent variables from their full conditional distribution.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::runGibbs(LatentPath& latentPath)
{
  if (latentPath.size() != observations_.n_rows)
  {
    latentPath.set_size(observations_.n_rows);
  }
  
  double shape = (getInverseTemperatureObs() + (modelParameters_.getDf() - 2) * getInverseTemperatureLat() + 2) / 2.0;
  
  arma::colvec scale = 2.0 / (getInverseTemperatureObs() * arma::pow(getObservations() - modelParameters_.getLocation(), 2.0) + modelParameters_.getDf() * getInverseTemperatureLat());
  latentPath = scale % arma::randg<arma::colvec>(observations_.n_rows, arma::distr_param(shape, 1.0));
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables using a (partially) non-centred parametrisation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihoodRepar
(
  const LatentPathRepar& latentPathRepar
)
{
  return evaluateLogCompleteLikelihood(latentPathRepar);
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Smc class
///////////////////////////////////////////////////////////////////////////////

/// Samples particles at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleParticles
(
  const unsigned int t, 
  std::vector<Particle>& particles
)
{
//   std::cout << "started sampleParticles()" << std::endl;
  
  double shape = (model_.getModelParameters().getDf()/2.0 - 1) * model_.getInverseTemperatureLat() + 1;
  double scale =  2.0 / (model_.getModelParameters().getDf() * model_.getInverseTemperatureLat());
  
  /*
  for (unsigned int n=0; n<nParticles_; n++)
  {
    particles[n] = rng_.randomGamma(shape, scale);
  }
  */
  
  particles = arma::conv_to<std::vector<Particle>>::from(arma::randg<arma::colvec>(nParticles_, arma::distr_param(shape, scale)));
  if (isConditional_) {particles[particleIndicesIn_(t)] = particlePath_[t];}
// std::cout << "finished sampleParticles()" << std::endl;
}
/// Computes a particle weight at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogParticleWeights
(
  const unsigned int t,
  const std::vector<Particle>& particles,
  arma::colvec& logWeights
)
{
//     std::cout << "started computeLogParticleWeights()" << std::endl;
    
  for (unsigned int n=0; n<nParticles_; n++)
  {
    logWeights(n) += 
      R::dnorm(model_.getObservations()(t), model_.getModelParameters().getLocation(), 1.0/std::sqrt(particles[n] * model_.getInverseTemperatureObs()), true) ;
  }
//     std::cout << "finished computeLogParticleWeights()" << std::endl;
}
/// Reparametrises particles at Step 0 to obtain the values of Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromParticles
(
  const unsigned int t,
  const std::vector<Particle>& particles, 
  std::vector<Aux>& aux1
)
{
  double shape = (model_.getModelParameters().getDf()/2.0 - 1) * model_.getInverseTemperatureLat() + 1;
  double scale =  2.0 / (model_.getModelParameters().getDf() * model_.getInverseTemperatureLat());
  
  for (unsigned int n=0; n<nParticles_; n++)
  {
    aux1[n] = R::qnorm(R::pgamma(particles[n], shape, scale, true, false), 0.0, 1.0, true, false);
  }
}
/// Reparametrises Gaussians at Step 0 to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineParticlesFromGaussians
(
  const unsigned int t,
  std::vector<Particle>& particles, 
  const std::vector<Aux>& aux1
)
{
  double shape = (model_.getModelParameters().getDf()/2.0 - 1) * model_.getInverseTemperatureLat() + 1;
  double scale =  2.0 / (model_.getModelParameters().getDf() * model_.getInverseTemperatureLat());

  for (unsigned int n=0; n<nParticles_; n++)
  {
    particles[n] = R::qgamma(R::pnorm(aux1[n], 0.0, 1.0, true, false), shape, scale, true, false);
  }
}

#endif
