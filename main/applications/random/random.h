/// \file
/// \brief Generating data and performing inference in a random-effects model. 
///
/// This file contains the functions for implementing the abstract model
/// class for a simple Gaussian random-effects model.

#ifndef __RANDOM_H
#define __RANDOM_H

#include "main/templates/static/univariate/univariate.h"
#include "main/algorithms/mcmc/Mcmc.h"
// #include "projects/optim/default/default.h" // TODO: do we really need this dependence?
#include "main/rng/gaussian.h"

// [[Rcpp::depends("RcppArmadillo")]]

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the model
////////////////////////////////////////////////////////////////////////////////

/// Holds all the static model parameters.
class ModelParameters
{
  
public:
  
  /// Returns the conditional prior mean of a single latent variable.
  double getA() const {return a_;};
  /// Returns the conditional prior standard deviation of a single latent variable.
  double getB() const {return b_;};
  /// Returns the conditional standard deviation of a single observation.
  double getD() const {return d_;};
  /// Returns the mean of the normal prior on a.
  double getMeanHyperA() const {return meanHyperA_;}
  /// Returns the variance of the normal prior on a.
  double getVarHyperA() const {return varHyperA_;}
  /// Returns the shape parameter of gamma the prior on b.
  double getShapeHyperB() const {return shapeHyperB_;}
  /// Returns the shape parameter of gamma the prior on d.
  double getShapeHyperD() const {return shapeHyperD_;}
  /// Returns the scale parameter of gamma the prior on b.
  double getScaleHyperB() const {return scaleHyperB_;}
  /// Returns the scale parameter of gamma the prior on d.
  double getScaleHyperD() const {return scaleHyperD_;}
  /// Specifies the conditional prior mean of a single latent variable.
  void setA(const double a) {a_ = a;};
  /// Specifies the conditional prior standard deviation of a single latent variable.
  void setB(const double b) {b_ = b;};
  /// Specifies the conditional standard deviation of a single observation.
  void setD(const double d) {d_ = d;};
  /// Determines the model parameters from arma::colvec theta.
  void setUnknownParameters(const arma::colvec& theta)
  {
    setA(theta(0));
    setB(theta(1));
    setD(theta(2));
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyperParameters)
  {
    meanHyperA_  = hyperParameters(0);
    varHyperA_   = hyperParameters(1);
    shapeHyperB_ = hyperParameters(2);
    scaleHyperB_ = hyperParameters(3);
    shapeHyperD_ = hyperParameters(4);
    scaleHyperD_ = hyperParameters(5);
  }
  
  /// Only needed for compatibility with the SmcSampler class.
//   arma::uvec getThetaIndicesSecond()
//   {
//     arma::uvec test;
//     return test;
//   } 
  
private:
  double meanHyperA_, varHyperA_;    // (known) hyper parameters of the normal prior on a
  double shapeHyperB_, scaleHyperB_; // (known) hyper parameters of the gamma prior on b
  double shapeHyperD_, scaleHyperD_; // (known) hyper parameters of the gamma prior on d 
  double a_, b_, d_; // (unknown) model parameters
  
  
  
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
  if (modelParameters_.getB() < 0 || modelParameters_.getD() < 0)
  {
    return - std::numeric_limits<double>::infinity();
  }
  else
  {
    return R::dnorm(modelParameters_.getA(), modelParameters_.getMeanHyperA(), std::sqrt(modelParameters_.getVarHyperA()), true) +
      dInverseGamma(modelParameters_.getB(), modelParameters_.getShapeHyperB(), modelParameters_.getScaleHyperB()) +
      dInverseGamma(modelParameters_.getD(), modelParameters_.getShapeHyperD(), modelParameters_.getScaleHyperD());
  }
}
/// Samples the set of parameters from the prior.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromPrior(arma::colvec& theta)
{
  theta(0) = rng_.randomNormal(modelParameters_.getMeanHyperA(), std::sqrt(modelParameters_.getVarHyperA()));
  theta(1) = 1.0 / rng_.randomGamma(modelParameters_.getShapeHyperB(), 1.0/modelParameters_.getScaleHyperB());
  theta(2) = 1.0 / rng_.randomGamma(modelParameters_.getShapeHyperD(), 1.0/modelParameters_.getScaleHyperD()); 
}
/// Samples a single latent variable its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromLatentPrior(const unsigned int t)
{
  return rng_.randomNormal(modelParameters_.getA(), modelParameters_.getB());
}
/// Samples a single observation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromObservationEquation(
  const unsigned int t, 
  Observations& observations,
  const LatentVariable& latentVariable
)
{
  observations_(t) = rng_.randomNormal(latentVariable, modelParameters_.getD());
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogLatentPriorDensity(const unsigned int t, const LatentVariable& latentVariable)
{
  return R::dnorm(latentVariable, modelParameters_.getA(), modelParameters_.getB()/std::sqrt(inverseTemperatureLat_), true);
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(
  const unsigned int t, 
  const LatentVariable& latentVariable
)
{
  return R::dnorm(getObservations()(t), latentVariable, modelParameters_.getD()/std::sqrt(inverseTemperatureObs_), true);
}
/// Increases the gradient by the gradient of the log-prior density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogPriorDensity(arma::colvec& gradient)
{
  gradient(0) += (2*modelParameters_.getMeanHyperA() - modelParameters_.getA()) / modelParameters_.getVarHyperA();
  gradient(1) += 1.0 / modelParameters_.getB() * (modelParameters_.getScaleHyperB() / modelParameters_.getB() - modelParameters_.getShapeHyperB() - 1.0);
  gradient(2) += 1.0 / modelParameters_.getD() * (modelParameters_.getScaleHyperD() / modelParameters_.getD() - modelParameters_.getShapeHyperD() - 1.0);
}
/// Increases the gradient by the gradient of the log-transition density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogLatentPriorDensity(
  const unsigned int t, 
  const LatentVariable& latentVariable,
  arma::colvec& gradient
)
{
  gradient(0) += inverseTemperatureLat_ * (latentVariable - modelParameters_.getA()) / std::pow(modelParameters_.getB(), 2.0);
  gradient(1) += inverseTemperatureLat_ * (1.0/ modelParameters_.getB() * std::pow(latentVariable, 2.0) - 1.0) / (modelParameters_.getB() * modelParameters_.getB());
}
/// Increases the gradient by the gradient of the log-observation density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogObservationDensity(
  const unsigned int t, 
  const LatentVariable& latentVariable, 
  arma::colvec& gradient
)
{
  gradient(2) += inverseTemperatureObs_ * (1.0/ modelParameters_.getD() * std::pow(observations_(t), 2.0) - 1.0) / (modelParameters_.getD() * modelParameters_.getD());
}
/// Evaluates the marginal likelihood of the parameters (with the latent 
/// variables integrated out). Note that analytical expressions for this 
/// will not be available for most models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihood()
{
  double logLike = 0;
  double stdDev = std::sqrt(std::pow(modelParameters_.getB(), 2.0) + std::pow(modelParameters_.getD(), 2.0));
  for (unsigned int t=0; t<observations_.n_rows; ++t)
  {
    logLike += R::dnorm(getObservations()(t), modelParameters_.getA(), stdDev, true);
  }
  return logLike;
}
/// Evaluates the score. Note that analytical expressions for this 
/// will not be available for most models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateScore(arma::colvec& score)
{
  score.set_size(dimTheta_); // NOTE: we probably don't need this line
  double sigma2 = std::pow(modelParameters_.getB(), 2.0) + std::pow(modelParameters_.getD(), 2.0);
//   for (unsigned int t=0; t<observations_.n_rows; ++t)
//   {
//     score(0) += (modelParameters_.getA() - observations_(t)) / sigma2;
//     score(1) += (2 * modelParameters_.getB() * std::pow(observations_(t) - modelParameters_.getA(), 2.0) / sigma2) / sigma2;
//     score(2) += (2 * modelParameters_.getD() * std::pow(observations_(t) - modelParameters_.getA(), 2.0) / sigma2) / sigma2;
//   }
  
    score(0) += arma::accu((modelParameters_.getA() - observations_)) / sigma2;
    score(1) += modelParameters_.getB() * (2 * arma::accu(arma::pow(observations_ - modelParameters_.getA(), 2.0)) / sigma2 - 1.0) / sigma2;
    score(2) += modelParameters_.getD() * (2 * arma::accu(arma::pow(observations_ - modelParameters_.getA(), 2.0)) / sigma2 - 1.0) / sigma2;
}
/// Generates latent variables from their full conditional distribution.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::runGibbs(LatentPath& latentPath)
{
  unsigned int T = observations_.n_rows;
  
  if (latentPath.size() != T)
  {
    latentPath.set_size(T);
  }
  double sigma2 = 1.0 / (inverseTemperatureLat_ / std::pow(modelParameters_.getB(), 2.0) + inverseTemperatureObs_ / std::pow(modelParameters_.getD(), 2.0));
  double sigma = std::sqrt(sigma2);
  
  latentPath = sigma2 * (inverseTemperatureLat_ * modelParameters_.getA() / std::pow(modelParameters_.getB(), 2.0) + inverseTemperatureObs_ * observations_ / std::pow(modelParameters_.getD(), 2.0)) + sigma * arma::randn<arma::colvec>(T);
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
/// Evaluates the log of the likelihood associated with some subset of the 
/// (static) model parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodFirst(LatentPath& latentPath)
{
  return 0.0;
}
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodSecond(LatentPath& latentPath)
{
  // Empty: the marginal likelihood in this model.
  return 0.0;
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
  
  particles = arma::conv_to<std::vector<Particle>>::from(model_.getModelParameters().getA() + model_.getModelParameters().getB() * arma::randn<arma::colvec>(nParticles_));
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
      R::dnorm(model_.getObservations()(t), particles[n], model_.getModelParameters().getD() / std::sqrt( model_.getInverseTemperatureObs()), true) ;
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
  for (unsigned int n=0; n<nParticles_; n++)
  {
    aux1[n] = R::qnorm(R::pnorm(particles[n], model_.getModelParameters().getA(), model_.getModelParameters().getB() / std::sqrt( model_.getInverseTemperatureLat()), true, false), 0.0, 1.0, true, false);
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
  for (unsigned int n=0; n<nParticles_; n++)
  {
    particles[n] = R::qnorm(R::pnorm(aux1[n], 0.0, 1.0, true, false), model_.getModelParameters().getA(), model_.getModelParameters().getB() / std::sqrt( model_.getInverseTemperatureLat()), true, false);
  }
}

#endif
