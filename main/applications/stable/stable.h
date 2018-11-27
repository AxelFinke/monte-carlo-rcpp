/// \file
/// \brief Generating data and performing inference in the Student-t toy model. 
///
/// This file contains the functions for implementing the abstract model
/// class for the Student-t toy model.

#ifndef __STABLE_H
#define __STABLE_H

#include <functional>
#include <math.h>

#include "main/templates/static/univariate/univariate.h"
#include "main/algorithms/mcmc/Mcmc.h"
#include "main/rng/gaussian.h"
#include "main/helperFunctions/envelope.h"
#include "main/helperFunctions/rootFinding.h"

// [[Rcpp::depends("RcppArmadillo")]]

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the model
////////////////////////////////////////////////////////////////////////////////

/// Holds all the static model parameters.
class ModelParameters
{
public:
  
  /// Returns the stability parameter.
  double getAlpha() const {return alpha_;}
  /// Returns the skewness parameter.
  double getBeta() const {return beta_;}
  /// Returns the scale parameter.
  double getGamma() const {return gamma_;}
  /// Returns the location parameter.
  double getDelta() const {return delta_;}
  /// Returns the first auxiliary parameter.
  double getEAux() const {return eAux_;}
  /// Returns the second auxiliary parameter.
  double getLAux() const {return lAux_;}
  
  /// Returns the minimum of the exclusion of the support of alpha.
  double getTabooMinAlpha() const {return tabooMinAlpha_;}
  /// Returns the maximum exclusion of the support of alpha.
  double getTabooMaxAlpha() const {return tabooMaxAlpha_;}
  /// Returns the shape of the gamm prior on the parameter gamma.
  double getShapeHyperGamma() const {return shapeHyperGamma_;}
  /// Returns the scale of the gamma prior on the parameter gamma.
  double getScaleHyperGamma() const {return scaleHyperGamma_;}
  /// Returns the prior mean of delta.
  double getMeanHyperDelta() const {return meanHyperDelta_;}
  /// Returns the prior variance of delta.
  double getVarHyperDelta() const {return varHyperDelta_;}
  
  /// Determines the model parameters from arma::colvec theta.
  void setUnknownParameters(const arma::colvec& theta)
  {
    if (theta(0) <= tabooMinAlpha_)
    {
      alpha_ = theta(0);
    }
    else 
    {
      alpha_ = theta(0) + tabooMaxAlpha_ - tabooMinAlpha_;
    }
    beta_  = theta(1);
    gamma_ = theta(2);
    delta_ = theta(3); 
    setAuxiliaryParameters();
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyp)
  {
    tabooMinAlpha_   = hyp(0); 
    tabooMaxAlpha_   = hyp(1);
    shapeHyperGamma_ = hyp(2);
    scaleHyperGamma_ = hyp(3);
    meanHyperDelta_  = hyp(4);
    varHyperDelta_   = hyp(5);
  }
  /// Computes some auxiliary parameters.
  void setAuxiliaryParameters()
  {
    eAux_ = M_PI * beta_ * std::min(alpha_, 2.0 - alpha_) / 2.0;
    lAux_ = - eAux_ / (M_PI * alpha_);
  }
  /// Computes the lower bound of the support of the latent variable.
  double computeLb(const double observation) const
  {
    if ((observation - delta_) / gamma_ <= 0)
    {
      return -0.5;
    }
    else 
    {
      return lAux_;
    }
  }
  /// Computes the upper bound of the support of the latent variable.
  double computeUb(const double observation) const
  {
    if ((observation - delta_) / gamma_ <= 0)
    {
      return lAux_;
    }
    else 
    {
      return 0.5;
    }
  }
  /// Computes an auxiliary parameter.
  double computeTAux(const double x) const
  {
    return (std::sin(M_PI * alpha_ * x + eAux_) * 
      std::pow(std::cos(M_PI * x), (- 1.0 / alpha_))) *
      std::pow(std::cos((alpha_-1.0) * M_PI * x + eAux_), ((1.0 - alpha_) / alpha_));
  }
  /// Computes the Jacobian associated with the transformation
  /// x -> tAux(x).
  double computeTAuxDerivative(const double x) const
  {  
    return M_PI*((std::pow(alpha_,2.0)-2.0*alpha_+1)*std::cos(M_PI*x)*std::sin(M_PI*alpha_*x+eAux_)*std::sin((M_PI*alpha_-M_PI)*x+eAux_)+(std::sin(M_PI*x)*std::sin(M_PI*alpha_*x+eAux_)+std::pow(alpha_,2.0)*std::cos(M_PI*x)*std::cos(M_PI*alpha_*x+eAux_))*std::cos((M_PI*alpha_-M_PI)*x+eAux_))/(alpha_*std::cos(M_PI*x)*std::pow((std::cos(M_PI*x)/std::cos((M_PI*alpha_-M_PI)*x+eAux_)),(1/alpha_))*std::pow(std::cos((M_PI*alpha_-M_PI)*x+eAux_),2.0));
  }
  /// Numerically inverts tAux.
  double invertTAux(const double observation, const double v, bool& isBracketing) const
  {
    // NOTE: Due to monotonicity,
    // we could accelerate this procedure by re-ordering
    // as proposed in Buckle (1995).
    
    double lb = computeLb(observation);
    double ub = computeUb(observation);

    // Computes tAux(x) for given parameters:
    auto fun   = [&] (double x) {return computeTAux(x) - v;};
    // Computes first derivative of tAux(x) for given parameters:
    auto deriv = [&] (double x) {return computeTAuxDerivative(x);};
    
    // Approximation of the root of fun():
    return rootFinding::saveGuardedNewton(isBracketing, fun, deriv, lb, ub, tolX_, tolF_, nIterations_);
  }
  
private:
  
  /// Parameters to be inferred:
  double alpha_, beta_, gamma_, delta_;
  
  // Auxiliary parameters:
  double eAux_, lAux_;
  
  /// Known hyper parameters:
  double tabooMinAlpha_, tabooMaxAlpha_, meanHyperDelta_, varHyperDelta_, shapeHyperGamma_, scaleHyperGamma_;
  
  // parameters for the safeguarded Netwon method:
  double tolX_ = 0.00001;
  double tolF_ = 0.00001;
  unsigned int nIterations_ = 50;

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
  
  /// Returns the support points of the envelope.
  const arma::colvec& getPoints() const {return points_;};
  /// Returns the normalised levels of the envelope.
  const arma::colvec& getLevels() const {return levels_;};
  /// Returns the probabilities associated with each piecewise-linear section of the envelope.
  const arma::colvec& getProbabilities() const {return probabilities_;};
  /// Returns number of support points of the envelope.
  unsigned int getnPoints() const {return nPoints_;};
  
  /// Returns the support points of the envelope.
  arma::colvec& getRefPoints() {return points_;};
  /// Returns the normalised levels of the envelope.
  arma::colvec& getRefLevels() {return levels_;};
  /// Returns the probabilities associated with each piecewise-linear section of the envelope.
  arma::colvec& getRefProbabilities() {return probabilities_;};
  
private:
  
  arma::colvec points_; // points forming the envelope
  arma::colvec levels_; // unnormalised levels forming the envelope
  arma::colvec probabilities_; // probabilities of falling in each of the piecewise-linear sections of the proposal density
  unsigned int nPoints_ = 50; // number of support points for adaptive envelope
 
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
  double logDensity = 
    -std::log(getSupportMax(0) - getSupportMin(0)) +
    -std::log(getSupportMax(1) - getSupportMin(1)) + 
    dInverseGamma(modelParameters_.getGamma(), modelParameters_.getShapeHyperGamma(), modelParameters_.getScaleHyperGamma()) +
    R::dnorm(modelParameters_.getDelta(), modelParameters_.getMeanHyperDelta(), std::sqrt(modelParameters_.getVarHyperDelta()), true);
    
  return logDensity;
}
/// Samples the set of parameters from the prior.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromPrior(arma::colvec& theta)
{
  theta(0) = getSupportMin(0) + arma::randu() * (getSupportMax(0) - getSupportMin(0));
  theta(1) = getSupportMin(1) + arma::randu() * (getSupportMax(1) - getSupportMin(1));
  theta(2) = 1.0 / rng_.randomGamma(modelParameters_.getShapeHyperGamma(), 1.0/modelParameters_.getScaleHyperGamma());
  theta(3) = rng_.randomNormal(modelParameters_.getMeanHyperDelta(), std::sqrt(modelParameters_.getVarHyperDelta()));
  
  /////////////////////
//   theta(0) = 0.5;
//   theta(1) = 0.7;
//   theta(2) = 1.0;
//    theta(3) = 0.0;
  /////////////////////
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogLatentPriorDensity(const unsigned int t, const LatentVariable& latentVariableNew)
{
  return 0;
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable)
{
  double z = (observations_(t) - modelParameters_.getDelta()) / modelParameters_.getGamma();
  double tAux = modelParameters_.computeTAux(latentVariable);
  double logYAux = std::log(std::abs(z / tAux)) * (modelParameters_.getAlpha() / (modelParameters_.getAlpha() - 1.0));
  
  double lb = modelParameters_.computeLb(observations_(t));
  double ub = modelParameters_.computeUb(observations_(t));
  
  if (lb < latentVariable && latentVariable < ub)
  {
    return getInverseTemperatureObs()* (std::log(modelParameters_.getAlpha()) - std::log(std::abs(modelParameters_.getAlpha() - 1.0)) - std::log(modelParameters_.getGamma()) - std::exp(logYAux) + logYAux - std::log(std::abs(z)));
  }
  else
  {
    return - std::numeric_limits<double>::infinity();
  }
}
/// Increases the gradient by the gradient of the log-prior density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogPriorDensity(arma::colvec& gradient)
{
  // Empty
}
/// Increases the gradient by the gradient of the log-transition density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogLatentPriorDensity(const unsigned int t, const LatentVariable& latentVariableNew, arma::colvec& gradient)
{
  // Empty
}
/// Increases the gradient by the gradient of the log-observation density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient)
{
  // NOTE: not currently implemented.
}
/// Evaluates the marginal likelihood of the parameters (with the latent 
/// variables integrated out). Note that analytical expressions for this 
/// will not be available for most models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihood()
{
  // NOTE: the marginal likelihood is intractable in this model
  return 0;
}
/// Evaluates the score. Note that analytical expressions for this 
/// will not be available for most models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateScore(arma::colvec& score)
{
  // NOTE: the score is intractable in this model.
}
/// Generates latent variables from their full conditional distribution.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::runGibbs(LatentPath& latentPath)
{
  // NOTE: the full conditional distribution of the latent variables 
  // is intractable in this model.
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of class <<Smc>>.
///////////////////////////////////////////////////////////////////////////////

/// Samples particles at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>  
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleParticles
(
  const unsigned int t, 
  std::vector<Particle>& particlesNew
)
{;
  double lb = model_.getModelParameters().computeLb(model_.getObservations()(t));
  double ub = model_.getModelParameters().computeUb(model_.getObservations()(t));

  // Approximation of the mode:
  bool isBracketing;
  
  double z = (model_.getObservations()(t) - model_.getModelParameters().getDelta()) / model_.getModelParameters().getGamma();
  double mode = model_.getModelParameters().invertTAux(model_.getObservations()(t), z, isBracketing);

  // Computes the unnormalised target density for a specific time step
  // for given parameters:
  auto logDensity = [=] (double x) {return model_.evaluateLogObservationDensity(t, x);};

  // Samples the envelope:
  envelope::create(smcParameters_.getRefPoints(), smcParameters_.getRefLevels(), smcParameters_.getRefProbabilities(), logDensity, smcParameters_.getnPoints(), lb, ub, mode, isBracketing); 

  for (unsigned int n=0; n<getNParticles(); n++)
  {
    particlesNew[n] = envelope::sample(smcParameters_.getPoints(), smcParameters_.getLevels(), smcParameters_.getProbabilities());
  }
  if (isConditional_) {particlesNew[particleIndicesIn_(t)] = particlePath_[t];}
}
/// Computes a particle weight at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>  
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogParticleWeights
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{ 
  for (unsigned int n=0; n<getNParticles(); n++)
  {  
    logWeights(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]) - envelope::evaluateLogDensity(particlesNew[n], smcParameters_.getPoints(), smcParameters_.getLevels());
  }
}
/// Reparametrises particles at Step 0 to obtain the values of Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>  
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromParticles
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew,
  std::vector<Aux>& aux1
)
{
  // The correlated pseudo-marginal approach cannot easily be used with the 
  // envelope proposal.
}
/// Reparametrises Gaussians at Step 0 to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>  
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineParticlesFromGaussians
(
  const unsigned int t,
  std::vector<Particle>& particlesNew, 
  const std::vector<Aux>& aux1
)
{
  // The correlated pseudo-marginal approach cannot easily be used with the 
  // envelope proposal.
}
/*
/// Simulates observations from the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::simulateData(const arma::colvec& extraParameters)
{
  // Empty (because we are using existing R packages to simulated data)
}
*/
/*
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihood
(
  const LatentPath& latentPath
)
{
  double logLike = 0.0;
  for (unsigned int t=0; t<observations_.n_rows; ++t)
  {
    logLike += evaluateLogObservationDensity(t, latentPath(t)); // - std::log(std::abs(getModelParameters().computeTAuxDerivative(latentPath(t))));
  }
  return logLike;
}
*/
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables using a (partially) non-centred parametrisation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihoodRepar
(
  const LatentPathRepar& latentPathRepar
)
{
  double logLike = 0.0;
  double z, logYAux, x;
  bool isBracketing = true;
  
//   double lb, ub;
  
  for (unsigned int t=0; t<observations_.n_rows; ++t)
  {
    
//     std::cout << "evaluateLogCompleteLikelihoodRepar at Time " << t << std::endl;
    
    x = modelParameters_.invertTAux(observations_(t), latentPathRepar(t) * (observations_(t) - modelParameters_.getDelta()), isBracketing);
    
//     lb = modelParameters_.computeLb(observations_(t)); // lower bound of the support of x
//     ub = modelParameters_.computeUb(observations_(t)); // upper bound of the support of x
    
//     std::cout << "lb: " << lb << "; x: " << x << "; ub: " << ub << "; isBracketing: " << isBracketing << std::endl;
    
    if (isBracketing)
    {
      z = (observations_(t) - modelParameters_.getDelta()) / modelParameters_.getGamma();
      logYAux = std::log(std::abs(1.0/ (latentPathRepar(t) * modelParameters_.getGamma()))) * (modelParameters_.getAlpha() / (modelParameters_.getAlpha() - 1.0));
      
      logLike += getInverseTemperatureObs()* (
        std::log(modelParameters_.getAlpha()) - std::log(std::abs(modelParameters_.getAlpha() - 1.0)) - std::log(modelParameters_.getGamma()) 
        - std::exp(logYAux) + logYAux 
        - std::log(std::abs(z))
      ) - std::log(std::abs(modelParameters_.computeTAuxDerivative(x)/(observations_(t) - modelParameters_.getDelta())));
    }
    else 
    {
//       std::cout << "WARNING: logCompleteLikelihood at proposed parameter is -INF" <<std::endl;
      logLike = - std::numeric_limits<double>::infinity();
      break;
    }
  }
  return logLike;
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Optim class
///////////////////////////////////////////////////////////////////////////////

/// Reparametrises latent variables from the standard (centred) parametrisation
/// to a (partially) non-centred parametrisation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>  
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::convertLatentPathToLatentPathRepar(const arma::colvec& theta, const LatentPath& latentPath, LatentPathRepar& latentPathRepar)
{
  model_.setUnknownParameters(theta);
  latentPathRepar.set_size(model_.getObservations().n_rows);
  for (unsigned int t=0; t<model_.getObservations().n_rows; t++)
  {
    latentPathRepar(t) = model_.getModelParameters().computeTAux(latentPath(t)) / (model_.getObservations()(t) - model_.getModelParameters().getDelta());
  }
}
/// Reparametrises latent variables from (partially) non-centred parametrisation
/// to the standard (centred) parametrisation
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>  
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::convertLatentPathReparToLatentPath(const arma::colvec& theta, LatentPath& latentPath, const LatentPathRepar& latentPathRepar)
{
  model_.setUnknownParameters(theta);
  latentPath.set_size(model_.getObservations().n_rows);
  bool isBracketing = true;

  for (unsigned int t=0; t<model_.getObservations().n_rows; t++)
  {
    latentPath(t)  = model_.getModelParameters().invertTAux(model_.getObservations()(t), latentPathRepar(t) * (model_.getObservations()(t) - model_.getModelParameters().getDelta()), isBracketing);
  }
}

#endif
