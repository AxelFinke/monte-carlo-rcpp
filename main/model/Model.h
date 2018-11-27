/// \file
/// \brief Generating data and performing (C)SMC-based inference in some model. 
///
/// This file contains the functions associated with an abstract model class
/// including generating simulated data and running (C)SMC algorithms.

#ifndef __MODEL_H
#define __MODEL_H

#include <omp.h> 
#include "main/rng/Rng.h"
#include "main/helperFunctions/helperFunctions.h"

// [[Rcpp::depends("RcppArmadillo")]]

/// Generic model class.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> class Model
{
  
public:
  
  /// Constructing the Model class using a specific vector of observations.
  Model
  (
    Rng& rng,
    const arma::colvec& hyperParameters,
    const Observations& observations,
    const unsigned int nCores 
  ) : 
    rng_(rng),
    observations_(observations), 
    inverseTemperatureObs_(1.0), 
    inverseTemperatureLat_(1.0), 
    nCores_(nCores)
  { 
    nObservations_ = 0; 
    std::cout << "Warning: nObservations is set to zero" << std::endl;
    modelParameters_.setKnownParameters(hyperParameters);
    marginaliseParameters_ = false;
  }
  /// Constructing the Model class using a specific vector of observations
  /// and also specifying the number of observations.
  Model
  (
    Rng& rng,
    const arma::colvec& hyperParameters,
    const Observations& observations,
    const unsigned int nObservations,
    const unsigned int nCores 
  ) : 
    rng_(rng),
    nObservations_(nObservations),
    observations_(observations), 
    inverseTemperatureObs_(1.0), 
    inverseTemperatureLat_(1.0), 
    nCores_(nCores)
  { 
    modelParameters_.setKnownParameters(hyperParameters);
    marginaliseParameters_ = false;
  }
/*  /// Constructing the Model class without data.
  Model
  (
    Rng& rng,
    const arma::colvec& hyperParameters,
    const arma::colvec& theta,
    const arma::colvec& extraParameters,
    const unsigned int nObservations,
    const unsigned int nCores
  ) : 
    rng_(rng), 
    nObservations_(nObservations),
    dimTheta_(theta.size()), 
    inverseTemperatureObs_(1.0), 
    inverseTemperatureLat_(1.0), 
    nCores_(nCores)
  { 
//     std::cout << "start setKNown" << std::endl;
    modelParameters_.setKnownParameters(hyperParameters);
//        std::cout << "start setUnknown" << std::endl;
    modelParameters_.setUnknownParameters(theta);
    marginaliseParameters_ = false;
//     std::cout << "end body of Model constructor" << std::endl;
  }*/ 
  /// Constructing the Model class without data and without
  /// specifying the number of observations nor the unknown parameters.
  Model
  (
    Rng& rng,
    const arma::colvec& hyperParameters,
    const unsigned int nCores
  ) : 
    rng_(rng), 
    inverseTemperatureObs_(1.0), 
    inverseTemperatureLat_(1.0), 
    nCores_(nCores)
  { 
    modelParameters_.setKnownParameters(hyperParameters);
    marginaliseParameters_ = false;
  } 
    /// Returns the reference to the class used for dealing with random-number generation.
  Rng& getRng() {return rng_;}
  /// Returns the length of the parameter vector.
  unsigned int getDimTheta() const {return dimTheta_;}
  /// Returns the model parameters.
  const ModelParameters& getModelParameters() const {return modelParameters_;}
  /// Returns the model parameters.
  ModelParameters& getRefModelParameters() {return modelParameters_;}
  /// Returns the minimum support of a model parameter.
  double getSupportMin(const unsigned int k) const {return support_(k,0);}
  /// Returns the minimum support of a model parameter.
  double getSupportMax(const unsigned int k) const {return support_(k,1);}
    /// Returns the inverse temperature used for tempering the observation density.
  double getNObservations() const {return nObservations_;}
  /// Returns the vector of latent variables.
  LatentPath getLatentPath() const {return latentPath_;}
  /// Returns the vector of latent variables.
  void getLatentPath(LatentPath& latentPath) const {latentPath = latentPath_;}
  /// Returns the observations.
  Observations& getRefObservations() {return observations_;}
  /// Returns the observations.
  const Observations& getObservations() const {return observations_;}
  /// Returns the observations.
  void getObservations(Observations& observations) const {observations = observations_;}
  /// Returns the inverse temperature used for tempering the observation density.
  double getInverseTemperatureObs() const {return inverseTemperatureObs_;}
  /// Returns the inverse temperature used for tempering the conditional prior density
  /// of the latent variables.
  double getInverseTemperatureLat() const {return inverseTemperatureLat_;}
  /// Returns whether model parameters should be integrated out analytically.
  bool getMarginaliseParameters() const {return marginaliseParameters_;};
  
  /// Specifies the length of the parameter vector.
  void setDimTheta(const unsigned int dimTheta) {dimTheta_ = dimTheta;}
  /// Specifies the support of the model parameters.
  void setSupport(const arma::mat& support) {support_ = support;}; 
  /// Specifies the model parameters.
  void setModelParameters(const ModelParameters& modelParameters) {modelParameters_ = modelParameters;} 
  /// Specifies the parameters which are to be inferred.
  void setUnknownParameters(const arma::colvec& theta) 
  {
    modelParameters_.setUnknownParameters(theta);
  }
  /// Specifies the known parameters (i.e. those which we do not infer).
  void setKnownParameters(const arma::colvec& hyperParameters) 
  {
    modelParameters_.setKnownParameters(hyperParameters);
  }
  /// Specifies the observations.
  void setObservations(const unsigned int nObservations, const Observations& observations)
  {
    nObservations_ = nObservations;
    observations_ = observations;
  }
  /// Specifies the observations.
  void setObservations(const Observations& observations) {observations_ = observations;}
  /// Specifies the inverse temperature used for tempering the model density.
  void setInverseTemperature(const double inverseTemperature)
  {
    inverseTemperatureObs_ = inverseTemperature;
    if (onlyTemperObservationDensity_)
    {
      inverseTemperatureLat_ = 1.0;
    }
    else
    {
      inverseTemperatureLat_ = inverseTemperature;
    }
  }
  /// Specifies whether only the observation density should be tempered (or also the conditional
  /// prior density of the latent variables).
  void setOnlyTemperObservationDensity(const bool onlyTemperObservationDensity)
  {
    onlyTemperObservationDensity_ = onlyTemperObservationDensity;
  }
  /// Specify whether model parameters should be integrated out analytically.
  void setMarginaliseParameters(const bool marginaliseParameters) 
  {
    marginaliseParameters_ = marginaliseParameters;
  };
  /// Simulates observations from the model.
  void simulateData(const arma::colvec& extraParameters);
  /// Evaluates the marginal likelihood of the parameters (with the latent 
  /// variables integrated out). Note that analytical expressions for this 
  /// will not be available for most models.
  double evaluateLogMarginalLikelihood();
  /// Evaluates the marginal likelihood of the parameters (with the latent 
  /// variables integrated out). Note that analytical expressions for this 
  /// will not be available for most models.
  double evaluateLogMarginalLikelihood(const arma::colvec& theta)
  {
    setUnknownParameters(theta);
    return evaluateLogMarginalLikelihood();
  }
  /// Evaluates the log of the likelihood of the parameters given the 
  /// latent variables.
  double evaluateLogCompleteLikelihood(const LatentPath& latentPath);
  /// Evaluates the log of the likelihood of the parameters given the 
  /// latent variables.
  double evaluateLogCompleteLikelihood(const arma::colvec& theta, const LatentPath& latentPath)
  {
    setUnknownParameters(theta);
    setInverseTemperature(1.0);
    return evaluateLogCompleteLikelihood(latentPath);
  }
  /// Evaluates the log of the likelihood of the parameters given the 
  /// latent variables.
  double evaluateLogCompleteLikelihood(const arma::colvec& theta, const LatentPath& latentPath, const double inverseTemperature)
  {
    setUnknownParameters(theta);
    setInverseTemperature(inverseTemperature);
    return evaluateLogCompleteLikelihood(latentPath);
  }
  /// Evaluates the log of the likelihood of the parameters given the 
  /// latent variables using a (partially) non-centred parametrisaion.
  double evaluateLogCompleteLikelihoodRepar(const LatentPathRepar& latentPathRepar);
  /// Evaluates the score. Note that analytical expressions for this 
  /// will not be available for most models.
  void evaluateScore(arma::colvec& score);
  /// Samples the set of parameters from the prior.
  void sampleFromPrior(arma::colvec& theta);
  /// Evaluates the log-prior density of the parameters.
  double evaluateLogPriorDensity();
  /// Evaluates the log-prior density of the parameters.
  double evaluateLogPriorDensity(const arma::colvec& theta)
  {
    modelParameters_.setUnknownParameters(theta);
    return evaluateLogPriorDensity();
  }
  /// Samples a single latent variable at Time t>0 from its conditional prior
  LatentVariable sampleFromInitialDistribution();
  /// Samples a single latent variable at Time t>0 from its conditional prior
  LatentVariable sampleFromTransitionEquation(const unsigned int t, const LatentVariable& latentVariableOld);
  /// Samples a single latent variable from its conditionall prior distribution
  /// in conditionally IID models.
  LatentVariable sampleFromLatentPrior(const unsigned int t);
  /// Samples a single observation at Time t>0 according to the observation equation
  /// and stores it in the vector observations.
  void sampleFromObservationEquation(const unsigned int t, Observations& observations, const LatentVariable& latentVariable);
  /// Evaluates the log-conditional prior density of the Time-t latent variable.
  double evaluateLogInitialDensity(const LatentVariable& latentVariable);
  /// Evaluates the log-conditional prior density of the Time-t latent variable.
  double evaluateLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld);
  /// Evaluates the log-conditional prior density of a latent variable
  /// in conditionally IID models.
  double evaluateLogLatentPriorDensity(const unsigned int t, const LatentVariable& latentVariable);
  /// Evaluates the log-observation density of the observations at Time t.
  double evaluateLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable);
  /// Increases the gradient by the gradient of the log-prior density.
  void addGradLogPriorDensity(arma::colvec& gradient);
  /// Increases the gradient by the gradient of the log-initial density of
  /// the latent states.
  void addGradLogInitialDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient);
  /// Increases the gradient by the gradient of the log-transition density.
  void addGradLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld, arma::colvec& gradient);
  /// Increases the gradient by the gradient of the prior density of a latent variable
  /// in conditionally IID models.
  void addGradLogLatentPriorDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient);
  /// Increases the gradient by the gradient of the log-observation density.
  void addGradLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient);
  /// Samples the latent variables from their full conditional distribution.
  void runGibbs(LatentPath& latentPath);
  /// Samples the latent variables from their full conditional distribution.
  void runGibbs(const arma::colvec& theta, LatentPath& latentPath, const double inverseTemperature)
  {
    setUnknownParameters(theta);
    setInverseTemperature(inverseTemperature);
    runGibbs(latentPath);
  }
  /// Evaluates the log of the likelihood associated with some subset of the 
  /// (static) model parameters.
  double evaluateLogMarginalLikelihoodFirst(LatentPath& latentPath);
  /// Evaluates the log of the likelihood associated with another subset of the 
  /// (static) model parameters.
  double evaluateLogMarginalLikelihoodSecond(LatentPath& latentPath);
  /// Evaluates the log of the likelihood associated with some subset of the 
  /// (static) model parameters.
  double evaluateLogMarginalLikelihoodFirst(const arma::colvec theta, LatentPath& latentPath)
  { 
    modelParameters_.setUnknownParameters(theta);
    return evaluateLogMarginalLikelihoodFirst(latentPath);
  }
  /// Evaluates the log of the likelihood associated with another subset of the 
  /// (static) model parameters.
  double evaluateLogMarginalLikelihoodSecond(const arma::colvec theta, LatentPath& latentPath)
  { 
    modelParameters_.setUnknownParameters(theta);
    return evaluateLogMarginalLikelihoodSecond(latentPath);
  }


  

private:
  
  Rng& rng_; // random number generation
  unsigned int nObservations_; // number of observations (unless this is random)
  arma::mat support_; // (dimTheta_, 2)-matrix containing the lower and upper bounds of the support of each parameter
  unsigned int dimTheta_; // length of the parameter vector theta
//   unsigned int dimLatentVariable_; // size of a single latent variable (if it can be represented as an arma::colvec)
//   unsigned int dimObservation_; // size of a single observation (if it can be represented as an arma::colvec)
  ModelParameters modelParameters_; // model parameters (a transformation of theta_)
  LatentPath latentPath_; // vector of latent variables (if observations are sampled from the model).
  Observations observations_; // vector of observations of length <nObservations>
  double inverseTemperatureObs_; // for tempering of the observation density; defaults to 1.0
  double inverseTemperatureLat_; // for tempering of the  conditional prior density of the latent states; defaults to 1.0
  bool onlyTemperObservationDensity_; // should only the observation density be tempered?
  bool marginaliseParameters_; // should certain model parameters be integrated out analytically?
  unsigned int nCores_; // number of cores to use (not currently used)
  
  
};

///////////////////////////////////////////////////////////////////////////////
/// Some non-member functions for use with in R.
///////////////////////////////////////////////////////////////////////////////
/// Returns the log-marginal likelihood.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double evaluateLogMarginalLikelihood
(
  const arma::colvec& hyperParameters, // hyper parameters
  const arma::colvec& theta, // parameters
  const Observations& observations, // vector of observations
  const unsigned int nCores // number of cores to use
)
{
  // TODO: need to make this static
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.getRefModelParameters().setUnknownParameters(theta);
  return model.evaluateLogMarginalLikelihood();
}
/// Simulates observations from the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void simulateData
( 
  const unsigned int nObservations, // number of observations
  const arma::colvec& hyperParameters, // hyper parameters
  const arma::colvec& theta, // parameters
  const arma::colvec& extraParameters, // additional parameters which are only used for generating the data
  LatentPath& latentPath, // latent variables
  Observations& observations, // observations
  const unsigned int nCores // number of cores to use
)
{
  // TODO: need to make this static
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, theta, nObservations, nCores);
  model.simulateData(extraParameters);
  model.getLatentPath(latentPath);
  model.getObservations(observations);
}
/// Simulates from the prior of the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void sampleFromPrior
( 
  arma::colvec& theta, // parameters
  const unsigned int dimTheta, // length of the parameter vector
  const arma::colvec& hyperParameters, // hyper parameters 
  const arma::mat& support, // matrix of boundaries of the support of the parameters 
  const unsigned int nCores // number of cores to use
)
{
//   std::cout << "dimTheta" << dimTheta << std::endl;
  theta.set_size(dimTheta);
  // TODO: need to make this static
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  /*
  std::uniform_int_distribution<unsigned int> unif;
  std::random_device rd;
  std::mt19937 engine(rd());
  std::function<unsigned int()> rnd = std::bind(unif, engine);
  RngDerived<std::mt19937> rngDerived(engine);
  */


  
  Observations observations;
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.sampleFromPrior(theta);
}

#endif
