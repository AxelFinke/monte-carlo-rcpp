#include <RcppArmadillo.h>

#include "optim/Optim.h"
#include "examples/random/random.h"

// TODO: disable range checks (by using at() for indexing elements of cubes/matrices/vectors)
// once the code is tested; 
// To that end, compile the code with ARMA_NO_DEBUG defined.

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

////////////////////////////////////////////////////////////////////////////////
// Returns log-marginal likelihood.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double evaluateLogMarginalLikelihoodCpp
(
  const arma::colvec& hyperParameters, // hyper parameters
  const arma::colvec& theta, // parameters
  const Observations& observations, // observations
  const unsigned int nCores = 1 // number of cores to use
)
{
  return evaluateLogMarginalLikelihood<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>(hyperParameters, theta, observations, nCores);
}
////////////////////////////////////////////////////////////////////////////////
// Samples from the prior of the parameters.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::colvec sampleFromPriorCpp
(
  const unsigned int dimTheta, // length of the parameter vector
  const arma::colvec& hyperParameters, // hyper parameters
  const arma::mat& support, // matrix containing the boundaries of the suppport of the parameters
  const unsigned int nCores = 1 // number of cores to use
)
{ 
  arma::colvec theta;
  sampleFromPrior<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>(theta, dimTheta, hyperParameters, support, nCores);
  return theta;
}
////////////////////////////////////////////////////////////////////////////////
// Simulates observations.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List simulateDataCpp
(
  const unsigned int nObservations, // number of observations
  const arma::colvec& hyperParameters, // hyper parameters 
  const arma::colvec& theta, // parameters
  const unsigned int nCores = 1// number of cores to use
)
{
  LatentPath latentPath;
  Observations observations;
  simulateData<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>(nObservations, hyperParameters, theta, latentPath, observations, nCores);
  return Rcpp::List::create(Rcpp::Named("x") = latentPath, Rcpp::Named("y") = observations);
}

////////////////////////////////////////////////////////////////////////////////
// Pseudo-marginal and pseuod-Gibbs sampling based
// optimisation and related algorithms (MCMC version)
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List optimMcmcCpp
(
  const unsigned int lower,                  // type of lower-level Monte Carlo algorithm to use
  const arma::colvec& inverseTemperatures,   // vector of inverse temperatures
  const bool areInverseTemperaturesIntegers, // are we only considering integer-valued inverse temperatures?
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const Observations& observations,          // observations
  const double proposalDownScaleProbability, // probability of using an RWMH proposal whose scale decreases in the inverse temperature
  const unsigned int kern,                   // type of proposal kernel for the random-walk Metropolis--Hastings kernels
  const bool useGradients,                   // are we using gradient information in the parameter proposals?
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const unsigned int fixedLagSmoothingOrder, // lag-order for fixed-lag smoothing (currently only used to approximate gradients).
  const double crankNicolsonScale,           // correlation parameter for Crank--Nicolson proposals
  const double proportionCorrelated,         // for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  const unsigned int smcProposalType,        // type of proposal kernel within the lower-level SMC sampler
  const unsigned int nThetaUpdates,          // number of parameter updates per iteration of Gibbs samplers of CSMC-based algorithms
  const bool onlyTemperObservationDensity,   // should only the observation densities be tempered?
  const unsigned int nSmcStepsLower,         // number of lower-level SMC steps
  const arma::uvec& nParticlesLower,         // number of particles per MCMC iteration within each lower-level SMC algorithm
  const double essResamplingThresholdLower,  // ESS-based resampling threshold for the lower-level SMC algorithms
  const unsigned int backwardSamplingType,   // type of backward-sampling scheme to use with the lower-level conditional SMC kernels
  const bool useNonCentredParametrisation,   // should Gibbs-sampling type algorithms use an NCP?
  const double nonCentringProbability,       // probability of using an NCP (if useNonCentredParametrisation == true)
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
  std::vector<arma::colvec> output;
  optimMcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>
  (
    output, lower, inverseTemperatures, areInverseTemperaturesIntegers, dimTheta,
    hyperParameters, support, observations, proposalDownScaleProbability, 
    kern, useGradients, rwmhSd, fixedLagSmoothingOrder, crankNicolsonScale, proportionCorrelated, smcProposalType, 
    nThetaUpdates, onlyTemperObservationDensity, nSmcStepsLower, nParticlesLower, 
    essResamplingThresholdLower, backwardSamplingType, useNonCentredParametrisation, nonCentringProbability, nCores
  );
  return Rcpp::List::create(Rcpp::Named("theta") = output);
}

////////////////////////////////////////////////////////////////////////////////
// Pseudo-marginal and pseuod-Gibbs sampling based
// optimisation and related algorithms (SMC version)
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List optimSmcCpp
(
  const unsigned int lower,                  // type of lower-level Monte Carlo algorithm to use
  const arma::colvec& inverseTemperatures,   // vector of inverse temperatures
  const bool areInverseTemperaturesIntegers, // are we only considering integer-valued inverse temperatures?
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const Observations& observations,          // observations
  const double proposalDownScaleProbability, // probability of using an RWMH proposal whose scale decreases in the inverse temperature
  const unsigned int kern,                   // type of proposal kernel for the random-walk Metropolis--Hastings kernels
  const bool useGradients,                   // are we using gradient information in the parameter proposals?
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const unsigned int fixedLagSmoothingOrder, // lag-order for fixed-lag smoothing (currently only used to approximate gradients).
  const double crankNicolsonScale,           // correlation parameter for Crank--Nicolson proposals
  const double proportionCorrelated,         // for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  const unsigned int smcProposalType,        // type of proposal kernel within the lower-level SMC sampler
  const unsigned int nThetaUpdates,          // number of parameter updates per iteration of Gibbs samplers of CSMC-based algorithms
  const bool onlyTemperObservationDensity,   // should only the observation densities be tempered?
  const unsigned int nSmcStepsLower,         // number of lower-level SMC steps
  const arma::uvec& nParticlesLower,         // number of particles per MCMC iteration within each lower-level SMC algorithm
  const double essResamplingThresholdLower,  // ESS-based resampling threshold for the lower-level SMC algorithms
  const unsigned int backwardSamplingType,   // type of backward-sampling scheme to use with the lower-level conditional SMC kernels
  const unsigned int nParticlesUpper,        // number of particles in the upper-level SMC sampler
  const double essResamplingThresholdUpper,  // ESS-based resampling threshold for the lower-level SMC algorithms
  const bool useNonCentredParametrisation,   // should Gibbs-sampling type algorithms use an NCP?
  const double nonCentringProbability,       // probability of using an NCP (if useNonCentredParametrisation == true)
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
  std::vector< std::vector<arma::colvec> > output;
  optimSmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>
  (
    output, lower, inverseTemperatures, areInverseTemperaturesIntegers, dimTheta,
    hyperParameters, support, observations, proposalDownScaleProbability, 
    kern, useGradients, rwmhSd, fixedLagSmoothingOrder, crankNicolsonScale, proportionCorrelated, smcProposalType, 
    nThetaUpdates, onlyTemperObservationDensity, nSmcStepsLower, nParticlesLower, 
    essResamplingThresholdLower, backwardSamplingType, nParticlesUpper, 
    essResamplingThresholdUpper, useNonCentredParametrisation, nonCentringProbability, nCores
  );
  return Rcpp::List::create(Rcpp::Named("theta") = output);
}

