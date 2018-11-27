#include <RcppArmadillo.h>

#include "projects/ensemble/comparison.h"
#include "main/applications/linear/linear.h"
#include "main/applications/linear/linearEnsemble.h"
#include "main/helperfunctions/kalman.h"

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
  arma::colvec extraParameters;
  LatentPath latentPath;
  Observations observations;
  simulateData<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>(nObservations, hyperParameters, theta, extraParameters, latentPath, observations, nCores);
  return Rcpp::List::create(Rcpp::Named("x") = latentPath, Rcpp::Named("y") = observations);
}

////////////////////////////////////////////////////////////////////////////////
// Approximates the marginal likelihood via SMC, EHMM, or alternative EHMM
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double approximateMarginalLikelihoodCpp
(
  const unsigned int samplerType,            // type of lower-level Monte Carlo algorithm to use
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& theta,                 // true static parameters (known in this case)
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const Observations& observations,          // observations
  const unsigned int smcProposalType,        // type of proposal kernel within the lower-level SMC sampler
  const unsigned int nSteps,                 // number of lower-level SMC steps
  const unsigned int nParticles,             // number of particles per MCMC iteration within each lower-level SMC algorithm
  const unsigned int ensembleNewInitialisationType, // type of initialisation of the MCMC chains at each time step of the sequential MCMC method
  const unsigned int nBurninSamples,         // number of additional samples discarded as burnin at each time step in the sequential MCMC method
  const double essResamplingThreshold,       // ESS-based resampling threshold for the lower-level SMC algorithms
  const arma::colvec& smcParameters, 
  const arma::colvec& ensembleOldParameters, 
  const arma::colvec& ensembleNewParameters,
  const unsigned int local,
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
  
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
  return approximateMarginalLikelihood<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, EnsembleOldParameters, EnsembleNewParameters>
    (static_cast<SamplerType>(samplerType), rngDerived, model, theta, smcProposalType, nSteps, nParticles, static_cast<EnsembleNewInitialisationType>(ensembleNewInitialisationType), nBurninSamples, essResamplingThreshold, smcParameters, ensembleOldParameters, ensembleNewParameters, static_cast<EnsembleNewLocalType>(local), nCores);
    
}


////////////////////////////////////////////////////////////////////////////////
// Runs a PMCMC algorithm (or related EHMM algorithm)
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runMcmcCpp
(
  const unsigned int mcmcType,               // 
  const unsigned int samplerType,            // 
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const Observations& observations,          // observations
  const unsigned int nIterations,            // number of MCMC iterations
  const unsigned int kern,                   // type of proposal kernel for the random-walk Metropolis--Hastings kernels
  const bool useGradients,                   // are we using gradient information in the parameter proposals?
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const unsigned int fixedLagSmoothingOrder, // lag-order for fixed-lag smoothing (currently only used to approximate gradients).
  const unsigned int smcProposalType,        // type of proposal kernel within the lower-level SMC sampler
  const unsigned int resampleType,           // type of resampling scheme to use in the lower-level SMC sampler
  const unsigned int nThetaUpdates,          // number of parameter updates per iteration of Gibbs samplers of CSMC-based algorithms
  const unsigned int nSteps,                 // number of lower-level SMC steps
  const unsigned int nParticles,             // number of particles per MCMC iteration within each lower-level SMC algorithm
  const unsigned int nSubsampledPaths,       // number of subsampled particle paths; only used by the asymmetric MH kernel(!)
  const double essResamplingThresholdLower,  // ESS-based resampling threshold for the lower-level SMC algorithms
  const unsigned int backwardSamplingType,   // type of backward-sampling scheme to use with the lower-level conditional SMC kernels
  const bool useNonCentredParametrisation,   // should Gibbs-sampling type algorithms use an NCP?
  const double nonCentringProbability,       // probability of using an NCP (if useNonCentredParametrisation == true)
  const arma::colvec& smcParameters, 
  const arma::colvec& ensembleOldParameters, 
  const arma::colvec& ensembleNewParameters,
  const unsigned int local,
  const bool estimateTheta,                  // should the static parameters be estimated? // TODO: only implemented for Gibbs samplers, so far
  const arma::colvec thetaInit,              // initial value for theta (if we keep theta fixed throughout) // TODO: only implemented for Gibbs samplers, so far
  const arma::uvec& times,                   // vector of length L which specifies the time steps of which state components are to be stored
  const arma::uvec& components,              // vector of length L which specifies the components which are to be stored
  const bool initialiseStatesFromStationarity, // should the CSMC-type algorithms initialise the state sequence from stationarity?
  const bool storeParentIndices,             // should the CSMC-type algorithms all their parent indices?
  const bool storeParticleIndices,           // should the CSMC-type algorithms store the indices of the (input and output) reference particle lineage?
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);

  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);

  // Class for running MCMC algorithms.
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, 
    static_cast<McmcKernelType>(kern), 
    useGradients, 0.0, rwmhSd, 0.0, nCores
  );
  
  std::vector<arma::colvec> theta(nIterations); // parameters sampled by the algorithm
  std::vector<arma::umat> parentIndices;
  std::vector<arma::uvec> particleIndicesIn;
  std::vector<arma::uvec> particleIndicesOut;
  
  arma::colvec ess(nSteps, arma::fill::zeros);  // storing the ess
  arma::colvec acceptanceRates(nSteps, arma::fill::zeros);  // storing the ess
  
  const unsigned int burnin = floor(0.1*nIterations); // TODO: make this accessible from R!

  arma::mat someStateComponents(components.size(), nIterations, arma::fill::zeros); // some components of the latent states sampled by the algorithm

  if (mcmcType == MCMC_MARGINAL)
  {
    runParticleMetropolis<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, EnsembleOldParameters, EnsembleNewParameters, McmcParameters>
      (theta, ess, acceptanceRates, static_cast<SamplerType>(samplerType), rngDerived, model, mcmc, nIterations, burnin, useGradients, fixedLagSmoothingOrder, 
      smcProposalType, nSteps, nParticles, essResamplingThresholdLower, smcParameters, ensembleOldParameters, ensembleNewParameters, static_cast<EnsembleNewLocalType>(local), nCores);
  }
  else if (mcmcType == MCMC_GIBBS)
  {
    if (storeParentIndices) 
    {
      parentIndices.resize(nIterations);
    }
    if (storeParticleIndices) 
    {
      particleIndicesIn.resize(nIterations);
      particleIndicesOut.resize(nIterations);
    }
    
    runParticleGibbs<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, EnsembleOldParameters, EnsembleNewParameters, McmcParameters>
      (theta, someStateComponents, parentIndices, particleIndicesIn, particleIndicesOut, storeParticleIndices, storeParentIndices, initialiseStatesFromStationarity, ess, acceptanceRates, times, components, static_cast<SamplerType>(samplerType), rngDerived, model, mcmc, nIterations, burnin,
      smcProposalType, nThetaUpdates, nSteps, nParticles, essResamplingThresholdLower, backwardSamplingType, useNonCentredParametrisation, nonCentringProbability,  smcParameters, ensembleOldParameters, ensembleNewParameters, static_cast<EnsembleNewLocalType>(local), static_cast<ResampleType>(resampleType), estimateTheta, thetaInit, nCores);

  }
  else if (mcmcType == MCMC_ASYMMETRIC)
  {
    runAsymmetricMh<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, EnsembleOldParameters, EnsembleNewParameters, McmcParameters>
    (
      theta, someStateComponents, initialiseStatesFromStationarity, times, components, static_cast<SamplerType>(samplerType), rngDerived, model, mcmc, nIterations, burnin, smcProposalType, nSteps, nParticles, nSubsampledPaths, essResamplingThresholdLower,
      smcParameters, ensembleOldParameters, ensembleNewParameters, static_cast<EnsembleNewLocalType>(local), static_cast<ResampleType>(resampleType), thetaInit, nCores
    );
  }

  if (mcmcType == MCMC_GIBBS && estimateTheta == false && storeParentIndices && storeParticleIndices)
  {
    return Rcpp::List::create(
      Rcpp::Named("parentIndices") = parentIndices, 
      Rcpp::Named("particleIndicesIn") = particleIndicesIn,
      Rcpp::Named("particleIndicesOut") = particleIndicesOut,
      Rcpp::Named("states") = someStateComponents, 
      Rcpp::Named("ess") = ess, 
      Rcpp::Named("acceptanceRates") = acceptanceRates
    );
  }
  else if (mcmcType == MCMC_GIBBS && estimateTheta == false && storeParticleIndices)
  {
    return Rcpp::List::create(
      Rcpp::Named("particleIndicesIn") = particleIndicesIn,
      Rcpp::Named("particleIndicesOut") = particleIndicesOut,
      Rcpp::Named("states") = someStateComponents, 
      Rcpp::Named("ess") = ess, 
      Rcpp::Named("acceptanceRates") = acceptanceRates
    );
  }
  else if (mcmcType == MCMC_GIBBS && estimateTheta == false && storeParentIndices)
  {
    return Rcpp::List::create(
      Rcpp::Named("parentIndices") = parentIndices, 
      Rcpp::Named("states") = someStateComponents, 
      Rcpp::Named("ess") = ess, 
      Rcpp::Named("acceptanceRates") = acceptanceRates
    );
  }
  else 
  {
    return Rcpp::List::create(
      Rcpp::Named("theta") = theta, 
      Rcpp::Named("states") = someStateComponents, 
      Rcpp::Named("ess") = ess, 
      Rcpp::Named("acceptanceRates") = acceptanceRates
    );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Runs a Kalman smoother to obtain the 
// predicted, updated and smoothed means and variances
// in a univariate linear-Gaussian state space model
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runKalmanSmootherUnivariateCpp
(
  const double A, 
  const double B, 
  const double C, 
  const double D, 
  const double m0, 
  const double C0, 
  const arma::colvec& y
)
{
  
  arma::colvec mP; // predicted means
  arma::colvec CP; // predicted variances
  arma::colvec mU; // updated means
  arma::colvec CU; // updated variances
  arma::colvec mS; // smoothed means
  arma::colvec CS; // smoothed variances

  kalman::runForwardFilteringBackwardSmoothing(mP, CP, mU, CU, mS, CS, A, B, C, D, m0, C0, y);
  
  return Rcpp::List::create(
    Rcpp::Named("predictedMeans")     = mP, 
    Rcpp::Named("predictedVariances") = CP, 
    Rcpp::Named("updatedMeans")       = mU, 
    Rcpp::Named("updatedVariances")   = CU,
    Rcpp::Named("smoothedMeans")      = mS, 
    Rcpp::Named("smoothedVariances")  = CS
  );
}
