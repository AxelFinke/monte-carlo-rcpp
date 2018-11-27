#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>

#include "mcmc/gibbsSampler.h"
#include "examples/levy/levy.h"
#include "time.h"

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

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
//////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List simulateDataCpp
(
  const unsigned int nObservationsCount, // number of count-data observations
  const arma::colvec& hyperParameters, // hyper parameters 
  const arma::colvec& theta, // parameters
  const unsigned int nCores = 1 // number of cores to use
)
{
  std::cout << "start simulate data" << std::endl;
  
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, theta, nObservations, nCores);
  model.simulateData();
  
  return Rcpp::List::create
  (
    Rcpp::Named("observations") = model.getObservations(),
    Rcpp::Named("jumpTimes")    = model.getLatentPath().jumpTimes_,
    Rcpp::Named("jumpSizes")    = model.getLatentPath().jumpSizes_
  );
}

////////////////////////////////////////////////////////////////////////////////
// Runs a Gibbs sampler
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runGibbsSamplerCpp
(
  const arma::colvec& observations,          // vector of observations
  const arma::colvec& observationTimes,      // times at which observations were recorded (
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const unsigned int nIterations,            // number of Gibbs-sampler sweeps (including burn-in)
  const unsigned int nParameterUpdates,      // number of model-parameter updates in between each set of latent-variable updates
  const double nonCentringProbability,       // probability of switching to a non-centred parametrisation for the parameter updates (if implemented)
  const arma::colvec& thetaInit,             // initial value for theta (if we keep theta fixed throughout) 
  const unsigned int samplerType,            // type of algorithm to be used for updating the latent variables; 0: Metropolis-within-Gibbs; 1: (conditional) SMC; 2: (conditional) EHMM; 3: sample latent variables from their full conditional posterior distribution (if available)
  const unsigned int marginalisationType,    // type analytical marginalisation of certain model parameters: 0: no marginalisation; 1: marginalisation only during parameter updates; 2: marginalisation during both latent-variable and parameter updates
  const arma::colvec& proposalScales,        // standard deviations of the uncorrelated Gaussian-random walk proposals for the full vector of model parameters
  const arma::colvec& proposalScalesMarginalised, // standard deviations of the uncorrelated Gaussian-random walk proposals for the vector of model parameters after certain parameters have been integrated out analytically (if possible)
  const unsigned int nLatentVariableUpdates, // number of times the Metropolis-within-Gibbs kernels are applied to update the latent variables in between each set of parameter updates
  const unsigned int moveProbabilities,      // probabilities for the five different reversible-jump moves
  const unsigned int nParticles,             // number of particles used by the SMC/EHMM updates
  const double essResamplingThreshold,       // ESS-based resampling threshold for the SMC/EHMM updates
  const arma::colvec& stepTimes,             // upper limit of the time intervals determining which observation are included in a particular SMC/EHMM step
  const unsigned int backwardSamplingType,   // 0: no backward simulation for the conditional SMC/EHMM updates; 1: backward sampling; 2: ancestor sampling
  const arma::colvec& mwgParameters,         // additional parameters to be passed to the Metropolis-within-Gibbs updates for the latent variables
  const arma::colvec& smcParameters,         // additional parameters to be passed to the standard conditional sequential Monte Carlo updates for the latent variables
  const arma::colvec& ehmmParameters,        // additional parameters to be passed to the embedded hidden Markov model updates for the latent variables
  const bool estimateTheta,                  // should the model parameters be estimated? Otherwise, they are kept fixed to their initial values
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{

  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Setting up the Rng class" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Setting up the Model class" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, nCores);
  model.setSupport(support);
  model.setDimTheta(thetaInit.size());
  model.getRefModelParameters().setObservationTimes(observationTimes);
  model.setObservations(observations.size(), observations);
 
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Setting up the Mwg class" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  Mwg<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, MwgParameters> mwg(
    rngDerived, model, nCores
  );
  mwg.setNUpdates(nLatentVariableUpdates);
  mwg.setMoveProbabilities(moveProbabilities);

  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Setting up the Smc class" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nCores
  );
  smc.setNParticles(nParticles);
  smc.getRefSmcParameters().setStepTimes(stepTimes);
  smc.getRefSmcParameters().computeBinnedObservations(observations, ObservationTimes);
  smc.setNSteps(stepTimes.size());
  smc.setSamplePath(true);
  smc.setEssResamplingThreshold(essResamplingThreshold);
  smc.setSmcBackwardSamplingType(static_cast<SmcBackwardSamplingType>(backwardSamplingType));
  
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Setting up the Ehmm class" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  Ehmm<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, EhmmParameters> ehmm(
    rngDerived, model, nCores
  );
  ehmm.setNParticles(nParticles);
  ehmm.setNSteps(stepTimes.size());
  ehmm.setSamplePath(true);
  ehmm.setEssResamplingThreshold(essResamplingThreshold);
  ehmm.setSmcBackwardSamplingType(static_cast<EhmmBackwardSamplingType>(backwardSamplingType));
  
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Setting up the GibbsSampler class" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, MwgParameters, SmcParameters, EhmmParameters> gibbsSampler(
    rngDerived, model, mwg, smc, ehmm, nCores
  );
  gibbsSampler.setSamplerType(static_cast<SamplerType>(samplerType));
  gibbsSampler.setNIterations(nIterations);
  gibbsSampler.setEstimateTheta(estimateTheta);
  gibbsSampler.setNParameterUpdates(nParameterUpdates);
  gibbsSampler.setNonCentringProbability(nonCentringProbability);
  gibbsSampler.setProposalScales(proposalScales);
  gibbsSampler.setProposalScalesMarginalised(proposalScalesMarginalised);
  
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Running the Gibbs sampler" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  std::vector<arma::colvec> output(nIterations); // parameters sampled by the algorithm
  double cpuTime; // total amount of time needed for running the algorithm
  gibbsSampler.run(output, thetaInit)
  
  /////////////////////////////////////////////////////////////////////////////
  std::cout << "Returning the output" << std::endl;
  /////////////////////////////////////////////////////////////////////////////
  return Rcpp::List::create(
    Rcpp::Named("output") = output,
    Rcpp::Named("cpuTime") = cpuTime             
  );
}
