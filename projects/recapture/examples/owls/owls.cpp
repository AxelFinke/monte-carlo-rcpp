#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>

#include "main/algorithms/mcmc/pmmh.h"
#include "main/algorithms/smc/SmcSampler.h"
#include "main/applications/owls/owls.h"
#include "time.h"

// TODO: disable range checks (by using at() for indexing elements of cubes/matrices/vectors)
// once the code is tested; 
// To that end, compile the code with ARMA_NO_DEBUG defined.

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
  
//           std::cout << "Start sample from prior" << std::endl;
  arma::colvec theta;
  sampleFromPrior<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>(theta, dimTheta, hyperParameters, support, nCores);
  
//           std::cout << "Finished sample from prior" << std::endl;
  return theta;
}

/*
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
*/


////////////////////////////////////////////////////////////////////////////////
// Runs a PMMH algorithm 
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runPmmhCpp
(
  const arma::umat& fecundity,               // fecundity data
  const arma::uvec& count,               // count data
  const arma::umat& capRecapFemaleFirst,     // capture-recapture matrix for first-year females
  const arma::umat& capRecapMaleFirst,       // capture-recapture matrix for first-year males
  const arma::umat& capRecapFemaleAdult,     // capture-recapture matrix for adult females
  const arma::umat& capRecapMaleAdult,       // capture-recapture matrix for adult males
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const unsigned int nIterations,            // number of MCMC iterations
  const unsigned int nParticles,             // number of particles per MCMC iteration within each lower-level SMC algorithm
  const double essResamplingThreshold,       // ESS-based resampling threshold for the lower-level SMC algorithms
  const arma::colvec& smcParameters,         // additional parameters to be passed to the particle filter
  const arma::colvec& mcmcParameters,        // additional parameters to be passed to the MCMC kernel
  const bool useDelayedAcceptance,           // should we combine the PMMH update with a delayed-acceptace step?
  const bool useAdaptiveProposal,            // should we adapt the proposal scale of the MCMC kernel as in Peters at al. (2010)?
  const bool useAdaptiveProposalScaleFactor1, // should we also adapt the constant by which the sample covariance matrix is multiplied?
  const arma::colvec& adaptiveProposalParameters, // parameters needed for the adaptive mixture proposal from Peters at al. (2010).
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const arma::colvec& thetaInit,             // initial value for theta (if we keep theta fixed throughout) 
  const double burninPercentage,             // percentage iterations to be thrown away as burnin
  const bool samplePath,                     // store particle paths?
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  unsigned int nObservations = count.size(); // number of observations
  arma::colvec observationTimes = arma::ones<arma::colvec>(nObservations); // intervals in which observations have been recorded (not used here)
  unsigned int nSteps = nObservations; // number of lower-level SMC steps

 
  Observations observations; // observations //TODO:
  observations.fecundity_           = fecundity;
  observations.count_               = count;
  observations.capRecapFemaleFirst_ = capRecapFemaleFirst;
  observations.capRecapMaleFirst_   = capRecapMaleFirst;
  observations.capRecapFemaleAdult_ = capRecapFemaleAdult;
  observations.capRecapMaleAdult_   = capRecapMaleAdult;
  
  observations.releasedFemaleFirst_ = arma::sum(capRecapFemaleFirst, 1);
  observations.releasedMaleFirst_   = arma::sum(capRecapMaleFirst, 1);
  observations.releasedFemaleAdult_ = arma::sum(capRecapFemaleAdult, 1);
  observations.releasedMaleAdult_   = arma::sum(capRecapMaleAdult, 1);
  
  
  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(0), 
    essResamplingThreshold,
    static_cast<SmcBackwardSamplingType>(1),
    false,
    1,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.setNParticles(nParticles);
  smc.setSamplePath(samplePath);

  // Class for running MCMC algorithms.
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, nCores
  );
  mcmc.setRwmhSd(rwmhSd);
  mcmc.setUseAdaptiveProposal(useAdaptiveProposal);
  mcmc.setUseDelayedAcceptance(useDelayedAcceptance);
  mcmc.setAdaptiveProposalParameters(adaptiveProposalParameters, nIterations);
  mcmc.setUseAdaptiveProposalScaleFactor1(useAdaptiveProposalScaleFactor1);
  mcmc.setNIterations(nIterations, burninPercentage);
  
  std::vector<arma::colvec> theta(nIterations); // parameters sampled by the algorithm
  std::vector<arma::umat> latentPath(nIterations); // one latent path sampled and stored at each iteration
  double cpuTime; // total amount of time needed for running the algorithm
  double acceptanceRateStage1; // first-stage acceptance rate after burn-in (if delayed-acceptance is used)
  double acceptanceRateStage2; // (second-stage) acceptance rate after burn-in
  
  runPmmh<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters,McmcParameters>
    (theta, latentPath, cpuTime, acceptanceRateStage1, acceptanceRateStage2, rngDerived, model, smc, mcmc, thetaInit, samplePath, nCores);
  
  
  return Rcpp::List::create(
    Rcpp::Named("theta") = theta, 
    Rcpp::Named("latentPath") = latentPath, 
    Rcpp::Named("cpuTime") = cpuTime, // total time needed for running the algorithm
    Rcpp::Named("acceptanceRateStage1") = acceptanceRateStage1, // first-stage acceptance rate after burn-in (if delayed-acceptance is used)
    Rcpp::Named("acceptanceRateStage2") = acceptanceRateStage2 // (second-stage) acceptance rate after burn-in
  );
}


////////////////////////////////////////////////////////////////////////////////
// Runs an SMC sampler to estimate the model evidence
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runSmcSamplerCpp
(
  const arma::umat& fecundity,               // fecundity data
  const arma::uvec& count,                   // count data
  const arma::umat& capRecapFemaleFirst,     // capture-recapture matrix for first-year females
  const arma::umat& capRecapMaleFirst,       // capture-recapture matrix for first-year males
  const arma::umat& capRecapFemaleAdult,     // capture-recapture matrix for adult females
  const arma::umat& capRecapMaleAdult,       // capture-recapture matrix for adult males
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const unsigned int lower,                  // type of algorithm for updating the count-data likelihood (0: pseudo-marginal; 2: MCWM)
  const unsigned int nParticlesUpper,        // number of upper-level particles
  const unsigned int nParticlesLower,        // number of lower-level (i.e. "filtering") particles
  const unsigned int nMetropolisHastingsUpdates, // number of MH updates per particle and SMC step
  const unsigned int nMetropolisHastingsUpdatesFirst, // number of MH updates per particle and SMC step at the first stage if we use dual tempering
  const double essResamplingThresholdUpper,  // ESS-based resampling threshold for the upper-level SMC sampler
  const double essResamplingThresholdLower,  // ESS-based resampling threshold for the lower-level SMC filter
  const arma::colvec& smcParameters,         // additional parameters to be passed to the particle filter
  const arma::colvec& mcmcParameters,        // additional parameters to be passed to the MCMC kernel
  const bool useAdaptiveTempering,           // should we determine the tempering schedule adaptively according to the CESS?
  const bool useAdaptiveCessTarget,          // should we adapt the CESS target based on the autocorrelation of the MCMC kernels?
  const bool useImportanceTempering,         // should we compute and return the importance-tempering weights?
  const bool useDelayedAcceptance,           // should we combine the PMMH update with a delayed-acceptace step?
  const bool useAdaptiveProposal,            // should we adapt the proposal scale of the MCMC kernel as in Peters at al. (2010)?
  const bool useAdaptiveProposalScaleFactor1, // should we also adapt the constant by which the sample covariance matrix is multiplied?
  const bool useDoubleTempering,             // should we temper both likelihood terms separately?
  const double cessTarget,                   // CESS target threshold (only used if useAdaptiveTempering == true).
  const double cessTargetFirst,              // CESS target threshold (only used if useAdaptiveTempering == true and useDoubleTempering == true).
  const arma::colvec& alpha,                 // manually specified tempering schedule (only used if useAdaptiveTempering == false)
  const arma::colvec& adaptiveProposalParameters, // parameters needed for the adaptive mixture proposal from Peters at al. (2010).
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
//     std::cout << "setting up the observations" << std::endl;
  
  // Starting the timer:
  clock_t t1,t2; // timing
  t1 = clock(); // start timer
  
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  unsigned int nObservations = count.size(); // number of observations
  arma::colvec observationTimes = arma::ones<arma::colvec>(nObservations); // intervals in which observations have been recorded (not used here)
  unsigned int nStepsLower = nObservations; // number of lower-level SMC steps


  
  Observations observations; // observations //TODO:
  observations.fecundity_           = fecundity;
  observations.count_               = count;
  observations.capRecapFemaleFirst_ = capRecapFemaleFirst;
  observations.capRecapMaleFirst_   = capRecapMaleFirst;
  observations.capRecapFemaleAdult_ = capRecapFemaleAdult;
  observations.capRecapMaleAdult_   = capRecapMaleAdult;
  
  observations.releasedFemaleFirst_ = arma::sum(capRecapFemaleFirst, 1);
  observations.releasedMaleFirst_   = arma::sum(capRecapMaleFirst, 1);
  observations.releasedFemaleAdult_ = arma::sum(capRecapFemaleAdult, 1);
  observations.releasedMaleAdult_   = arma::sum(capRecapMaleAdult, 1);
  
//   std::cout << "setting up the model" << std::endl;
  
  /////////////////////////////////////////////////////////////////////////////
  // Model class.
  /////////////////////////////////////////////////////////////////////////////
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
//     std::cout << "setting up the SMC filter class" << std::endl;
  
  /////////////////////////////////////////////////////////////////////////////
  // Class for running the lower-level SMC algorithm.
  /////////////////////////////////////////////////////////////////////////////
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nStepsLower,
    static_cast<SmcProposalType>(0), 
    essResamplingThresholdLower,
    static_cast<SmcBackwardSamplingType>(1),
    false,
    1,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.setNParticles(nParticlesLower);
  smc.setSamplePath(false);
  
  /////////////////////////////////////////////////////////////////////////////
  // Class for running MCMC algorithms.
  /////////////////////////////////////////////////////////////////////////////
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, nCores
  );
  mcmc.setRwmhSd(rwmhSd);
  mcmc.setUseAdaptiveProposal(useAdaptiveProposal);
  mcmc.setUseDelayedAcceptance(useDelayedAcceptance);
  mcmc.setAdaptiveProposalParameters(adaptiveProposalParameters);
  mcmc.setUseAdaptiveProposalScaleFactor1(useAdaptiveProposalScaleFactor1);
  mcmc.setIsWithinSmcSampler(true);
  
  /////////////////////////////////////////////////////////////////////////////
  // Class for running the upper-level SMC sampler.
  /////////////////////////////////////////////////////////////////////////////
  
  SmcSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters> smcSampler(rngDerived, model, smc, mcmc, nCores);
  smcSampler.setNParticles(nParticlesUpper);
  smcSampler.setEssResamplingThreshold(essResamplingThresholdUpper);
  smcSampler.setUseAdaptiveTempering(useAdaptiveTempering);
  smcSampler.setLower(static_cast<SmcSamplerLowerType>(lower)); // use lower-level pseudo-marginal approach
  smcSampler.setNMetropolisHastingsUpdates(nMetropolisHastingsUpdates);
  smcSampler.setUseAdaptiveCessTarget(useAdaptiveCessTarget);
  
  smcSampler.setUseDoubleTempering(useDoubleTempering);
  if (useDoubleTempering)
  {
     smcSampler.setNMetropolisHastingsUpdatesFirst(nMetropolisHastingsUpdatesFirst);
  }

  if (smcSampler.getUseAdaptiveTempering()) 
  {
    std::cout << "using adaptive tempering" << std::endl;
    smcSampler.setCessTarget(cessTarget); // specify the sequence of temperatures (and the number of steps) adaptively based on the CESS.
    if (useDoubleTempering) 
    {
      smcSampler.setCessTargetFirst(cessTargetFirst); // specify the sequence of temperatures (and the number of steps) adaptively based on the CESS.
    }
  }
  else // i.e. if we specifiy the sequence of temperatures manually
  {
     std::cout << "manually specified tempering schedule" << std::endl;
    smcSampler.setAlpha(alpha); // manually specify the sequence of temperatures (and in particular, the number of steps).
  }

  smcSampler.runSmcSampler(); // running the SMC sampler
  
  if (useImportanceTempering)
  {
    smcSampler.computeImportanceTemperingWeights();
  }
  
  t2 = clock(); // stop timer 
  double cpuTime = (static_cast<double>(t2)-static_cast<double>(t1)) / CLOCKS_PER_SEC; // elapsed time in seconds
  std::cout << "Running the SMC sampler took " << cpuTime << " seconds." << std::endl;
  
  if (useImportanceTempering)
  {
    return Rcpp::List::create(             
      Rcpp::Named("inverseTemperatures")                       = smcSampler.getAlpha(),
      Rcpp::Named("logUnnormalisedWeights")                    = smcSampler.getLogUnnormalisedWeightsFull(),
      Rcpp::Named("selfNormalisedWeights")                     = smcSampler.getSelfNormalisedWeightsFull(),  
      Rcpp::Named("theta")                                     = smcSampler.getThetaFull(), 
      Rcpp::Named("cpuTime")                                   = cpuTime,
      Rcpp::Named("logEvidenceEstimate")                       = smcSampler.getLogEvidenceEstimate(),
//       Rcpp::Named("logEvidenceEstimateEss")                    = smcSampler.getLogEvidenceEstimateEss(),
      Rcpp::Named("logEvidenceEstimateEssAlternate")           = smcSampler.getLogEvidenceEstimateEssAlternate(),
//       Rcpp::Named("logEvidenceEstimateEssResampled")           = smcSampler.getLogEvidenceEstimateEssResampled(),
      Rcpp::Named("logEvidenceEstimateEssResampledAlternate")  = smcSampler.getLogEvidenceEstimateEssResampledAlternate(),
      Rcpp::Named("logUnnormalisedReweightedWeights")          = smcSampler.getLogUnnormalisedReweightedWeights(), 
      Rcpp::Named("logUnnormalisedReweightedWeightsResampled") = smcSampler.getLogUnnormalisedReweightedWeightsResampled(),
      Rcpp::Named("selfNormalisedReweightedWeights")           = smcSampler.getSelfNormalisedReweightedWeights(), 
      Rcpp::Named("selfNormalisedReweightedWeightsResampled")  = smcSampler.getSelfNormalisedReweightedWeightsResampled(),
      Rcpp::Named("selfNormalisedWeightsEss")                  = smcSampler.getSelfNormalisedWeightsEss(),
      Rcpp::Named("selfNormalisedWeightsEssResampled")         = smcSampler.getSelfNormalisedWeightsEssResampled(),
      Rcpp::Named("ess")                                       = smcSampler.getEss(),
      Rcpp::Named("essResampled")                              = smcSampler.getEssResampled(),
      Rcpp::Named("acceptanceRates")                           = smcSampler.getAcceptanceRates(),
      Rcpp::Named("maxParticleAutocorrelations")               = smcSampler.getMaxParticleAutocorrelations()
    );
  }
  else 
  {
    return Rcpp::List::create(
      Rcpp::Named("logEvidenceEstimate")                       = smcSampler.getLogEvidenceEstimate(),
      Rcpp::Named("logUnnormalisedWeights")                    = smcSampler.getLogUnnormalisedWeightsFull(),                        
      Rcpp::Named("selfNormalisedWeights")                     = smcSampler.getSelfNormalisedWeightsFull(),
      Rcpp::Named("inverseTemperatures")                       = smcSampler.getAlpha(),
      Rcpp::Named("theta")                                     = smcSampler.getThetaFull(), 
      Rcpp::Named("cpuTime")                                   = cpuTime,
      Rcpp::Named("acceptanceRates")                           = smcSampler.getAcceptanceRates(),
      Rcpp::Named("maxParticleAutocorrelations")               = smcSampler.getMaxParticleAutocorrelations()
    );
  }

}
