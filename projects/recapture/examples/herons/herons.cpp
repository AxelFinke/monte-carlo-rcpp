#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
// #include <gperftools/profiler.h>

#include "mcmc/pmmh.h"
#include "smc/SmcSampler.h"
#include "examples/herons/herons.h"
// #include "examples/herons/heronsContinuous.h"
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
  const arma::umat& ringRecovery,        // capture-recapture matrix for first-year females
  const unsigned int nCores = 1// number of cores to use
)
{
  LatentPath latentPath;
  Observations observations;
  
  // TODO: this is not actually necessary but is done here to avoid some errors for the moment
  observations.ringRecovery_ = ringRecovery;
  
  observations.nRinged_ = arma::sum(observations.ringRecovery_, 1);
  
//   std::cout << "start simulate data" << std::endl;
  
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, theta, nObservationsCount, nCores);
  model.simulateData();
  
//   model.getLatentPath(latentPath);
//   model.getObservations(observations);
  
  
//   simulateData<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>(nObservationsCount,  hyperParameters, theta, latentPath, observations, nCores);
  
//   std::cout << "finish simulate data" << std::endl;
  
  return Rcpp::List::create
  (
    Rcpp::Named("latentTrue")   = model.getLatentPath().trueCounts_, 
    Rcpp::Named("count")        = model.getObservations().count_, 
    Rcpp::Named("ringRecovery") = model.getObservations().ringRecovery_,
    Rcpp::Named("nRinged")      = model.getObservations().nRinged_,
    Rcpp::Named("rhoTrue")      = model.getModelParameters().getRho(),
    Rcpp::Named("lambdaTrue")   = model.getModelParameters().getLambda(),
    Rcpp::Named("phiTrue")      = model.getModelParameters().getPhi()
  );
}


////////////////////////////////////////////////////////////////////////////////
// Runs a PMMH algorithm 
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runPmmhCpp
(
  const arma::uvec& count,                   // count data
  const arma::umat& ringRecovery,            // capture-recapture matrix for first-year females
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
  
//   std::cout << "setting up observations" << std::endl;
  
  unsigned int nObservationsCount = count.size(); // number of observations
  unsigned int nSteps = nObservationsCount; // number of lower-level SMC steps

 
  Observations observations; // observations 
  observations.count_ = count;
  observations.ringRecovery_ = ringRecovery;
  observations.nRinged_ = arma::sum(ringRecovery, 1);
  
  
//   std::cout << "setting up model class" << std::endl;
  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
//   std::cout << "setting up SMC class" << std::endl;
  
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(0), 
    essResamplingThreshold,
    static_cast<SmcBackwardSamplingType>(0),
    false,
    1,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.setNParticles(nParticles);
  smc.setSamplePath(samplePath);
  smc.setNLookaheadSteps(smcParameters(0));
  
//   std::cout << "setting up MCMC class" << std::endl;

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
  
//   std::vector<arma::umat> latentPath(nIterations); // one latent path sampled and stored at each iteration
  std::vector<LatentPath> latentPaths(nIterations); // one latent path sampled and stored at each iteration
  double cpuTime; // total amount of time needed for running the algorithm
  double acceptanceRateStage1; // first-stage acceptance rate after burn-in (if delayed-acceptance is used)
  double acceptanceRateStage2; // (second-stage) acceptance rate after burn-in
  
//   std::cout << "running PMMH" << std::endl;
  
  runPmmh<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters,McmcParameters>
    (theta, latentPaths, cpuTime, acceptanceRateStage1, acceptanceRateStage2, rngDerived, model, smc, mcmc, thetaInit, samplePath, nCores);
  
  return Rcpp::List::create(
    Rcpp::Named("theta")                = theta, 
//     Rcpp::Named("latentPath")           = latentPath.trueCounts_, 
    Rcpp::Named("cpuTime")              = cpuTime, // total time needed for running the algorithm
    Rcpp::Named("acceptanceRateStage1") = acceptanceRateStage1, // first-stage acceptance rate after burn-in (if delayed-acceptance is used)
    Rcpp::Named("acceptanceRateStage2") = acceptanceRateStage2 // (second-stage) acceptance rate after burn-in
  );
}

/*
////////////////////////////////////////////////////////////////////////////////
// Runs a particle Gibbs sampler
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runPgCpp
(
  const arma::uvec& count,                   // count data
  const arma::umat& ringRecovery,            // capture-recapture matrix for first-year females
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const unsigned int nIterations,            // number of MCMC iterations
  const unsigned int nParticles,             // number of particles per MCMC iteration within each lower-level SMC algorithm
  const double essResamplingThreshold,       // ESS-based resampling threshold for the lower-level SMC algorithms
  const arma::colvec& smcParameters,         // additional parameters to be passed to the particle filter
  const arma::colvec& mcmcParameters,        // additional parameters to be passed to the MCMC kernel
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const arma::colvec& thetaInit,             // initial value for theta (if we keep theta fixed throughout) 
  const double burninPercentage,             // percentage iterations to be thrown away as burnin
  const bool estimateTheta,                  // should the static parameters be estimated?
  const unsigned int nThetaUpdates,          // number of static-parameter updates per iteration
  const unsigned int csmc,                   // type of backward or ancestor sampling used in the Gibbs samplers
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
  


  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  std::cout << "setting up observations" << std::endl;
  
  unsigned int nObservationsCount = count.size(); // number of observations
  unsigned int nSteps = nObservationsCount; // number of lower-level SMC steps

 
  Observations observations; // observations 
  observations.count_ = count;
  observations.ringRecovery_ = ringRecovery;
  observations.nRinged_ = arma::sum(ringRecovery, 1);
  
  
  std::cout << "setting up model class" << std::endl;
  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
  std::cout << "setting up SMC class" << std::endl;
  
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(0), 
    essResamplingThreshold,
    static_cast<SmcBackwardSamplingType>(csmc),
    false,
    1,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.setNParticles(nParticles);
//   smc.setNLookaheadSteps(smcParameters(0));
  
  std::cout << "setting up MCMC class" << std::endl;

  // Class for running MCMC algorithms.
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, nCores
  );
  mcmc.setRwmhSd(rwmhSd);
  mcmc.setUseAdaptiveProposal(false);
  mcmc.setUseDelayedAcceptance(false);
//   mcmc.setAdaptiveProposalParameters(adaptiveProposalParameters);
  mcmc.setNIterations(nIterations, burninPercentage);
  
  std::vector<arma::colvec> theta(nIterations); // parameters sampled by the algorithm
  double cpuTime; // total amount of time needed for running the algorithm
  double acceptanceRate; // acceptance rate after burn-in 
  
  std::cout << "running PG" << std::endl;
  
  runPg<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters,McmcParameters>
    (theta, cpuTime, acceptanceRate, rngDerived, model, smc, mcmc, thetaInit, estimateTheta, nThetaUpdates, csmc, nCores);
    
  
  return Rcpp::List::create(
    Rcpp::Named("theta") = theta, 
    Rcpp::Named("cpuTime") = cpuTime, // total time needed for running the algorithm
    Rcpp::Named("acceptanceRate") = acceptanceRate // (second-stage) acceptance rate after burn-in
  );
}

*/

////////////////////////////////////////////////////////////////////////////////
// Runs an SMC sampler to estimate the model evidence
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runSmcSamplerCpp
(
  const arma::uvec& count,                   // count data
  const arma::umat& ringRecovery,            // capture-recapture matrix for first-year females
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const unsigned int lower,                  // type of algorithm for updating the count-data likelihood (0: pseudo-marginal; 2: MCWM, 3: exact evaluation of (approximate) likelihood)
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
  
  // Starting the timer:
  clock_t t1,t2; // timing
  t1 = clock(); // start timer
  
    std::cout << "setting up RNG" << std::endl;
  
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  unsigned int nObservationsCount = count.size(); // number of observations
  unsigned int nStepsLower = nObservationsCount; // number of lower-level SMC steps
  
      std::cout << "setting up observations" << std::endl;

  Observations observations; // observations 
  observations.count_ = count;
  observations.ringRecovery_ = ringRecovery;
  observations.nRinged_ = arma::sum(ringRecovery, 1);

  /////////////////////////////////////////////////////////////////////////////
  // Model class.
  /////////////////////////////////////////////////////////////////////////////
      std::cout << "setting up Model class" << std::endl;
  
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);

  /////////////////////////////////////////////////////////////////////////////
  // Class for running the lower-level SMC algorithm.
  /////////////////////////////////////////////////////////////////////////////
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nStepsLower,
    static_cast<SmcProposalType>(0), // 0: sample from prior; 5: lookahead
    essResamplingThresholdLower,
    static_cast<SmcBackwardSamplingType>(1),
    false,
    1,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.setNParticles(nParticlesLower);
  smc.setNLookaheadSteps(smcParameters(1));
  smc.setSamplePath(true);
  

  /////////////////////////////////////////////////////////////////////////////
  // Class for running MCMC algorithms.
  /////////////////////////////////////////////////////////////////////////////
  
      std::cout << "setting up MCMC class" << std::endl;
  
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
  
      std::cout << "setting up SMC sampler class" << std::endl;
  
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
    smcSampler.setCessTarget(cessTarget); // specify the sequence of temperatures (and the number of steps) adaptively based on the CESS.
    if (useDoubleTempering) 
    {
      smcSampler.setCessTargetFirst(cessTargetFirst); // specify the sequence of temperatures (and the number of steps) adaptively based on the CESS.
    }
  }
  else // i.e. if we specifiy the sequence of temperatures manually
  {
    smcSampler.setAlpha(alpha); // manually specify the sequence of temperatures (and in particular, the number of steps).
  }
  
      std::cout << "running the SMC sampler" << std::endl;
  
      
 /////////////////////////////
  /////////////////////////////
//   ProfilerStart("/home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/recapture/herons/profile.output.file.log");
  /////////////////////////////
  /////////////////////////////
      
  smcSampler.runSmcSampler(); // running the SMC sampler
  
  /////////////////////////////
  /////////////////////////////
//   ProfilerStop();
  /////////////////////////////
  /////////////////////////////
  
  if (useImportanceTempering)
  {
    smcSampler.computeImportanceTemperingWeights();
  }
  
  t2 = clock(); // stop timer 
  double cpuTime = (static_cast<double>(t2)-static_cast<double>(t1)) / CLOCKS_PER_SEC; // elapsed time in seconds
  std::cout << "Running the SMC sampler took " << cpuTime << " seconds." << std::endl;
  
  
  
  
  
  
  // Computing some additional output
  std::vector<ParticleUpper<LatentPath, Aux>> finalParticles(nParticlesUpper);
  smcSampler.getFinalParticles(finalParticles);
  unsigned int nAgeGroups = static_cast<unsigned int>(hyperParameters(2*dimTheta+6));
//   ModelType modelType = static_cast<ModelType>(hyperParameters(0));
  
  arma::colvec timeNormCovar = hyperParameters(arma::span(2*dimTheta+18, 2*dimTheta+16 + nObservationsCount)); // "normalised" years (i.e. nObservationsCount - 1 elements)
  arma::colvec fDaysCovar    = hyperParameters(arma::span(2*dimTheta+17 + nObservationsCount, 2*dimTheta+16 + 2*nObservationsCount)); // (i.e. nObservationsCount elements)
      
  arma::mat  productivityRates(nObservationsCount-1, nParticlesUpper);
  arma::mat  recoveryProbabilities(nObservationsCount-1, nParticlesUpper);
  arma::cube survivalProbabilities(nAgeGroups, nObservationsCount-1, nParticlesUpper);
  
  arma::ucube trueCounts(nAgeGroups, nObservationsCount, nParticlesUpper, arma::fill::zeros);
  arma::cube smoothedMeans(nAgeGroups, nObservationsCount, nParticlesUpper, arma::fill::zeros);
  arma::cube smoothedVariances(nAgeGroups, nObservationsCount, nParticlesUpper, arma::fill::zeros);
  
  double alpha0, beta0;
  arma::colvec alpha1, beta1;
  alpha1.set_size(nAgeGroups);
  beta1.set_size(nAgeGroups);
  
  for (unsigned int n=0; n<nParticlesUpper; n++)
  {
    alpha0 = finalParticles[n].theta_(0);
    beta0  = finalParticles[n].theta_(1);
    alpha1 = finalParticles[n].theta_(arma::span(2,1+nAgeGroups));
    beta1  = finalParticles[n].theta_(arma::span(2+nAgeGroups,1+2*nAgeGroups));
    
    productivityRates.col(n)     = finalParticles[n].latentPath_.productivityRates_;
    recoveryProbabilities.col(n) = inverseLogit(alpha0 + beta0 * timeNormCovar); 
    for (unsigned int a=0; a<nAgeGroups; a++)
    {
      survivalProbabilities.slice(n).row(a) = arma::trans(inverseLogit(alpha1(a) + beta1(a) * fDaysCovar(arma::span(1,fDaysCovar.size()-1)))); 
    }
  }
  
  if (lower == 3) // i.e. if we approximate/calculate the marginal count-data likelihood analytically
  {
    for (unsigned int n=0; n<nParticlesUpper; n++)
    {
      smoothedMeans.slice(n) = finalParticles[n].latentPath_.smoothedMeans_;
      for (unsigned int t=0; t<nObservationsCount; t++)
      {
        smoothedVariances.slice(n).col(t) = finalParticles[n].latentPath_.smoothedCovarianceMatrices_.slice(t).diag();
      }
    }
  }
  else // i.e. if we use a standard SMC algorithm to approximate the marginal count-data likelihood
  {
    for (unsigned int n=0; n<nParticlesUpper; n++)
    {
      trueCounts.slice(n) = finalParticles[n].latentPath_.trueCounts_(arma::span(0,nAgeGroups-1), arma::span(0,nObservationsCount-1));
    }
  }
  
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
      Rcpp::Named("maxParticleAutocorrelations")               = smcSampler.getMaxParticleAutocorrelations(),
      Rcpp::Named("productivityRates")                         = productivityRates,
      Rcpp::Named("survivalProbabilities")                     = survivalProbabilities,
      Rcpp::Named("recoveryProbabilities")                     = recoveryProbabilities, 
      Rcpp::Named("trueCounts")                                = trueCounts,
      Rcpp::Named("smoothedMeans")                             = smoothedMeans,
      Rcpp::Named("smoothedVariances")                         = smoothedVariances
    );
  }

}

////////////////////////////////////////////////////////////////////////////////
// Approximates the marginal (count-data) likelihood 
// via the SMC filter
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runSmcFilterCpp
(
  const arma::uvec& count,                   // count data
  const arma::umat& ringRecovery,            // capture-recapture matrix for first-year females
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const unsigned int nParticles,             // number of particles per MCMC iteration within each lower-level SMC algorithm
  const double essResamplingThreshold,       // ESS-based resampling threshold for the lower-level SMC algorithms
  const arma::colvec& smcParameters,         // additional parameters to be passed to the particle filter
  const arma::colvec& thetaInit,             // initial value for theta (if we keep theta fixed throughout) 
  unsigned int smcProposalType,              // type of proposal kernel used by the SMC filter
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
//   std::cout << "setting up observations" << std::endl;
  
  unsigned int nObservationsCount = count.size(); // number of observations
  unsigned int nSteps = nObservationsCount; // number of lower-level SMC steps

 
  Observations observations; // observations 
  observations.count_ = count;
  observations.ringRecovery_ = ringRecovery;
  observations.nRinged_ = arma::sum(ringRecovery, 1);
  
  
//   std::cout << "setting up model class" << std::endl;
  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
//   std::cout << "setting up SMC class" << std::endl;
  
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSteps,
    static_cast<SmcProposalType>(smcProposalType), 
    essResamplingThreshold,
    static_cast<SmcBackwardSamplingType>(1),
    false,
    1,
    nCores
  );
  smc.setUseGaussianParametrisation(false);
  smc.setNParticles(nParticles);
  smc.setNLookaheadSteps(smcParameters(0)); 
  smc.setSamplePath(true);
  
//   std::cout << "setting up MCMC class" << std::endl;
/*
  // Class for running MCMC algorithms.
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, nCores
  );
  mcmc.setRwmhSd(rwmhSd);
  mcmc.setAdaptProposal(adaptProposal);
  mcmc.setUseDelayedAcceptance(useDelayedAcceptance);
  mcmc.setAdaptiveProposalParameters(adaptiveProposalParameters);
  mcmc.setNIterations(nIterations, burninPercentage);
  */

    // TODO: implement support for use of gradient information
  LatentPath latentPath;
  AuxFull<Aux> aux; // TODO: implement support for correlated psuedo-marginal approaches

//   std::cout << "running the SMC algorithm!" << std::endl;
  double logLikelihoodEstimate = smc.runSmc(smc.getNParticles(), thetaInit, latentPath, aux, 1.0);
  
  return Rcpp::List::create(
      Rcpp::Named("logLikelihoodEstimate") = logLikelihoodEstimate,
      Rcpp::Named("trajectory") = latentPath.trueCounts_
    );
}
