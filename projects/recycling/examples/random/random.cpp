#include <RcppArmadillo.h>

#include "main/algorithms/mcmc/pmmh.h"
#include "main/algorithms/smc/SmcSampler.h"
#include "time.h"

#include "main/applications/random/random.h"

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
// Testing R's use of the gamma density
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double dInverseGammaCpp
(
  const double x,
  const double shape,
  const double scale
)
{
  return dInverseGamma(x, shape, scale);
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
  arma::colvec extraParameters; // not used here
  simulateData<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>(nObservations, hyperParameters, theta, extraParameters, latentPath, observations, nCores);
  return Rcpp::List::create(Rcpp::Named("x") = latentPath, Rcpp::Named("y") = observations);
}

////////////////////////////////////////////////////////////////////////////////
// Runs an SMC sampler to estimate the model evidence
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runSmcSamplerCpp
(
  const arma::colvec& observations,          // data
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperParameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (par.size(), 2)-matrix containing the lower and upper bounds of the support of each parameter
  const unsigned int lower,                  // type of algorithm for updating the count-data likelihood (0: pseudo-marginal; 2: MCWM)
  const unsigned int nParticlesUpper,        // number of upper-level particles
  const unsigned int nParticlesLower,        // number of lower-level (i.e. "filtering") particles
  const unsigned int nMetropolisHastingsUpdates, // number of MH updates per particle and SMC step
  const double essResamplingThresholdUpper,  // ESS-based resampling threshold for the upper-level SMC sampler
  const double essResamplingThresholdLower,  // ESS-based resampling threshold for the lower-level SMC filter
  const arma::colvec& smcParameters,         // additional parameters to be passed to the particle filter
  const arma::colvec& mcmcParameters,        // additional parameters to be passed to the MCMC kernel
  const bool useAdaptiveTempering,           // should we determine the tempering schedule adaptively according to the CESS?
  const bool useAdaptiveCessTarget,          // should we adapt the CESS target based on the autocorrelation of the MCMC kernels?
  const bool useImportanceTempering,         // should we compute and return the importance-tempering weights?
  const bool useAdaptiveProposal,            // should we adapt the proposal scale of the MCMC kernel as in Peters at al. (2010)?
  const bool useAdaptiveProposalScaleFactor1, // should we also adapt the constant by which the sample covariance matrix is multiplied?
  const double cessTarget,                   // CESS target threshold (only used if useAdaptiveTempering == true).
  const arma::colvec& alpha,                 // manually specified tempering schedule (only used if useAdaptiveTempering == false)
  const arma::colvec& adaptiveProposalParameters, // parameters needed for the adaptive mixture proposal from Peters at al. (2010).
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
  // Starting the timer:
  clock_t t1,t2; // timing
  t1 = clock(); // start timer
  
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  unsigned int nObservations = observations.size(); // number of observations
  unsigned int nStepsLower = nObservations; // number of lower-level SMC steps

  Observations obs; // observations 
  obs = observations;

  /////////////////////////////////////////////////////////////////////////////
  // Model class.
  /////////////////////////////////////////////////////////////////////////////
  
//     std::cout << "set up Model class" << std::endl;
  
  
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperParameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);

  /////////////////////////////////////////////////////////////////////////////
  // Class for running the lower-level SMC algorithm.
  /////////////////////////////////////////////////////////////////////////////
  
//     std::cout << "set up Smc class" << std::endl;
  
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
  
//     std::cout << "set up MCMC class" << std::endl;
  
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, nCores
  );
  mcmc.setRwmhSd(rwmhSd);
  mcmc.setUseAdaptiveProposal(useAdaptiveProposal);
  mcmc.setUseDelayedAcceptance(false);
  mcmc.setAdaptiveProposalParameters(adaptiveProposalParameters);
  mcmc.setUseAdaptiveProposalScaleFactor1(useAdaptiveProposalScaleFactor1);
  mcmc.setIsWithinSmcSampler(true);
  
  /////////////////////////////////////////////////////////////////////////////
  // Class for running the upper-level SMC sampler.
  /////////////////////////////////////////////////////////////////////////////
  
//     std::cout << "set up SmcSampler class" << std::endl;
  
  SmcSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters> smcSampler(rngDerived, model, smc, mcmc, nCores);
  smcSampler.setNParticles(nParticlesUpper);
  smcSampler.setEssResamplingThreshold(essResamplingThresholdUpper);
  smcSampler.setUseAdaptiveTempering(useAdaptiveTempering);
  smcSampler.setLower(static_cast<SmcSamplerLowerType>(lower));
  smcSampler.setNMetropolisHastingsUpdates(nMetropolisHastingsUpdates);
  smcSampler.setUseAdaptiveCessTarget(useAdaptiveCessTarget);
  
  smcSampler.setUseDoubleTempering(false);

  
  if (smcSampler.getUseAdaptiveTempering()) 
  {
    smcSampler.setCessTarget(cessTarget); // specify the sequence of temperatures (and the number of steps) adaptively based on the CESS.
  }
  else // i.e. if we specifiy the sequence of temperatures manually
  {
    smcSampler.setAlpha(alpha); // manually specify the sequence of temperatures (and in particular, the number of steps).
  }

  smcSampler.runSmcSampler(); // running the SMC sampler
  
  if (useImportanceTempering)
  {
    smcSampler.computeImportanceTemperingWeights();
  }
  
  t2 = clock(); // stop timer 
  double cpuTime = (static_cast<double>(t2)-static_cast<double>(t1)) / CLOCKS_PER_SEC; // elapsed time in seconds
  
  if (useImportanceTempering)
  {
    return Rcpp::List::create(             
      Rcpp::Named("inverseTemperatures")                       = smcSampler.getAlpha(),
//       Rcpp::Named("logUnnormalisedWeights")                    = smcSampler.getLogUnnormalisedWeightsFull(),
      Rcpp::Named("selfNormalisedWeights")                     = smcSampler.getSelfNormalisedWeightsFull(),  
      Rcpp::Named("theta")                                     = smcSampler.getThetaFull(), 
      Rcpp::Named("thetaResampled")                            = smcSampler.getThetaFullResampled(), 
      Rcpp::Named("cpuTime")                                   = cpuTime,
      Rcpp::Named("logEvidenceEstimate")                       = smcSampler.getLogEvidenceEstimate(),
//       Rcpp::Named("logEvidenceEstimateEss")                    = smcSampler.getLogEvidenceEstimateEss(),
      Rcpp::Named("logEvidenceEstimateEssAlternate")           = smcSampler.getLogEvidenceEstimateEssAlternate(),
//       Rcpp::Named("logEvidenceEstimateEssResampled")           = smcSampler.getLogEvidenceEstimateEssResampled(),
      Rcpp::Named("logEvidenceEstimateEssResampledAlternate")  = smcSampler.getLogEvidenceEstimateEssResampledAlternate(),
//       Rcpp::Named("logEvidenceEstimateCess")                   = smcSampler.getLogEvidenceEstimateCess(),
      Rcpp::Named("logEvidenceEstimateCessAlternate")          = smcSampler.getLogEvidenceEstimateCessAlternate(),
      Rcpp::Named("logUnnormalisedReweightedWeights")          = smcSampler.getLogUnnormalisedReweightedWeights(), 
      Rcpp::Named("logUnnormalisedReweightedWeightsResampled") = smcSampler.getLogUnnormalisedReweightedWeightsResampled(),
      Rcpp::Named("selfNormalisedReweightedWeights")           = smcSampler.getSelfNormalisedReweightedWeights(), 
      Rcpp::Named("selfNormalisedReweightedWeightsResampled")  = smcSampler.getSelfNormalisedReweightedWeightsResampled(),
      Rcpp::Named("selfNormalisedWeightsEss")                  = smcSampler.getSelfNormalisedWeightsEss(),
      Rcpp::Named("selfNormalisedWeightsEssResampled")         = smcSampler.getSelfNormalisedWeightsEssResampled(),
      Rcpp::Named("selfNormalisedWeightsCess")                 = smcSampler.getSelfNormalisedWeightsCess(),
      Rcpp::Named("acceptanceRates")                           = smcSampler.getAcceptanceRates(),
      Rcpp::Named("maxParticleAutocorrelations")               = smcSampler.getMaxParticleAutocorrelations()
                              
//       Rcpp::Named("ess")                                       = smcSampler.getEss(),
//       Rcpp::Named("essResampled")                              = smcSampler.getEssResampled(),
//       Rcpp::Named("cess")                                      = smcSampler.getCess()
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
