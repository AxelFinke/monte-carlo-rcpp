## Specifies default values for some of the tuning parameters in SMC and MCMC algorithms

###############################################################################
## Returns a number of auxiliary parameters for the MCMC and SMC algorithms.
###############################################################################
getAuxiliaryAlgorithmParameters <- function(dimTheta, nObservations) {

  ## See Peters et al. (2010, Section 3.4) for more information on the following mixture proposal kernel
  ## used by the pseudo-marginal MCMC algorithm (but not used within the SMC sampler for
  ## model selection!)
  mixtureProposalWeight1 <- 0.95 # probability of proposing parameters from the adapted multivariate normal distribution
  proposalScaleFactor1   <- 2.38^2/dimTheta # scalar multiplier for the covariance matrix in the first mixture component (which is the adapted normal)
  proposalScaleFactor2   <- 0.1^2/dimTheta # scalar multiplier for the covariance matrix in the second mixture component
  
  # Parameters for adaptively increasing/decreasing the constant proposalScaleFactor1 as a 
  # function of the acceptance rates observed over
  # previous iterations of the MCMC algorithm/the previous step of the SMC sampler
  proposalScaleFactor1IncreaseFactor <- 2   # increase proposalScaleFactor1 by multiplying it by this factor
  proposalScaleFactor1DecreaseFactor <- 1/2 # decrease proposalScaleFactor1 by multiplying it by this factor
  acceptanceRateLowerBound <- 0.2 # increase proposalScaleFactor1 if the acceptance rate falls below this threshold
  acceptanceRateUpperBound <- 0.5  # decrease proposalScaleFactor1 by multiplying it by this factor

  rwmhSd <- sqrt(rep(1, times=dimTheta)/(dimTheta*nObservations)) # random-walk MH scale

  essResamplingThresholdUpper <- 0.9
  essResamplingThresholdLower <- 0.9
  burninPercentage            <- 0.5 # percentage of total no. of MCMC iterations discarded as burn-in
  nonAdaptPercentage          <- 0.1 # percentage of total no. of MCMC iterations to take place before the MH-proposal covariance matrix will be adapted
 
  # additional parameters used by the adaptive mixture proposal from Peters et al. (2010)
  # potentially with adaptive choice of proposalScaleFactor1.
  adaptiveProposalParameters  <- c(
     mixtureProposalWeight1, proposalScaleFactor1, proposalScaleFactor2, 
     nonAdaptPercentage, 
     proposalScaleFactor1DecreaseFactor, proposalScaleFactor1IncreaseFactor, 
     acceptanceRateLowerBound, acceptanceRateUpperBound) 
    
  nMetropolisHastingsUpdates  <- 1 # number of MH updates per particle and per SMC step
  
  # Parameters used for the first part of the SMC sampler if we employ "double" tempering:
  nMetropolisHastingsUpdatesFirst <- 1 # at stage 1 and with dual tempering

  return(list(adaptiveProposalParameters=adaptiveProposalParameters, rwmhSd=rwmhSd, essResamplingThresholdUpper=essResamplingThresholdUpper, essResamplingThresholdLower=essResamplingThresholdLower, cessTarget=0.99, cessTargetFirst=0.9999, burninPercentage=burninPercentage, nMetropolisHastingsUpdates=nMetropolisHastingsUpdates, nMetropolisHastingsUpdatesFirst=nMetropolisHastingsUpdatesFirst))
}
