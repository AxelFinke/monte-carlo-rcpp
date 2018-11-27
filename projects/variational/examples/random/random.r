## Comparison of variational Monte Carlo approaches in a simple random-effects model
## Model: X_t ~ N(a, b^2); Y_t ~ N(x_t, d^2); 
## Priors: a has a normal prior; b and d have inverse-gamma priors 
## (all parameters are independent, a-priori)

rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp" # put the path to the monte-carlo-rcpp directory here
pathToOutputBase  <- "/home/axel/Dropbox/research/ouTerminal 1tput/cpp/monte-carlo-rcpp" # put the path to the folder which whill contain the simulation output here

exampleName  <- "random"
projectName  <- "variational"
jobName      <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))

## ========================================================================= ##
## MODEL
## ========================================================================= ##

dimTheta   <- 3 # length of the parameter vector
supportMin <- c(-Inf, 0, 0) # minimum of the support for the mean and the variance-parameters
supportMax <- c(Inf, Inf, Inf) # maximum of the support for the mean and the variance-parameters
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

## Hyper parameters:
meanHyperA  = 0
varHyperA   = 1
shapeHyperB = 1
scaleHyperB = 1
shapeHyperD = 1
scaleHyperD = 1
hyperParameters <- c(meanHyperA, varHyperA, shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD)

aTrue <- 0
bTrue <- 1
dTrue <- 0.1

nObservations <- 10 # number of time steps/observations
thetaTrue     <- c(aTrue, bTrue, dTrue) # "true" parameter values used for generating the data
thetaNames    <- c("a", "b", "d") # names of the parameters
DATA          <- simulateDataCpp(nObservations,  hyperParameters, thetaTrue, nCores)
observations  <- DATA$y # simulated observations

## Auxiliary parameters for the SMC and MCMC algorithms:
ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(dimTheta, nObservations)

smcParameters  <- numeric(0) # additional parameters to be passed to the particle filter 
mcmcParameters <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates


testFun <- function(x) {(x^3)*(x > 0.25) + 0.01*(x <= 0.25)}


## SMC sampler for a simple random-effects model
## Model: X_t ~ N(a, b^2); Y_t ~ N(x_t, d^2); 
## Priors: a has a normal prior; b and d have inverse-gamma priors 
## (all parameters are independent, a-priori)

## ========================================================================= ##
## Approximating the model evidence via standard importance sampling
## ========================================================================= ##

simpleImportanceSampling2 <- function(nSamples, observations) {
  theta     <- matrix(NA, nrow=dimTheta, ncol=nSamples)
  theta[1,] <- rnorm(nSamples, mean=meanHyperA, sd=sqrt(varHyperA))
  theta[2,] <- 1/rgamma(nSamples, shape=shapeHyperB, scale=1/scaleHyperB)
  theta[3,] <- 1/rgamma(nSamples, shape=shapeHyperD, scale=1/scaleHyperD)
  
  logW <- rep(0, times=nSamples)
  
  for (tt in 1:nObservations) {
    x    <- rnorm(nSamples, mean=theta[1,], sd=theta[2,])
    logW <- logW + dnorm(rep(observations[tt], times=nSamples), mean=x, sd=theta[3,], log=TRUE)
  }

  logEvidence <- log(sum(exp(logW))) - log(nSamples)
  
  return(list(logEvidence=logEvidence))
}

simpleImportanceSampling <- function(nSamples, observations) {
  theta     <- matrix(NA, nrow=dimTheta, ncol=nSamples)
  theta[1,] <- rnorm(nSamples, mean=meanHyperA, sd=sqrt(varHyperA))
  theta[2,] <- 1/rgamma(nSamples, shape=shapeHyperB, scale=1/scaleHyperB)
  theta[3,] <- 1/rgamma(nSamples, shape=shapeHyperD, scale=1/scaleHyperD)

  logW <- rep(0, times=nSamples)
  for (tt in 1:nObservations) {
    logW <- logW + dnorm(rep(observations[tt], times=nSamples), mean=theta[1,], sd=sqrt(theta[2,]^2 + theta[3,]^2), log=TRUE)
  }

  logEvidence <- log(sum(exp(logW))) - log(nSamples)
  
  return(list(logEvidence=logEvidence))
}


  

aux  <- simpleImportanceSampling(nSamples=1000000, observations=observations) 
aux2 <- simpleImportanceSampling2(nSamples=10000, observations=observations) 

logEvidenceTrue <- aux$logEvidence
logEvidenceTrue2 <- aux2$logEvidence





## ========================================================================= ##
## Estimating the model evidence
## ========================================================================= ##

nParticlesUpper            <- 1000
nParticlesLower            <- 100
useImportanceTempering     <- TRUE
nSteps                     <- 27
nSimulations               <- 50
nMetropolisHastingsUpdates <- 1

ALPHA                      <- seq(from=0, to=1, length=nSteps)
USE_ADAPTIVE_TEMPERING     <- c(0,1,0,1)
ADAPT_PROPOSAL             <- c(0,0,1,1)
SMC_LOWER_TYPE             <- c(3,3,3,3)

MM     <- nSimulations # number of independent replicates
LL     <- length(USE_ADAPTIVE_TEMPERING)
MODELS <- 0 ## not actually needed here

outputLogEvidenceEstimate   <- array(NA, c(length(MODELS), LL, 6, MM))
outputPosteriorMeanEstimate <- array(NA, c(dimTheta, length(MODELS), LL, 6, MM))



for (mm in 1:MM) {

  for (ll in 1:LL) {

    for (ii in 1:length(MODELS)) {
    
      print(paste("Random effects model, model selection: ", mm , ", " , ii, sep=''))
    
      ## Auxiliary (known) model parameters:
#       MODEL_PAR     <- getAuxiliaryModelParameters(MODELS[ii], MISC_PAR)
      ## Auxiliary parameters for the SMC and MCMC algorithms:
      ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(dimTheta, nObservations)

      aux <- runSmcSamplerCpp(observations, 
            dimTheta, hyperParameters, support, nParticlesUpper, nParticlesLower, SMC_LOWER_TYPE[ll], nMetropolisHastingsUpdates, ALGORITHM_PAR$essResamplingThresholdUpper, ALGORITHM_PAR$essResamplingThresholdLower,
            smcParameters, mcmcParameters, 
            USE_ADAPTIVE_TEMPERING[ll], 0, useImportanceTempering, ADAPT_PROPOSAL[ll], ALGORITHM_PAR$cessTarget, ALPHA, ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, nCores
          )
          
      ############################################
      ## Evidence estimators
      ############################################
      
      ## Standard stimator:
      outputLogEvidenceEstimate[ii,ll,1,mm] <- aux$logEvidenceEstimate                
      
      ## Estimators which reweight using the (generalised) ESS:
#       outputLogEvidenceEstimate[ii,ll,2,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeights) # generalised ESS 
      outputLogEvidenceEstimate[ii,ll,3,mm] <- aux$logEvidenceEstimateEssAlternate                        # standard ESS 
      
      ## Estimators which first resample to create obtain sets of evenly weighted particles: 
#       outputLogEvidenceEstimate[ii,ll,4,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeightsResampled) # generalised ESS (computed from resampled particles)
      outputLogEvidenceEstimate[ii,ll,5,mm] <- aux$logEvidenceEstimateEssResampledAlternate                        # standard ESS (computed from resampled particles)
      
      ## Estimator which reweights using the CESS:
      outputLogEvidenceEstimate[ii,ll,6,mm] <- aux$logEvidenceEstimateCessAlternate 
      
      
      
      nSteps <- length(aux$inverseTemperatures)
      THETA  <- array(unlist(aux$theta), c(dimTheta, nParticlesUpper, nSteps))
      THETA_RESAMPLED  <- array(unlist(aux$thetaResampled), c(dimTheta, nParticlesUpper, nSteps))
      
      
            
      for (jj in 1:dimTheta) {
        outputPosteriorMeanEstimate[jj,ii,ll,1,mm] <- aux$selfNormalisedWeights[,nSteps] %*% testFun(THETA[jj,,nSteps])
        outputPosteriorMeanEstimate[jj,ii,ll,2,mm] <- sum(aux$selfNormalisedWeightsEss * testFun(THETA[jj,,]))
        outputPosteriorMeanEstimate[jj,ii,ll,3,mm] <- sum(aux$selfNormalisedWeightsEssResampled * testFun(THETA_RESAMPLED[jj,,]))
        outputPosteriorMeanEstimate[jj,ii,ll,4,mm] <- sum(aux$selfNormalisedWeightsCess * testFun(THETA[jj,,]))
        outputPosteriorMeanEstimate[jj,ii,ll,5,mm] <- sum(W0 * testFun(THETA[jj,,]))
        
        ############################################
        # Using the general definition of the ESS which 
        # allows for unnormalised weights 
        # but also incorporating the test function:
        W1 <- aux$selfNormalisedReweightedWeights
        logL1 <- rep(NA, nSteps)
        
#         for (tt in 1:nSteps) {
#           logL1[tt] <- - log(var.welford(u[,tt] + log(testFun(THETA[jj,,tt]))) + 1)
#         }

 
        for (tt in 1:nSteps) {
          maxU <- max(u[,tt])
#           logL1[tt] <- 2*log(sum(exp(u[,tt] - maxU))) - log(sum(exp(2*(u[,tt] - maxU))*testFun(THETA[jj,,tt])^2))
          logL1[tt] <- 2*log(sum(exp(u[,tt] - maxU))) - 
                       log(
                         nParticlesUpper * sum(exp(2*(u[,tt] - maxU))*testFun(THETA[jj,,tt])^2) - 
                         (sum(exp(u[,tt] - maxU)*testFun(THETA[jj,,tt])))^2
                       )
        }


        logL1[is.nan(logL1)] <- -Inf
        logL1[is.infinite(logL1)] <- -Inf
        normalisedL1 <- exp(logL1) / sum(exp(logL1))
        for (tt in 1:nSteps) {
          W1[,tt] <- W1[,tt] * normalisedL1[tt]
        }
        
        ############################################
        outputPosteriorMeanEstimate[jj,ii,ll,6,mm] <- sum(W1 * testFun(THETA[jj,,]))
      }
      
      
      print(outputLogEvidenceEstimate[ii,ll,,mm])
    }
  
  }
  
  save(
    list  = ls(envir = environment(), all.names = TRUE), 
    file  = file.path(pathToResults, paste("smc_estimating_model_evidence_nParticles_", nParticlesUpper, "_nSimulations_", nSimulations, sep='')),
    envir = environment()
  ) 
  
}







