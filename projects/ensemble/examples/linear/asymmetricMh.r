## The asymmetric Metropolis--Hastings algorithm based around a 
## subsampled CSMC approach from Yildirim, Andrieu, Doucet & Chopin (2017)
## We compare performance of the approach 
## in a simple multivariate linear-Gaussian state-space model
## in the case that the algorithm is based around a
## conventional and based around a "local" random-walk CSMC algorithm


## ========================================================================= ##
## SETUP
## ========================================================================= ##

## ------------------------------------------------------------------------- ##
## Directories
## ------------------------------------------------------------------------- ##

rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp"
exampleName       <- "linear"
projectName       <- "ensemble"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))


## ------------------------------------------------------------------------- ##
## Global model parameters
## ------------------------------------------------------------------------- ##

nOffDiagonals <- 0 # number of non-zero off-diagonals on each side of the main diagonal of A
dimTheta <- nOffDiagonals + 3 # length of the parameter vector

# Support of the unknown parameters:
supportMin <- c(-1, 0, 0) # minimum
supportMax <- c( 1, Inf, Inf) # maximum
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

# Inverse-gamma prior on b:
shapeHyperB <- 1 # shape parameter
scaleHyperB <- 0.5 # scale parameter

# Inverse-gamma prior on d:
shapeHyperD <- 1 # shape parameter
scaleHyperD <- 0.5 # scale parameter

# Collecting all the hyper parameters in a vector:
# hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)

nObservations <- 50 # number of time steps/observations
thetaTrue     <- c(1, 1, 1) # "true" parameter values used for generating the data
thetaNames    <- c("a", "b", "d") # names of the parameters


## ------------------------------------------------------------------------- ##
## Global algorithm parameters
## ------------------------------------------------------------------------- ##

## Parameters for the model parameter updates
ESTIMATE_THETA <- TRUE
nThetaUpdates  <- 10
useNonCentredParametrisation <- FALSE
nonCentringProbability <- 0

## Other parameters
kern <- 0
useGradients <- FALSE
fixedLagSmoothingOrder <- 0
essResamplingThreshold <- 1.0
resampleType <- 0 # 0: multinomial; 1: systematic
nSteps <- nObservations # number of SMC steps

DIMENSIONS <- c(1,2) 
BACK  <- c(1,1,1,1,0) # type of backward simulation scheme: 0: none; 1: backward; 2: ancestor
PROP  <- c(0,6,0,6,0) # lower-level proposals (PROP must have the same length as LOWER)
LOWER <- c(0,0,0,0,3) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
LOCAL <- c(0,0,0,0,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach
UPPER <- c(1,1,2,2,0)

LL <- length(LOWER) # number of lower-level sampler configurations to test
DD <- length(DIMENSIONS)


## ========================================================================= ##
## SIMULATION STUDIES
## ========================================================================= ##

nIterations      <- 200000
nSimulations     <- 2
nParticles       <- 31
nSubsampledPaths <- 5

initialiseStatesFromStationarity <- TRUE # should the CSMC-type algorithms initialise the state sequence from stationarity?
storeParentIndices   <- FALSE # should the CSMC-type algorithms all their parent indices?
storeParticleIndices <- FALSE # should the CSMC-type algorithms store the indices of the (input and output) reference particle lineage?
TIMES      <- c(0) # c(0, nObservations-1) # time steps at which components of the latent states are to be stored
COMPONENTS <- c(0) # rep(0, times=length(TIMES)) # components which are to be stored

MM <- nSimulations  # number of independent replicates

# outputStates  <- array(NA, c(length(TIMES), nIterations, LL, DD, MM))
if (storeParentIndices) {
  outputParentIndices <- array(NA, c(nParticles, nSteps-1, nIterations, LL, DD, MM))
}
if (storeParticleIndices) {
  outputParticleIndicesIn  <- array(NA, c(nSteps, nIterations, LL, DD, MM))
  outputParticleIndicesOut <- array(NA, c(nSteps, nIterations, LL, DD, MM))
}

outputStates <- array(NA, c(length(TIMES), nIterations, LL, DD, MM))
outputTheta  <- array(NA, c(dimTheta, nIterations, LL, DD, MM))


for (dd in 1:DD) { 
  
  dimX <- DIMENSIONS[dd]
  dimY <- dimX
  
  smcParameters         <- rep(sqrt(1.0/dimX), times=nObservations); # additional parameters for the SMC algorithms
  ensembleOldParameters <- c(rep(0, dimX), 1.0/dimX, 0.5); # additional parameters for the original EHMM algorithms
  ensembleNewParameters <- c(sqrt(1.0/dimX), sqrt(1-1.0/dimX), 1.0, 1.0) # additional parameters for the alternative EHMM algorithms: c("scale for random-walk proposals", "correlation parameter for autoregressive updates)
  rwmhSd                <- sqrt(rep(1, times=dimTheta)/(100*dimTheta*dimX*nObservations))
  
  hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)
  DATA            <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
  observations    <- DATA$y # simulated observations
  
  thetaInit <- sampleFromPriorCpp(dimTheta, hyperparameters, support, nCores); ## NOTE: this is not used!
  
  for (mm in 1:MM) {
    
    for (ll in 1:LL) {
    
      print(paste("ll:", ll, "; dd: ", dd, "; mm: ", mm, sep=''))
  
      aux <- runMcmcCpp(UPPER[ll], LOWER[ll], dimTheta, hyperparameters, support, observations, nIterations, kern, useGradients, rwmhSd, fixedLagSmoothingOrder, PROP[ll], resampleType, nThetaUpdates, nSteps, nParticles, nSubsampledPaths, essResamplingThreshold, BACK[ll], useNonCentredParametrisation, nonCentringProbability, smcParameters, ensembleOldParameters, ensembleNewParameters, LOCAL[ll], ESTIMATE_THETA, thetaInit, TIMES, COMPONENTS,
      initialiseStatesFromStationarity, storeParentIndices, storeParticleIndices, nCores)
      
      outputStates[,,ll,dd,mm] <- matrix(unlist(aux$states), length(TIMES), nIterations)
      outputTheta[,,ll,dd,mm]  <- matrix(unlist(aux$theta), dimTheta, nIterations)
      rm(aux)
    }
  }
  
  save(
    list  = ls(envir = environment(), all.names = TRUE), 
    file  = file.path(pathToResults, paste("ASYMMETRIC_MH_nIterations", nIterations, "nObservations", nObservations, "nSimulations", nSimulations, "nParticles", nParticles, "nSubsampledPaths", nSubsampledPaths, "estimateTheta", ESTIMATE_THETA, sep='_')),
    envir = environment()
  ) 
}

##############################################################
## PLOT OUTPUT
##############################################################

## -------------------------------------------------------- ##
## Density estimates
## -------------------------------------------------------- ##

myCol <- c("red", "blue", "red", "blue", "gray") # c(colour of default CSMC, colour of random-walk CSMC)
myLegend <- c("MH-within-PG (CSMC)", "MH-within-PG (RW-CSMC)", "Asymmetric MH (CSMC)", "Asymmetric MH (RW-CSMC)", "Idealised MH")
myLty <- c(2,2,1,1,1)

lagMax <- 100
burnin <- floor(nIterations/2)
xLimMin <- c(0,0.5)
xLimMax <- c(2,1.5)
yLimMax <- c(5,12)

op <- par(mfrow=c(dimTheta,DD))
for (ii in 1:dimTheta) {
  for (dd in 1:DD) {
    plot(c(0,0), c(0,0), col="white", type="l", xlim=c(xLimMin[dd],xLimMax[dd]), ylim=c(0, yLimMax[dd]))
    for (mm in 1:MM) {
      for (ll in 1:LL) {
        lines(density(outputTheta[ii,(burnin+1):nIterations,ll,dd,mm]), col=myCol[ll], type="l", lty=myLty[ll])
      }
    }
    legend("topleft", legend=myLegend, col=myCol, bty='n', lty=myLty)
  }
}
par(op)



locator(1)


## -------------------------------------------------------- ##
## Autocorrelations
## -------------------------------------------------------- ##

op <- par(mfrow=c(dimTheta,DD))
for (ii in 1:dimTheta) {
  for (dd in 1:DD) {
    plot(c(0,0), c(0,0), col="white", type="l", xlim=c(0,lagMax), ylim=c(0, 1))
    for (mm in 1:MM) {
      for (ll in 1:LL) {
        ACF <- acf(outputTheta[ii,(burnin+1):nIterations,ll,dd,mm], lag.max=lagMax, plot=FALSE)$acf
        lines(x=0:lagMax, y=ACF, col=myCol[ll], type="l", lty=myLty[ll])
      }
    }
    legend("topleft", legend=myLegend, col=myCol, bty='n', lty=myLty)
  }
}
par(op)
