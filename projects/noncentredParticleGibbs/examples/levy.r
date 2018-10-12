## Non-centred (particle) Gibbs samplers for the 
## Levy-driven stochastic volatility model

rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"

exampleName       <- "levy"
projectName       <- "nonCentredParticleGibbs"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))

## Miscellaneous parameters:
miscParameters    <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)

## ========================================================================= ##
## MODEL PARAMETERS
## ========================================================================= ##

## Selecting the model:
modelType        <- 3

## Loading observations:
observations     <- as.numeric(unlist((read.table(paste(file.path(pathToData, "SP500GriffinSteel"), ".R", sep='')))))
observationTimes <- 1:length(observations)
nObservations    <- length(observationTimes)

## Auxiliary (known) model parameters and data:
modelParameters  <- getAuxiliaryModelParameters(modelType, miscParameters, observationTimes=observationTimes)

## ========================================================================= ##
## ALGORITHM PARAMETERS
## ========================================================================= ##

algorithmParameters <- getAuxiliaryAlgorithmParameters(modelParameters$dimTheta, modelParameters$nObservations)

nIterations            <- 100 # number of Gibbs-sampler sweeps (including burn-in)
nParameterUpdates      <- 10 # number of model-parameter updates in between each set of latent-variable updates
nonCentringProbability <- 0.5 # probability of switching to a non-centred parametrisation for the parameter updates (if implemented)
thetaInit              <- rnorm(modelParameters$dimTheta) # initial value for theta (if we keep theta fixed throughout) 
samplerType            <- 0 # type of algorithm to be used for updating the latent variables; 0: Metropolis-within-Gibbs; 1: (conditional) SMC; 2: (conditional) EHMM; 3: sample latent variables from their full conditional posterior distribution (if available)
marginalisationType    <- 0 # type analytical marginalisation of certain model parameters: 0: no marginalisation; 1: marginalisation only during parameter updates; 2: marginalisation during both latent-variable and parameter updates
proposalScales         <- sqrt(rep(1, times=modelParameters$dimTheta)/(modelParameters$dimTheta*nObservations)) # standard deviations of the uncorrelated Gaussian-random walk proposals for the full vector of model parameters
proposalScalesMarginalised <- sqrt(rep(1, times=modelParameters$dimThetaMarginalised)/(modelParameters$dimThetaMarginalised*nObservations)) # standard deviations of the uncorrelated Gaussian-random walk proposals for the vector of model parameters after certain parameters have been integrated out analytically (if possible)
nLatentVariableUpdates <- 500 # number of times the Metropolis-within-Gibbs kernels are applied to update the latent variables in between each set of parameter updates
nParticles             <- 50 # number of particles used by the SMC/EHMM updates
stepTimes              <- #######
essResamplingThreshold <- algorithmParameters$essResamplingThresholdLower # ESS-based resampling threshold for the SMC/EHMM updates
backwardSamplingType   <- 2 # 0: no backward simulation for the conditional SMC/EHMM updates; 1: backward sampling; 2: ancestor sampling
mwgParameters          <- numeric(0) # additional parameters to be passed to the Metropolis-within-Gibbs updates for the latent variables
smcParameters          <- numeric(0) # additional parameters to be passed to the standard conditional sequential Monte Carlo updates for the latent variables
ehmmParameters         <- numeric(0) # additional parameters to be passed to the embedded hidden Markov model updates for the latent variables
estimateTheta          <- TRUE # should the model parameters be estimated? Otherwise, they are kept fixed to their initial values


## ========================================================================= ##
## RUNNING THE ALGORITHM
## ========================================================================= ##

aux <- runGibbsSamplerCpp(
   observationms, modelParameters$hyperParameters, modelParameters$support, 
   nIterations, nParameterUpdates, nonCentringProbability, thetaInit, samplerType, 
   marginalisationType, proposalScales, proposalScalesMarginalised, nLatentVariableUpdates,
   nParticles, stepTimes, essResamplingThreshold, backwardSamplingType,
   mwgParameters, smcParameters, ehmmParameters, estimateTheta, nCores
   )

