## SMC sampler for a simple random-effects model
## Model: X_t ~ N(a, b^2); Y_t ~ N(x_t, d^2); 
## Priors: a has a normal prior; b and d have inverse-gamma priors 
## (all parameters are independent, a-priori)

## WARNING: especially for small to moderate numbers of observations and small 
## inverse temperatures, the likelihood-tempered posterior distributions
## for the log of the variance parameters has very heavy right tails!

rm(list = ls())
DEBUG <- TRUE

if (DEBUG) {
  pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
  pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"
  jobName           <- "smc_sge_array_debug"
} else {
  pathToInputBase   <- "/home/ucakafi/code/cpp/mc"
  pathToOutputBase  <- "/home/ucakafi/Scratch/output/cpp/mc"
  jobName           <- "smc_sge_array_2017-05-22"
}
projectName <- "recycling"
exampleName <- "random"

source(file=file.path(pathToInputBase, "setupRCpp.r")) # loads generic R functions and compiles the C++ code






## ========================================================================= ##
## TASK ARRAY PARAMETERS
## ========================================================================= ##

# if (DEBUG) {
#   taskId <- 1
# } else {
#   taskId <- as.numeric(Sys.getenv("SGE_TASK_ID"))
# }

for (taskId in 1:1000) {

if (taskId %% N_CONFIGS > 0) {
  configId  <- taskId %%  N_CONFIGS
  replicaId <- taskId %/% N_CONFIGS + 1 # the simulation run for this particular model
} else if (taskId %% N_CONFIGS == 0){
  configId  <- N_CONFIGS
  replicaId <- taskId %/% N_CONFIGS # the simulation run for this particular model
}

modelType             <- MODELS[configId]
lower                 <- LOWER[configId]
nParticlesUpper       <- N_PARTICLES_UPPER[configId]
nParticlesLower       <- N_PARTICLES_LOWER[configId]
cessTarget            <- CESS_TARGET[configId]
useAdaptiveCessTarget <- USE_ADAPTIVE_CESS_TARGET[configId]
nMetropolisHastingsUpdates <- N_METROPOLIS_HASTINGS_UPDATES[configId]
useAdaptiveProposal   <- USE_ADAPTIVE_PROPOSAL[configId]
useAdaptiveProposalScaleFactor1 <- USE_ADAPTIVE_PROPOSAL_SCALE_FACTOR1[configId]

set.seed(replicaId)

## ========================================================================= ##
## THE SMC SAMPLER
## ========================================================================= ##

# Miscellaneous parameters:
miscParameters      <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)
# Auxiliary (known) model parameters:
modelParameters     <- getAuxiliaryModelParameters(modelType, miscParameters)
# Auxiliary parameters for the SMC and MCMC algorithms:
algorithmParameters <- getAuxiliaryAlgorithmParameters(modelParameters$dimTheta, modelParameters$nObservations)

aux <- runSmcSamplerCpp(
      modelParameters$observations,
      modelParameters$dimTheta, modelParameters$hyperParameters, modelParameters$support, lower, nParticlesUpper, nParticlesLower, nMetropolisHastingsUpdates, algorithmParameters$essResamplingThresholdUpper, algorithmParameters$essResamplingThresholdLower, 
      SMC_PARAMETERS, MCMC_PARAMETERS, 
      USE_ADAPTIVE_TEMPERING, useAdaptiveCessTarget, USE_IMPORTANCE_TEMPERING, useAdaptiveProposal, useAdaptiveProposalScaleFactor1, cessTarget, ALPHA, algorithmParameters$adaptiveProposalParameters, algorithmParameters$rwmhSd, nCores
    )    

if (USE_IMPORTANCE_TEMPERING) {
  alternateLogEvidenceEstimate    <- rep(NA, times=3)
  alternateLogEvidenceEstimate[1] <- aux$logEvidenceEstimate
  alternateLogEvidenceEstimate[2] <- aux$logEvidenceEstimateEssAlternate ## uses importance-tempering weights proportional to the ESS 
  alternateLogEvidenceEstimate[3] <- aux$logEvidenceEstimateEssResampledAlternate ## based importance-tempering weights proportional to the ESS (after additional resampling step)
} 

## ========================================================================= ##
## STORING THE OUTPUT
## ========================================================================= ##

# if (!DEBUG) {
  
  nStepsAux  <- length(aux$inverseTemperatures)
  thetaAux   <- array(unlist(aux$theta), c(modelParameters$dimTheta, nParticlesUpper, nStepsAux))
  
  saveRDS(thetaAux[,,nStepsAux],                 file.path(pathToResults, paste("finalParameters_", configId ,"_", replicaId, ".rds", sep='')))
  if (USE_IMPORTANCE_TEMPERING) {
    saveRDS(alternateLogEvidenceEstimate,        file.path(pathToExtra, paste("alternateLogEvidenceEstimate_", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(aux$selfNormalisedReweightedWeights, file.path(pathToExtra, paste("selfNormalisedReweightedWeights_", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(thetaAux,                            file.path(pathToExtra, paste("parameters_", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(aux$selfNormalisedWeights,           file.path(pathToExtra, paste("selfNormalisedWeights_", configId ,"_", replicaId, ".rds", sep='')))
  }
  rm(thetaAux)
  
  saveRDS(aux$selfNormalisedWeights[,nStepsAux], file.path(pathToResults, paste("finalSelfNormalisedWeights_", configId ,"_", replicaId, ".rds", sep='')))
  saveRDS(aux$cpuTime,                           file.path(pathToResults, paste("cpuTime_", configId ,"_", replicaId, ".rds", sep='')))
  saveRDS(aux$inverseTemperatures,               file.path(pathToResults, paste("inverseTemperatures_", configId ,"_", replicaId, ".rds", sep='')))
  saveRDS(aux$acceptanceRates,                   file.path(pathToResults, paste("acceptanceRates_", configId ,"_", replicaId, ".rds", sep='')))
#   if (useAdaptiveCessTarget) {
  saveRDS(aux$maxParticleAutocorrelations,       file.path(pathToResults, paste("maxParticleAutocorrelations_", configId ,"_", replicaId, ".rds", sep='')))
#   }
  saveRDS(aux$logEvidenceEstimate,               file.path(pathToResults, paste("standardLogEvidenceEstimate_", configId ,"_", replicaId, ".rds", sep='')))
  


  
#   save(
#     list  = ls(envir = environment(), all.names = TRUE), 
#     file  = file.path(
#       pathToFull, 
#       paste("fullOutput_configId_", configId, 
#       "_replicaId_", replicaId, 
#       sep='')),
#     envir = environment()
#   ) 
  rm(aux)
#   q(save="no")
# }

}


