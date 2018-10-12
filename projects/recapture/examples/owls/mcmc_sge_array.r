## PMMH with delayed acceptance in the "little-owls" capture-recapture model.
## This file is meant to run the SMC sampler for a single model as part
## of a simulation study conducted via an SGE cluster

rm(list = ls())
DEBUG <- FALSE

if (DEBUG) {
  pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
  pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"
  jobName           <- "mcmc_sge_array_2017-08-05" ##"mcmc_sge_array_2017-07-30" ####"mcmc_sge_array_debug"
} else {
  pathToInputBase   <- "/home/ucakafi/code/cpp/mc"
  pathToOutputBase  <- "/home/ucakafi/Scratch/output/cpp/mc"
  jobName           <- "mcmc_sge_array_2017-08-05"
}

projectName <- "recapture"
exampleName <- "owls"

source(file=file.path(pathToInputBase, "setupRCpp.r")) # loads generic R functions and compiles the C++ code



## ========================================================================= ##
## TASK ARRAY PARAMETERS
## ========================================================================= ##

# for (taskId in 1:2) {
# for (taskId in 3:4) {
# for (taskId in 5:6) {
# for (taskId in 7:8) {
# for (taskId in 9:10) {
# for (taskId in 11:13) {
# for (taskId in 14:16) {


# for (taskId in 1:16) {

## need to comment this out unless we run this on the cluster
if (DEBUG) {
  taskId <- 10 #################
} else {
  taskId <- as.numeric(Sys.getenv("SGE_TASK_ID"))
}

if (taskId %% N_CONFIGS > 0) {
  configId  <- taskId %% N_CONFIGS
  replicaId <- taskId %/% N_CONFIGS + 1 # the simulation run for this particular model
} else if (taskId %% N_CONFIGS == 0){
  configId  <- N_CONFIGS
  replicaId <- taskId %/% N_CONFIGS # the simulation run for this particular model
}

modelType             <- MODELS[configId]
lower                 <- LOWER[configId]
nParticles            <- N_PARTICLES[configId]
nIterations           <- N_ITERATIONS[configId]
useAdaptiveProposal   <- USE_ADAPTIVE_PROPOSAL[configId]
useAdaptiveProposalScaleFactor1 <- USE_ADAPTIVE_PROPOSAL_SCALE_FACTOR1[configId]
useDelayedAcceptance  <- USE_DELAYED_ACCEPTANCE[configId]
samplePath            <- SAMPLE_PATH[configId]
thinningInterval      <- THINNING_INTERVAL[configId]



set.seed(replicaId*44)

## ========================================================================= ##
## THE SMC SAMPLER
## ========================================================================= ##

# Miscellaneous parameters:
miscParameters      <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)
# Auxiliary (known) model parameters:
modelParameters     <- getAuxiliaryModelParameters(modelType, miscParameters)
# Auxiliary parameters for the SMC and MCMC algorithms:
algorithmParameters <- getAuxiliaryAlgorithmParameters(modelParameters$dimTheta, modelParameters$nObservations)

# Initial value for the chain sampled from the prior.
thetaInit <- sampleFromPriorCpp(modelParameters$dimTheta, modelParameters$hyperParameters, modelParameters$support, nCores); 

aux <- runPmmhCpp(
  modelParameters$fecundity, modelParameters$count, modelParameters$capRecapFemaleFirst, modelParameters$capRecapMaleFirst, modelParameters$capRecapFemaleAdult, modelParameters$capRecapMaleAdult,
  modelParameters$dimTheta, modelParameters$hyperParameters, modelParameters$support, nIterations, nParticles, algorithmParameters$essResamplingThresholdLower, SMC_PARAMETERS, 
  MCMC_PARAMETERS, useDelayedAcceptance, useAdaptiveProposal, useAdaptiveProposalScaleFactor1, algorithmParameters$adaptiveProposalParameters, algorithmParameters$rwmhSd, thetaInit, algorithmParameters$burninPercentage, samplePath, nCores
)



## ========================================================================= ##
## STORING THE OUTPUT
## ========================================================================= ##


# if (!DEBUG) {
  
  nBurnin  <- ceiling(algorithmParameters$burninPercentage*nIterations)
  nSamples <- nIterations - nBurnin
  lagMax   <- min(LAG_MAX[configId], nSamples-1)
  
  thetaAux      <- matrix(unlist(aux$theta), modelParameters$dimTheta, nIterations)
  latentPathAux <- array(unlist(aux$latentPath), c(modelParameters$dimLatentVariable, modelParameters$nObservationsCount, nIterations))
  
  acfAux <- matrix(NA, modelParameters$dimTheta, lagMax+1)
  for (jj in 1:modelParameters$dimTheta) {
#     print(length(as.numeric(acf(thetaAux[jj,((nBurnin+1):nIterations)], lag.max=lagMax, plot=FALSE)$acf)))
#     print(length(acfAux[jj,]))
    acfAux[jj,] <- as.numeric(acf(thetaAux[jj,((nBurnin+1):nIterations)], lag.max=lagMax, plot=FALSE)$acf)
  }
  
  if (thinningInterval > 1 && thinningInterval < nSamples) {
    thinIdx <- seq(from=nBurnin+1, to=nIterations, by=thinningInterval)
  } else {
    thinIdx <- seq(from=nBurnin+1, to=nIterations, by=1)
  }
  saveRDS(thetaAux[,thinIdx],       file.path(pathToResults, paste("parameters_", configId ,"_", replicaId, ".rds", sep='')))
  saveRDS(latentPathAux[,,thinIdx], file.path(pathToResults, paste("states_", configId ,"_", replicaId, ".rds", sep='')))
  saveRDS(acfAux, file.path(pathToResults, paste("acf_", configId ,"_", replicaId, ".rds", sep='')))
  rm(thetaAux)
  rm(latentPathAux)
  rm(acfAux)
  rm(thinIdx)
  
  saveRDS(unlist(aux$cpuTime),                    file.path(pathToResults, paste("cpuTime_", configId ,"_", replicaId, ".rds", sep='')))
  saveRDS(unlist(aux$outputAcceptanceRateStage1), file.path(pathToResults, paste("acceptanceRatesStage1_", configId ,"_", replicaId, ".rds", sep='')))
  saveRDS(unlist(aux$outputAcceptanceRateStage2), file.path(pathToResults, paste("acceptanceRatesStage2_", configId ,"_", replicaId, ".rds", sep='')))

  rm(aux)
  q(save="no")
# }

# }

