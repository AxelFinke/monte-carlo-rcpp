## PMMH with delayed acceptance in the "herons" capture-recapture model
## This file is meant to run the SMC sampler for a single model as part
## of a simulation study conducted via SGE task arrays (one task for each model)

rm(list = ls())
DEBUG <- TRUE

if (DEBUG) {
  pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp" # put the path to the monte-carlo-rcpp directory here
  pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp" # put the path to the folder which whill contain the simulation output here
  jobName           <- "smc_sge_array_debug"
} else {
  pathToInputBase   <- "/home/ucakafi/code/cpp/mc"
  pathToOutputBase  <- "/home/ucakafi/Scratch/output/cpp/mc"
  jobName           <- "smc_sge_array_2017-05-14"
}

projectName <- "recapture"
exampleName <- "herons"

source(file=file.path(pathToInputBase, "setupRCpp.r")) # loads generic R functions and compiles the C++ code

##########
minSim <- 2
maxSim <- 2
minConfig <- 7
maxConfig <- 9


for (taskId in ((minSim-1)*length(MODELS)+minConfig):((maxSim-1)*length(MODELS)+maxConfig)) {
set.seed(taskId)

## ========================================================================= ##
## TASK ARRAY PARAMETERS
## ========================================================================= ##

if (DEBUG) {
####   taskId <- rr
} else {
  taskId <- as.numeric(Sys.getenv("SGE_TASK_ID"))
}

if (taskId %% length(MODELS) > 0) {
  configId  <- taskId %% length(MODELS)
  replicaId <- taskId %/% length(MODELS) + 1 # the simulation run for this particular model
} else if (taskId %% length(MODELS) == 0){
  configId  <- length(MODELS)
  replicaId <- taskId %/% length(MODELS) # the simulation run for this particular model
}

modelType          <- MODELS[configId]
nLevels            <- N_LEVELS[configId]
nAgeGroups         <- N_AGE_GROUPS[configId]
lower              <- LOWER[configId]
useDoubleTempering <- USE_DOUBLE_TEMPERING[configId]
nParticlesUpper    <- N_PARTICLES_UPPER[configId]
nParticlesLower    <- N_PARTICLES_LOWER[configId]
cessTarget         <- CESS_TARGET[configId]
cessTargetFirst       <- CESS_TARGET_FIRST[configId]
useAdaptiveCessTarget <- USE_ADAPTIVE_CESS_TARGET[configId]
useAdaptiveProposal   <- USE_ADAPTIVE_PROPOSAL[configId]
useAdaptiveProposalScaleFactor1 <- USE_ADAPTIVE_PROPOSAL_SCALE_FACTOR1[configId]
modelName <- MODEL_NAMES[configId]

set.seed(replicaId)


## ========================================================================= ##
## THE SMC SAMPLER
## ========================================================================= ##

# Miscellaneous parameters:
miscParameters      <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)
# Auxiliary (known) model parameters and data:
modelParameters     <- getAuxiliaryModelParameters(modelType, nAgeGroups, nLevels, miscParameters)
# Auxiliary parameters for the SMC and MCMC algorithms:
algorithmParameters <- getAuxiliaryAlgorithmParameters(modelParameters$dimTheta, modelParameters$nObservationsCount)

###################
if (SIMULATE_DATA) {

  count <- 100000
  while (mean(count) < 1000 || mean(count) > 10000) {
  
    thetaTrue <- sampleFromPriorCpp(modelParameters$dimTheta, modelParameters$hyperParameters, modelParameters$support, nCores); # making sure that all algorithms are initialised in the same way
    parTrueAux <- simulateDataCpp(modelParameters$nObservationsCount, modelParameters$hyperParameters, thetaTrue, modelParameters$ringRecovery, nCores)
    count <- parTrueAux$count
  }
  print(count)
  modelParameters$thetaTrue    <- thetaTrue
  modelParameters$latentTrue   <- parTrueAux$latentTrue
  modelParameters$count        <- parTrueAux$count
  modelParameters$ringRecovery <- parTrueAux$ringRecovery
  modelParameters$phiTrue      <- parTrueAux$phiTrue
  modelParameters$lambdaTrue   <- parTrueAux$lambdaTrue
  modelParameters$rhoTrue      <- parTrueAux$rhoTrue
  if (nAgeGroups == 2) {
    modelParameters$countTrue    <- parTrueAux$latentTrue[nAgeGroups,]
  } else {
    modelParameters$countTrue    <- colSums(parTrueAux$latentTrue[2:nAgeGroups,])
  }
}
print(t(modelParameters$rhoTrue))
###################

aux <- runSmcSamplerCpp(
  modelParameters$count, modelParameters$ringRecovery, 
  modelParameters$dimTheta, modelParameters$hyperParameters, modelParameters$support, lower, nParticlesUpper, nParticlesLower, algorithmParameters$nMetropolisHastingsUpdates, algorithmParameters$nMetropolisHastingsUpdatesFirst, algorithmParameters$essResamplingThresholdUpper, algorithmParameters$essResamplingThresholdLower, 
  SMC_PARAMETERS, MCMC_PARAMETERS, 
  USE_ADAPTIVE_TEMPERING, useAdaptiveCessTarget, USE_IMPORTANCE_TEMPERING, USE_DELAYED_ACCEPTANCE, useAdaptiveProposal, useAdaptiveProposalScaleFactor1, useDoubleTempering, cessTarget, cessTargetFirst, ALPHA, algorithmParameters$adaptiveProposalParameters, algorithmParameters$rwmhSd, nCores
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
  
  if (!USE_IMPORTANCE_TEMPERING) {
    saveRDS(aux$productivityRates,               file.path(pathToResults, paste("productivityRates_", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(aux$recoveryProbabilities,           file.path(pathToResults, paste("recoveryProbabilities_", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(aux$survivalProbabilities,           file.path(pathToResults, paste("survivalProbabilities", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(aux$smoothedMeans,                   file.path(pathToResults, paste("smoothedMeans_", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(aux$smoothedVariances,               file.path(pathToResults, paste("smoothedVariances_", configId ,"_", replicaId, ".rds", sep='')))
    saveRDS(aux$trueCounts,                      file.path(pathToResults, paste("trueCounts_", configId ,"_", replicaId, ".rds", sep='')))
  }

  
  
### Diagnostic plots:
  
addShaded <- function(x, probs=c(1,0.9), alpha=c(0.15,0.3), col="magenta3", lwd=1) {
  # x: (N,T)-matrix where we average over N samples and display the results for T time steps.
  # probs: a vector of coverage probabilities.
  # col: a single string indicating the colour.
  # lwd: a single value indicating the width of the line representing the median.
  # alpha: transparency parameters (must be the same length as probs).

  T <- dim(x)[2]
  colNumeric <- as.numeric(col2rgb(col))/256
  nProbs  <- length(probs)
  xProbs  <- array(NA, c(T, 2, nProbs))
  xMedian <- rep(NA, times=T)
  
  for (t in 1:T) {
    xMedian[t] <- median(x[,t])
    for (j in 1:nProbs) {
      xProbs[t,1,j] <- quantile(x[,t], probs=(1-probs[j])/2)
      xProbs[t,2,j] <- quantile(x[,t], probs=1-(1-probs[j])/2)
    }
  }

  for (j in 1:nProbs) {
    polygon(c(1:T, T:1), c(xProbs[1:T,2,j], xProbs[T:1,1,j]), border=NA, col=rgb(colNumeric[1], colNumeric[2], colNumeric[3], alpha[j]))
  }
  lines(1:T, xMedian, col=col, type='l')
}


col1 <- "forestgreen" 
col2 <- "magenta3" 
colTrue <- "black"
oma <- c(3.5,4,1.5,0.5) # bottom, left, top, right
mar <- c(0.7,0.7,0.9,0.7)+0.1

survivalProbabilities <- aux$survivalProbabilities
recoveryProbabilities <- aux$recoveryProbabilities
productivityRates     <- aux$productivityRates

T <- modelParameters$nObservationsCount
N <- dim(survivalProbabilities)[2]

if (lower == 3) {
  trueCounts <- array(NA, c(nAgeGroups, T, N))
  for (t in 1:T) {
    for (n in 1:N) {
      trueCounts[,t,n] <- rnorm(nAgeGroups, aux$smoothedMeans[,t,n], sqrt(aux$smoothedVariances[,t,n]))
    }
  }
} else {
  trueCounts <- aux$trueCounts
}

adultCounts <- matrix(NA, T, N)
for (t in 1:T) {
  for (n in 1:N) {
    adultCounts[t,n] <- sum(trueCounts[2:nAgeGroups,t,n])
  }
}

# 
# op <- par(mfrow=c(nAgeGroups,1))
#   for (a in 1:nAgeGroups) {
#     plot(1:T, rep(1,T), col="white", xlab="Time", ylab=paste("Heron counts (age ", a, ")", sep=''), ylim=c(0,max(modelParameters$count)))
#     addShaded(t(trueCounts[a,,]), col=col1)
#   }
#   abline(h=0, lty=3, col="gray")
# par(op)
#   

pdf(file=paste(file.path(pathToFigures, "overview_"), configId ,"_", replicaId, ".pdf", sep=''), width=10, height=14)
op <- par(mfrow=c(4,1), oma=oma, mar=mar)
  
  ## Heron Counts:

  plot(1:T, rep(1,T), col="white", xlab='', ylab='', ylim=c(0,max(modelParameters$count)), yaxs='i', xaxs='i', xaxt="n", yaxt="n")
  addShaded(t(adultCounts), col=col1)
  lines(1:T, modelParameters$count, type='p', pch=1, col=colTrue) # observed counts
  if (lower == 3) {
    mtext(paste("Model for productivity rate: ", modelName, " (with Gaussian approximation)", sep=''), side=3, outer=FALSE, line=0.5)
  } else {
    mtext(paste("Model for productivity rate: ", modelName, " (without Gaussian approximation)", sep=''), side=3, outer=FALSE, line=0.5)
  }
  axis(side=2)
  mtext("Population size", side=2, outer=FALSE, line=3)
  legend("bottomright", legend=c("estimated count (age > 1)", "observed count (age > 1)"), col=c(col1, colTrue), pch=c(-1,1), lty=c(1,-1), bty='n')

  ## Productivity Rates:
  plot(1:T, rep(1,T), col="white", xlab='', ylab='', ylim=c(0,2), yaxs='i', xaxs='i', xaxt="n", yaxt="n")
  addShaded(t(productivityRates), col=col1)
  axis(side=2)
  mtext("Productivity rate", side=2, outer=FALSE, line=3)
  legend("topright", legend=c("estimated productivity rate"), col=c(col1), lty=c(1), bty='n')
  
  ## Survival Probabilities:
  plot(1:T, rep(1,T), col="white", xlab='', ylab='', ylim=c(0,1), yaxs='i', xaxs='i', xaxt="n", yaxt="n")
  for (a in 1:nAgeGroups) {
    addShaded(t(survivalProbabilities[a,,]), col=col1)
  }
  axis(side=2)
  mtext("Survival probability", side=2, outer=FALSE, line=3)
  legend("bottomright", legend=c("estimated survival probabilities (for age groups A, A-1,...,2,1 from top to bottom)"), col=c(col1), lty=c(1), bty='n')
  
  ## Recovery Probabilities:
  plot(1:T, rep(1,T), col="white", xlab='', ylab='', ylim=c(0,1), yaxs='i', xaxs='i', xaxt="n", yaxt="n")
  addShaded(t(recoveryProbabilities), col=col1)
  axis(side=2)
  axis(side=1)
  mtext("Recovery probability", side=2, outer=FALSE, line=3)
  mtext("Time", side=1, outer=FALSE, line=2.5)
  legend("topright", legend=c("estimated recovery probability"), col=c(col1), lty=c(1), bty='n')
  
par(op)
dev.off()


    
     
  
## TODO: also plot the individual age groups here!
  
  
#   save(
#     list  = ls(envir = environment(), all.names = TRUE), 
#     file  = file.path(
#       pathToFull, 
#       paste("fullOutput_configId_", configId, 
#       "_replicaId_", replicaId, 
#       sep='')),
#     envir = environment()
#   ) 
#   rm(aux)
#   q(save="no")
# }


###################
}
