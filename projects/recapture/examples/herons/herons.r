## PMMH with delayed acceptance in the "herons" capture-recapture model

print("WARNING: WE HAVE SHIFTED fDaysCovar by one time period!")


## TODO:

## -- IMPLEMENT & RUN MODEL WITH CONSTANT PRODUCTIVITY
## -- IMPLEMENT GAUSSIAN APPROXIMATION-VERSIONS OF ALL MODELS
## -- ONLY USE DISCRETE UNIFORM INITIAL DISTRIBUTION! (remove parameters delta0 and delta1)
## -- try double tempering again! (probably doesn't work)
## -- check ESJD of the MCMC kernels
## -- try MCWM updates
## -- test SMC samplers on simulated data

#### more speculative:
## -- CSMC updates (would require a different tempering approach?)
## -- can we use Pieralberto's stuff as a proposal?
## -- some other sequence of target distributions?



rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"

exampleName       <- "herons"
projectName       <- "recapture"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))


## ========================================================================= ##
## MODEL
## ========================================================================= ##

## Selecting the model:

# Maximum number of age groups:
nAgeGroups <- 4

# Model to be estimated (needed for model comparison):
# 0: constant productivity rate.
# 1: productivity rates regressed on fDays,
# 2: direct density dependence (log-productivity rates specified as a linear function of abundance),
# 3: threshold dependence (of the productivity rate) on the observed heron counts (with nLevels+1 levels),
# 4: threshold dependence (of the productivity rate) on the true heron counts (with nLevels+1 levels),
# 5: latent Markov regime-switching dynamics for the productivity rates (with nLevels regimes).

modelType <- 1

# Maximum number of levels in the step functions for the productivity parameter
# used by some models:
nLevels <- 2

smcParameters  <- c(3) # additional parameters to be passed to the particle filter;
# Here, the only element of smcParameters determines the number of lookahead steps 
# used if we employ the Kalman filtering/smoothing approximation-based 
# proposal kernel.

mcmcParameters <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates

## Miscellaneous parameters:
MISC_PAR      <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)
## Auxiliary (known) model parameters and data:
MODEL_PAR     <- getAuxiliaryModelParameters(modelType, nAgeGroups, nLevels, MISC_PAR)
## Auxiliary parameters for the SMC and MCMC algorithms:
ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(MODEL_PAR$dimTheta, MODEL_PAR$nObservationsCount)


# plot(MODEL_PAR$count, type='o', ylim=c(0,8000), xaxt='n', xlim=c(1,71), xaxs='i', xlab="year", ylab="no. of individuals")
# lines(1000+1000*MODEL_PAR$fDaysCovar, type='o', col="blue", lty=2)
# axis(1, at=1:71, labels=1928:1998, las=2)
# grid(nx=70, ny=NULL, lwd = 1)
# legend("topleft", legend=c("observed counts", "no. of below-freezing days covariate (scaled)"), lty=c(1,2), col=c("black", "blue"), bty='n')



###############################################################################
##
## Approximating the Marginal (Count-Data) Likelihood
##
###############################################################################

nThetaValues <- 4
nSimulations <- 100
simulateData <- FALSE

prop <- c(0)
nParticlesLower <- c(1000,10000,25000,50000)

N_PARTICLES_LOWER <- rep(nParticlesLower, each=length(prop))
PROP <- rep(prop, times=length(nParticlesLower))

MM <- nSimulations
KK <- nThetaValues
NN <- length(PROP)

logZ <- array(NA, c(KK, NN, MM))
thetaInit <- c()


for (kk in 1:KK) {

  ###################

  
    count <- 100000
    while (mean(count) < 1000 || mean(count) > 10000 || min(count) == 0) {
    
      thetaTrue <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
    
      ## TODO: simulate ring-recovery-data, too
    
    

      parTrueAux <- simulateDataCpp(MODEL_PAR$nObservationsCount, MODEL_PAR$hyperParameters, thetaTrue, MODEL_PAR$ringRecovery, nCores)
      count <- parTrueAux$count
    }
    
  if (simulateData) {
    print(count)
    MODEL_PAR$thetaTrue    <- thetaTrue
    MODEL_PAR$latentTrue   <- parTrueAux$latentTrue
    MODEL_PAR$count        <- parTrueAux$count
    MODEL_PAR$ringRecovery <- parTrueAux$ringRecovery
    MODEL_PAR$phiTrue      <- parTrueAux$phiTrue
    MODEL_PAR$lambdaTrue   <- parTrueAux$lambdaTrue
    MODEL_PAR$rhoTrue      <- parTrueAux$rhoTrue
    MODEL_PAR$countTrue    <- colSums(parTrueAux$latentTrue[2:nAgeGroups,])
  }
  ###################
  
  thetaInit <- cbind(thetaInit, thetaTrue)

  for (nn in 1:NN) {
    for (mm in 1:MM) {
    
      print(paste("kk: ", kk, "; PROP: ", nn))
      
      
      aux <- runSmcFilterCpp(
          MODEL_PAR$count, MODEL_PAR$ringRecovery, 
          MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, N_PARTICLES_LOWER[nn], ALGORITHM_PAR$essResamplingThresholdLower, smcParameters, 
          thetaInit[,kk], PROP[nn], nCores 
        )
        
      logZ[kk,nn,mm] <- aux$logLikelihoodEstimate
      
    }
  }
}






op <- par(mfrow=c(KK,1))

logZ[logZ == -Inf] <- 0
for (kk in 1:KK) {
  boxplot(t(logZ[kk,,]), range=0, names=PROP, xlab="Proposal kernel type") ## ylim=c(1.001,0.999)*range(logZ[kk,dim(logZ)[2],]))
}
par(op)


###############################################################################
##
## PMMH algorithms
##
###############################################################################

nIterations     <- 500000
nSimulations    <- 1
nParticlesLower <- 3000
simulateData   <- FALSE

samplePath <- TRUE # should we store one path of latent variables per iteration
USE_DELAYED_ACCEPTANCE <- c(1) # should delayed acceptance be used?
ADAPT_PROPOSAL <- c(1) # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)

ll <- 1

MM <- nSimulations # number of independent replicates
LL <- length(USE_DELAYED_ACCEPTANCE)

outputTheta   <- array(NA, c(MODEL_PAR$dimTheta, nIterations, LL, MM))
outputLatentPath <- array(NA, c(MODEL_PAR$dimLatentVariable, MODEL_PAR$nObservationsCount, nIterations, LL, MM))
outputCpuTime <- matrix(NA, LL, MM)
outputAcceptanceRateStage1 <- matrix(NA, LL, MM)
outputAcceptanceRateStage2 <- matrix(NA, LL, MM)
mm <- 1


for (mm in 1:MM) {

  ###################
  if (simulateData) {
  
    count <- 100000
    while (mean(count) < 1000 || mean(count) > 10000) {
    
      thetaTrue <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
    
      ## TODO: simulate ring-recovery-data, too
    
    

      parTrueAux <- simulateDataCpp(MODEL_PAR$nObservationsCount, MODEL_PAR$hyperParameters, thetaTrue, MODEL_PAR$ringRecovery, nCores)
      count <- parTrueAux$count
    }
    print(count)
    MODEL_PAR$thetaTrue    <- thetaTrue
    MODEL_PAR$latentTrue   <- parTrueAux$latentTrue
    MODEL_PAR$count        <- parTrueAux$count
    MODEL_PAR$ringRecovery <- parTrueAux$ringRecovery
    MODEL_PAR$phiTrue      <- parTrueAux$phiTrue
    MODEL_PAR$lambdaTrue   <- parTrueAux$lambdaTrue
    MODEL_PAR$rhoTrue      <- parTrueAux$rhoTrue
    MODEL_PAR$countTrue    <- colSums(parTrueAux$latentTrue[2:nAgeGroups,])
  }
  ###################
  
  thetaInit <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
  
  print(t(MODEL_PAR$rhoTrue))
  
  
  for (ll in 1:LL) {
  
    print(ll)
  
    aux <- runPmmhCpp(
      MODEL_PAR$count, MODEL_PAR$ringRecovery, 
      MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nIterations, nParticlesLower, ALGORITHM_PAR$essResamplingThresholdLower, smcParameters, 
      mcmcParameters, USE_DELAYED_ACCEPTANCE[ll], ADAPT_PROPOSAL[ll], FALSE,  ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, thetaInit, ALGORITHM_PAR$burninPercentage, samplePath, nCores 
    )
    
    outputTheta[,,ll,mm] <- matrix(unlist(aux$theta), MODEL_PAR$dimTheta, nIterations)
    
    outputLatentPath[,,,ll,mm] <- array(unlist(aux$latentPath), c(MODEL_PAR$dimLatentVariable, MODEL_PAR$nObservationsCount, nIterations))
    
    
    outputCpuTime[ll,mm] <- unlist(aux$cpuTime)
    outputAcceptanceRateStage1[ll,mm] <- unlist(aux$acceptanceRateStage1)
    outputAcceptanceRateStage2[ll,mm] <- unlist(aux$acceptanceRateStage2)
      
  }
  
  save(
    list  = ls(envir = environment(), all.names = TRUE), 
    file  = paste(pathToResults, "pmmh_nIterations_", nIterations, "_nSimulations_", nSimulations, "_modelType_", modelType, "_nAgeGroups_", nAgeGroups, "_nLevels_", nLevels, "_simulateData_", simulateData, sep=''),
    envir = environment()
  ) 
  
  
}


##### plotting the implied productivity and survival rates:

#load(paste(pathToOutputBase, "pmmh_nIterations_350000_nSimulations_1_modelType_11_nAgeGroups_4_nLevels_3", sep=''))
#load(paste(pathToOutputBase, "pmmh_nIterations_350000_nSimulations_1_modelType_5_nAgeGroups_4_nLevels_3", sep=''))
#load(paste(pathToOutputBase, "pmmh_nIterations_350000_nSimulations_1_modelType_9_nAgeGroups_4_nLevels_3", sep=''))
#load(paste(pathToOutputBase, "pmmh_nIterations_350000_nSimulations_1_modelType_12_nAgeGroups_4_nLevels_3", sep=''))




probs   <- c(0.025,0.05,0.95,0.975)

burnin  <- ceiling(nIterations * ALGORITHM_PAR$burninPercentage)
latent  <- colSums(outputLatentPath[2:nAgeGroups,,(burnin+1):nIterations,1,1])

quantX    <- matrix(NA, length(probs), dim(outputLatentPath)[2])
quantRho  <- matrix(NA, length(probs), dim(outputLatentPath)[2]-1)

quantPhi <- array(NA, c(length(probs), dim(outputLatentPath)[2]-1, nAgeGroups))
phiAux   <- array(NA, c(dim(outputLatentPath)[2]-1, dim(latent)[2], nAgeGroups))

rhoAux  <- matrix(NA, dim(outputLatentPath)[2]-1, dim(latent)[2])

alpha <- outputTheta[4:(nAgeGroups+3),(burnin+1):nIterations,1,1]
beta  <- outputTheta[(nAgeGroups+4):(2*nAgeGroups+3),(burnin+1):nIterations,1,1]

for (tt in 1:dim(phiAux)[1]) {
  for (kk in 1:nAgeGroups) {
    phiAux[tt, ,kk] <- plogis(alpha[kk,] + beta[kk,] * MODEL_PAR$fDaysCovar[tt])
  }
}


## TODO: deal with the case that nLevels == 0; also plot the "true" (i.e. based on thetaInit and the simulated latent variables, for modelType == 3) and also simulate the ring-recovery data.

if (modelType == 0) {

  ## Regression of fDays
  gamma <- outputTheta[(2*(nAgeGroups+1)+4):(2*(nAgeGroups+1)+5),(burnin+1):nIterations,1,1]
  for (tt in 1:dim(rhoAux)[1]) {
    rhoAux[tt, ] <- exp(gamma[1,] + gamma[2,] * MODEL_PAR$fDaysCovar[tt])
  }
  
  modelName <- paste("Productivity rate regressed on the fDays covariate", sep='')
  
} else if (modelType == 1) {

  ## Direct density dependence

  epsilon <- outputTheta[(2*(nAgeGroups+1)+4):(2*(nAgeGroups+1)+5),(burnin+1):nIterations,1,1]
  
  normalisedCounts <- (MODEL_PAR$count - mean(MODEL_PAR$count))/sd(MODEL_PAR$count)
  
  for (tt in 1:dim(rhoAux)[1]) {
    rhoAux[tt, ] <- exp(epsilon[1,] + epsilon[2,] * normalisedCounts[tt])
  }
  
  ## PROBLEM: if we work with normalised count data it's impossible to simulate those data
  
  modelName <- paste("Direct density dependence", sep='')

} else if (modelType == 2) {
  
  ## Threshold dependence on the observed counts

  zeta <- outputTheta[(2*(nAgeGroups+1)+4):(2*(nAgeGroups+1)+4+(nLevels-1)),(burnin+1):nIterations,1,1]
  eta  <- outputTheta[(2*(nAgeGroups+1)+5+(nLevels-1)):(2*(nAgeGroups+1)+4+2*(nLevels-1)),(burnin+1):nIterations,1,1]
  
  for (gg in 1:dim(rhoAux)[2]) {
  
    # Computing the thresholds
    if ((nLevels-1) > 1) {
      tau <- c(0, cumsum(exp(eta[,gg])), .Machine$integer.max)
    } else if ((nLevels-1) == 1) {
      tau <- c(0, cumsum(exp(eta[gg])), .Machine$integer.max)
    }
    
    # Computing the levels
    nu  <- rep(NA, times=(nLevels-1)+1)
    nu[(nLevels-1)+1] <- exp(zeta[(nLevels-1)+1,gg])
    for (kk in (nLevels-1):1) {
      nu[kk] <- nu[kk+1] + exp(zeta[kk,gg])
    }

    # Evaluating the step function:
    if ((nLevels-1) > 0) {
      rhoAux[,gg] <- nu[.bincode(x=MODEL_PAR$count[1:(dim(outputLatentPath)[2]-1)], breaks=tau, TRUE, TRUE)]
    } else if ((nLevels-1) == 0) {
      ## TODO
    }
  }
  
  modelName <- paste("Threshold dependence of the productivity rate on observed counts (", nLevels , " levels)", sep='')
  
} else if (modelType == 3) {

  ## Threshold dependence on the true counts
  
  zeta <- outputTheta[(2*(nAgeGroups+1)+4):(2*(nAgeGroups+1)+4+(nLevels-1)),(burnin+1):nIterations,1,1]
  eta  <- outputTheta[(2*(nAgeGroups+1)+5+(nLevels-1)):(2*(nAgeGroups+1)+4+2*(nLevels-1)),(burnin+1):nIterations,1,1]
  
  for (gg in 1:dim(rhoAux)[2]) {
  
    # Computing the thresholds
    if (nLevels > 2) {
      tau <- c(0, cumsum(exp(eta[,gg])), .Machine$integer.max)
    } else if (nLevels == 2) {
      tau <- c(0, cumsum(exp(eta[gg])), .Machine$integer.max)
    }
    
    
    # Computing the levels
    nu  <- rep(NA, times=nLevels)
    nu[nLevels] <- exp(zeta[nLevels,gg])
    for (kk in (nLevels-1):1) {
      nu[kk] <- nu[kk+1] + exp(zeta[kk,gg])
    }

    # Evaluating the step function:
    for (tt in 1:dim(rhoAux)[1]) {
      rhoAux[tt,gg] <- nu[.bincode(x=latent[tt,gg], breaks=tau, TRUE, TRUE)]
    }
  }
  modelName <- paste("Threshold dependence of the productivity rate on true counts (", nLevels , " levels)", sep='')

} else if (modelType == 4) {

  ## Markov regime-switching model for the productivity
  
  zeta <- outputTheta[(2*(nAgeGroups+1)+4):(2*(nAgeGroups+1)+3+nLevels),(burnin+1):nIterations,1,1]
  nu <- exp(zeta)
  for (kk in 2:nLevels) {
    nu[kk,] = exp(zeta[kk,]) + nu[kk-1] 
  }
  
  regimes <- outputLatentPath[nAgeGroups+1,,(burnin+1):nIterations,1,1]
  for (tt in 1:dim(rhoAux)[1]) {
    for (kk in 1:nLevels) {
      if (length(regimes[tt+1, regimes[tt+1,]==kk-1]) > 0) {
        rhoAux[tt, regimes[tt+1,]==kk-1] <- nu[kk, regimes[tt+1,]==kk-1]
      }
    }
  }
  
  modelName <- paste("Markov regime-switching dynamics for productivity (", nLevels , " regimes)", sep='')
}

for (tt in 1:dim(quantX)[2]) {
  quantX[,tt] <- quantile(x=latent[tt,], probs=probs)
}
for (tt in 1:dim(quantRho)[2]) {
  quantRho[,tt] <- quantile(x=rhoAux[tt,], probs=probs)
}
for (tt in 1:dim(quantPhi)[1]) {
  for (kk in 1:nAgeGroups) {
    quantPhi[,tt,kk] <- quantile(x=phiAux[tt,,kk], probs=probs)
  }
}


meanRho  <- rowMeans(rhoAux)
meanX    <- rowMeans(latent)

meanPhi <- matrix(NA, nAgeGroups, dim(phiAux)[1])
for (kk in 1:nAgeGroups) {
  meanPhi[kk,] <- rowMeans(phiAux[,,kk])
}

colX        <- as.numeric(col2rgb("red"))/256
colRho      <- as.numeric(col2rgb("red"))/256
colRhoTrue  <- as.numeric(col2rgb("blue"))/256
colPhi      <- c("chocolate1", "chocolate2", "chocolate3", "chocolate4")
alphaPlot   <- c(0.1, 0.2, 0.5, 0.9)

colTrue <- "blue"

T <- MODEL_PAR$nObservationsCount

pdf(file=paste(pathToFigures, "pmmh_nIterations_", nIterations, "_nSimulations_", nSimulations, "_modelType_", modelType, "_nAgeGroups_", nAgeGroups, "_nLevels_", nLevels, "_simulateData_", simulateData, ".pdf", sep=''), width=10, height=18)
op <- par(mfrow=c(4,1))
plot(MODEL_PAR$count, type='l', ylim=c(0,8000), xaxt='n', xlim=c(1,T), xaxs='i', xlab="year", ylab="no. of individuals", col="white", main=modelName)
polygon(c(1:T, T:1), c(quantX[4,], quantX[1,T:1]), border=NA, col=rgb(colX[1], colX[2], colX[3], alphaPlot[1]))
polygon(c(1:T, T:1), c(quantX[3,], quantX[2,T:1]), border=NA, col=rgb(colX[1], colX[2], colX[3], alphaPlot[2]))
lines(1:length(meanX), meanX, type='l', col="red")
points(MODEL_PAR$count, pch=1, col="black")
axis(1, at=1:T, labels=1928:1998, las=2)
if (simulateData) {
  lines(1:length(MODEL_PAR$countTrue), MODEL_PAR$countTrue, type='l', col=colTrue)
  legend("topleft", legend=c("observed counts", "true counts of herons aged > 1", "mean estimated counts of herons aged > 1 (shaded areas: 95 % and 90 % of realisations)"), lty=c(-1,1,1), pch=c(1,-1,-1), col=c("black", colTrue, "red"), bty='n')
} else {
  legend("topleft", legend=c("observed counts", "mean estimated counts of herons aged > 1 (shaded areas: 95 % and 90 % of realisations)"), lty=c(-1,1), pch=c(1,-1), col=c("black", "red"), bty='n')
}
grid(nx=T-1, ny=NULL, lwd = 1)

plot(MODEL_PAR$fDaysCovar, type='l', col="red", lty=1, xaxs='i', xaxt='n', xlab="year", ylab="fDays covariate (normalised)", xlim=c(1,T))
axis(1, at=1:T, labels=1928:1998, las=2)
grid(nx=T-1, ny=NULL, lwd = 1)

plot(1:length(meanRho), meanRho, type='l', ylim=c(0,10), xaxt='n', xlim=c(1,T), xaxs='i', xlab="year", ylab="productivity rate", col="red")
polygon(c(1:(T-1), (T-1):1), c(quantRho[4,], quantRho[1,(T-1):1]), border=NA, col=rgb(colRho[1], colRho[2], colRho[3], alphaPlot[1]))
polygon(c(1:(T-1), (T-1):1), c(quantRho[3,], quantRho[2,(T-1):1]), border=NA, col=rgb(colRho[1], colRho[2], colRho[3], alphaPlot[2]))
if (simulateData) {
  lines(1:length(MODEL_PAR$rhoTrue), MODEL_PAR$rhoTrue, type='l', col=colTrue)
  legend("topleft", legend=c("true productivity rate", "mean productivity rate (shaded areas: 95 % and 90 % of realisations)"), lty=c(1,1), pch=c(1,1), col=c(colTrue, "red"), bty='n')
} else {
  legend("topleft", legend=c("mean estimated productivity rate (shaded areas: 95 % and 90 % of realisations)"), lty=c(1), pch=c(1), col=c("red"), bty='n')
}
axis(1, at=1:T, labels=1928:1998, las=2)
grid(nx=T-1, ny=NULL, lwd = 1)

plot(1:length(meanPhi[1,]), meanPhi[1,], type='l', ylim=c(0,1), xaxt='n', xlim=c(1,71), xaxs='i', xlab="year", ylab="mean survival probabilities", col="red", lty=1)

for (kk in 2:nAgeGroups) {
  lines(meanPhi[kk,], lty=kk, col="red")
}

axis(1, at=1:T, labels=1928:1998, las=2)

if (simulateData) {

  for (kk in 1:nAgeGroups) {
    lines(1:length(MODEL_PAR$phiTrue[kk,]), MODEL_PAR$phiTrue[kk,], type='l', lty=kk, col=colTrue)
  }

  legend("topleft", legend=c("first-years", "second-years", "third-years", "adults", "first-years (true)", "second-years (true)", "third-years (true)", "adults (true)"), lty=c(1:nAgeGroups, 1:nAgeGroups), col=c(rep("red", times=nAgeGroups), rep(colTrue, times=nAgeGroups)), bty='n')
  
} else {
  legend("topleft", legend=c("first-years", "second-years", "third-years", "adults"), lty=1:nAgeGroups, col=rep("red", times=nAgeGroups), bty='n')
}

grid(nx=T-1, ny=NULL, lwd = 1)

# legend("topleft", legend=c("observed counts", "estimated marginal productivity rate", "no. of below-freezing days covariate (scaled)"), lty=c(1,1,2), col=c("black", "red", "blue"), bty='n')

par(op)
dev.off()


###############################################################################
##
## PG algorithms
##
###############################################################################

nIterations     <- 200000
nSimulations    <- 1
nParticlesLower <- 500

estimateTheta <- TRUE
nThetaUpdates <- 200
csmc <- 2

ll <- 1
MM <- nSimulations # number of independent replicates
LL <- 1

outputTheta   <- array(NA, c(MODEL_PAR$dimTheta, nIterations, LL, MM))
outputCpuTime <- matrix(NA, LL, MM)
outputAcceptanceRate <- matrix(NA, LL, MM)

for (mm in 1:MM) {

#   thetaInit <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
  
  for (ll in 1:LL) {
  
    print(ll)
  
    aux <- runPgCpp(
      MODEL_PAR$count, MODEL_PAR$ringRecovery, 
      MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nIterations, nParticlesLower, ALGORITHM_PAR$essResamplingThresholdLower, smcParameters, 
      mcmcParameters, ALGORITHM_PAR$rwmhSd/100, thetaInit, ALGORITHM_PAR$burninPercentage, estimateTheta, nThetaUpdates, csmc, nCores 
    )
    
      outputLogEvidenceEstimate[ii,ll,1,mm] <- aux$logEvidenceEstimate
      outputLogEvidenceEstimate[ii,ll,2,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeights) ## uses importance-tempering weights proportional to the generalised ESS 
      outputLogEvidenceEstimate[ii,ll,3,mm] <- aux$logEvidenceEstimateEssAlternate ## uses importance-tempering weights proportional to the ESS 
      outputLogEvidenceEstimate[ii,ll,4,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeightsResampled)
      outputLogEvidenceEstimate[ii,ll,5,mm] <- aux$logEvidenceEstimateEssResampledAlternate ## based importance-tempering weights proportional to the ESS 
      
      ## Checking calculations (results should be numerically 0)
      print(computeLogZEss(aux$logUnnormalisedReweightedWeights) - aux$logEvidenceEstimateEssAlternate)
      print(computeLogZEss(aux$logUnnormalisedReweightedWeightsResampled) - aux$logEvidenceEstimateEssResampledAlternate)
      
  }
  
  save(
    list  = ls(envir = environment(), all.names = TRUE), 
    file  = paste(pathToResults, "pg_nIterations_", nIterations, "_nSimulations_", nSimulations, "_modelType_", modelType, sep=''), ## TODO
    envir = environment()
  ) 
}


## ========================================================================= ##
## PLOT RESULTS
## ========================================================================= ##

#load(paste(pathToOutputBase, "pmmh_nIterations_5e+05_nSimulations_1_modelType_2_nAgeGroups_4_nLevels_2_simulateData_FALSE", sep=''))

#########################
## Summary statistics
# X <- outputTheta[,50000:100000,3,1,1]
# Z <- summary(t(X))
# postMean <- round(apply(X,1,mean), digits=2)
# postSd <- round(apply(X,1,sd), digits=2)
# 
# 
# TAB <- data.frame(postMean, postSd)
# rownames(TAB) <- thetaNames
# colnames(TAB) <- c("mean", "sd")
#########################


COL_LL  <- c("royalblue4", "red4", "royalblue", "red2")
# LTY_NN  <- c(2,1)
LTY_NN  <- c(1)
LAG_MAX <- 500
PAR_LIM <- cbind(rep(-10, times=MODEL_PAR$dimTheta), rep(10, times=MODEL_PAR$dimTheta))
KDE_LIM <- rep(5, times=MODEL_PAR$dimTheta)
CEX_LEGEND <- 0.9
ALGORITHMS <- c("Standard PMMH", "Standard PMMH with Delayed Acceptance", "Adaptive PMMH", "Adaptive PMMH with Delayed Acceptance")

WIDTH  <- 15
HEIGHT <- 3.5


plotThetaMcmc <- function(title, theta, idx, burnin, KDE=TRUE, ACF=TRUE, TRACE=TRUE)
{
  II <- length(idx)
  JJ <- sum(KDE, ACF, TRACE)
  
  GG <- dim(theta)[2] # number of iterations
  LL <- dim(theta)[3] # number of different algorithms to compare
  MM <- dim(theta)[4] # number of independent replicates
  
  pdf(file=paste(pathToFigures, title, ".pdf", sep=''), width=WIDTH, height=HEIGHT)
      
  for (ii in 1:II) {
    op <- par(mfrow=c(1, JJ))
    if (KDE) {
      plot(density(theta[ii,burnin:GG,LL,mm]), type='l', col="white", xlab=MODEL_PAR$thetaNames[idx[ii]], ylab="Density", xlim=PAR_LIM[idx[ii],], ylim=c(0, KDE_LIM[idx[ii]]), main='')
      for (ll in 1:LL) {
        for (mm in 1:MM) {
          lines(density(theta[ii,burnin:GG,ll,mm]), type='l', col=COL_LL[ll], lty=1, main='')
        }
      }
      grid <- seq(from=min(PAR_LIM[idx[ii],]), to=max(PAR_LIM[idx[ii],]), length=10000)
      lines(grid, dnorm(grid, mean=MODEL_PAR$meanHyper[idx[ii]], sd=MODEL_PAR$sdHyper[idx[ii]]), col="black", lty=2)
      if (simulateData) {
        abline(v=MODEL_PAR$thetaTrue[idx[ii]], col="red")
      }
      legend("topleft", legend=c(ALGORITHMS, "Prior"), col=c(rep(COL_LL,2), "black"), bty='n', lty=c(rep(LTY_NN, each=LL), 2), cex=CEX_LEGEND)
    }
    if (ACF) {
      plot(0:LAG_MAX, rep(1, times=LAG_MAX+1), type='l', col="white", xlab=paste("Lag (",MODEL_PAR$thetaNames[idx[ii]],")", sep=''), ylab="ACF", ylim=c(0,1), xlim=c(0,LAG_MAX))
      mtext(paste("No. of iterations: ", GG, "; of which burn-in: ", burnin, sep=''), side = 3, line = 1, outer = FALSE)
      for (ll in 1:LL) {
        for (mm in 1:MM) {    
          ACF_PLOT <-as.numeric(acf(theta[ii,burnin:GG,ll,mm], lag.max=LAG_MAX, plot=FALSE)$acf)
          lines(0:LAG_MAX, ACF_PLOT, type='l', col=COL_LL[ll], lty=1)
        }
      }
    }
    if (TRACE) {
      plot(1:GG, rep(1, times=GG), type='l', col="white", ylab=MODEL_PAR$thetaNames[idx[ii]], xlab="Iteration", ylim=PAR_LIM[idx[ii],], xlim=c(1,GG))
      for (ll in 1:LL) {
        for (mm in 1:MM) {    
          lines(1:GG, theta[ii,,ll,mm], type='l', col=COL_LL[ll], lty=1)
        }
      }
      if (simulateData) {
        abline(h=MODEL_PAR$thetaTrue[idx[ii]], col="red")
      }
    }
    par(op)
  }
  dev.off()

}


plotThetaMcmc(paste("pmmh_theta_modelType_", modelType, "_simulateData_", simulateData, sep=''), outputTheta, 1:MODEL_PAR$dimTheta, burnin=ceiling(nIterations * ALGORITHM_PAR$burninPercentage))


plotThetaMcmc(paste("pg_theta_modelType_", modelType, sep=''), outputTheta, 1:MODEL_PAR$dimTheta, burnin=ceiling(nIterations * ALGORITHM_PAR$burninPercentage))


## Plots an overview of the posterior correlations

# load(paste(pathToOutputBase, "pmmh_nIterations_1e+05_nSimulations_1_modelType_1", sep=''))

pdf(file=paste(pathToFigures, "mcmc_estimated_posterior_correlations_in_model_", modelType, ".pdf", sep=''), width=WIDTH, height=WIDTH)
op <- par(oma=c(1,4,4,1), mar=c(1,4,4,1))
corrMat <- cor(t(outputTheta[,,1,1]))
corrMat <- corrMat[MODEL_PAR$dimTheta:1,]
image(corrMat, main="", xaxt='n', yaxt='n', col=terrain.colors(500))
for (i in 1:MODEL_PAR$dimTheta) {
  for (j in (MODEL_PAR$dimTheta-i+1):MODEL_PAR$dimTheta) {
    text(x=(i-1)/(MODEL_PAR$dimTheta-1), y=(j-1)/(MODEL_PAR$dimTheta-1), labels=round(corrMat[i,j], digits=2), cex=0.5)
  }
}
axis(3, at=seq(from=0, to=1, length=MODEL_PAR$dimTheta), labels=MODEL_PAR$thetaNames, las=2)
axis(2, at=seq(from=0, to=1, length=MODEL_PAR$dimTheta), labels=MODEL_PAR$thetaNames[MODEL_PAR$dimTheta:1], las=1)
par(op)
dev.off()



# ###############################################################################
# ##                                                                           
# ## SMC Samplers for Model Selection                                          
# ##                                                                           
# ###############################################################################
# 
# 
# ## ========================================================================= ##
# ## Estimating the model evidence
# ## ========================================================================= ##

# load(paste(pathToOutputBase,  "smc_estimating_model_evidence_nParticlesUpper_1000nParticlesLowerr_4000_nSimulations_3_all_models_",sep=''))

nParticlesUpper <- 100
nParticlesLower <- 500
useImportanceTempering <- TRUE
nSteps <- 
nMetropolisHastingsUpdates <- 1
nSimulations <- 2
simulateData <- FALSE


MODELS     <- c(3)
NLEVELS    <- c(3)
NAGEGROUPS <- rep(4, times=length(MODELS))

ALPHA                  <- seq(from=0, to=1, length=nSteps)
USE_DELAYED_ACCEPTANCE <- c(1) # should delayed acceptance be used?
USE_ADAPTIVE_TEMPERING <- c(1) ### c(1)
ADAPT_PROPOSAL         <- c(1) # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
USE_DOUBLE_TEMPERING   <- c(0)

MM <- nSimulations # number of independent replicates
LL <- length(USE_DELAYED_ACCEPTANCE)


outputLogEvidenceEstimate <- array(NA, c(length(MODELS), LL, 5, MM))

for (mm in 1:MM) {

  for (ll in 1:LL) {

    for (ii in 1:length(MODELS)) {
    
      ### TODO: we need to use NLEVELS and NAGEGROUPS here!!!!!!!!!!!!!!!!!!!!
    
      print(paste("Herons model selection: ", mm , ", " , ii, sep=''))
    
      ## Auxiliary (known) model parameters:
      nAgeGroups <- NAGEGROUPS[ii]
      nLevels <- NLEVELS[ii]
      modelType <- MODELS[ii]
      
      MODEL_PAR     <- getAuxiliaryModelParameters(MODELS[ii], NAGEGROUPS[ii], NLEVELS[ii], MISC_PAR)
      ## Auxiliary parameters for the SMC and MCMC algorithms:
      ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(MODEL_PAR$dimTheta, MODEL_PAR$nObservationsCount)
      
      
      
      ###################
      if (simulateData) {
      
        count <- 100000
        while (mean(count) < 1000 || mean(count) > 10000) {
        
          thetaTrue <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
        
          ## TODO: simulate ring-recovery-data, too
        
        

          parTrueAux <- simulateDataCpp(MODEL_PAR$nObservationsCount, MODEL_PAR$hyperParameters, thetaTrue, MODEL_PAR$ringRecovery, nCores)
          count <- parTrueAux$count
        }
        print(count)
        MODEL_PAR$thetaTrue    <- thetaTrue
        MODEL_PAR$latentTrue   <- parTrueAux$latentTrue
        MODEL_PAR$count        <- parTrueAux$count
        MODEL_PAR$ringRecovery <- parTrueAux$ringRecovery
        MODEL_PAR$phiTrue      <- parTrueAux$phiTrue
        MODEL_PAR$lambdaTrue   <- parTrueAux$lambdaTrue
        MODEL_PAR$rhoTrue      <- parTrueAux$rhoTrue
        MODEL_PAR$countTrue    <- colSums(parTrueAux$latentTrue[2:nAgeGroups,])
      }
      print(t(MODEL_PAR$rhoTrue))
      ###################

      


      aux <- runSmcSamplerCpp(
            MODEL_PAR$count, MODEL_PAR$ringRecovery, 
            MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, 0, nParticlesUpper, nParticlesLower, ALGORITHM_PAR$nMetropolisHastingsUpdates, ALGORITHM_PAR$nMetropolisHastingsUpdatesFirst, ALGORITHM_PAR$essResamplingThresholdUpper, ALGORITHM_PAR$essResamplingThresholdLower, 
            smcParameters, mcmcParameters, 
            USE_ADAPTIVE_TEMPERING[ll], useImportanceTempering, USE_DELAYED_ACCEPTANCE[ll], ADAPT_PROPOSAL[ll], USE_DOUBLE_TEMPERING[ll], ALGORITHM_PAR$cessTarget, ALGORITHM_PAR$cessTargetFirst, ALPHA, ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, nCores
          )
          
      outputLogEvidenceEstimate[ii,ll,1,mm] <- aux$logEvidenceEstimate
      outputLogEvidenceEstimate[ii,ll,2,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeights) ## uses importance-tempering weights proportional to the generalised ESS 
      outputLogEvidenceEstimate[ii,ll,3,mm] <- aux$logEvidenceEstimateEssAlternate ## uses importance-tempering weights proportional to the ESS 
      outputLogEvidenceEstimate[ii,ll,4,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeightsResampled)
      outputLogEvidenceEstimate[ii,ll,5,mm] <- aux$logEvidenceEstimateEssResampledAlternate ## based importance-tempering weights proportional to the ESS 
      
      ## Checking calculations (results should be numerically 0)
      print(computeLogZEss(aux$logUnnormalisedReweightedWeights) - aux$logEvidenceEstimateEssAlternate)
      print(computeLogZEss(aux$logUnnormalisedReweightedWeightsResampled) - aux$logEvidenceEstimateEssResampledAlternate)
      
      save(
        list  = ls(envir = environment(), all.names = TRUE), 
        file  = paste(
          pathToResults, 
          "smc_nParticlesUpper_", nParticlesUpper, 
          "_nParticlesLower_", nParticlesLower, 
          "_nSimulations_", nSimulations, 
          "_model_", MODELS[ii], 
          "_simulation_run_", mm, 
          sep=''),
        envir = environment()
      ) 
      rm(aux)
    }
    
  }
}



## ========================================================================= ##
## Plotting parameter estimates from final step of one of the SMC samplers
## ========================================================================= ##

# nSteps <- length(aux$inverseTemperatures)
# theta  <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nSteps, nParticlesUpper))
# theta  <- theta[,nSteps,] ## only using the particles from the last step here! TODO: we may need to resample these first or use the weights attached to them!




nSteps <- length(aux$inverseTemperatures)
theta  <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nParticlesUpper, nSteps))



WIDTH  <- 10
HEIGHT <- 6
COL_LL  <- c("royalblue4", "red4", "royalblue", "red2")
LTY_NN  <- c(1)
LAG_MAX <- 500
PAR_LIM <- cbind(rep(-10, times=MODEL_PAR$dimTheta), rep(10, times=MODEL_PAR$dimTheta))
KDE_LIM <- rep(5, times=MODEL_PAR$dimTheta)
CEX_LEGEND <- 0.9


plotThetaSmc <- function(title, theta, idx)
{
  RR <- length(idx)

  pdf(file=paste(MISC_PAR$pathToFigures, title, ".pdf", sep=''), width=WIDTH, height=HEIGHT)
      
  for (rr in 1:RR) {
  
    plot(density(theta[rr,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]), type='l', col="white", xlab=MODEL_PAR$thetaNames[idx[rr]], ylab="Density", xlim=PAR_LIM[idx[rr],], ylim=c(0, KDE_LIM[idx[rr]]), main='')
    
    for (jj in 1:5) {
  
      load(paste(pathToOutputBase, "smc_estimating_model_evidence_modelType_2_nParticlesUpper_1000_nParticlesLower_2000_nSimulations_25_simulateData_FALSE_nLevels_3_ii_1_mm_", jj, sep=''))
      
      nSteps <- length(aux$inverseTemperatures)
      theta  <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nParticlesUpper, nSteps))
    
      lines(density(theta[rr,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]), type='l', col="red", lty=1, main='')
      lines(density(c(theta[rr,,]), weights=c(aux$importanceTemperingWeights)), type='l', col="red", lty=2, main='')
    }
    
    grid <- seq(from=min(PAR_LIM[idx[rr],]), to=max(PAR_LIM[idx[rr],]), length=10000)
    lines(grid, dnorm(grid, mean=MODEL_PAR$meanHyper[idx[rr]], sd=MODEL_PAR$sdHyper[idx[rr]]), col="black", lty=2)
    if (simulateData) {
      abline(v=thetaTrue[rr,], col="blue")
    }
    legend("topleft", legend=c("SMC sampler", "Prior"), col=c("red", "black"), bty='n', lty=c(1, 2), cex=CEX_LEGEND)
    
    
  }
  dev.off()
}

plotThetaSmc(paste("smc_theta_modelType_", modelType, "nParticlesUpper_", nParticlesUpper, "_nParticlesLower_", nParticlesLower, "_nLevels_", nLevels, "_simulateData_", simulateData, sep=''), theta, 1:MODEL_PAR$dimTheta)





for (jjj in 1:4) {
  for (iii in 0:9) {
  load(paste("/home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/recapture/herons/legion_2017-05-09/herons_smc_nParticlesUpper_500_nParticlesLower_2000_model_", jjj, "_simulation_run_", iii, sep=''))
#   print(outputLogEvidenceEstimate)
  aux$logUnnormalisedWeights
    print(length(aux$inverseTemperatures))
  }
}


  load("/home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/recapture/herons/testtest/herons_smc_nParticlesUpper_3000_nParticlesLower_3000_model_0_nLevels_2_nAgeGroups_4_simulation_run_0")
#   print(outputLogEvidenceEstimate)
  aux$logUnnormalisedWeights
    print(length(aux$inverseTemperatures))






## ========================================================================= ##
## Plotting the log-model evidence
## ========================================================================= ##

modelIndices <- 0:6

# 
# load(paste(pathToOutputBase, "smc_estimating_model_evidence_modelType_2_nParticlesUpper_250_nParticlesLower_3000_nSimulations_25_simulateData_FALSE_nLevels_2_ii_1_mm_1", sep=''))

# library(tikzDevice)
# options( 
#   tikzDocumentDeclaration = c(
#     "\\documentclass[12pt]{beamer}",
#     "\\usepackage{amssymb,amsmath,graphicx,mathtools,mathdots,stmaryrd}",
#     "\\usepackage{tikz}" 
#   )
# )

pdf(file=paste(pathToFigures, "owls_evidence_boxplot.pdf", sep=''), width=5, height=7)

boxplot(t(outputLogEvidenceEstimate[,1:mm]), 
    ylab="Log-Evidence",
    xlab="Model",
    range=0,
    las=2, 
    names=modelIndices+1,
    main=""
  )
dev.off()



## ========================================================================= ##
## Plotting the log-model evidence
## ========================================================================= ##

MODEL_IDX     <- c(0,1,2,2,3,3,4,4)
N_SIMULATIONS <- 4
N_LEVELS      <- c(2,2,2,3,2,3,2,3)
RR            <- length(MODEL_IDX)
MODEL_NAMES   <- c("1", "2", "3 (K=2)", "3 (K=3)", "4 (K=2)", "4 (K=3)","5 (K=2)", "5 (K=3)")

logZ <- matrix(NA, RR, N_SIMULATIONS)

for (rr in 1:RR) {
  for (ss in 1:N_SIMULATIONS) {
  
    
    load(paste(pathToOutputBase, "smc_estimating_model_evidence_modelType_", MODEL_IDX[rr], "_nParticlesUpper_1000_nParticlesLower_2000_nSimulations_25_simulateData_FALSE_nLevels_",N_LEVELS[rr],"_ii_1_mm_", ss, sep=''))

  
    logZ[rr,ss] <- aux$logEvidenceEstimate
  }
}


pdf(file=paste(pathToFigures, "herons_evidence_boxplot.pdf", sep=''), width=5, height=7)

boxplot(t(logZ), 
    ylab="Log-Evidence",
    xlab="Model",
    range=0,
    las=2, 
    names=MODEL_NAMES,
    main=""
  )
dev.off()




## Plotting parameter estimates obtained from single SMC sampler:

FILE       <- "/home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/recapture/herons/testtest/herons_smc_nParticlesUpper_3000_nParticlesLower_3000_model_0_nLevels_2_nAgeGroups_4_simulation_run_0"
WIDTH      <- 10
HEIGHT     <- 6
LAG_MAX    <- 500    # maximum lag
PAR_MIN    <- -10    # minimum of first axis
PAR_MAX    <- 10     # maximum of first axis
KDE_MAX    <- 5      # limit of the 2nd axis of kernel density plots
LEGEND_CEX <- 0.9

COL_SMC_DEFAULT <- "red"
COL_SMC_REUSE   <- "red"
LTY_SMC_DEFAULT <- 1
LTY_SMC_REUSE   <- 2
COL_PRIOR       <- "black"
LTY_PRIOR       <- 1


plotThetaSmc <- function(input, output, imageType)
{
  load(input) 
  nSteps  <- length(aux$inverseTemperatures)
  theta   <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nParticlesUpper, nSteps))
  par_lim <- cbind(rep(PAR_MIN, times=MODEL_PAR$dimTheta), rep(PAR_MAX, times=MODEL_PAR$dimTheta))
  kde_lim <- rep(KDE_MAX, times=MODEL_PAR$dimTheta) 
  RR      <- MODEL_PAR$dimTheta # number of parameters to plot
  
  if (imageType == "pdf") {
    pdf(file=paste(output, ".pdf", sep=''), width=WIDTH, height=HEIGHT)
  } else if (imageType == "tex") {
    #tikz(file=paste(output, ".tex", sep=''), width=WIDTH, height=HEIGHT)
  } else if (imageType == "none") {
    # EMPTY
  }
  

  
  for (rr in 1:RR) {

    plot(density(theta[rr,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]), type='l', col="white", xlab=MODEL_PAR$thetaNames[rr], ylab="Density", xlim=par_lim[rr,], ylim=c(0, kde_lim[rr]), main='')
    
    lines(density(theta[rr,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]),     type='l', col=COL_SMC_DEFAULT, lty=LTY_SMC_DEFAULT, main='')
    lines(density(c(theta[rr,,]),    weights=c(aux$selfNormalisedWeightsESS)), type='l', col=COL_SMC_REUSE,   lty=LTY_SMC_REUSE,   main='')
    
#     for (ss in 1:nSteps) {
#       lines(density(c(theta[rr,,ss]),    weights=c(aux$selfNormalisedWeights[,ss])), type='l', col="blue",   lty=LTY_SMC_REUSE,   main='')
#     }

    grid <- seq(from=min(par_lim[rr,]), to=max(par_lim[rr,]), length=10000)
    lines(grid, dnorm(grid, mean=MODEL_PAR$meanHyper[rr], sd=MODEL_PAR$sdHyper[rr]), col=COL_PRIOR, lty=LTY_PRIOR)
    
    # Plot true values (if simulated data is used)
    if (simulateData) {
      abline(v=thetaTrue[rr,], col="blue")
    }
    
    # Add legend
    legend("topleft", 
      legend=c("SMC sampler (using only final-step samples)", "SMC sampler (using all particles)", "Prior"), 
      col=c(COL_SMC_DEFAULT, COL_SMC_REUSE, COL_PRIOR), bty='n', lty=c(LTY_SMC_DEFAULT, LTY_SMC_REUSE, LTY_PRIOR), cex=LEGEND_CEX
    )
    
  }
  
  if (imageType != "none") {
    dev.off()
  }


}

plotThetaSmc(FILE, paste("/home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/recapture/herons/","smc_test", sep=''), "pdf")



