## SMC sampler for a class of multivariate stochastic volatility models
## with copula dependence

rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"

exampleName       <- "factorCopulaSv"
projectName       <- "copulaSv"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))


## NOTE: 
## log kappa shouldn't get too large (> 4)
## log sigma shouldn't get too small (<0)

## ========================================================================= ##
##
## MODEL
##
## ========================================================================= ##

## Selecting the model
## ------------------------------------------------------------------------- ##

# Model to be estimated:
# 0: exactly measured log-volatility
# 1: noisily measured log-volatility
modelType <- 0

# Type of copulas used for modelling the dependence structure
# between the noise variables and the latent factors:
# 0: Gaussian
# 1: rotated by 90 degrees
# 2: rotated by 270 degrees
copulaTypeH  <- 0 # dependence between exchange-rate noise eta and latent factor H
copulaTypeZ  <- 0 # dependence between volatility noise zeta and latent factor Z
copulaTypeHZ <- 0 # dependence between latent factors H and Z

smcParameters  <- c(3) # additional parameters to be passed to the particle filter;
# Here, the only element of smcParameters determines the number of lookahead steps 
# used if we employ the Kalman filtering/smoothing approximation-based 
# proposal kernel.

mcmcParameters <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates

# Miscellaneous parameters:
MISC_PAR  <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)


## Specifying the data set to be used
## ------------------------------------------------------------------------- ##

## Simulating new data set (with 3 exchange rates)
nObservationsSimulated  <- 100
nExchangeRatesSimulated <- 3

alphaTrue    <- rep(0, times=nExchangeRatesSimulated)
betaTrue     <- rep(1/2, times=nExchangeRatesSimulated)
kappaTrue    <- rep(10, times=nExchangeRatesSimulated)
muTrue       <- rep(1/100, times=nExchangeRatesSimulated)
sigmaTrue    <- rep(1/10, times=nExchangeRatesSimulated)
lambdaHTrue  <- rep(1/2, times=nExchangeRatesSimulated)
lambdaZTrue  <- rep(1/2, times=nExchangeRatesSimulated)
lambdaHZTrue <- 1/2
omegaTrue    <- 1


# varthetaAux <- c(0, 1/2, 2.5, -1.7, -2.5, 1, -1)
# thetaTrue   <- c(rep(varthetaAux, each=nExchangeRatesSimulated), 0)
# if (modelType == 1) {
#   thetaTrue <- c(thetaTrue, 0)
# }

thetaTrue        <- transformParametersToTheta(alpha=alphaTrue, beta=betaTrue, kappa=kappaTrue, mu=muTrue, sigma=sigmaTrue, lambdaH=lambdaHTrue, lambdaZ=lambdaZTrue, lambdaHZ=lambdaHZTrue, omega=omegaTrue, copulaTypeH=copulaTypeH, copulaTypeZ=copulaTypeZ, copulaTypeHZ=copulaTypeHZ, modelType=modelType)

parametersTrue   <- unlist(transformThetaToParameters(theta=thetaTrue, copulaTypeH=copulaTypeH, copulaTypeZ=copulaTypeZ, copulaTypeHZ=copulaTypeHZ, modelType=modelType))



# simulateData(nObservations=nObservationsSimulated, nExchangeRates=nExchangeRatesSimulated, modelType, copulaTypeH, copulaTypeZ, copulaTypeHZ, dataSetName="simulated_data", thetaTrue, MISC_PAR, nCores)

dataSetName <-   paste("simulated_data_nObservations", nObservationsSimulated, "nExchangeRates",  nExchangeRatesSimulated, "modelType", modelType, "copulaTypeH", copulaTypeH, "copulaTypeZ", copulaTypeZ, "copulaTypeHZ", copulaTypeHZ, sep='_')
isSimulatedData <- TRUE
# 
# ## Selecting real data set:
# dataSetName <- "eur_gbp_jpy_2008_to_2013"
# isSimulatedData <- FALSE
# 
# 
# 
if (isSimulatedData) {
  thetaTrue  <- data.matrix(read.table(file.path(MISC_PAR$pathToData, dataSetName, "thetaTrue.dat")))
  latentTrue <- data.matrix(read.table(file.path(MISC_PAR$pathToData, dataSetName, "latentTrue.dat")))
}

## Auxiliary (known) model parameters and data
## ------------------------------------------------------------------------- ##

MODEL_PAR <- getAuxiliaryModelParameters(modelType, copulaTypeH, copulaTypeZ, copulaTypeHZ, MISC_PAR, dataSetName)
nExchangeRates <- MODEL_PAR$nExchangeRates # number of modelled/observed exchange rates
nObservations  <- MODEL_PAR$nObservations # number of observations


## Output for Kostas
## ------------------------------------------------------------------------- ##
# 
# parMat           <- cbind(parametersTrue, thetaTrue, MODEL_PAR$meanHyper, MODEL_PAR$sdHyper)
# colnames(parMat) <- c("Original Parameter", "Transformed parameter", "Prior mean", "Prior std. dev.")
# rownames(parMat) <- names(parametersTrue)
# 
# latentMat           <- t(latentTrue)
# colnames(latentMat) <- c("Simulated H's", "Simulated Z's")
# 
# logExchangeRatesMat        <- t(MODEL_PAR$logExchangeRates)
# logVolatilitiesMat         <- t(MODEL_PAR$logVolatilities)
# initialLogExchangeRatesMat <- t(MODEL_PAR$initialLogExchangeRates)
# initialLogVolatilitiesMat  <- t(MODEL_PAR$initialLogVolatilities)
# 
# parInitialLogVolatilitiesMat           <- cbind(MODEL_PAR$meanInitialLogVolatility, MODEL_PAR$sdInitialLogVolatility)
# colnames(parInitialLogVolatilitiesMat) <- c("Mean of initial log-volatility (i.e. of the distribution denoted nu)", "Std. dev. of initial log-volatility (i.e. of the distribution denoted nu)")
# 
# customName <- paste("nObservations", nObservations, "nExchangeRates", nExchangeRates, "copulaTypeH", getCopulaName(copulaTypeH),"copulaTypeZ", getCopulaName(copulaTypeZ), "copulaTypeHZ", getCopulaName(copulaTypeHZ), sep="_")
# 
# if (modelType == 0) {
#   pathToFilesForKostas <- paste("/home/axel/Dropbox/CopulaSV/data/simulatedDataForComparison/exactlyObservedVolatilities", customName, sep="/")
#   } else if (modelType == 1) {
#   pathToFilesForKostas <- paste("/home/axel/Dropbox/CopulaSV/data/simulatedDataForComparison/noisilyObservedVolatilities", customName, sep="/")
# }
# dir.create(pathToFilesForKostas, showWarnings = FALSE)
# 
# write.csv(x=parMat, file=file.path(pathToFilesForKostas, "parametersAndPriors.csv"))
# write.csv(x=latentMat, file=file.path(pathToFilesForKostas, "latentVariables.csv"))
# write.csv(x=logExchangeRatesMat, file=file.path(pathToFilesForKostas, "observedLogExchangeRates.csv"))
# write.csv(x=logVolatilitiesMat, file=file.path(pathToFilesForKostas, "observedLogVolatilities.csv"))
# write.csv(x=initialLogExchangeRatesMat, file=file.path(pathToFilesForKostas, "initialObservedLogExchangeRates.csv"))
# write.csv(x=initialLogVolatilitiesMat, file=file.path(pathToFilesForKostas, "initialObservedLogVolatilities.csv"))
# write.csv(x=parInitialLogVolatilitiesMat, file=file.path(pathToFilesForKostas, "parametersOfInitialLogVolatilities.csv"))
# 
# 
# rm(parMat)
# rm(latentMat)
# rm(logExchangeRatesMat)
# rm(logVolatilitiesMat)
# rm(initialLogExchangeRatesMat)
# rm(initialLogVolatilitiesMat)
# rm(parInitialLogVolatilitiesMat)
# rm(customName)
# rm(pathToFilesForKostas)

## Auxiliary parameters for the SMC and MCMC algorithms
## ------------------------------------------------------------------------- ##

ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(MODEL_PAR$dimTheta, MODEL_PAR$nObservations)

# If we measure the log-volatility exactly, the posterior distribution of the
# latent variables factorises into a product of the marginal posterior distributions
# for each time step. Therefore, to ensure that the estimator for the marginal likelihood
# factorises in the same way, we need to enforce resampling at every time step.
if (modelType == 0) {
  ALGORITHM_PAR$essResamplingThresholdLower <- 99
}


## Plot the data
## ------------------------------------------------------------------------- ##

if (isSimulatedData) {
op <- par(mfcol=c(2,nExchangeRates+1))
  for (kk in 1:nExchangeRates) {
    plot(c(MODEL_PAR$InitialLogExchangeRates[kk], MODEL_PAR$logExchangeRates[kk,]), type="l", main="S")#, ylim=range(MODEL_PAR$logExchangeRates))
    plot(c(MODEL_PAR$initialLogVolatilities[kk], MODEL_PAR$logVolatilities[kk,]), type="l", main="X")#, ylim=range(MODEL_PAR$logVolatilities))
  }
    plot(latentTrue[1,], type="l", main="H")#, ylim=range(MODEL_PAR$latentTrue[1,]))
    plot(latentTrue[2,], type="l", main="Z")#, ylim=range(MODEL_PAR$latentTrue[2,]))
  par(op)
} else if (!isSimulatedData) {
  op <- par(mfcol=c(2,nExchangeRates))
  for (kk in 1:nExchangeRates) {
    plot(c(MODEL_PAR$InitialLogExchangeRates[kk], MODEL_PAR$logExchangeRates[kk,]), type="l", main="S")#, ylim=range(MODEL_PAR$logExchangeRates))
    plot(c(MODEL_PAR$initialLogVolatilities[kk], MODEL_PAR$logVolatilities[kk,]), type="l", main="X")#, ylim=range(MODEL_PAR$logVolatilities))
  }
  par(op)
}


###############################################################################
##
## Approximating the Marginal (Count-Data) Likelihood
##
###############################################################################
# 
# nThetaValues <- 25
# nSimulations <- 2
# simulateData <- FALSE ###
# 
# if (simulateData == TRUE) {
#   MODEL_PAR$nObservations <- nObservations
# }
# 
# nParticlesLower <- c(50)
# N_PARTICLES_LOWER <- nParticlesLower
# 
# MM <- nSimulations
# KK <- nThetaValues
# NN <- length(N_PARTICLES_LOWER)
# 
# logZ <- array(NA, c(KK, NN, MM))
# 
# # varthetaAux <- c(0, 1/2, 2.5, 1/100, 1/10, 1/2, 1/2)
# # thetaTrue   <- c(rep(varthetaAux, each=nExchangeRates), 1/2)
# 
# if (simulatedData == TRUE) {
# 
#   parTrueAux  <- simulateDataCpp(MODEL_PAR$nObservations, MODEL_PAR$hyperParameters, thetaTrue, nCores)
# 
#   ## Plot the data
#   op <- par(mfcol=c(2,nExchangeRates+1))
#     for (kk in 1:nExchangeRates) {
#       plot(parTrueAux$logExchangeRates[kk,], type="l", main="S")#, ylim=range(parTrueAux$logExchangeRates))
#       plot(parTrueAux$logVolatilities[kk,], type="l", main="X")#, ylim=range(parTrueAux$logVolatilities))
#     }
#     plot(parTrueAux$latentTrue[1,], type="l", main="H")#, ylim=range(parTrueAux$latentTrue[1,]))
#     plot(parTrueAux$latentTrue[2,], type="l", main="Z")#, ylim=range(parTrueAux$latentTrue[2,]))
#   par(op)
# 
# }
# 
# # thetaTrueAlt <- c(-3.24038694, 1.02829161, 0.05153242, 1.04795999, 1.60363194, -1.47496723, 3.34622335, 2.02180357, 6.32475097, -0.29780072, -2.29990749, -0.90167026, 0.32698761, 6.56932172, 2.06929042, -0.91697960, 0.03597698, 2.23593859, -0.76347658, 0.06306005, -0.21990431, -3.48845359)
# 
# # thetaTrueAlt <- c(0, 0, 0, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 0.01,0.01,0.01, 0.32698761, 6.56932172, 2.06929042, -0.91697960, 0.03597698, 2.23593859, -0.76347658, 0.06306005, -0.21990431, -3.48845359)
# # 
# # thetaTrueAlt <- c(
# # 0,0,0, #-3.24038694, 1.02829161, 0.05153242, 
# # 0.5, 0.5, 0.5, #1.04795999, 1.60363194, -1.47496723
# # 2.5, 2.5, 2.5, #3.34622335, 2.02180357, 6.32475097, 
# # 0.01,0.01,0.01, #-0.29780072, -2.29990749, -0.90167026, 
# # 0.1, 0.1, 0.1, #0.32698761, 6.56932172, 2.06929042, ### this seems to make the difference between low and high likelihoods
# # 0.5, 0.5, 0.5, #-0.91697960, 0.03597698, 2.23593859, 
# # 0.5, 0.5, 0.5, #-0.76347658, 0.06306005, -0.21990431, 
# # 0.5 #-3.48845359
# # )
# 
# 
# 
# thetaInit <- c()
# 
# for (kk in 1:KK) {
# 
# 
# #   if (simulateData) {
# #   
# # #   thetaTrue <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
# # #   varthetaAux <- c(0, 1/2, 2.5, 1/100, 1/10, 1/2, 1/2)
# # #   thetaTrue <- c(rep(varthetaAux, each=nExchangeRates), 1/2)
# #   
# # #   parTrueAux <- simulateDataCpp(MODEL_PAR$nObservations, MODEL_PAR$hyperParameters, thetaTrue, nCores)
# # 
# #   
# #     MODEL_PAR$thetaTrue               <- thetaTrue
# #     MODEL_PAR$latentTrue              <- parTrueAux$latentTrue
# #     MODEL_PAR$logExchangeRates        <- parTrueAux$logExchangeRates
# #     MODEL_PAR$logVolatilities         <- parTrueAux$logVolatilities
# #     MODEL_PAR$initialLogExchangeRates <- parTrueAux$initialLogExchangeRates
# #     MODEL_PAR$initialLogVolatilities  <- parTrueAux$initialLogVolatilities
# #   }
# 
#   thetaTrueAlt <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores)
# #  thetaTrueAlt[7:9] <- thetaTrue[7:9]
# #  thetaTrueAlt[13:15] <- thetaTrue[13:15]
# 
#   thetaTrueAlt[7:9]   <- -1
#   thetaTrueAlt[13:15] <- -0.1
#   
#   thetaInit <- cbind(thetaInit, thetaTrueAlt)
#   
# 
# #   thetaInit <- cbind(thetaInit, thetaTrueAlt)
# #   thetaInit <- cbind(thetaInit, thetaTrueAlt)
# 
# 
#   for (mm in 1:MM) {
#     for (nn in 1:NN) {
#       print(paste("kk: ", kk, "nn: ", nn))
#       aux <- runSmcFilterCpp(
#           MODEL_PAR$logExchangeRates, MODEL_PAR$logVolatilities, MODEL_PAR$initialLogExchangeRates, MODEL_PAR$initialLogVolatilities, 
#           MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, N_PARTICLES_LOWER[nn], ALGORITHM_PAR$essResamplingThresholdLower, smcParameters, 
#           thetaInit[,kk], 0, nCores 
#         )
#         
#       logZ[kk,nn,mm] <- aux$logLikelihoodEstimate
#     }
#   }
# }
# 
# op <- par(mfrow=c(KK,1))
# logZ[logZ == -Inf] <- 0
# for (kk in 1:KK) {
# #   boxplot(t(logZ[kk,,]), range=0, xlab="number of particles", ylim=range(logZ), names=nParticlesLower)
#     boxplot(t(logZ[kk,,]), range=0, xlab="number of particles",  names=nParticlesLower)
# }
# par(op)
# 
# 


###############################################################################
##
## PMMH algorithm
##
###############################################################################
# 
# nIterations     <- 5
# nSimulations    <- 1
# nParticlesLower <- 300

# samplePath <- TRUE # should we store one path of latent variables per iteration
# USE_DELAYED_ACCEPTANCE <- c(0) # should delayed acceptance be used?
# ADAPT_PROPOSAL <- c(1) # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
# useAdaptiveProposalScaleFactor1 <- TRUE
# 
# ll <- 1
# MM <- nSimulations # number of independent replicates
# LL <- length(USE_DELAYED_ACCEPTANCE)
# 
# outputTheta   <- array(NA, c(MODEL_PAR$dimTheta, nIterations, LL, MM))
# outputLatentPath <- array(NA, c(MODEL_PAR$dimLatentVariable, MODEL_PAR$nObservationsCount, nIterations, LL, MM))
# outputCpuTime <- matrix(NA, LL, MM)
# outputAcceptanceRate <- matrix(NA, LL, MM)
# mm <- 1
# 
# 
# for (mm in 1:MM) {
# 
#   thetaInit <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
#   
#   for (ll in 1:LL) {
#   
#     print(ll)
#   
#     aux <- runPmmhCpp(
#       MODEL_PAR$logExchangeRates, MODEL_PAR$logVolatilities, MODEL_PAR$initialLogExchangeRates, MODEL_PAR$initialLogVolatilities, 
#       MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nIterations, nParticlesLower, ALGORITHM_PAR$essResamplingThresholdLower, smcParameters, 
#       mcmcParameters, ADAPT_PROPOSAL[ll], useAdaptiveProposalScaleFactor1, ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, thetaInit, ALGORITHM_PAR$burninPercentage, samplePath, nCores 
#     )
#     
#     outputTheta[,,ll,mm] <- matrix(unlist(aux$theta), MODEL_PAR$dimTheta, nIterations)
#     
# #     outputLatentPath[,,,ll,mm] <- array(unlist(aux$latentPath), c(MODEL_PAR$dimLatentVariable, MODEL_PAR$nObservationsCount, nIterations))
#     
#     outputCpuTime[ll,mm] <- unlist(aux$cpuTime)
#     outputAcceptanceRate[ll,mm] <- unlist(aux$acceptanceRate)
#       
#   }
#   
# #   save(
# #     list  = ls(envir = environment(), all.names = TRUE), 
# #     file  = paste(pathToResults, "pmmh_nIterations_", nIterations, "_nSimulations_", nSimulations, "_modelType_", modelType, "_nExchangeRates_", nExchangeRates, sep=''),
# #     envir = environment()
# #   ) 
# #   
#   
# }
# 
# 
# ii <- 4
# plot(outputTheta[ii,,1,1], type='l')


###############################################################################
##                                                                           
## SMC Samplers for Model Selection                                          
##                                                                           
###############################################################################


## ========================================================================= ##
## Estimating the model evidence
## ========================================================================= ##

# load(paste(pathToResults,  "smc_estimating_model_evidence_nParticlesUpper_1000nParticlesLowerr_4000_nSimulations_3_all_models_",sep=''))

nParticlesUpper <- 1000
nParticlesLower <- 1000
nSteps          <- 100
nSimulations    <- 5

useImportanceTempering     <- TRUE
nMetropolisHastingsUpdates <- 1

MODEL_TYPE     <- c(modelType)
COPULA_TYPE_H  <- c(copulaTypeH)
COPULA_TYPE_Z  <- c(copulaTypeZ)
COPULA_TYPE_HZ <- c(copulaTypeHZ)

ALPHA                  <- seq(from=0, to=1, length=nSteps)
USE_ADAPTIVE_TEMPERING <- c(1) 
ADAPT_PROPOSAL         <- c(1) # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
useAdaptiveProposalScaleFactor1 <- TRUE

MM <- nSimulations # number of independent replicates
LL <- length(COPULA_TYPE_H)

outputLogEvidenceEstimate <- array(NA, c(length(MODEL_TYPE), LL, MM))

for (mm in 1:MM) {

  for (ll in 1:LL) {

    for (ii in 1:length(MODEL_TYPE)) {
    
      print(paste("Computing model evidence: ", mm , ", " , ii, sep=''))
    
      ## Auxiliary (known) model parameters:
#       modelType     <- MODEL_TYPE[ii]
#       copulaTypeH   <- COPULA_TYPE_H[ll]
#       copulaTypeZ   <- COPULA_TYPE_Z[ll]
#       copulaTypeHZ  <- COPULA_TYPE_HZ[ll]
      MODEL_PAR     <- getAuxiliaryModelParameters(MODEL_TYPE[ii], COPULA_TYPE_H[ll], COPULA_TYPE_Z[ll], COPULA_TYPE_HZ[ll], MISC_PAR, dataSetName)
      
      ## Auxiliary parameters for the SMC and MCMC algorithms:
      ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(MODEL_PAR$dimTheta, MODEL_PAR$nObservations)
      # If we measure the log-volatility exactly, the posterior distribution of the
      # latent variables factorises into a product of the marginal posterior distributions
      # for each time step. Therefore, to ensure that the estimator for the marginal likelihood
      # factorises in the same way, we need to enforce resampling at every time step.
      if (modelType == 0) {
        ALGORITHM_PAR$essResamplingThresholdLower <- 99
      }

      
      aux <- 
      runSmcSamplerCpp(
        MODEL_PAR$logExchangeRates, MODEL_PAR$logVolatilities, MODEL_PAR$initialLogExchangeRates, MODEL_PAR$initialLogVolatilities, 
        MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, 0, nParticlesUpper, nParticlesLower, ALGORITHM_PAR$nMetropolisHastingsUpdates, ALGORITHM_PAR$nMetropolisHastingsUpdatesFirst, ALGORITHM_PAR$essResamplingThresholdUpper, ALGORITHM_PAR$essResamplingThresholdLower, 
        smcParameters, mcmcParameters, 
        USE_ADAPTIVE_TEMPERING[ll], 
        useImportanceTempering, 
        ADAPT_PROPOSAL[ll], 
        useAdaptiveProposalScaleFactor1,
        0.999, ###ALGORITHM_PAR$cessTarget, 
        ALPHA, ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, nCores
      )
      
      outputLogEvidenceEstimate[ii,ll,mm] <- aux$logEvidenceEstimate

      save(
        list  = ls(envir = environment(), all.names = TRUE), 
        file  = file.path(
          pathToResults, 
          paste(
            "smc_nParticlesUpper", nParticlesUpper, 
            "nParticlesLower", nParticlesLower, 
            "nSimulations", nSimulations, 
            "modelType", modelType, 
            "copulaTypeH", copulaTypeH,
            "copulaTypeZ", copulaTypeZ,
            "copulaTypeHZ", copulaTypeHZ,
            "simulation_run", mm, 
          sep='_')),
        envir = environment()
      ) 
#       rm(aux)
    }
    
  }
}

## ========================================================================= ##
## Plotting parameter estimates from final step of one of the SMC samplers
## ========================================================================= ##

  load(file.path(pathToResults, 
  paste(
  "smc_nParticlesUpper", nParticlesUpper, "nParticlesLower", nParticlesLower,
  "nSimulations", nSimulations, "modelType", modelType, "copulaTypeH", copulaTypeH, "copulaTypeZ", copulaTypeZ, "copulaTypeHZ", copulaTypeHZ, "simulation_run_1", sep='_')))

nSteps <- length(aux$inverseTemperatures)
theta  <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nParticlesUpper, nSteps))

WIDTH      <- 10
HEIGHT     <- 6
# COL_LL     <- c("royalblue4", "red4", "royalblue", "red2")
PAR_LIM    <- cbind(rep(-10, times=MODEL_PAR$dimTheta), rep(10, times=MODEL_PAR$dimTheta))
KDE_LIM    <- rep(5, times=MODEL_PAR$dimTheta)
CEX_LEGEND <- 0.9

plotThetaSmc <- function(title, theta, idx, nSim)
{
  RR <- length(idx)

  pdf(file=file.path(MISC_PAR$pathToFigures, paste(title, ".pdf", sep='')), width=WIDTH, height=HEIGHT)
      
  for (rr in 1:RR) {
  
    plot(density(theta[rr,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]), type='l', col="white", xlab=MODEL_PAR$thetaNames[idx[rr]], ylab="Density", xlim=PAR_LIM[idx[rr],], ylim=c(0, KDE_LIM[idx[rr]]), main='')

    ## Add estimates based on the particles at the last step of the SMC sampler
    for (mm in 1:nSim) {
      
      load(file.path(pathToResults, 
      paste(
      "smc_nParticlesUpper", nParticlesUpper, "nParticlesLower", nParticlesLower,
      "nSimulations", nSimulations, "modelType", modelType, "copulaTypeH", copulaTypeH, "copulaTypeZ", copulaTypeZ, "copulaTypeHZ", copulaTypeHZ, "simulation_run", mm, sep='_')))
      
      if (rr == 1) {print(aux$logEvidenceEstimate)}
      
      nSteps <- length(aux$inverseTemperatures)
      theta  <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nParticlesUpper, nSteps))
    
      lines(density(theta[rr,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]), type='l', col="red", lty=1, main='')
    }
    
    ## Add estimates based on importance tempering
#     lines(density(c(theta[rr,,]), weights=c(aux$importanceTemperingWeights)), type='l', col="red", lty=2, main='')
  
    ## Add prior densities
    grid <- seq(from=min(PAR_LIM[idx[rr],]), to=max(PAR_LIM[idx[rr],]), length=10000)
    lines(grid, dnorm(grid, mean=MODEL_PAR$meanHyper[idx[rr]], sd=MODEL_PAR$sdHyper[idx[rr]]), col="black", lty=2)
    if (isSimulatedData) {
      abline(v=thetaTrue[rr], col="blue")
    }
    legend("topleft", legend=c("SMC sampler", "Prior"), col=c("red", "black"), bty='n', lty=c(1, 2), cex=CEX_LEGEND)
 
  }
  dev.off()
}


plotThetaSmc(paste("smc_theta_modelType", modelType, "nParticlesUpper", nParticlesUpper, "nParticlesLower", nParticlesLower, "nObservations", nObservations, "isSimulatedData", isSimulatedData, sep='_'), theta, 1:MODEL_PAR$dimTheta, nSim=5)

