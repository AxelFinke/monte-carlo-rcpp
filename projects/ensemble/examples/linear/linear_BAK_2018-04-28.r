## Particle vs. Ensemble MCMC in a multivariate linear Gaussian state-space model

########## TODO: allow the use of multinomial resampling!


rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"
exampleName       <- "linear"
projectName       <- "ensemble"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))


## ========================================================================= ##
## MODEL
## ========================================================================= ##

dimX     <- 5 # dimension of the states
#############
dimY     <- dimX # dimension of the observations 
# dimY <- 1
##############
nOffDiagonals <- 1 # number of non-zero off-diagonals on each side of the main diagonal of A
dimTheta <- nOffDiagonals + 3 # length of the parameter vector

# Support of the unknown parameters:
supportMin <- c(-1, -1, 0, 0) # minimum
supportMax <- c( 1,  1, Inf, Inf) # maximum
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

# Inverse-gamma prior on b:
shapeHyperB <- 1 # shape parameter
scaleHyperB <- 0.5 # scale parameter

# Inverse-gamma prior on d:
shapeHyperD <- 1 # shape parameter
scaleHyperD <- 0.5 # scale parameter

# Collecting all the hyper parameters in a vector:
hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)

nObservations <- 10 # number of time steps/observations
thetaTrue     <- c(0.5, 0.2, 1, 1) # "true" parameter values used for generating the data
# thetaTrue     <- c(0,0, 1, 1) # "true" parameter values used for generating the data 

thetaNames    <- c("a0", "a1", "b", "d") # names of the parameters

DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
observations  <- DATA$y # simulated observations

smcParameters <- rep(sqrt(1.0/dimX), times=nObservations); # additional parameters for the SMC algorithms
ensembleOldParameters <- c(rep(0, dimX), 1.0/dimX, 0.5);# additional parameters for the original EHMM algorithms
####################
## The original values (scaling the variance as 1/d)
ensembleNewParameters <- c(sqrt(1.0/dimX), sqrt(1-1.0/dimX), 1.0, 1.0) # additional parameters for the alternative EHMM algorithms: c("scale for random-walk proposals", "correlation parameter for autoregressive updates)
####################

####################
## scaling the variance the only as 1/sqrt(d)
# ensembleNewParameters <- c(sqrt(1.0/dimX), sqrt(1-(1.0/dimX)^(1/2)), 1.0, 1.0); # additional parameters for the alternative EHMM algorithms: c("scale for random-walk proposals", "correlation parameter for autoregressive updates)
#####################
 
rwmhSd = sqrt(rep(1, times=dimTheta)/(100*dimTheta*dimX*nObservations))

initialiseStatesFromStationarity <- FALSE  # should the CSMC-type algorithms initialise the state sequence from stationarity?
storeParentIndices <- FALSE                # should the CSMC-type algorithms all their parent indices?
storeParticleIndices <- FALSE              # should the CSMC-type algorithms store the indices of the (input and output) reference particle lineage?
 
## ========================================================================= ##
## Approximating the marginal likelihood
## ========================================================================= ##

nSteps <- nObservations
nParticlesLinear <- 500 ######## 1000
nParticlesQuadratic <- 10 ############## 100
essResamplingThreshold <- 1.0

nSimulations <- 4 ############
MM <- nSimulations # number of independent replicates

PROP  <- c(0,0,0,0,2,2,2,2,2,3) # lower-level proposals (PROP must have the same length as LOWER)
LOWER <- c(0,2,2,2,0,2,2,2,1,1) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
LOCAL <- c(0,0,1,2,0,0,1,2,0,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach
LL <- length(LOWER) # number of lower-level sampler configurations to test

lowerNames <- c(
"BPF + systematic", "BPF + multinomial", "MCMC BPF (random-walk)", "MCMC BPF (autoregressive)", 
"FA-APF + systematic", "FA-APF + multinomial", "MCMC FA-APF (random-walk)", "MCMC FA-APF (autoregressive)", 
"original (i.e. O(N^2)) EHMM I", 
"original (i.e. O(N^2)) EHMM II")


# logLikeTrue <- evaluateLogMarginalLikelihoodCpp(hyperparameters, thetaTrue, observations, nCores)
centredLogLikeHat <- matrix(NA, LL, MM)
# ess <- array(NA, c(nSteps, LL, MM))
# acceptanceRates <- array(NA, c(nSteps, LL, MM))

for (mm in 1:MM) {

  DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
  observations  <- DATA$y # simulated observations
  logLikeTrue   <- evaluateLogMarginalLikelihoodCpp(hyperparameters, thetaTrue, observations, nCores)

  for (ll in 1:LL) {
  
    print(ll)
    
    if (LOWER[ll] == 1) {
      nParticlesAux <- nParticlesQuadratic
    } else {
      nParticlesAux <- nParticlesLinear
    }
    centredLogLikeHat[ll, mm] <- approximateMarginalLikelihoodCpp(LOWER[ll], dimTheta, thetaTrue, hyperparameters, support, observations, PROP[ll], nSteps, nParticlesAux, essResamplingThreshold, smcParameters, ensembleOldParameters, ensembleNewParameters, LOCAL[ll], nCores) - logLikeTrue ### NOTE: subtracting the true loglikelihood here
    
#     centredLogLikeHat[ll, mm] <- aux$logLikeHat - logLikeTrue ### NOTE: subtracting the true loglikelihood here
#     ess[,ll, mm] <- aux$ess
#     acceptanceRates[ll, mm] <- aux$acceptanceRates
 
  }
}

save(
  list  = ls(envir = environment(), all.names = TRUE), 
  file  = paste(pathToResults, "likelihood_estimates---nObservations_", nObservations, "_nSimulations_", nSimulations, "_nParticles_", nParticlesLinear, "_dimension_", dimX, sep=''),
  envir = environment()
) 

## Plot the results
#load(file=paste(pathToOutput, "likelihood_estimates---nObservations_25_nSimulations_10000_nParticles_1000_dimension_2", sep=''))
#load(file=paste(pathToOutput, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_5", sep=''))


pdf(file=paste(pathToFigures, "likelihood_estimates.pdf", sep=''), width=8, height=10)   

  op <- par(mar=c(20, 3, 2, 2)+1) ## c(bottom, left, top, right)
  boxplot(exp(t(centredLogLikeHat))-1, 
    ylab="hat{Z}/Z - 1", ylim=c(-1,1),
    range=0,
    las=2, 
    names=lowerNames,
    main="Relative estimates of marginal likelihood (denoted Z)"
    )
  
  mtext(paste("N =", nParticlesLinear, "(for O(N) methods); N =", nParticlesQuadratic, "(for O(N^2) methods); T =", nObservations, "; d =", dimX), side = 3, line = 0, outer = FALSE)
  abline(h=0, col="red")
  par(op)
  
dev.off()


## ========================================================================= ##
## MCMC algorithms
## ========================================================================= ##

nIterations <- 1000000
kern = 0
useGradients <- FALSE
fixedLagSmoothingOrder <- 0
nSteps <- nObservations

essResamplingThreshold <- 1.0
backwardSamplingType <- 2 ##########################
useNonCentredParametrisation <- FALSE
nonCentringProbability <- 0
nSimulations <- 5


# UPPER <- c(1) # type of "upper-level" MCMC algorithm to use (0: PMMH; 1: particle Gibbs)
# PROP  <- c(0,2,0,2,0,2,0,2,0,1,0) # lower-level proposals (PROP must have the same length as LOWER)
# LOWER <- c(0,0,2,2,2,2,2,2,1,1,3) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
# LOCAL <- c(0,0,0,0,1,1,2,2,0,0,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach

## Only testing MCMC BPFs:
nParticlesLinearMetropolis      <- 1000
nParticlesQuadraticMetropolis   <- ceiling(sqrt(nParticlesLinearMetropolis))
############ nParticlesLinearGibbs           <- 100
nParticlesLinearGibbs           <- 100
nParticlesQuadraticGibbs        <- ceiling(sqrt(nParticlesLinearGibbs))
nThetaUpdates                   <- 100

UPPER <- c(1) # type of "upper-level" MCMC algorithm to use (0: PMMH; 1: particle Gibbs)

################################
## Using Hilbert-sort
# PROP  <- c(0,0,0,0,0,2,2,2,2,2,0) # lower-level proposals (PROP must have the same length as LOWER)
# LOWER <- c(2,2,2,2,2,2,2,2,2,2,3) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
# LOCAL <- c(0,1,2,3,4,0,1,2,3,4,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach
################################

################################
## not using Hilbert-sort
PROP  <- c(0,0,0,2,2,2,0) # lower-level proposals (PROP must have the same length as LOWER)
LOWER <- c(2,2,2,2,2,2,3) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
LOCAL <- c(0,1,2,0,1,2,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach
################################

################################
## not using Hilbert-sort but using the original EHMM approach
# PROP  <- c(0,0,0,2,2,2,0,1,0) # lower-level proposals (PROP must have the same length as LOWER)
# LOWER <- c(2,2,2,2,2,2,1,1,3) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
# LOCAL <- c(0,1,2,0,1,2,0,0,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach
################################


# PROP  <- c(0,0,0) # lower-level proposals (PROP must have the same length as LOWER)
# LOWER <- c(3,2,3) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
# LOCAL <- c(0,1,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach

ESTIMATE_THETA <- TRUE ###############

## Testing all:
# nParticlesLinearMetropolis      <- ceiling(nObservations*exp(dimX))
# nParticlesQuadraticMetropolis   <- ceiling(sqrt(nParticlesLinearMetropolis))
# nParticlesLinearGibbs           <- 10*exp(dimX)
# nParticlesQuadraticGibbs        <- ceiling(sqrt(nParticlesLinearGibbs))
# nThetaUpdates                   <- 50
# UPPER <- c(0,1) # type of "upper-level" MCMC algorithm to use (0: PMMH; 1: particle Gibbs)
# PROP  <- c(0,0,0,0,0,0,2,2,2,2,2,2,0,1,0) # lower-level proposals (PROP must have the same length as LOWER)
# LOWER <- c(0,2,2,2,2,2,0,2,2,2,2,2,1,1,3) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
# LOCAL <- c(0,0,1,2,3,4,0,0,1,2,3,4,0,0,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach
#ESTIMATE_THETA <- TRUE ###############

TIMES      <- c(0, ceiling(nObservations/4), ceiling(nObservations/2), nObservations-1) # time steps at which components of the latent states are to be stored
COMPONENTS <- rep(1, times=4) # components which are to be stored

MM <- nSimulations # number of independent replicates
LL <- length(LOWER) # number of lower-level sampler configurations to test
UU <- length(UPPER) # number of upper-level MCMC algorithms (here: PP = 2 because we test (pseudo-)marginal and (pseudo-)Gibbs kernels)




outputTheta  <- array(NA, c(dimTheta, nIterations, LL, UU, MM))
outputStates <- array(NA, c(length(TIMES), nIterations, LL, UU, MM))
outputEss             <- array(NA, c(nObservations, LL, UU, MM))
outputAcceptanceRates <- array(NA, c(nObservations, LL, UU, MM))

for (mm in 1:MM) {
  for (uu in 1:UU) {
    for (ll in 1:LL) {
    
      if (LOWER[ll] == 1) {
        if (UPPER[uu] == 0) {
          nParticlesAux <- nParticlesQuadraticMetropolis
        } else {
          nParticlesAux <- nParticlesQuadraticGibbs
        }
      } else {
        if (UPPER[uu] == 0) {
          nParticlesAux <- nParticlesLinearMetropolis
        } else {
          nParticlesAux <- nParticlesLinearGibbs
        }
      }
    
      aux <- runMcmcCpp(UPPER[uu], LOWER[ll], dimTheta, hyperparameters, support, observations, nIterations, kern, useGradients, rwmhSd, fixedLagSmoothingOrder, PROP[ll], nThetaUpdates, nSteps, nParticlesAux, essResamplingThreshold, backwardSamplingType, useNonCentredParametrisation, nonCentringProbability, smcParameters, ensembleOldParameters, ensembleNewParameters, LOCAL[ll], ESTIMATE_THETA, thetaTrue, TIMES, COMPONENTS,
      initialiseStatesFromStationarity, storeParentIndices, storeParticleIndices, backwardSamplingType, nCores)
      
      outputTheta[,,ll,uu,mm]  <- matrix(unlist(aux$theta), dimTheta, nIterations)
      outputStates[,,ll,uu,mm] <- matrix(unlist(aux$states), length(TIMES), nIterations)
      outputEss[,ll,uu,mm]             <- unlist(aux$ess)
      outputAcceptanceRates[,ll,uu,mm] <- unlist(aux$acceptanceRates)
      
    }
  }
  
  save(
    list  = ls(envir = environment(), all.names = TRUE), 
    file  = paste(pathToResults, "nIterations_", nIterations, "_nObservations_", nObservations, "_nSimulations_", nSimulations, "_nParticles_", nParticlesLinearGibbs, "_dimension_", dimX, "_backwardSampling_", backwardSamplingType, "_UPPER1_", UPPER[1], "_ESTIMATE_THETA_" , ESTIMATE_THETA, sep=''),
    envir = environment()
  ) 
}


## ========================================================================= ##
## SIMULATION STUDY OF THE GIBBS SAMPLERS IN VARIOUS MODEL DIMENSIONS
## ========================================================================= ##
# 
# 
# library(coda)
# 
# nIterations <- 20000
# nSimulations <- 5
# DIMS <- c(25,50,75,100) # model dimensions to test
# # dimY <- 1 ###############
# # PARTICLES <- c(10,20,30) # number of particles
# nParticles <- 25
# ALPHA <- (0:15)/10
# # ALPHA <- c(0.2,0.4,0.6,0.8,1.0,1.2,1.4)
# ALPHA_AUX <- c(ALPHA, ALPHA, 1,1)
# CONSTANT <- c(1,2,3)
# 
# 
# burnin <- ceiling(nIterations*0.1)
# kern = 0
# useGradients <- FALSE
# fixedLagSmoothingOrder <- 0
# nSteps <- nObservations
# 
# essResamplingThreshold <- 1.0
# backwardSamplingType <- 2 ##########################
# useNonCentredParametrisation <- FALSE
# nonCentringProbability <- 0
# nThetaUpdates<- 100
# 
# PROP  <- c(rep(2, times=2*length(ALPHA)),0,2) # lower-level proposals (PROP must have the same length as LOWER)
# LOWER <- c(rep(2, times=2*length(ALPHA)),2,2) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
# LOCAL <- c(rep(1, times=length(ALPHA)),rep(2, times=length(ALPHA)),0,0)  # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach
# 
# TIMES      <- c(0, ceiling(nObservations/4), ceiling(nObservations/2), nObservations-1) # time steps at which components of the latent states are to be stored
# COMPONENTS <- rep(1, times=4) # components which are to be stored
# 
# MM <- nSimulations # number of independent replicates
# LL <- length(LOWER) # number of lower-level sampler configurations to test
# 
# ESTIMATE_THETA <- FALSE ###############
# 
# 
# 
# ESS <- array(NA, c(length(DIMS), LL, length(CONSTANT), length(TIMES), MM))
# ESJD <- array(NA, c(length(DIMS), LL, length(CONSTANT), length(TIMES), MM))
# 
# 
# for (mm in 1:MM) {
#   for (cc in 1:length(CONSTANT)) {
#     for (dd in 1:length(DIMS)) {
#     
#       dimX <- DIMS[dd]
#       dimY <- dimX
#       
# #       if (cc == 1) {
# #         dimY <- dimX
# #       } else if (cc == 2) {
# #         dimY <- 1
# #       }
#       
#       hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)
#       DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
#       observations  <- DATA$y # simulated observations
#     
#       for (ll in 1:LL) {
#       
#         alpha <- ALPHA_AUX[ll]
#         ensembleNewParameters <- c(sqrt((CONSTANT[cc]/dimX)^(alpha)), sqrt(1-(CONSTANT[cc]/dimX)^alpha), 1.0, 1.0)
# 
#         aux <- runMcmcCpp(1, LOWER[ll], dimTheta, hyperparameters, support, observations, nIterations, kern, useGradients, rwmhSd, fixedLagSmoothingOrder, PROP[ll], nThetaUpdates, nSteps, nParticles, essResamplingThreshold, backwardSamplingType, useNonCentredParametrisation, nonCentringProbability, smcParameters, ensembleOldParameters, ensembleNewParameters, LOCAL[ll], ESTIMATE_THETA, thetaTrue, TIMES, COMPONENTS, initialiseStatesFromStationarity, storeParentIndices, storeParticleIndices, backwardSamplingType, nCores)
#         
#         xx <- matrix(unlist(aux$states), length(TIMES), nIterations)
#         ESS[dd,ll,cc,,mm] <- effectiveSize(mcmc(t(xx[,(burnin+1):nIterations])))
#         for (ii in 1:length(TIMES)) {
#           ESJD[dd,ll,cc,ii,mm] <- mean(diff(xx[ii,(burnin+1):nIterations])^2)
#         }
#         
# 
#       }
#     }
#   }
#   save(
#     list  = ls(envir = environment(), all.names = TRUE), 
#     file  = paste(pathToResults, "nIterations_", nIterations, "_nObservations_", nObservations, "_nSimulations_", nSimulations, "_nParticles_", nParticles, "_maxDimension_", max(DIMS), "_backwardSampling_", backwardSamplingType, "_ESTIMATE_THETA_" , ESTIMATE_THETA, sep=''),
#     envir = environment()
#   )
# }
# 
# 
# 

## ========================================================================= ##
## PLOT RESULTS
## ========================================================================= ##

# source(file=paste(pathToOutputBase, "ensemble/generic_functions.r", sep=''))
# setwd(paste(pathToOutputBase, "ensemble/examples/", exampleName, "/", sep=''))

# 
# # MCMC algorithms:
# inputPathPlot <- paste(pathToOutput, "nIterations_10000_nObservations_10_nSimulations_3_nParticles_25_dimension_25_backwardSampling_2_UPPER1_1", sep='')
# yLim <- matrix(c(0,1.5,0,1.5,0,3,0,3), 2, dimTheta)
# yLimBoxplot <- matrix(c(0,1.5,0,1.5,0,3,0,3), 2, dimTheta)
# 
# lowerNames <- c( 
# "PF", "MCMC BPF Ia", "MCMC BPF Ib", "MCMC BPF IIa", "MCMC BPF IIb", 
# "FA-APF", "MCMC FA-APF Ia", "MCMC FA-APF Ib", "MCMC FA-APF IIa", "MCMC FA-APF IIb",
# "idealised")
# 
# 
# 
# load(inputPathPlot)
# 
# LOWER_TITLE_AUX <- matrix(c( 
# "PF", "MCMC BPF Ia", "MCMC BPF Ib", "MCMC BPF IIa", "MCMC BPF IIb",
# "FA-APF", "MCMC FA-APF Ia", "MCMC FA-APF Ib", "MCMC FA-APF IIa", "MCMC FA-APF IIb"), 5, 2)
# LOWER_AUX <- matrix(c(1:5,6:10), 5, 2)
# LTY_AUX <- matrix(c(1,1,2,1,2,1,1,2,1,2), 5, 2)
# COL_AUX <- matrix(c("blue", "magenta3", "magenta3", "forestgreen", "forestgreen", "blue", "magenta3", "magenta3", "forestgreen", "forestgreen"), 5 , 2)
# 
# 
# par(mfrow=c(2,1), mar=c(1,1,0.2,0.2))
# for (jj in 1:2) {
#   plot(outputEss[,ll,1,1], col="white", type='l', ylim=c(0,1))
#   legend("topright",  legend=LOWER_TITLE_AUX[,jj], col=COL_AUX[,jj], lty=LTY_AUX[,jj], bty='n')
#   for (ll in 1:5) {
#     lines(outputEss[,(5*(jj-1))+ll,1,1], col=COL_AUX[ll,jj], type='l', lty=LTY_AUX[ll,jj], ylim=c(0,1))
#     if (mm > 1) {
#       for (m in 2:mm) {
#         lines(outputEss[,(5*(jj-1))+ll,1,m], col=COL_AUX[ll,jj], lty=LTY_AUX[ll,jj], type='l')
#       }
#     }
#   }
# }
# 
# 
# 
# par(mfrow=c(2,1), mar=c(1,1,0.2,0.2))
# for (jj in 1:2) {
#   plot(outputEss[,ll,1,1], col="white", type='l', ylim=c(0,1))
#   legend("topright",  legend=LOWER_TITLE_AUX[,jj], col=COL_AUX[,jj], lty=LTY_AUX[,jj], bty='n')
#   for (ll in 1:5) {
#     lines(outputAcceptanceRates[,(5*(jj-1))+ll,1,1], col=COL_AUX[ll,jj], type='l', lty=LTY_AUX[ll,jj], ylim=c(0,1))
#     if (mm > 1) {
#       for (m in 2:mm) {
#         lines(outputAcceptanceRates[,(5*(jj-1))+ll,1,m], col=COL_AUX[ll,jj], lty=LTY_AUX[ll,jj], type='l')
#       }
#     }
#   }
# }
# 
# upperNames = "Gibbs"
# 
# # SMC algorithms:
# plotResults(
#   inputName = inputPathPlot,
#   outputName = paste(pathToOutput, "linearGaussianSSM", sep=''),
#   thetaTrue = thetaTrue,
#   dimTheta = dimTheta,
#   yLim = yLim,
#   yLimBoxplot = yLimBoxplot,
#   yLabel = thetaNames,
#   lowerTitle = lowerNames,
#   upperTitle = upperNames,
#   burnin = 10000
# )



##############################################################################
## Graphics for the paper
##############################################################################

library(tikzDevice)
options( 
  tikzDocumentDeclaration = c(
    "\\documentclass[12pt]{beamer}",
    "\\usepackage{amssymb,amsmath,graphicx,mathtools,mathdots,stmaryrd}",
    "\\usepackage{tikz}" 
  )
)

colPlot <- "magenta3"
colPlotAlternate <- "forestgreen"
outputPathPlot <- "/home/axel/Dropbox/ensemble_interpretation/figures/"

cex.legend <- 0.8

##############################################################################
## Dependence of proposal scale on dimension
##############################################################################
# 
# library(coda)
# inputPathPlot <- paste(pathToOutput, "nIterations_20000_nObservations_10_nSimulations_5_nParticles_25_maxDimension_100_backwardSampling_2_ESTIMATE_THETA_FALSE", sep='')
# load(inputPathPlot)
# 
# ##############################
# ## not using the MCMC-PF
# LOWER_TITLE_AUX <- c("MCMC FA-APF I","MCMC FA-APF II","PF", "FA-APF")
# LOWER_AUX <- c(1:11)
# LTY_AUX <- c(1,1,2,1)
# COL_AUX <- c( "blue", "forestgreen", "magenta3", "magenta3")
# 
# LL <- length(COL_AUX)
# ##############################
# 
# plotMar <- c(0.7, 0.6, 0.9, 1) + 0.1
# plotOma <- c(2.85,3,0.5,0.5)
# 
# NAMES <- c("ESS", "ESJD")
# 
# for (jj in 1:length(NAMES)) {
# 
#   if (jj == 1) {
#     XX <- ESS
#     YLIM <- c(0,ceiling(nIterations/3))
#   } else {
#     XX <- ESJD
#     YLIM <- c(0,0.5)
#   }
# 
#   for (ii in 1:length(TIMES)) {
#     tikz(file=paste(outputPathPlot, NAMES[jj], "_scaling_in_d_TIME_", ii, ".tex", sep=''), width=6, height=5)
# 
#     op <- par(mfrow=c(length(CONSTANT),length(DIMS)), mar=plotMar, oma=plotOma)
# 
#     for (cc in 1:length(CONSTANT)) {
#     
#       for (dd in 1:length(DIMS)) {
#       
#         plot(ALPHA, 1:length(ALPHA), col="white", ylim=YLIM, xlab="", ylab="", 
#             main="", 
#             xaxt='n', yaxt='n', yaxs='i', xaxs='i')
#             
#         mtext(paste("$d = ", DIMS[dd], "; \\alpha = ", CONSTANT[cc], "$", sep=''), side = 3, line = 0, outer = FALSE)
#         
# #         abline(v=1.0, col="black", lty=2)
#             
#         if (cc == length(CONSTANT)) {
#           axis(side=1, at=ALPHA)
#         }
#         if (dd == 1) {
#           axis(side=2, at=YLIM, las=2)
#         }
#         
#         if (cc == ceiling(length(CONSTANT)/2) && dd == 1) {
#           mtext("$\\beta$", side = 1, line = 2, outer = TRUE)
#         }
#         
#         if (cc == length(CONSTANT) && dd == ceiling(length(DIMS)/2)) {
#           mtext(NAMES[jj], side = 2, line = 2, outer = TRUE)
#         }
#         
#         if (cc == 1 && dd == length(DIMS)){
#           legend("topright", legend=LOWER_TITLE_AUX, col=COL_AUX, lty=LTY_AUX, 
#           bty='n', 
#           cex=cex.legend)
#         }
#         
#         for (ll in 1:2) {
#           if (ll == 1) {
#             idx <- 1:length(ALPHA)
#           } else if (ll == 2) {
#             idx <- (length(ALPHA)+1):(2*length(ALPHA))
#           }
#           if (mm > 1) {
#             lines(ALPHA, apply(XX[dd,idx,cc,ii,1:mm],1,mean), type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
#           } else {
#             lines(ALPHA, XX[dd,idx,cc,ii,1], type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
#           }
#         }
#         
#         for (ll in 3:4) {
#           if (ll == 3) {
#             idx <- 2*length(ALPHA)+1
#           } else if (ll == 4) {
#             idx <- 2*length(ALPHA)+2
#           }
#           if (mm > 1) {
#             abline(h=mean(XX[dd,idx,cc,ii,1:mm]), col=COL_AUX[ll], lty=LTY_AUX[ll])
#           } else {
#             abline(h=XX[dd,idx,cc,ii,1], col=COL_AUX[ll], lty=LTY_AUX[ll])
#           }
#         }
#         
#         
#         
#       }
#       
#   
#       
#     }
# 
#     par(op)
#     dev.off()
#   }
# }

##############################################################################
## Marginal-ikelihood estimates
##############################################################################



for (ii in 1:4) {

  if (ii == 1) {
    inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_2")
  } else if (ii == 2) {
    inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_5")
  } else if (ii == 3) {
    inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_10")
  } else if (ii == 4) {
    inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_25")
  }
  load(inputPathPlot)

  lowerNames <- c(
  "BPF", "FA-APF", "MCMC BPF I", "MCMC BPF II", 
  "MCMC FA-APF I", "MCMC FA-APF II", "original EHMM I", "original EHMM II")
  
  lowerPlot <- c(2,8, 3:4, 9:10, 13:14)

  tikz(paste(outputPathPlot, "likelihood_estimates_in_dimension_", dimX, ".tex", sep=''),  width=2.2, height=3)

    op <- par(mar=c(7, 3.5, 0, 0)+1) ## c(bottom, left, top, right)
    
    if (ii == 1) {
      boxplot(exp(t(centredLogLikeHat[lowerPlot,])), 
        ylab="$\\hat{p}_\\theta(y_{1:T}) / p_\\theta(y_{1:T}) $", ylim=c(0,2),
        range=0,
        las=2, 
        names=lowerNames,
        main="",
        yaxt='n'
      )
      axis(side=2, at=c(0,1,2), labels=c(0,1,2), las=2)
    } else {
      boxplot(exp(t(centredLogLikeHat[lowerPlot,])), 
        ylab="", ylim=c(0,2),
        range=0,
        las=2, 
        names=lowerNames,
        main="",
        yaxt="n"
      )
    }
    abline(h=1.0, col=colPlot)
    par(op)

  dev.off()
}


##############################################################################
## Gibbs Sampling-type algorithms
##############################################################################

inputPathPlot <- file.path(pathToResults, "nIterations_5e+05_nObservations_10_nSimulations_3_nParticles_100_dimension_100_backwardSampling_2_UPPER1_1_ESTIMATE_THETA_FALSE")

load(inputPathPlot)

yLim <- matrix(c(0,1.5,0,1.5,0,3,0,3), 2, dimTheta)
yLimBoxplot <- matrix(c(0,1.5,0,1.5,0,3,0,3), 2, dimTheta)

colPlot   <- "magenta3"
widthPlot <- 2.4
heightPlot <- 2.0
colTrue   <- "black"
mycol     <- as.numeric(col2rgb(colPlot))/256
marPlot   <- c(2, 2, 0.1, 0.1) ## c(bottom, left, top, right)
omaPlot   <- c(0,0,0,0)+0.5 ## c(bottom, left, top, right)
padLeft   <- 2.5
padBottom <- 1.5
alphaPlot <- c(0.1, 0.15, 0.2, 0.25)
quantPlot <- matrix(c(0,1,0.05,0.95,0.1,0.9,0.25,0.75), 2, length(alphaPlot))
lag.max.theta <- 500
####### lag.max.states <- 200

lag.max.states <- 100
###########

############ burnin <- ceiling(nIterations*0.1) ##############
burnin <- ceiling(nIterations*0.2)

upperTitle <- c("mh", "gibbs")

##############################
## Using the Hilbert-sort algorithms
# LOWER_TITLE_AUX <- c(
# "PF", "MCMC BPF Ia", "MCMC BPF Ib", "MCMC BPF IIa", "MCMC BPF IIb",
# "FA-APF", "MCMC FA-APF Ia", "MCMC FA-APF Ib", "MCMC FA-APF IIa", "MCMC FA-APF IIb","Idealised"
# )
# LOWER_AUX <- c(1:11)
# LTY_AUX <- c(2,2,2,2,2,1,1,1,1,1,1)
# COL_AUX <- c("red", "blue", "yellow", "forestgreen","magenta3", "red", "blue", "yellow", "forestgreen","magenta3", "black")
##############################

##############################
## not using the Hilbert-sort algorithms
LOWER_TITLE_AUX <- c("BPF", "MCMC BPF I", "MCMC BPF II", "FA-APF", "MCMC FA-APF I", "MCMC FA-APF II", "Idealised")
LOWER_AUX <- c(1:3,4:6,7)
LTY_AUX <- c(2,2,2,1,1,1,1)
COL_AUX <- c("magenta3", "blue", "forestgreen", "magenta3", "blue", "forestgreen", "black")
##############################

##############################
## not using the Hilbert-sort algorithms but using the original EHMM
# LOWER_TITLE_AUX <- c("PF", "MCMC BPF I", "MCMC BPF II", "FA-APF", "MCMC FA-APF I", "MCMC FA-APF II", "orig. EHMM I", "orig. EHMM II", "Idealised")
# LOWER_AUX <- 1:9
# LTY_AUX <- c(2,2,2,1,1,1,1,2,1)
# COL_AUX <- c("magenta3", "blue", "forestgreen", "magenta3", "blue", "forestgreen", "gray", "gray", "black")
##############################


### ACF of states:

for (uu in 1:length(UPPER)) {    
  for (kk in 1:length(TIMES)) {
    
    tikz(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_acf_of_states_TIME_", kk, "_dimX_", dimX, ".tex", sep=''), width=widthPlot, height=heightPlot)
    op <- par(oma=omaPlot, mar=marPlot)
      
    plot(0:lag.max.states, rep(1, times=lag.max.states+1), type='l', col="white",
    xlim=c(0,lag.max.states), ylim=c(0,1.05),
    ylab="", xlab="",
    yaxs='i', xaxs='i', xaxt='n', yaxt='n')
  
    for (ll in 1:length(LOWER_AUX)) {
    
      if (mm == 1) {
        X <- outputStates[kk,(burnin+1):nIterations,LOWER_AUX[ll],uu,1]
      } else {
        X <- apply(outputStates[kk,(burnin+1):nIterations,LOWER_AUX[ll],uu,1:mm], 1, mean)
      }
      ACF <- as.numeric(acf(X, lag.max=lag.max.states, plot=FALSE)$acf)      
      lines(0:lag.max.states, ACF, type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
    }
    axis(side=2, at=c(0,1), labels=c(0,1), las=2)
    axis(side=1, at=c(0,lag.max.states))
    mtext("Lag", side = 1, line = 1, outer = FALSE)
    mtext("Autocorrelation", side = 2, line = 1, outer = FALSE)
    
    if (kk == 1){
      legend("topright", legend=LOWER_TITLE_AUX, col=COL_AUX, lty=LTY_AUX, 
      bty='n', 
      cex=cex.legend)
    }
    
    
    par(op)
    dev.off()
  }
}




### ESS:

for (uu in 1:length(UPPER)) {    

  tikz(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_ess_dimX_", dimX, ".tex", sep=''), width=5.5, height=1.5) 
  op <- par(oma=omaPlot, mar=marPlot + c(0,1,0,0))
  
  plot(1:nObservations, rep(1, times=nObservations), type='l', col="white", xlim=c(1,nObservations), ylim=c(0,1.05), ylab="", xlab="", yaxs='i', xaxs='i', xaxt='n', yaxt='n')

  for (ll in 1:length(LOWER_AUX)) {
  
    if (mm == 1) {
      XMean <- outputEss[,LOWER_AUX[ll],uu,1]
    } else {
      XMean <- apply(outputEss[,LOWER_AUX[ll],uu,1:mm], 1, mean)
    }

    lines(1:nObservations, XMean, type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
  }

  axis(side=2, at=c(0,1), labels=c(0,1), las=2)
  axis(side=1, at=c(1,nObservations))
  mtext("Time", side = 1, line = 1, outer = FALSE)
  mtext("ESS", side = 2, line = 2, outer = FALSE)
 
  par(op)
  dev.off()
}

### Acceptance rates:

# outputAcceptanceRates[,c(1,4),,] <- 1

for (uu in 1:length(UPPER)) {    

  tikz(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_acceptance_rates_dimX_", dimX, ".tex", sep=''), width=5.5, height=1.5) 
  op <- par(oma=omaPlot, mar=marPlot + c(0,1,0,0))
  
  plot(1:nObservations, rep(1, times=nObservations), type='l', col="white", xlim=c(1,nObservations), ylim=c(0,1.05), ylab="", xlab="", yaxs='i', xaxs='i', xaxt='n', yaxt='n')

  for (ll in 1:length(LOWER_AUX)) {
  
    if (mm == 1) {
      XMean <- outputAcceptanceRates[,LOWER_AUX[ll],uu,1]
    } else {
      XMean <- apply(outputAcceptanceRates[,LOWER_AUX[ll],uu,1:mm], 1, mean)
    }
    if (UPPER[uu] == 1) {
      XMean <- XMean*(nParticlesLinearGibbs)/(nParticlesLinearGibbs-1)
    }
    
    lines(1:nObservations, XMean, type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
  }

  axis(side=2, at=c(0,1), labels=c(0,1), las=2)
  axis(side=1, at=c(1,nObservations))
  mtext("Time", side = 1, line = 1, outer = FALSE)
  mtext("Acceptance rate", side = 2, line = 2, outer = FALSE)
 
  par(op)
  dev.off()
}

### KDE of the states:

for (uu in 1:length(UPPER)) {
  for (kk in 1:dimTheta) {
  
    if (dimX == 100 && UPPER[uu] == 1) {
    ### ESTIMATE_THETA = TRUE:
#       DENSITY_LIM_STATES <- c(7,14,4,7)
#       LB_STATES <- c(-1,0,0,0)
#       UB_STATES <- c(1,1,2,2)
### ESTIMATE_THETA == FALSE:
      DENSITY_LIM_STATES <- c(0.7,1,1,0.6)
      LB_STATES <- c(-3,-3,-3,-2)
      UB_STATES <- c(3,3,3,4)
    } else {
      DENSITY_LIM_STATES <- c(3,3,3,3)
      LB_STATES <- c(-1,0,0,0)
      UB_STATES <- c(1,1,2,2)
    }

    
    tikz(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_kde_of_states_TIME_", kk, "_dimX_", dimX, ".tex", sep=''), width=widthPlot, height=heightPlot)
  
    op <- par(oma=omaPlot, mar=marPlot)
    
    plot(0:1, 0:1, type='l', col="white",
         xlim=c(LB_STATES[kk], UB_STATES[kk]), ylim=c(0,DENSITY_LIM_STATES[kk]), ylab="", xlab="", yaxs='i', xaxs='i', xaxt='n', yaxt='n')
  
    
    for (m in 1:mm) {
    
      for (ll in 1:length(LOWER_AUX)) {
        if (ll > 1) { ############
          X <- outputStates[kk,(burnin+1):nIterations,LOWER_AUX[ll],uu,m]
          lines(density(X), type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
        } ###############
      }
    } 
    
    axis(side=1, at=c(LB_STATES[kk], UB_STATES[kk]), labels=c(LB_STATES[kk], UB_STATES[kk]))
    axis(side=2, at=c(0,DENSITY_LIM_STATES[kk]), labels=c(0,DENSITY_LIM_STATES[kk]), las=2)
    mtext("Value", side = 1, line = 1, outer = FALSE)
    mtext("Density", side = 2, line = 1, outer = FALSE)
    
#     if (kk == 1){
#       legend("topleft", legend=LOWER_TITLE_AUX, col=COL_AUX, lty=LTY_AUX, 
#       bty='n', 
#       cex=cex.legend)
#     }
    
    
    par(op)
    dev.off()
  }
}

###############################################################################
## Parameter estimation
###############################################################################

# PMMH with dimension 25
# inputPathPlot <- file.path(pathToResults, "nIterations_1e+06_nObservations_10_nSimulations_5_nParticles_100_dimension_25_backwardSampling_2_UPPER1_0_ESTIMATE_THETA_TRUE")

# # Particle Gibbs with dimension 100
# inputPathPlot <- file.path(pathToResults, "nIterations_1e+06_nObservations_10_nSimulations_5_nParticles_100_dimension_100_backwardSampling_2_UPPER1_1_ESTIMATE_THETA_TRUE")

# Particle Gibbs with dimension 25
inputPathPlot <- file.path(pathToResults, "nIterations_1e+06_nObservations_10_nSimulations_5_nParticles_100_dimension_25_backwardSampling_2_UPPER1_1_ESTIMATE_THETA_TRUE")







load(inputPathPlot)



if (UPPER[uu] == 0) {
  ###plotIdx <- c(1,2,4)##############
  plotIdx <- 1:mm
} else {
  plotIdx <- 1:mm
}

LOWER_TITLE_AUX <- c("BPF", "MCMC BPF I", "MCMC BPF II", "FA-APF", "MCMC FA-APF I", "MCMC FA-APF II", "Idealised")
LOWER_AUX <- c(1:3,4:6,7)
LTY_AUX <- c(2,2,2,1,1,1,1)
COL_AUX <- c("magenta3", "blue", "forestgreen", "magenta3", "blue", "forestgreen", "black")

widthPlotParameters <- widthPlot
heightPlotParameters <- heightPlot

### ACF of parameters:

for (uu in 1:length(UPPER)) {    
  for (kk in 1:dimTheta) {
    
    tikz(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_acf_of_parameters_", kk, "_dimX_", dimX, ".tex", sep=''), width=widthPlotParameters, height=heightPlotParameters) 
    
#     pdf(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_acf_of_parameters_", kk, "_dimX_", dimX, ".pdf", sep=''), width=widthPlotParameters, height=heightPlotParameters) 
  
    op <- par(oma=omaPlot, mar=marPlot)
    
    plot(0:lag.max.theta, rep(1, times=lag.max.theta+1), type='l', col="white",
    xlim=c(0,lag.max.theta), ylim=c(0,1.05),
    ylab="", xlab="",
    yaxs='i', xaxs='i', xaxt='n', yaxt='n')
  
    for (ll in 1:length(LOWER_AUX)) {
      if (length(plotIdx) == 1) {
        X <- outputTheta[kk,(burnin+1):nIterations,LOWER_AUX[ll],uu,1]
      } else {
        X <- apply(outputTheta[kk,(burnin+1):nIterations,LOWER_AUX[ll],uu,plotIdx], 1, mean)
      }
      ACF <- as.numeric(acf(X, lag.max=lag.max.theta, plot=FALSE)$acf)      
      lines(0:lag.max.theta, ACF, type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
    }
    axis(side=2, at=c(0,1), labels=c(0,1), las=2)
    axis(side=1, at=c(0,lag.max.theta))
    mtext("Lag", side = 1, line = 1, outer = FALSE)
    mtext("Autocorrelation", side = 2, line = 1, outer = FALSE)
    
    if (UPPER[uu] == 0 && kk == 3){
      legend("bottomleft", legend=LOWER_TITLE_AUX, col=COL_AUX, lty=LTY_AUX, 
      bty='n', 
      cex=cex.legend)
    } else if (UPPER[uu] == 1 && kk == 1 && dimX == 25){
      legend("topright", legend=LOWER_TITLE_AUX, col=COL_AUX, lty=LTY_AUX, 
      bty='n', 
      cex=cex.legend)
    }
        
    par(op)
    dev.off()
  }
}


# ### TRACE of parameters:
# nIterationsMax <- 5000
# 
# for (uu in 1:length(UPPER)) {
#   for (kk in 1:dimTheta) {
#     
#     tikz(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_traces_of_parameters_", kk, "_dimX_", dimX, ".tex", sep=''), width=widthPlotParameters, height=heightPlotParameters) 
#   
#     op <- par(oma=omaPlot, mar=marPlot)
#     
#     plot(0:1, 0:1, type='l', col="white",
#          xlim=c(1,nIterationsMax), ylim=yLim[,kk], ylab="", xlab="", yaxs='i', xaxs='i', xaxt='n', yaxt='n')
#   
#     for (ll in 1:length(LOWER_AUX)) {
#       X <- outputTheta[kk,1:nIterationsMax,LOWER_AUX[ll],uu,1]    
#       lines(1:nIterationsMax, X, type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
#     }
#     axis(side=2, at=yLim[,kk], labels=yLim[,kk], las=2)
#     axis(side=1, at=c(1,nIterationsMax), labels=c(1,nIterationsMax))
#     mtext("Parameter value", side = 2, line = 1, outer = FALSE)
#     mtext("Iteration", side = 1, line = 1, outer = FALSE)
#     
#     par(op)
#     dev.off()
#   }
# }

### KDE of parameters:

for (uu in 1:length(UPPER)) {

  if (UPPER[uu] == 1) {
    DENSITY_LIM_PARAMETERS <- c(9,16,4,8)
    LB_PARAMETERS <- c(0,0,0,0)
    UB_PARAMETERS <- c(1,0.5,2,2)
  } else {
    DENSITY_LIM_PARAMETERS <- c(5,10,5,4)
    LB_PARAMETERS <- c(-1,0,0,0)
    UB_PARAMETERS <- c(1,1,2,2)
  }

  for (kk in 1:dimTheta) {
    
    tikz(file=paste(outputPathPlot, upperTitle[UPPER[uu]+1], "_kde_of_parameters_", kk, "_dimX_", dimX, ".tex", sep=''), width=widthPlotParameters, height=heightPlotParameters) 
  
    op <- par(oma=omaPlot, mar=marPlot)
    
    plot(0:1, 0:1, type='l', col="white",
         xlim=c(LB_PARAMETERS[kk], UB_PARAMETERS[kk]), ylim=c(0,DENSITY_LIM_PARAMETERS[kk]), ylab="", xlab="", yaxs='i', xaxs='i', xaxt='n', yaxt='n')
  
    for (m in plotIdx) { ##############
      for (ll in 1:length(LOWER_AUX)) {
      
        if (UPPER[uu] == 1) {
        #         if (ll > 3) { ##########
          X <- outputTheta[kk,(burnin+1):nIterations,LOWER_AUX[ll],uu,m]
          if (kk >= dimTheta - 1) {
            lines(density(X, from=0), type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
          } else {
            lines(density(X, from=-1, to=1), type='l', col=COL_AUX[ll], lty=LTY_AUX[ll]) 
          }
#         } ############
        } else {
                if (ll > 3) { ##########
          X <- outputTheta[kk,(burnin+1):nIterations,LOWER_AUX[ll],uu,m]
          if (kk >= dimTheta - 1) {
            lines(density(X, from=0), type='l', col=COL_AUX[ll], lty=LTY_AUX[ll])
          } else {
            lines(density(X, from=-1, to=1), type='l', col=COL_AUX[ll], lty=LTY_AUX[ll]) 
          }
        } ############
        }

        
      }
    } ############
    
    axis(side=2, at=c(0,DENSITY_LIM_PARAMETERS[kk]), labels=c(0,DENSITY_LIM_PARAMETERS[kk]),las=2)
    axis(side=1, at=c(LB_PARAMETERS[kk],UB_PARAMETERS[kk]), labels=c(LB_PARAMETERS[kk],UB_PARAMETERS[kk]))
    mtext("Value", side = 1, line = 1, outer = FALSE)
    mtext("Density", side = 2, line = 1, outer = FALSE)
    
    ##############
    if (UPPER[uu] == 0 && kk == 1){
      legend("topleft", legend=LOWER_TITLE_AUX, col=COL_AUX, lty=LTY_AUX, 
      bty='n', 
      cex=cex.legend)
    }
    #############
    
    par(op)
    dev.off()
  }
}

