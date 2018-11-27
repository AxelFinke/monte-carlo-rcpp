## Standard PFs vs. MCMC PFs in a simple linear-Gaussian state-space model

rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp"
exampleName       <- "linear"
projectName       <- "ensemble"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))


## ========================================================================= ##
## MODEL
## ========================================================================= ##

dimX     <- 5 # dimension of the states
dimY     <- dimX # dimension of the observations 
nOffDiagonals <- 0 # number of non-zero off-diagonals on each side of the main diagonal of A
dimTheta <- nOffDiagonals + 3 # length of the parameter vector

# Support of the unknown parameters:
supportMin <- c(-1, 0, 0) # minimum
supportMax <- c( 1,  Inf, Inf) # maximum
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
thetaTrue     <- c(0.5, 1, 1) # "true" parameter values used for generating the data
thetaNames    <- c("a0", "b", "d") # names of the parameters

DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
observations  <- DATA$y # simulated observations

smcParameters <- rep(sqrt(1.0/dimX), times=nObservations); # additional parameters for the SMC algorithms
ensembleOldParameters <- c(rep(0, dimX), 1.0/dimX, 0.5);# additional parameters for the original EHMM algorithms
####################
## The original values (scaling the variance as 1/d)
ensembleNewParameters <- c(sqrt(1.0/dimX), sqrt(1-1.0/dimX), 1.0, 1.0) # additional parameters for the alternative EHMM algorithms: c("scale for random-walk proposals", "correlation parameter for autoregressive updates)
####################

 
rwmhSd = sqrt(rep(1, times=dimTheta)/(100*dimTheta*dimX*nObservations))

initialiseStatesFromStationarity <- FALSE  # should the CSMC-type algorithms initialise the state sequence from stationarity?
storeParentIndices <- FALSE                # should the CSMC-type algorithms all their parent indices?
storeParticleIndices <- FALSE              # should the CSMC-type algorithms store the indices of the (input and output) reference particle lineage?

backwardSamplingType <- 2 # 0: none; 1: backward; 2: ancestor
resampleType <- 1 # 0: multinomial; 1: systematic
 
## ========================================================================= ##
## Approximating the marginal likelihood
## ========================================================================= ##



nSteps <- nObservations
nParticles       <- 1000
nBurninSamples   <- 100
essResamplingThreshold <- 1.0 # unused here

nSimulations <- 1000
MM <- nSimulations # number of independent replicates

PROP  <- c(0,2,0,2) # distribution flow, i.e. BPF or APF (PROP must have the same length as LOWER)
LOWER <- c(2,2,2,2) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
LOCAL <- c(0,0,1,1) # 0: conditionally IID updates; 1: alternative EHMM with Gaussian random-walk proposals


LL <- length(LOWER) # number of lower-level sampler configurations to test

lowerNames <- c("BPF", "FA-APF", "MCMC BPF", "MCMC FA-APF")


# logLikeTrue <- evaluateLogMarginalLikelihoodCpp(hyperparameters, thetaTrue, observations, nCores)
centredLogLikeHat <- matrix(NA, LL, MM)
# ess <- array(NA, c(nSteps, LL, MM))
# acceptanceRates <- array(NA, c(nSteps, LL, MM))

for (mm in 1:MM) {

  print(paste(mm, "of", MM, sep=' '))

  DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
  observations  <- DATA$y # simulated observations
  logLikeTrue   <- evaluateLogMarginalLikelihoodCpp(hyperparameters, thetaTrue, observations, nCores)

  for (ll in 1:LL) {
    
    if (PROP[ll] == 2 && LOWER[ll] == 2 && LOCAL[ll] == 1) {
      nParticlesAux <-  nParticles - nBurninSamples
      ensembleNewInitialisationType <- 1 # using burn-in for the MCMC-FA-APF
    } else {
      nParticlesAux <-  nParticles
      ensembleNewInitialisationType <- 0 # initialising from stationarity
    }
    
    centredLogLikeHat[ll, mm] <- approximateMarginalLikelihoodCpp(LOWER[ll], dimTheta, thetaTrue, hyperparameters, support, observations, PROP[ll], nSteps, nParticlesAux, ensembleNewInitialisationType, nBurninSamples, essResamplingThreshold, smcParameters, ensembleOldParameters, ensembleNewParameters, LOCAL[ll], nCores) - logLikeTrue ### NOTE: subtracting the true loglikelihood here
 
  }
}

save(
  list  = ls(envir = environment(), all.names = TRUE), 
  file  = paste(pathToResults, "/likelihood_estimates_for_clt_paper---nObservations_", nObservations, "_nSimulations_", nSimulations, "_nParticles_", nParticles, "_dimension_", dimX, sep=''),
  envir = environment()
) 

## Plot the results
#load(file=paste(pathToOutput, "likelihood_estimates---nObservations_25_nSimulations_10000_nParticles_1000_dimension_2", sep=''))


# for (dd in c(1,2,5,10)) {

#     load(file.path(pathToResults, paste("likelihood_estimates_for_clt_paper---nObservations_10_nSimulations_10000_nParticles_10000_dimension_", dd, sep='')))

#     pdf(file=paste(pathToFigures, "likelihood_estimates_in_dimension_", dimX,".pdf", sep=''), width=8, height=10)   

    op <- par(mar=c(20, 3, 2, 2)+1) ## c(bottom, left, top, right)
    boxplot(exp(t(centredLogLikeHat))-1, 
        ylab="hat{Z}/Z - 1", ylim=c(-1,1),
        range=0,
        las=2, 
        names=lowerNames,
        main="Relative estimates of marginal likelihood (denoted Z)"
        )
    
    mtext(paste("N =", nParticlesLinear, "T =", nObservations, "; d =", dimX), side = 3, line = 0, outer = FALSE)
    abline(h=0, col="red")
    par(op)
    
#     dev.off()


# }



# 
# 
# ## ========================================================================= ##
# ## PLOT RESULTS
# ## ========================================================================= ##
# 
# ##############################################################################
# ## Graphics for the paper
# ##############################################################################
# 
# library(tikzDevice)
# options( 
#   tikzDocumentDeclaration = c(
#     "\\documentclass[12pt]{beamer}",
#     "\\usepackage{amssymb,amsmath,graphicx,mathtools,mathdots,stmaryrd}",
#     "\\usepackage{tikz}" 
#   )
# )
# 
# colPlot <- "magenta3"
# colPlotAlternate <- "forestgreen"
# outputPathPlot <- "/home/axel/Dropbox/ensemble_interpretation/figures/"
# 
# cex.legend <- 0.8
# 
# ##############################################################################
# ## Marginal-ikelihood estimates
# ##############################################################################
# 
# 
# 
# 
# 
# 
# for (ii in 1:4) {
# 
#   if (ii == 1) {
#     inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_2")
#   } else if (ii == 2) {
#     inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_5")
#   } else if (ii == 3) {
#     inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_10")
#   } else if (ii == 4) {
#     inputPathPlot <- file.path(pathToResults, "likelihood_estimates---nObservations_10_nSimulations_1000_nParticles_1000_dimension_25")
#   }
#   load(inputPathPlot)
# 
# #   lowerNames <- c(
# #   "BPF", "FA-APF", "MCMC BPF I", "MCMC BPF II", 
# #   "MCMC FA-APF I", "MCMC FA-APF II", "original EHMM I", "original EHMM II")
#   
#   lowerNames <- c(
#   "BPF", "FA-APF", "MCMC BPF", 
#   "MCMC FA-APF", "original EHMM I", "original EHMM II")
#   
# #   lowerPlot <- c(2,8, 3:4, 9:10, 13:14)
#   lowerPlot <- c(2,8, 3, 9, 13:14)
#   
# 
#   tikz(paste(outputPathPlot, "likelihood_estimates_in_dimension_", dimX, ".tex", sep=''),  width=1.9, height=2.7)
# 
#     op <- par(mar=c(7, 3.5, 0, 0)+1) ## c(bottom, left, top, right)
#     
#     if (ii == 1) {
#       boxplot(exp(t(centredLogLikeHat[lowerPlot,])), 
#         ylab="", ylim=c(0,2),
#         range=0,
#         las=2, 
#         names=lowerNames,
#         main="",
#         yaxt='n'
#       )
#       axis(side=2, at=c(0,1,2), labels=c(0,1,2), las=2)
#       mtext("$\\hat{p}_\\theta(y_{1:T}) / p_\\theta(y_{1:T}) $", side = 2, line = 2, outer = FALSE)
#     
#     } else {
#       boxplot(exp(t(centredLogLikeHat[lowerPlot,])), 
#         ylab="", ylim=c(0,2),
#         range=0,
#         las=2, 
#         names=lowerNames,
#         main="",
#         yaxt="n"
#       )
#     }
#     abline(h=1.0, col=colPlot)
#     par(op)
# 
#   dev.off()
# }
# 
