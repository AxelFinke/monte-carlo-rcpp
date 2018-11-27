## Maximum--Likelihood estimation for a stable distribution


rm(list = ls())
set.seed(123)

pathToBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp"
exampleName  <- "stable"
projectName  <- "optim"

source(file=file.path(pathToBase, "setupRCpp.r"))
source(file=file.path(pathToBase, projectName,  paste(projectName, ".r", sep='')))


# rm(list = ls())
# 
# exampleName   <- "stable"
# pathToInput   <- "/home/axel/Dropbox/phd/code/cpp/generic/"
# pathToOutput  <- paste(pathToInput, "results/optim/", exampleName, "/", sep='') 
# nCores <- 1 # number of cores to use (not currently implemented)
# set.seed(123)
# 
# ## ========================================================================= ##
# ## SETUP JUST-IN-TIME COMPILER
# ## ========================================================================= ##
# 
# R_COMPILE_PKGS=TRUE
# R_ENABLE_JIT=3
# 
# ## ========================================================================= ##
# ## SETUP RCPP
# ## ========================================================================= ##
# 
# library("Rcpp")
# Sys.setenv("PKG_CXXFLAGS"=paste("-Wall -std=c++11 -I\"", pathToInput, "\" -I/usr/include -lCGAL -fopenmp -O3 -ffast-math -march=native", sep=''))
#            
# Sys.setenv("PKG_LIBS"="-fopenmp -lCGAL")
# sourceCpp(paste(pathToInput, "optim/examples/", exampleName, "/", exampleName, ".cpp", sep=''), rebuild=TRUE)
# 
# ## ========================================================================= ##
# ## SETUP OTHER
# ## ========================================================================= ##
# 
# source(file=paste(pathToInput, "optim/generic_functions.r", sep=''))
# setwd(paste(pathToInput, "optim/examples/", exampleName, "/", sep=''))

## ========================================================================= ##
## MODEL
## ========================================================================= ##



dimTheta   <- 4 # length of the parameter vector

tabooMinAlpha   <- 0.9 # (here: additional restriction of the support for the tail parameter)
tabooMaxAlpha   <- 1.1
shapeHyperGamma <- 1
scaleHyperGamma <- 0.5
meanHyperDelta  <- 0
varHyperDelta   <- 10
hyperparameters <- c(tabooMinAlpha, tabooMaxAlpha, shapeHyperGamma, scaleHyperGamma, meanHyperDelta, varHyperDelta) # hyper parameters 

supportMin <- c(0.1, -1, 0, -Inf)   # minimum of the support for the (tail, skewness, location, scale)-parameters
supportMax <- c(2 - (tabooMaxAlpha-tabooMinAlpha), 1, Inf, Inf) # maximum of the support for the (tail, skewness, location, scale)-parameters
# NOTE: the first component of theta (corresponding to alpha) has been reparametrised to have contiguous support
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

alphaTrue  <- 0.5
betaTrue   <- 0.7
gammaTrue  <- 1
deltaTrue  <- 0

thetaTrue     <- c(alphaTrue, betaTrue, gammaTrue, deltaTrue) # "true" parameter values used for generating the data
if (alphaTrue > tabooMaxAlpha) {
  thetaTrue[1] <- alphaTrue - (tabooMaxAlpha-tabooMinAlpha)
}

thetaNames    <- c("alpha", "beta", "gamma", "delta") # names of the parameters

## Option 1: sample new data set
## library("stabledist")
## nObservations <- 10
## observations  <- rstable(nObservations, alpha=alphaTrue, beta=betaTrue, gamma=gammaTrue, delta=deltaTrue, pm=1) # simulated observations

## Option 2: load data set used by Riabiz et al. (2015)
library("R.matlab")
observations <- readMat("data_1.mat")$z
nObservations <- length(observations) # number of time steps/observations


## ========================================================================= ##
## Numerically approximate the MLE
## ========================================================================= ##

# Maybe we can use John Nolan's stuff here (Matlab)

## ========================================================================= ##
## ALGORITHM
## ========================================================================= ##

LOWER <- c(0,2)
UPPER <- c(0,1) 

runSimulationStudy(
  pathToOutput = pathToOutput,
  dimTheta = dimTheta, 
  hyperparameters = hyperparameters,
  observations = observations,
  nSmcStepsLower = nObservations,
  useGradients = FALSE, # MALA kernels don't work well in this model due to multimodality
  support = support,
  fixedLagSmoothingOrder = 0,
  rwmhSd = c(1, 1, 10, 1)*10/nObservations, # Metropolis--Hastings proposal scale
  nStepsMcmc = 50000,
  nSimulations = 30, 
  proportionCorrelated = 0.5, # for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  crankNicolsonScaleParameter = 0.00001, # the proposal is Normal(rho*u, (1-rho^2)*I), where rho_p = exp(-crankNicolsonScaleParameter*N_p/beta_p)
  nParticlesLowerMin = 100,
  betaMinMcmc = 1, #########
  betaMaxMcmc = 1, #########
  betaMinSmc  = 0.2, ########
  betaMaxSmc  = 1, ##########
  proportionBetaFixedMcmc = 0.2, ##########
  proportionBetaFixedSmc = 0.2, ########
  nThetaUpdates = 50, ##########
  nParticlesUpper = 5000, #########
  alpha = 1,
  prop = 2, # 2: adaptive envelope proposal
  useNonCentredParametrisation = TRUE, # should the (pseudo-)Gibbs samplers use an NCP?
  nonCentringProbability = 1, # probability of using the NCP if useNonCentredParametrisation = TRUE
  lower = LOWER,
  upper = UPPER,
  essResamplingThresholdLower = 1.0
)

# TODO: maybe run the MCMC and SMC versions separately

## ========================================================================= ##
## PLOT RESULTS
## ========================================================================= ##

# MCMC algorithms:
inputPathPlot <- paste(pathToOutput, "optim_nObservations_1000_nSimulations_25_betaMaxMcmc_1_alpha_1_nParticlesLowerMin_100", sep='')
yLim <- matrix(c(0.1,2,-1,1,0,5,-2,2), 2, dimTheta)
yLimBoxplot <- matrix(c(0.1,2,-1,1,0,3,-1,1), 2, dimTheta)
load(inputPathPlot)


plotSimulationStudyResults(
  inputName=inputPathPlot,
  outputName=paste(pathToOutput, "results_mcmc", sep=''),
  thetaMle = thetaTrue,
  dimTheta = dimTheta,
  yLim = yLim,
  yLimBoxplot = yLimBoxplot,
  yLabel = thetaNames,
  upperPlot = 0, 
  lowerPlot = LOWER
)

# SMC algorithms:
plotSimulationStudyResults(
  inputName=inputPathPlot,
  outputName=paste(pathToOutput, "results_smc", sep=''),
  thetaMle = thetaTrue,
  dimTheta = dimTheta,
  yLim = yLim,
  yLimBoxplot = yLimBoxplot,
  yLabel = thetaNames,
  upperPlot = 1,
  lowerPlot = LOWER
)
#


#################
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
# 
# setwd("~/Dropbox/applications/2016-03-06_beskos_ucl/slides/")
# 
# width <- 5
# height <- 3.3
# oma <- c(7,3,0,0)
# mar <- c(0,3,2,0.5)
# cex <- 0.6
# ylim <- c(-0.3,0.3)
# 
# load(inputPathPlot)
# X <- outputMcmc[1,,,,1:mm]
# Y <- matrix(NA, 9, mm)
# for (ii in 1:9) {
#   for (m in 1:mm) {
#     Y[ii,m] <- mean(X[4501:5000,ii,m])
#   }
# }
# 
# tikz("optim_mcmc.tex",  width=width, height=height)
# op <- par(oma=oma, mar=mar)
# boxplot(t(Y)-thetaMle, ylim=ylim, range=0, las=2, names=c("{Pseudo-marginal SAME}", "{(correlated)}", "{(noisy)}", "{Rubenthaler}", "{(correlated)}", "{(noisy)}", "{Simulated annealing}", "{Pseudo-Gibbs SAME}", "{Gibbs SAME}"), main="", cex.axis=cex)
#   mtext("Error", side=2, outer=TRUE, line=0, cex=cex)
#   abline(h=0, col="magenta")
# par(op)
# dev.off()
# 
# X <- outputSmc[1,,,,,1:mm]
# Y <- matrix(NA, 9, mm)
# for (ii in 1:9) {
#   for (m in 1:mm) {
#     Y[ii,m] <- mean(X[,91:100,ii,m])
#   }
# }
# 
# tikz("optim_smc.tex",  width=width, height=height)
# op <- par(oma=oma, mar=mar)
# boxplot(t(Y)-thetaMle, ylim=ylim, range=0, las=2, names=c("{Pseudo-marginal SAME}", "{(correlated)}", "{(noisy)}", "{Rubenthaler}", "{(correlated)}", "{(noisy)}", "{Simulated annealing}", "{Pseudo-Gibbs SAME}", "{Gibbs SAME}"), main="", cex.axis=cex)
#   mtext("Error", side=2, outer=TRUE, line=0, cex=cex)
#     abline(h=0, col="magenta")
# par(op)
# dev.off()
# 


# Display the amount of work required for all the particle-based algorithms
# load(inputPathPlot)
# calculateWork(nParticlesLowerMcmc, nParticlesUpper, nParticlesLowerSmc, betaMcmc, betaSmc, nStepsMcmc, nStepsSmc)




