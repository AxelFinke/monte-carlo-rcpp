## Optimisation in a multivariate linear Gaussian state-space model

rm(list = ls())
set.seed(123)

pathToBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
exampleName  <- "linear"
projectName  <- "optim"

source(file=file.path(pathToBase, "setupRCpp.r"))
source(file=file.path(pathToBase, projectName,  paste(projectName, ".r", sep='')))
# 
# rm(list = ls())
# 
# exampleName  <- "linear"
# pathToInput  <- "/home/axel/Dropbox/phd/code/cpp/generic/"
# pathToOutput <- paste(pathToInput, "results/optim/", exampleName, "/", sep='') 
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
# library(Rcpp)
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


dimX     <- 2 # dimension of the states
dimY     <- dimX # dimension of the observations
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

rwmhSd = rep(0.5, times=dimTheta)

nObservations <- 500 # number of time steps/observations
thetaTrue     <- c(0.5, 0.2, 1, 1) # "true" parameter values used for generating the data
thetaNames    <- c("a0", "a1", "b", "d") # names of the parameters

DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, numeric(0), nCores)
observations  <- DATA$y # simulated observations

## ========================================================================= ##
## Numerically approximate the MLE
## ========================================================================= ##

# Applies a reparametrisation needed for numerical optimisation
repar <- function(theta) { 
  theta[(length(theta)-1):length(theta)] <- log(theta[(length(theta)-1):length(theta)])
  return(theta)
}
# Reverts to original parametrisation
inverseRepar <- function(theta) {
  theta[(length(theta)-1):length(theta)] <- exp(theta[(length(theta)-1):length(theta)])
  return(theta)
}

thetaMle <- numericallyApproximateMle(
  nRuns=10, dimTheta=dimTheta, 
  hyperparameters=hyperparameters, 
  support=support, 
  observations=observations, 
  repar=repar, 
  inverseRepar=inverseRepar, 
  method="Nelder-Mead", nCores)$theta
  

## ========================================================================= ##
## ALGORITHM
## ========================================================================= ##

LOWER <- c(0,1,3,4,6)

runSimulationStudy(
  pathToOutput = pathToOutput,
  dimTheta = dimTheta, 
  hyperparameters = hyperparameters,
  observations = observations,
  nSmcStepsLower = nObservations,
  useGradients = FALSE,
  support = support,
  fixedLagSmoothingOrder = 0,
  rwmhSd = rwmhSd, # Metropolis--Hastings proposal scale
  nStepsMcmc = 5000,
  nSimulations = 50,
  proportionCorrelated = 0.90, # for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  crankNicolsonScaleParameter = 0.00001, # the proposal is Normal(rho*u, (1-rho^2)*I), where rho_p = exp(-crankNicolsonScaleParameter*N_p/beta_p)
  nParticlesLowerMin = 500,
  betaMaxMcmc = 10,
  alpha = 1,
  prop = c(0,1),
  lower = c(0,1,3,4,6) ######################
)


## ========================================================================= ##
## PLOT RESULTS
## ========================================================================= ##

# MCMC algorithms:
inputPathPlot <- paste(pathToOutput, "optim_nObservations_250_nSimulations_50_betaMaxMcmc_10_alpha_1_nParticlesLowerMin_300", sep='')
yLim <- matrix(c(0,1.5,0,1.5,0,3,0,3), 2, dimTheta)
yLimBoxplot <- matrix(c(0,1.5,0,1.5,0,3,0,3), 2, dimTheta)

load(inputPathPlot)

plotSimulationStudyResults(
  inputName = inputPathPlot,
  outputName = paste(pathToOutput, "results_mcmc", sep=''),
  thetaMle = thetaMle,
  dimTheta = dimTheta,
  yLim = yLim,
  yLimBoxplot = yLimBoxplot,
  yLabel = thetaNames,
  upperPlot = 0, 
  lowerPlot = LOWER
)

# SMC algorithms:
plotSimulationStudyResults(
  inputName = inputPathPlot,
  outputName = paste(pathToOutput, "results_smc", sep=''),
  thetaMle = thetaMle,
  dimTheta = dimTheta,
  yLim = yLim,
  yLimBoxplot = yLimBoxplot,
  yLabel = thetaNames,
  upperPlot = 1,
  lowerPlot = LOWER
)

