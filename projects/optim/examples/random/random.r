## Optimisation in a random-effects model
## Model: X_t ~ N(a, b^2); Y_t ~ N(x_t, d^2); 
## Priors: a has a normal prior; b and d have inverse-gamma priors 
## (all parameters are independent, a-priori)


rm(list = ls())
set.seed(123)

pathToBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp"
exampleName  <- "random"
projectName  <- "optim"

source(file=file.path(pathToBase, "setupRCpp.r"))
source(file=file.path(pathToBase, projectName,  paste(projectName, ".r", sep='')))

# 
# rm(list = ls())
# 
# exampleName  <- "random"
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

dimTheta   <- 3 # length of the parameter vector
supportMin <- c(-Inf, 0, 0) # minimum of the support for the mean and the variance-parameters
supportMax <- c(Inf, Inf, Inf) # maximum of the support for the mean and the variance-parameters
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

## Hyper parameters:
meanHyperA  = 0
varHyperA   = 10
shapeHyperB = 1
scaleHyperB = 1
shapeHyperD = 1
scaleHyperD = 1
hyperparameters <- c(meanHyperA, varHyperA, shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD)


aTrue <- 0
bTrue <- 1
dTrue <- 0.1

nObservations <- 10 # number of time steps/observations
thetaTrue     <- c(aTrue, bTrue, dTrue) # "true" parameter values used for generating the data
thetaNames    <- c("a", "b", "d") # names of the parameters
DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, numeric(0), nCores)
observations  <- DATA$y # simulated observations

## ========================================================================= ##
## Numerically approximate the MLE
## ========================================================================= ##

thetaMle <- numericallyApproximateMle(
  nRuns=10, 
  dimTheta=dimTheta, 
  hyperparameters=hyperparameters, 
  support=support,
  observations=observations,
  repar=(function(x) x), 
  inverseRepar=(function(x) x),
  method="Nelder-Mead",
  nCores)$theta

## ========================================================================= ##
## ALGORITHM
## ========================================================================= ##

LOWER <- c(0,1,2,3,4,5,6,7,8)
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
  rwmhSd = c(10,10,10)/nObservations, # Metropolis--Hastings proposal scale
  nStepsMcmc = 2000,
  nSimulations = 5,
  proportionCorrelated = 0.9, # for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  crankNicolsonScaleParameter = 0.00001, # the proposal is Normal(rho*u, (1-rho^2)*I), where rho_p = exp(-crankNicolsonScaleParameter*N_p/beta_p)
  nParticlesLowerMin = 50,
  betaMaxMcmc = 50,
  alpha = 0.5,
  prop = 0,
  nParticlesUpper = 100,
  useNonCentredParametrisation = FALSE, # should the (pseudo-)Gibbs samplers use an NCP?
  nonCentringProbability = 0, # probability of using the NCP if useNonCentredParametrisation = TRUE
  lower = LOWER,
  upper = UPPER,
  essResamplingThresholdLower = 0
)

# TODO: maybe run the MCMC and SMC versions separately

## ========================================================================= ##
## PLOT RESULTS
## ========================================================================= ##

# MCMC algorithms:
inputPathPlot <- paste(pathToOutput, "optim_nObservations_10_nSimulations_5_betaMaxMcmc_50_alpha_0.5_nParticlesLowerMin_50", sep='')
yLim <- matrix(c(-5, 5), 2, dimTheta)
yLimBoxplot <- matrix(c(1.8, 2.2), 2, dimTheta)
load(inputPathPlot)

plotSimulationStudyResults(
  inputName=inputPathPlot,
  outputName=paste(pathToOutput, "results_mcmc", sep=''),
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
  inputName=inputPathPlot,
  outputName=paste(pathToOutput, "results_smc", sep=''),
  thetaMle = thetaMle,
  dimTheta = dimTheta,
  yLim = yLim,
  yLimBoxplot = yLimBoxplot,
  yLabel = thetaNames,
  upperPlot = 1,
  lowerPlot = LOWER
)
