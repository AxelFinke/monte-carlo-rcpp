## Optimisation in a Levy-driven stochastic volatility model

rm(list = ls())
set.seed(123)

pathToBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
exampleName  <- "levy"
projectName  <- "optim"

source(file=file.path(pathToBase, "setupRCpp.r", sep=''))
source(file=file.path(pathToBase, projectName,  paste(projectName, ".r", sep='')))

# 
# rm(list = ls())
# 
# exampleName  <- "ssm"
# pathToInput  <- "/home/axel/Dropbox/research/code/cpp/mc/"
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
# Sys.setenv("PKG_CXXFLAGS"=paste("-Wall -std=c++11 -I\"", inputPathName, "\" -I/usr/include -lCGAL -fopenmp -O3 -ffast-math -march=native", sep=''))
#            
# Sys.setenv("PKG_LIBS"="-fopenmp -lCGAL")
# sourceCpp(paste(inputPathName, "optim/examples/", exampleName, "/", exampleName, ".cpp", sep=''), rebuild=TRUE)
# 
# ## ========================================================================= ##
# ## SETUP OTHER
# ## ========================================================================= ##
# 
# source(file=paste(inputPathName, "optim/generic_functions.r", sep=''))
# setwd(paste(inputPathName, "optim/examples/", exampleName, "/", sep=''))

## ========================================================================= ##
## MODEL
## ========================================================================= ##

## Parameters for the latent Levy processes (and some useful reparametrisations):
kappaTrue <- c(0.0.5, 0.5) # decay rate for each component process (elements must be strictly increasing)
wTrue <- c(0.7, 0.3) # weights for each component process (elements must sum to 1)
zetaTrue <- 0.7 # rate for the exponential jump-size distribution
xiTrue <- 2 # stationary mean-parameter of each component process
nComponents <- length(kappaTrue)

## Reparametrisations (not used by the algorithm)
deltaTrue <- kappaTrue - c(0, cumsum(kappaTrue[1:(length(kappaTrue)-1)]))
sumEpsilonTrue <- xiTrue * zetaTrue
epsilonTrue <- wTrue * sumEpsilonTrue
lambdaTrue <- kappaTrue * epsilonTrue # rates for the exponential jump-times distribution
invZetaTrue <- 1/zetaTrue
omega2True  <- sumEpsilonTrue / zetaTrue^2 # stationary variance-parameter for each component process

## Parameters for the observation equation (these are integrated out analytically!):
muTrue <- 0
betaTrue <- c(0.05, 0.05) # risk-premium parameters  (potentially different for each component process)
rhoTrue  <- c(-0.5, -0.5) # linear-leverage parameters (potentially different for each component process)
nBeta <- length(betaTrue)
nRho  <- length(rhoTrue)

thetaTrue <- c(kappaTrue, wTrue[1:(nComponents-1), xiTrue, invZetaTrue]
dimTheta <- length(thetaTrue) # length of the parameter vector (excludes the last weight as the weights need to sum to one)

# Support of the unknown parameters:
supportMin <- c(0,0,0,-Inf,0) # minimum
supportMax <- c(Inf, Inf, 1, Inf, Inf) # maximum
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

# Gamma prior on each element of delta:
shapeHyperDelta <- 1 # shape parameter
scaleHyperDelta <- 1 # scale parameter

# Symmetric Dirichlet prior on the weights:
hyperW <- 1

# Gamma prior on xi:
shapeHyperXi <- 2
scaleHyperXi <- 5

# Gamma prior on 1/zeta:
shapeHyperInvZeta <- 2
scaleHyperInvZeta <- 5

# Normal priors on the parameters of the observation equation:
meanHyperObsEq <- 0
varHyperObsEq  <- 100

# Collecting all the hyper parameters in a vector:
hyperparameters <- c(nComponents, nBeta, nRho, shapeHyperDelta, scaleHyperDelta, hyperW, shapeHyperXi, scaleHyperXi, shapeHyperInvZeta, scaleHyperInvZeta, meanHyperObsEq, varHyperObsEq)

nObservations <- 200 # number of time steps/observations

DATA          <- simulateDataCpp(nObservations,  hyperparameters, thetaTrue, nCores)
observations  <- DATA$y # simulated observations

rwmhSd = rep(0.5, times=dimTheta)
thetaNames <- c(paste("kappa", 0:(nComponents-1), sep=''), paste("w", 0:(nComponents-2), sep=''), "xi", "1/zeta") # names of the parameters

## ========================================================================= ##
## ALGORITHM
## ========================================================================= ##

LOWER <- c(0,2,3,5,7)

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
  nStepsMcmc = 1000,
  nSimulations = 100,
  proportionCorrelated = 0.5, # for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  crankNicolsonScaleParameter = 0.00001, # the proposal is Normal(rho*u, (1-rho^2)*I), where rho_p = exp(-crankNicolsonScaleParameter*N_p/beta_p)
  nParticlesLowerMin = 100,
  betaMaxMcmc = 10,
  alpha = 1,
  prop = c(0,1),
  lower = LOWER
)


## ========================================================================= ##
## PLOT RESULTS
## ========================================================================= ##

# MCMC algorithms:
inputPathPlot <- paste(pathToOutput, "optim_nObservations_300_nSimulations_100_betaMaxMcmc_10_alpha_1_nParticlesLowerMin_300", sep='')
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

