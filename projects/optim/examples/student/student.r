## Optimisation in the Student-t toy model

rm(list = ls())
set.seed(123)

pathToBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp"
exampleName  <- "student"
projectName  <- "optim"

source(file=file.path(pathToBase, "setupRCpp.r"))
source(file=file.path(pathToBase, projectName,  paste(projectName, ".r", sep='')))

# rm(list = ls())
# 
# exampleName  <- "student"
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

dimTheta   <- 1 # length of the parameter vector
df         <- 0.05 # degrees of freedom
supportMin <- -50 # minimum of the support for the location parameter
supportMax <-  50 # maximum of the support for the location parameter
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)
hyperparameters <- c(df) # hyper parameters

#nObservations <- 25 # number of time steps/observations
thetaTrue      <- 2 # "true" parameter values used for generating the data
thetaNames     <- "theta" # names of the parameters
#DATA          <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
#observations  <- DATA$y # simulated observations
observations   <- c(-20, 1, 2, 3)
nObservations  <- length(observations)

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
  rwmhSd = 10, # Metropolis--Hastings proposal scale
  nStepsMcmc = 5000,
  nSimulations = 50,
  proportionCorrelated = 0.9, # for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  crankNicolsonScaleParameter = 0.00001, # the proposal is Normal(rho*u, (1-rho^2)*I), where rho_p = exp(-crankNicolsonScaleParameter*N_p/beta_p)
  nParticlesLowerMin = 100,
  betaMaxMcmc = 50,
  alpha = 0.5,
  prop = 0,
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
inputPathPlot <- paste(pathToOutput, "optim_nObservations_4_nSimulations_50_betaMaxMcmc_50_alpha_0.5_nParticlesLowerMin_100", sep='')
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
