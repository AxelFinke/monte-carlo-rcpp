## Blocked particle smoothing in a high-dimensional
## linear Gaussian state-space model

## V-dimensional linear Gaussian state-space model:
## X_1 ~ N(m0, C0),
## X_t = A X_{t-1} + B U_t,
## Y_t = C X_{t-1} + D V_t,
## where A is a symmetric banded diagonal matrix of size c(V, V) with diagonal elements a_0 
## and elements a_i on the ith band and negative first band;
## B = b^2*diag(V), D = d^2*diag(V), C = diag(V), 
## m0 = rep(0, times=V), C0 = diag(V), 
## U_t and V_t are IID V-dimensional standard Gaussian noise.

## This script runs a single replicate in a simulation study
## showing the behaviour of the various particle smoothers
## when used within offline stochastic gradient-ascent or 
## offline stochastic expectation--maximisation algorithms for 
## approximating the MLE of the model parameters.


## ========================================================================= ##
## SETUP
## ========================================================================= ##

rm(list = ls())

DEBUG <- TRUE

if (DEBUG) {
  pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp" # put the path to the monte-carlo-rcpp directory here
  pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp" # put the path to the folder which whill contain the simulation output here
  jobName           <- "smc_sge_array_debug"
} else {
  pathToInputBase   <- "/home/ucakafi"
  pathToOutputBase  <- "/home/ucakafi/Scratch/output"
  jobName           <- "smc_sge_array_2017-05-14"
}

exampleName  <- "linearGaussianHmm"
projectName  <- "blockedSmoothing"

source(file=file.path(pathToInputBase, "setupRCpp.r"))

## ========================================================================= ##
## TASK-ID PARAMETERS
## ========================================================================= ##

if (DEBUG) {
  task_id <- 1
} else {
  task_id <- as.numeric(Sys.getenv("SGE_TASK_ID"))
}

set.seed(task_id)

## ========================================================================= ##
## PARAMETERS
## ========================================================================= ##

## Model parameters
## ------------------------------------------------------------------------- ##
para <- 0 # type of reparametrisation: 0: unbounded (for gradient/EM algorithms); 1: bounded (for MCMC/SMC)
V    <- 100 # number of vertices/dimension of the state space
T    <- 20 # number of time steps
b    <- 1
d    <- 1
a    <- c(0.5, 0.2)
K    <- length(a)-1

hyperParameters <- c(1,1,1,1)

A  <- toeplitzCpp(c(a, rep(0,length=V-length(a))))
B  <- b*diag(V) 
C  <- diag(V)
D  <- d*diag(V)
m0 <- rep(0, V)
C0 <- diag(V)

## Checking stationarity of the latent chain:
if (any(abs(eigen(A, symmetric=TRUE)$values) >= 1)) {
  print("WARNING: Latent process is not stationary!")
}

if (para == 0) {
  thetaTrue <- c(a, log(b), log(d)) # true parametes in the parametrisation used by the algorithm
} else if (para == 1) {
  thetaTrue <- c(a, b, d) # true parametes in the parametrisation used by the algorithm
}

## Parameters of the SMC algorithm
## ------------------------------------------------------------------------- ##
filt <- 0  # type of SMC algorithm to be used; 0: standard; 1: blocked PF; 2: Exact marginal filter
prop <- 1  # type of SMC proposal kernel to be used; 0: bootstrap/prior; 1: conditionally locally optimal
H    <- 1 # effective sample size-based adaptive resampling threshold
# In high dimensions, it may be sensible to resample at every step; 
# otherwise, unless the number of particles scales exponentially in the dimension of the state space, 
# there is a risk that the effective sample size will be overestimated
# because all particles miss important regions of the state space;


## ========================================================================= ##
## SIMULATION STUDY: ESTIMATING STATIC PARAMETERS
## ========================================================================= ##

## Simulated data
## ------------------------------------------------------------------------- ##

# DATA <- simulateDataCpp(T, A, B, C, D, m0, C0)
# y    <- DATA$y # simulated observations
# write.table(x=y, file=paste(getwd(), "/data/data_for_mle_simulation_study.dat", sep=''))

y <- data.matrix(read.table(file.path(getwd(), "data", "data_for_mle_simulation_study.dat")))

## Numerically computing the maximum--likelihood estimate:
## ------------------------------------------------------------------------- ##

# Here, we use numerical methods to find the maximum of the marginal likelihood.
# This is used as a reference for the stochastic gradient-ascent and stochastic
# EM algorithms. The marginal likelihood is evaluated analytically. 
# (this is usually called the Kalman filter). For greater robustness, 
# we run the numerical optimisers multiple times from different initial values.
# In addition, we perform these operations for two different Kalman-filter
# implementations, one custom implementation and one implementation from the 
# R library FKF.
# 
# # Exact log-likelihood (computed using a custom function implemented in RCpp/Armadillo):
# objectiveFunctionCpp <- function(theta) {
#   
#   A <- toeplitzCpp(c(theta[1:(K+1)], rep(0, times=V-K-1)))
#   C <- diag(V)
#    
#   B <- exp(theta[K+2])*diag(V)
#   D <- exp(theta[K+3])*diag(V)
# 
#   logLike <- kalmanCpp(A, B, C, D, m0, C0, y)
#   return(-logLike)
# }
# 
# # Exact log-likelihood (computed using Kalman-filtering routines from the R package "FKF"):
# library("FKF")
# objectiveFunctionFkf <- function(theta) {
# 
#   A <- toeplitzCpp(c(theta[1:(K+1)], rep(0, times=V-K-1)))
#   C <- diag(V)
#   
#   B <- exp(theta[K+2])*diag(V)
#   D <- exp(theta[K+3])*diag(V)
# 
# 
#   logLike <- fkf(m0, C0, matrix(0,V,T), matrix(0,V,T), array(A, c(V,V,1)), array(diag(V), c(V,V,1)), array(B*t(B), c(V,V,1)), array(D*t(D), c(V,V,1)), y)$logLik
#   return(-logLike)
# }
# 
# exactML <- function(thetaInit, F, method="BFGS") {
# 
#   nML <- dim(thetaInit)[2]
#   argminF <- matrix(NA,K+3,nML)
#   minF <- rep(NA, nML)
#   for (nn in 1:nML) {
#     aux <- optim(thetaInit[,nn], F, method)
#     argminF[,nn] <- aux$par
#     minF[nn] <- aux$value
#   }
#   
#   kk <- which.min(minF)
#   
#   return(list(par=argminF[,kk], value=minF[kk], allPars=argminF, allValues=minF))
# }
# 
# nML <- 5
# 
# thetaInit <- thetaTrue
# if (para == 1) {
#   thetaInit[(K+2):(K+3)] <- log(thetaTrue[(K+2):(K+3)])
# }
# thetaInitMat <- matrix(thetaTrue + rnorm((K+3)*nML, sd=0.1), K+3, nML)
# aux1 <- exactML(thetaInitMat, objectiveFunctionCpp) # numerical optima found when using the custom Kalman-filter implementation
# aux2 <- exactML(thetaInitMat, objectiveFunctionFkf) # numerical optima found when using the Kalman-filter implementation from the R library FKF
# thetaML <- aux1$par
# if (para == 1) {
#   thetaML[(K+2):(K+3)] <- exp(thetaML[(K+2):(K+3)])
# }


## Parameters of the simulation study
## ------------------------------------------------------------------------- ##

bInn  <- c(3) # diameter of the blocks that partition the space
bOut  <- c(2) # # size by which each block is extended in each direction to create the enlarged blocks
N     <- 1000 # number of filtering particles
M     <- 250 # number of smoothing paths (if M>=N we compute the sum in Equation 11 exactly)

FILT  <- c(1) # types of filtering schemes to test

L1 <- length(FILT) # number of different combinations of filtering and backward-sampling schemes to test
I1 <- length(bInn) # number of different inner block configurations
J1 <- length(bOut) # number of different outer block configurations

M1 <- 1 # number of replicates

## Parameters for maximum-likelihood estimation
## ------------------------------------------------------------------------- ##

G         <- 2000 # number of iterations of offline (stochastic) EM or (stochastic) gradient ascent algorithms.
stepSizes <- (1:G)^(-0.8) # step size for (stochastic) gradient ascent algorithms.

## Parameters for simulation studies
## ------------------------------------------------------------------------- ##

MLE  <- c(0, 1) # types of algorithms used for approximating the MLE:
# 0: offline stochastic gradient ascent
# 1: offline stochastic EM
K1 <- length(MLE)


## Running the simulation study
## ------------------------------------------------------------------------- ##

# thetaMleTrue     <- array(NA, c(K+3, G, K1, M1))

thetaMleFs       <- array(NA, c(K+3, G, L1, K1, M1))
thetaMleBs       <- array(NA, c(K+3, G, L1, K1, M1))

thetaMleBlockFs  <- array(NA, c(K+3, G, L1, K1, M1))
thetaMleBlockBs  <- array(NA, c(K+3, G, L1, K1, M1))

aux <- setBlocks(bInn, bOut, V)
blockInn <- aux$blockInn # boundaries of the actual blocks, i.e. of $K$ in the paper
blockOut <- aux$blockOut # boundaries of the enlarged blocks, i.e. of $\overline{K}$ in the paper

mm <- 1
ll <- 1

thetaInit      <- rnorm(4,mean=0,sd=1)
thetaInit[1:2] <- thetaInit[1:2] / (2*sum(abs(thetaInit[1:2])))

# print("GA:")
# thetaMleTrue[,,1,mm]  <- runOgaCpp(thetaInit, stepSizes, m0, C0, y, nCores)$theta
# print("EM:")
# thetaMleTrue[,,2,mm]  <- runOemCpp(thetaInit, stepSizes, m0, C0, y, nCores)$theta

# 
# load("/home/axel/Dropbox/research/output/cpp/mc/blockedSmoothing/linearGaussianHmm/for_revised_tsp_paper/results/suff_maxV_500_T_20_N_500_M1_1_prop_1_task_id_1")

print("SGA-FS:")
thetaMleFs[,,ll,1,mm] <- runOsgaCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta
print("SEM-FS:")
thetaMleFs[,,ll,2,mm] <- runOsemCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta
print("SGA-BS:")
thetaMleBs[,,ll,1,mm] <- runOsgaCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta
print("SEM-BS:")
thetaMleBs[,,ll,2,mm] <- runOsemCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta

print("SGA-BlockFS:")
thetaMleBlockFs[,,ll,1,mm] <- runOsgaCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta
print("SEM-BlockFS:")
thetaMleBlockFs[,,ll,2,mm] <- runOsemCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta
print("SGA-BlockBS:")
thetaMleBlockBs[,,ll,1,mm] <- runOsgaCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta
print("SEM-BlockBS:")
thetaMleBlockBs[,,ll,2,mm] <- runOsemCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta

## Saving output: 
save(
  list  = ls(envir = environment(), all.names = TRUE), 
  file  = paste("osem_osga_T_", T, "_V_", V, "_N_", N, "_M_", M, "_M1_", M1, "_G_", G, "_task_id_", task_id, sep=''),
  envir = environment()
) 
