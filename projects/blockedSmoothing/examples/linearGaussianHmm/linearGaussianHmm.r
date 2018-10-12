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

## ========================================================================= ##
## SETUP
## ========================================================================= ##

rm(list = ls())
# set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"
exampleName       <- "linearGaussianHmm"
projectName       <- "blockedSmoothing"
jobName           <- "for_revised_tsp_paper"

source(file=file.path(pathToInputBase, "setupRCpp.r"))


## ========================================================================= ##
## PARAMETERS
## ========================================================================= ##

## Model parameters
## ------------------------------------------------------------------------- ##
para <- 0 # type of reparametrisation: 0: unbounded (for gradient/EM algorithms); 1: bounded (for MCMC/SMC)
V    <- 100 # number of vertices/dimension of the state space
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
## SIMULATION STUDY I: ESTIMATING SUFFICIENT STATISTICS & GRADIENTS
## AS A FUNCTION OF THE DIMENSION OF THE STATE SPACE
## ========================================================================= ##

## Parameters of the simulation study
## ------------------------------------------------------------------------- ##

bInn  <- c(1,3,20) # inner block sizes
bOut  <- c(0,1) # size of the extension of each inner block on each side

SPACE <- c(20,50,100,150)  ### c(5,10,25,50,75,100,200,300,400,500) ##seq(from=50, to=100, by=2) # model dimensions to test
T     <- 20 # number of time steps
N     <- 100 ### 500 # number of filtering particles
M     <- 25 ### 100 # number of smoothing paths (if M>=N we compute the sum in Equation 11 exactly)

FILT  <- c(0, 1, 2) # types of filtering schemes to test: 
# 0: standard particle filter; 
# 1: blocked particle filter; 
# 2: IID samples from exact filter.

L1 <- length(FILT) # number of different combinations of filtering and backward-sampling schemes to test
I1 <- length(bInn) # number of different inner block configurations
J1 <- length(bOut) # number of different outer block configurations

V1 <- length(SPACE)
M1 <- 10 ## 100 # number of replicates





## Running the simulation study
## ------------------------------------------------------------------------- ##

# True sufficient statistics:
suffHTrue         <- array(NA, c(K+1, K+1, V1, M1))
suffQTrue         <- array(NA, c(K+1, V1, M1))
suffRTrue         <- matrix(NA, V1, M1)
suffSTrue         <- matrix(NA, V1, M1)
gradCompTrue      <- array(NA, c(K+3, V1, M1))

# Standard forward smoothing:
suffHFs           <- array(NA, c(K+1, K+1, I1, J1, V1, L1, M1))
suffQFs           <- array(NA, c(K+1, I1, J1, V1, L1, M1))
suffRFs           <- array(NA, c(I1, J1, V1, L1, M1))
suffSFs           <- array(NA, c(I1, J1, V1, L1, M1))
gradCompFs        <- array(NA, c(K+3, I1, J1, V1,L1, M1))

# Standard backward sampling:
suffHBs           <- array(NA, c(K+1, K+1, I1, J1, V1, L1, M1))
suffQBs           <- array(NA, c(K+1, I1, J1, V1, L1, M1))
suffRBs           <- array(NA, c(I1, J1, V1, L1, M1))
suffSBs           <- array(NA, c(I1, J1, V1, L1, M1))
gradCompBs        <- array(NA, c(K+3, I1, J1, V1, L1, M1))

# Blocked forward smoothing:
suffHBlockFs      <- array(NA, c(K+1, K+1, I1, J1, V1, L1, M1))
suffQBlockFs      <- array(NA, c(K+1, I1, J1, V1, L1, M1))
suffRBlockFs      <- array(NA, c(I1, J1, V1, L1, M1))
suffSBlockFs      <- array(NA, c(I1, J1, V1, L1, M1))
gradCompBlockFs   <- array(NA, c(K+3, I1, J1, V1,L1, M1))

# Blocked backward sampling:
suffHBlockBs      <- array(NA, c(K+1, K+1, I1, J1, V1, L1, M1))
suffQBlockBs      <- array(NA, c(K+1, I1, J1, V1, L1, M1))
suffRBlockBs      <- array(NA, c(I1, J1, V1, L1, M1))
suffSBlockBs      <- array(NA, c(I1, J1, V1, L1, M1))
gradCompBlockBs   <- array(NA, c(K+3, I1, J1, V1,L1, M1))


for (mm in 1:M1) {

  for (vv in 1:V1) {
  
    print(paste("mm: ", mm, " vv: ", vv))
    
    ## Increasing the state space and simulating new data 
    V <- SPACE[vv]
    A  <- toeplitzCpp(c(a, rep(0,length=V-length(a))))
    B  <- b*diag(V) 
    C  <- diag(V)
    D  <- d*diag(V)
    m0 <- rep(0, V)
    C0 <- diag(V)
    
    print("simulate data")
    y <- simulateDataCpp(T, A, B, C, D, m0, C0)$y
    
    ## Calculating exact sufficient statistics and gradients:
    
    print("exact smoothing")
    aux <- suffExactCpp(thetaTrue, m0, C0, y, nCores)
    suffHTrue[,,vv,mm]   <- aux$suffHTrue
    suffQTrue[,vv,mm]    <- aux$suffQTrue
    suffRTrue[vv,mm]     <- aux$suffRTrue
    suffSTrue[vv,mm]     <- aux$suffSTrue
    gradCompTrue[,vv,mm] <- aux$gradTrue
    
    ## Approximating the sufficient statistics and gradients:
    for (ii in 1:I1) {
      for (jj in 1:J1) {    
      
        aux      <- setBlocks(bInn[ii], bOut[jj], SPACE[vv])
        blockInn <- aux$blockInn
        blockOut <- aux$blockOut
        
        for (ll in 1:L1) {
        
          print(paste("(ii,jj,ll): (", ii, ",", jj, ",", ll ,")", sep=''))
          
 
          if ((ii == 1 && jj == 1) || (FILT[ll] == 1 && jj == 1)) {
          
            # Standard forward smoothing:
            aux <- suffCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 0, thetaTrue, m0, C0, y, nCores)
            suffHFs[,,ii,jj,vv,ll,mm]        <- aux$suffHEst
            suffQFs[,ii,jj,vv,ll,mm]         <- aux$suffQEst
            suffRFs[ii,jj,vv,ll,mm]          <- aux$suffREst
            suffSFs[ii,jj,vv,ll,mm]          <- aux$suffSEst
            gradCompFs[,ii,jj,vv,ll,mm]      <- aux$gradEst
            
            # Standard backward sampling:
            aux <- suffCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 0, thetaTrue, m0, C0, y, nCores)
            suffHBs[,,ii,jj,vv,ll,mm]        <- aux$suffHEst
            suffQBs[,ii,jj,vv,ll,mm]         <- aux$suffQEst
            suffRBs[ii,jj,vv,ll,mm]          <- aux$suffREst
            suffSBs[ii,jj,vv,ll,mm]          <- aux$suffSEst
            gradCompBs[,ii,jj,vv,ll,mm]      <- aux$gradEst
            
          } else if (FILT[ll] == 1 && jj > 1) {
          
            # Standard forward smoothing:
            suffHFs[,,ii,jj,vv,ll,mm]        <- suffHFs[,,ii,1,vv,ll,mm]
            suffQFs[,ii,jj,vv,ll,mm]         <- suffQFs[,ii,1,vv,ll,mm] 
            suffRFs[ii,jj,vv,ll,mm]          <- suffRFs[ii,1,vv,ll,mm] 
            suffSFs[ii,jj,vv,ll,mm]          <- suffSFs[ii,1,vv,ll,mm]
            gradCompFs[,ii,jj,vv,ll,mm]      <- gradCompFs[,ii,1,vv,ll,mm] 
            
            # Standard backward sampling:
            suffHBs[,,ii,jj,vv,ll,mm]        <- suffHBs[,,ii,1,vv,ll,mm]
            suffQBs[,ii,jj,vv,ll,mm]         <- suffQBs[,ii,1,vv,ll,mm] 
            suffRBs[ii,jj,vv,ll,mm]          <- suffRBs[ii,1,vv,ll,mm] 
            suffSBs[ii,jj,vv,ll,mm]          <- suffSBs[ii,1,vv,ll,mm]
            gradCompBs[,ii,jj,vv,ll,mm]      <- gradCompBs[,ii,1,vv,ll,mm] 
            
          } else {
          
            # Standard forward smoothing:
            suffHFs[,,ii,jj,vv,ll,mm]        <- suffHFs[,,1,1,vv,ll,mm]
            suffQFs[,ii,jj,vv,ll,mm]         <- suffQFs[,1,1,vv,ll,mm] 
            suffRFs[ii,jj,vv,ll,mm]          <- suffRFs[1,1,vv,ll,mm] 
            suffSFs[ii,jj,vv,ll,mm]          <- suffSFs[1,1,vv,ll,mm]
            gradCompFs[,ii,jj,vv,ll,mm]      <- gradCompFs[,1,1,vv,ll,mm] 
            
            # Standard backward sampling:
            suffHBs[,,ii,jj,vv,ll,mm]        <- suffHBs[,,1,1,vv,ll,mm]
            suffQBs[,ii,jj,vv,ll,mm]         <- suffQBs[,1,1,vv,ll,mm] 
            suffRBs[ii,jj,vv,ll,mm]          <- suffRBs[1,1,vv,ll,mm] 
            suffSBs[ii,jj,vv,ll,mm]          <- suffSBs[1,1,vv,ll,mm]
            gradCompBs[,ii,jj,vv,ll,mm]      <- gradCompBs[,1,1,vv,ll,mm]
            
          }
          
          # Blocked forward smoothing:
          aux <- suffCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 1, thetaTrue, m0, C0, y, nCores)
          suffHBlockFs[,,ii,jj,vv,ll,mm]   <- aux$suffHEst
          suffQBlockFs[,ii,jj,vv,ll,mm]    <- aux$suffQEst
          suffRBlockFs[ii,jj,vv,ll,mm]     <- aux$suffREst
          suffSBlockFs[ii,jj,vv,ll,mm]     <- aux$suffSEst
          gradCompBlockFs[,ii,jj,vv,ll,mm] <- aux$gradEst
          
          # Blocked backward sampling:
          aux <- suffCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 1, thetaTrue, m0, C0, y, nCores)
          suffHBlockBs[,,ii,jj,vv,ll,mm]   <- aux$suffHEst
          suffQBlockBs[,ii,jj,vv,ll,mm]    <- aux$suffQEst
          suffRBlockBs[ii,jj,vv,ll,mm]     <- aux$suffREst
          suffSBlockBs[ii,jj,vv,ll,mm]     <- aux$suffSEst
          gradCompBlockBs[,ii,jj,vv,ll,mm] <- aux$gradEst
                
        }
      }
    }
  }
  
  ## Saving output: 
  save(
    list  = ls(envir = environment(), all.names = TRUE), 
    file  = file.path(pathToOutput, paste("suff_fixed_", max(SPACE), "_T_", T, "_N_", N, "_M1_", M1, "_prop_", prop, "_05", sep='')),
    envir = environment()
  ) 
}


# 
# ## Issue sourceCpp with the option verbose=TRUE
# ## Then we can see the file path and generated object file, 
# ## under the heading "Building shared library", e.g.
# /tmp/RtmpOtFMHL/sourcecpp_569b542f6368
# /tmp/RtmpOtFMHL/sourcecpp_569b542f6368/sourceCpp_2.so
# 
# 
# pprof --pdf /tmp/RtmpOtFMHL/sourcecpp_569b542f6368/sourceCpp_2.so /home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/blockedSmoothing/linearGaussianHmm/profile_runParticleSmoother.log > /home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/blockedSmoothing/linearGaussianHmm/profiling_output.pdf
# 
# pprof --text /tmp/RtmpOtFMHL/sourcecpp_569b542f6368/sourceCpp_2.so /home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/blockedSmoothing/linearGaussianHmm/profile_runParticleSmoother.log













## ========================================================================= ##
## SIMULATION STUDY II: ESTIMATING STATIC PARAMETERS
## ========================================================================= ##

set.seed(7)

## Model parameters
## ------------------------------------------------------------------------- ##
para <- 0 # type of reparametrisation: 0: unbounded (for gradient/EM algorithms); 1: bounded (for MCMC/SMC)
V    <- 100 # number of vertices/dimension of the state space
T    <- 10 # number of time steps
b    <- 1
d    <- 1
a    <- c(0.5, 0.2)

K    <- length(a)-1
A    <- toeplitzCpp(c(a, rep(0,length=V-length(a))))
B    <- b*diag(V) 
C    <- diag(V)
D    <- d*diag(V)
m0   <- rep(0, V)
C0   <- diag(V)


## Checking stationarity of the latent chain:
if (any(abs(eigen(A, symmetric=TRUE)$values) >= 1)) {
  print("WARNING: Latent process is not stationary!")
}

if (para == 0) {
  thetaTrue <- c(a, log(b), log(d)) # true parametes in the parametrisation used by the algorithm
} else if (para == 1) {
  thetaTrue <- c(a, b, d) # true parametes in the parametrisation used by the algorithm
}


## Simulated data
## ------------------------------------------------------------------------- ##

# DATA <- simulateDataCpp(T, A, B, C, D, m0, C0)
# y    <- DATA$y # simulated observations
# write.table(x=y, file=paste(getwd(), "/data/data_for_mle_simulation_study.dat", sep=''))

y <- data.matrix(read.table(file.path(getwd(), "data/data_for_mle_simulation_study.dat")))
V <- dim(y)[1]
T <- dim(y)[2]
m0   <- rep(0, V)
C0   <- diag(V)


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

# Exact log-likelihood (computed using a custom function implemented in RCpp/Armadillo):
objectiveFunctionCpp <- function(theta) {
  
  A <- toeplitzCpp(c(theta[1:(K+1)], rep(0, times=V-K-1)))
  C <- diag(V)
   
  B <- exp(theta[K+2])*diag(V)
  D <- exp(theta[K+3])*diag(V)

  logLike <- kalmanCpp(A, B, C, D, m0, C0, y)
  return(-logLike)
}

# Exact log-likelihood (computed using Kalman-filtering routines from the R package "FKF"):
library("FKF")
objectiveFunctionFkf <- function(theta) {

  A <- toeplitzCpp(c(theta[1:(K+1)], rep(0, times=V-K-1)))
  C <- diag(V)
  
  B <- exp(theta[K+2])*diag(V)
  D <- exp(theta[K+3])*diag(V)


  logLike <- fkf(m0, C0, matrix(0,V,T), matrix(0,V,T), array(A, c(V,V,1)), array(diag(V), c(V,V,1)), array(B*t(B), c(V,V,1)), array(D*t(D), c(V,V,1)), y)$logLik
  return(-logLike)
}

exactML <- function(thetaInit, F, method="BFGS") {

  nML <- dim(thetaInit)[2]
  argminF <- matrix(NA,K+3,nML)
  minF <- rep(NA, nML)
  for (nn in 1:nML) {
    aux <- optim(thetaInit[,nn], F, method)
    argminF[,nn] <- aux$par
    minF[nn] <- aux$value
  }
  
  kk <- which.min(minF)
  
  return(list(par=argminF[,kk], value=minF[kk], allPars=argminF, allValues=minF))
}

nML <- 5

thetaInit <- thetaTrue
if (para == 1) {
  thetaInit[(K+2):(K+3)] <- log(thetaTrue[(K+2):(K+3)])
}
thetaInitMat <- matrix(thetaTrue + rnorm((K+3)*nML, sd=0.1), K+3, nML)
aux1 <- exactML(thetaInitMat, objectiveFunctionCpp) # numerical optima found when using the custom Kalman-filter implementation
aux2 <- exactML(thetaInitMat, objectiveFunctionFkf) # numerical optima found when using the Kalman-filter implementation from the R library FKF
thetaML <- aux1$par
if (para == 1) {
  thetaML[(K+2):(K+3)] <- exp(thetaML[(K+2):(K+3)])
}


## Parameters of the simulation study
## ------------------------------------------------------------------------- ##

bInn  <- c(3) # diameter of the blocks that partition the space
bOut  <- c(2) # # size by which each block is extended in each direction to create the enlarged blocks
N     <- 500 # number of filtering particles
M     <- 200 # number of smoothing paths (if M>=N we compute the sum in Equation 11 exactly)

FILT  <- c(1, 2) # types of filtering schemes to test: 
# 0: standard particle filter; 
# 1: blocked particle filter; 
# 2: IID samples from exact filter.

L1 <- length(FILT) # number of different combinations of filtering and backward-sampling schemes to test
I1 <- length(bInn) # number of different inner block configurations
J1 <- length(bOut) # number of different outer block configurations

M1 <- 50 # number of replicates

## Parameters for maximum-likelihood estimation
## ------------------------------------------------------------------------- ##

G         <- 1000 # number of iterations of offline (stochastic) EM or (stochastic) gradient ascent algorithms.
stepSizes <- (1:G)^(-0.8) # step size for (stochastic) gradient ascent algorithms.

## Parameters for simulation studies
## ------------------------------------------------------------------------- ##

MLE  <- c(0, 1) # types of algorithms used for approximating the MLE:
# 0: offline stochastic gradient ascent
# 1: offline stochastic EM
K1 <- length(MLE)

useFS <- TRUE
useBS <- FALSE


## Running the simulation study
## ------------------------------------------------------------------------- ##

thetaMleTrue <- array(NA, c(K+3, G, K1, M1))

if (useFS) {
  thetaMleFs       <- array(NA, c(K+3, G, L1, K1, M1))
  thetaMleBlockFs  <- array(NA, c(K+3, G, L1, K1, M1))
}

if (useBS) {
  thetaMleBs       <- array(NA, c(K+3, G, L1, K1, M1))
  thetaMleBlockBs  <- array(NA, c(K+3, G, L1, K1, M1))
}

aux <- setBlocks(bInn, bOut, V)
blockInn <- aux$blockInn # boundaries of the actual blocks, i.e. of $K$ in the paper
blockOut <- aux$blockOut # boundaries of the enlarged blocks, i.e. of $\overline{K}$ in the paper

for (mm in 1:M1) {

  print(paste("mm: ", mm))

  thetaInit <- rnorm(4,mean=0,sd=1)
  thetaInit[1:2] <- thetaInit[1:2] / (2*sum(abs(thetaInit[1:2])))
  
  thetaMleTrue[,,1,mm] <- runOgaCpp(thetaInit, stepSizes, m0, C0, y, nCores)$theta
  thetaMleTrue[,,2,mm] <- runOemCpp(thetaInit, stepSizes, m0, C0, y, nCores)$theta
  
  for (ll in 1:L1) {
    if (useFS) {
      print("SGA-FS:")
      thetaMleFs[,,ll,1,mm] <- runOsgaCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta
      print("SEM-FS:")
      thetaMleFs[,,ll,2,mm] <- runOsemCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta
    }
    if (useBS) {
      print("SGA-BS:")
      thetaMleBs[,,ll,1,mm] <- runOsgaCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta
      print("SEM-BS:")
      thetaMleBs[,,ll,2,mm] <- runOsemCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 0, thetaInit, stepSizes, m0, C0, y, nCores)$theta
    }
    
    if (useFS) {
      print("SGA-BlockFS:")
      thetaMleBlockFs[,,ll,1,mm] <- runOsgaCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta
      print("SEM-BlockFS:")
      thetaMleBlockFs[,,ll,2,mm] <- runOsemCpp(N, N, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta
    }
    if (useBS) {
      print("SGA-BlockBS:")
      thetaMleBlockBs[,,ll,1,mm] <- runOsgaCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta
      print("SEM-BlockBS:")
      thetaMleBlockBs[,,ll,2,mm] <- runOsemCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 1, thetaInit, stepSizes, m0, C0, y, nCores)$theta
    }
  }

  ## Saving output: 
  save(
    list  = ls(envir = environment(), all.names = TRUE), 
    file  = paste("osem_osga_new_T_", T, "_V_", V, "_N_", N, "_M_", M, "_M1_", M1, "_G_", G, "_07", sep=''),
    envir = environment()
  ) 
}





















## ========================================================================= ##
## SIMULATION STUDY III: BLOCKED ONLINE STOCHASTIC GRADIENT ASCENT
## ========================================================================= ##

M1 <- 50
N <- 500


bInn <- 3
bOut <- 2
aux <- setBlocks(bInn, bOut, V)
blockInn <- aux$blockInn
blockOut <- aux$blockOut
nBlocks <- dim(blockInn)[2] # TODO: check this!


FILT <- c(0,0,1,1) ############################## c(0,1,2) ### WARNING: FILT == 2 is not implemented yet
BACK <- c(0,1,0,1)
L1   <- length(FILT)

prop <- 1

alpha <- 0.8
# T0 <- ceiling(T/10) # number of iterations in which the step sizes are kept constant
# stepSizesTime  <- c(rep(3^(-alpha), times=T0+2), (3:(T-T0))^(-alpha))

stepSizesTime  <- (1:T)^(-alpha)

# padding <- 1
# stepSizesBlock <- (1:(T*nBlocks))^(-alpha)


## Running the simulation study
## ------------------------------------------------------------------------- ##

ESTIMATE_THETA <- TRUE # should theta be estimated
NORMALISE_BY_L2_NORM <- TRUE # should the gradient approximations be normalised by the L2 norm

thetaTime  <- array(NA, c(K+3, T, L1, M1))
# thetaBlock <- array(NA, c(K+3, T*nBlocks, L1, M1))

for (mm in 1:M1) {

  print(mm)
  DATA <- simulateDataCpp(T, A, B, C, D, m0, C0)
  y <- DATA$y # simulated observations
  thetaInit      <- rnorm(4, mean=0, sd=0.1)
  thetaInit[1:2] <- thetaInit[1:2] / (2*sum(abs(thetaInit[1:2])))
  
  
  
  for (ll in 1:L1) {
    print(ll)
#     thetaBlock[,,ll,mm] <- runOnlineStochasticGradientAscentCpp(N, blockInn, blockOut, padding, FILT[ll], prop, thetaInit, stepSizesBlock, m0, C0, y, nCores)$theta
    thetaTime[,,ll,mm]  <- runOnlineStochasticGradientAscentEnlargedCpp(N, NORMALISE_BY_L2_NORM, ESTIMATE_THETA, blockInn, blockOut, FILT[ll], BACK[ll], prop, thetaInit, stepSizesTime, m0, C0, y, nCores)$theta

  }

  save.image(file=paste("online_fixed_step_size_T_", T, "_V_", V, "_N_", N, "_M1_", M1, sep=""))
}


# load("online_fixed_step_size_T_500_V_100_N_500_M1_50")
ltyPlot <- c(2,2,1,1)
op <- par(mfrow=c(K+3,1))
for (k in 1:(K+3)) {

  X <- thetaTime[k,,1,1:mm]
  plot(apply(X, 1, mean), type='l', ylim=c(0,0.5), col="white", ylab=paste("RMSE of Parameter ", k, sep=''))

  for (ll in 1:L1) {
    RMSE <- sqrt(apply((thetaTime[k,,ll,1:mm] - thetaTrue[k])^2, 1, mean))
    lines(RMSE, type='l', col=colPlot[ll], lty=ltyPlot[ll])
  }
  
  legend("topright", legend=c("standard PF + standard FS", "standard PF + blocked FS","blocked PF + standard FS","blocked PF + blocked FS"), col=colPlot[1:L1], lty=ltyPlot[1:L1], bty='n')
}

par(op)





