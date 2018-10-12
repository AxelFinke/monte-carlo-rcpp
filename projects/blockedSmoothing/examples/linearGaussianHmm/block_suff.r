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
## when estimating certain sufficient statistics.


## ========================================================================= ##
## SETUP
## ========================================================================= ##

rm(list = ls())
# set.seed(123)

DEBUG <- TRUE

if (DEBUG) {
  pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
  pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"
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

para <- 0 # type of reparametrisation: 0: unbounded (for gradient/EM algorithms); 1: bounded (for MCMC/SMC)
hyperParameters <- c(1,1,1,1)

b    <- 1
d    <- 1
a    <- c(0.5, 0.2)
K    <- length(a)-1

if (para == 0) {
  thetaTrue <- c(a, log(b), log(d)) # true parametes in the parametrisation used by the algorithm
} else if (para == 1) {
  thetaTrue <- c(a, b, d) # true parametes in the parametrisation used by the algorithm
}

prop <- 1  # type of SMC proposal kernel to be used; 0: bootstrap/prior; 1: conditionally locally optimal
H    <- 1 # effective sample size-based adaptive resampling threshold
# In high dimensions, it may be sensible to resample at every step; 
# otherwise, unless the number of particles scales exponentially in the dimension of the state space, 
# there is a risk that the effective sample size will be overestimated
# because all particles miss important regions of the state space;


## ========================================================================= ##
## SIMULATION STUDY: ESTIMATING SUFFICIENT STATISTICS & GRADIENTS
## AS A FUNCTION OF THE DIMENSION OF THE STATE SPACE
## ========================================================================= ##

## Parameters of the simulation study
## ------------------------------------------------------------------------- ##

bInn  <- c(1,3,20) # inner block sizes
bOut  <- c(0,1) # size of the extension of each inner block on each side

SPACE <- c(5,10,25,50,75,100,200,300,400,500) ##seq(from=50, to=100, by=2) # model dimensions to test
T     <- 20 # number of time steps
N     <- 500 # number of filtering particles
M     <- 100 # number of smoothing paths (if M>=N we compute the sum in Equation 11 exactly)

FILT  <- c(0, 1, 2) # types of filtering schemes to test: 
# 0: standard particle filter; 
# 1: blocked particle filter; 
# 2: IID samples from exact filter.

L1 <- length(FILT) # number of different combinations of filtering and backward-sampling schemes to test
I1 <- length(bInn) # number of different inner block configurations
J1 <- length(bOut) # number of different outer block configurations

V1 <- length(SPACE)
M1 <- 1 # number of replicates


## Running the simulation study
## ------------------------------------------------------------------------- ##

# True sufficient statistics:
suffHTrue         <- array(NA, c(K+1, K+1, V1, M1))
suffQTrue         <- array(NA, c(K+1, V1, M1))
suffRTrue         <- matrix(NA, V1, M1)
suffSTrue         <- matrix(NA, V1, M1)
gradTrue          <- array(NA, c(K+3, V1, M1))

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
gradBlockFs       <- array(NA, c(K+3, I1, J1, V1,L1, M1))

# Blocked backward sampling:
suffHBlockBs      <- array(NA, c(K+1, K+1, I1, J1, V1, L1, M1))
suffQBlockBs      <- array(NA, c(K+1, I1, J1, V1, L1, M1))
suffRBlockBs      <- array(NA, c(I1, J1, V1, L1, M1))
suffSBlockBs      <- array(NA, c(I1, J1, V1, L1, M1))
gradBlockBs       <- array(NA, c(K+3, I1, J1, V1,L1, M1))

mm <- 1 ######

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
  
  ## Calculating exact sufficient statistics and gradients
  
  print("exact smoothing")
  aux <- suffExactCpp(thetaTrue, m0, C0, y, nCores)
  suffHTrue[,,vv,mm]   <- aux$suffHTrue
  suffQTrue[,vv,mm]    <- aux$suffQTrue
  suffRTrue[vv,mm]     <- aux$suffRTrue
  suffSTrue[vv,mm]     <- aux$suffSTrue
  gradTrue[,vv,mm]     <- aux$gradTrue
  

  ## Approximating the sufficient statistics and gradients
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
        gradBlockFs[,ii,jj,vv,ll,mm]     <- aux$gradEst
        
        # Blocked backward sampling:
        aux <- suffCpp(N, M, H, blockInn, blockOut, FILT[ll], prop, 1, thetaTrue, m0, C0, y, nCores)
        suffHBlockBs[,,ii,jj,vv,ll,mm]   <- aux$suffHEst
        suffQBlockBs[,ii,jj,vv,ll,mm]    <- aux$suffQEst
        suffRBlockBs[ii,jj,vv,ll,mm]     <- aux$suffREst
        suffSBlockBs[ii,jj,vv,ll,mm]     <- aux$suffSEst
        gradBlockBs[,ii,jj,vv,ll,mm]     <- aux$gradEst
              
      }
    }
  }
}

## Saving output: 
save(
  list  = ls(envir = environment(), all.names = TRUE), 
  file  = file.path(pathToResults, paste("suff_maxV_", max(SPACE), "_T_", T, "_N_", N, "_M1_", M1, "_prop_", prop, "_task_id_", task_id, sep='')),
  envir = environment()
) 

