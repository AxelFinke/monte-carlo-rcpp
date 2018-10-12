## Conventional vs. random-walk CSMC algorithms
## in a simple multivariate linear-Gaussian state-space model


### TODO: change dimensions to c(1,2,5,10,20,50)
### 4628450035255489

## ========================================================================= ##
## SETUP
## ========================================================================= ##

## ------------------------------------------------------------------------- ##
## Directories
## ------------------------------------------------------------------------- ##

rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"
exampleName       <- "linear"
projectName       <- "ensemble"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))

RUN_ACCEPTANCE_RATES  <- FALSE
RUN_AUTOCORRELATIONS  <- FALSE

PLOT_ACCEPTANCE_RATES <- FALSE
PLOT_AUTOCORRELATIONS <- FALSE

## ------------------------------------------------------------------------- ##
## Global model parameters
## ------------------------------------------------------------------------- ##

nOffDiagonals <- 0 # number of non-zero off-diagonals on each side of the main diagonal of A
dimTheta <- nOffDiagonals + 3 # length of the parameter vector

# Support of the unknown parameters:
supportMin <- c(-1, 0, 0) # minimum
supportMax <- c( 1, Inf, Inf) # maximum
support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

# Inverse-gamma prior on b:
shapeHyperB <- 1 # shape parameter
scaleHyperB <- 0.5 # scale parameter

# Inverse-gamma prior on d:
shapeHyperD <- 1 # shape parameter
scaleHyperD <- 0.5 # scale parameter

# Collecting all the hyper parameters in a vector:
# hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)

nObservations <- 25 # number of time steps/observations
thetaTrue     <- c(1, 1, 1) # "true" parameter values used for generating the data
thetaNames    <- c("a", "b", "d") # names of the parameters


## ------------------------------------------------------------------------- ##
## Global algorithm parameters
## ------------------------------------------------------------------------- ##

## Parameters for the model parameter updates
ESTIMATE_THETA <- FALSE 
nThetaUpdates  <- 1
useNonCentredParametrisation <- FALSE
nonCentringProbability <- 0

## Other parameters
kern <- 0
useGradients <- FALSE
fixedLagSmoothingOrder <- 0
essResamplingThreshold <- 1.0
resampleType <- 0 # 0: multinomial; 1: systematic
nSteps <- nObservations # number of SMC steps

DIMENSIONS <- c(1,2,5,10,20,50) 
BACK  <- c(0,0,1,1) # type of backward simulation scheme: 0: none; 1: backward; 2: ancestor
PROP  <- c(0,6,0,6) # lower-level proposals (PROP must have the same length as LOWER)
LOWER <- c(0,0,0,0) # type of "lower-level (0: SMC; 1: original EHMM; 2: alternative EHMM; 3: exact)
LOCAL <- c(0,0,0,0) # type of (potentially) local updates for the particles and parent indices in the alternative EHMM approach

LL <- length(LOWER) # number of lower-level sampler configurations to test
DD <- length(DIMENSIONS)


## ========================================================================= ##
## SIMULATION STUDIES
## ========================================================================= ##

## ------------------------------------------------------------------------- ##
## Convergence of the distribution of the particle lineages
## ------------------------------------------------------------------------- ##

if (RUN_ACCEPTANCE_RATES) {

  nIterations  <- 100
  nSimulations <- 200
  nParticles   <- 31

  initialiseStatesFromStationarity <- TRUE # should the CSMC-type algorithms initialise the state sequence from stationarity?
  storeParentIndices   <- TRUE # should the CSMC-type algorithms all their parent indices?
  storeParticleIndices <- TRUE # should the CSMC-type algorithms store the indices of the (input and output) reference particle lineage?
  TIMES      <- c(0) # c(0, nObservations-1) # time steps at which components of the latent states are to be stored
  COMPONENTS <- c(0) # rep(0, times=length(TIMES)) # components which are to be stored

  MM <- nSimulations  # number of independent replicates

  # outputStates  <- array(NA, c(length(TIMES), nIterations, LL, DD, MM))
  if (storeParentIndices) {
    outputParentIndices <- array(NA, c(nParticles, nSteps-1, nIterations, LL, DD, MM))
  }
  if (storeParticleIndices) {
    outputParticleIndicesIn  <- array(NA, c(nSteps, nIterations, LL, DD, MM))
    outputParticleIndicesOut <- array(NA, c(nSteps, nIterations, LL, DD, MM))
  }

  for (mm in 1:MM) {
  
    print(paste("RUN_ACCEPTANCE_RATES; mm: ", mm, sep=''))

    for (dd in 1:DD) { 
    
      dimX <- DIMENSIONS[dd]
      dimY <- dimX
      
      smcParameters         <- rep(sqrt(1.0/dimX), times=nObservations); # additional parameters for the SMC algorithms
      ensembleOldParameters <- c(rep(0, dimX), 1.0/dimX, 0.5); # additional parameters for the original EHMM algorithms
      ensembleNewParameters <- c(sqrt(1.0/dimX), sqrt(1-1.0/dimX), 1.0, 1.0) # additional parameters for the alternative EHMM algorithms: c("scale for random-walk proposals", "correlation parameter for autoregressive updates)
      rwmhSd                <- sqrt(rep(1, times=dimTheta)/(100*dimTheta*dimX*nObservations))
      
      hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)
      DATA            <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
      observations    <- DATA$y # simulated observations
      
      for (ll in 1:LL) {
    
        aux <- runMcmcCpp(1, LOWER[ll], dimTheta, hyperparameters, support, observations, nIterations, kern, useGradients, rwmhSd, fixedLagSmoothingOrder, PROP[ll], resampleType, nThetaUpdates, nSteps, nParticles, 0, essResamplingThreshold, BACK[ll], useNonCentredParametrisation, nonCentringProbability, smcParameters, ensembleOldParameters, ensembleNewParameters, LOCAL[ll], ESTIMATE_THETA, thetaTrue, TIMES, COMPONENTS,
        initialiseStatesFromStationarity, storeParentIndices, storeParticleIndices, nCores)
        
  #       outputStates[,,ll,dd,mm] <- matrix(unlist(aux$states), length(TIMES), nIterations)
        
        if (LOWER[ll] != 3) {
          if (storeParentIndices && LOWER[ll] != 1) {
            outputParentIndices[,,,ll,dd,mm] <- array(unlist(aux$parentIndices), c(nParticles, nSteps-1, nIterations))
          }
          if (storeParticleIndices) {
            outputParticleIndicesIn[,,ll,dd,mm]  <- matrix(unlist(aux$particleIndicesIn), nSteps, nIterations)
            outputParticleIndicesOut[,,ll,dd,mm] <- matrix(unlist(aux$particleIndicesOut), nSteps, nIterations)
          }
        }
        
  #       rm(aux)
      }
    }
    
    save(
      list  = ls(envir = environment(), all.names = TRUE), 
      file  = file.path(pathToResults, paste("ACCEPTANCE_RATES_nIterations", nIterations, "nObservations", nObservations, "nSimulations", nSimulations, "nParticles", nParticles, "estimateTheta", ESTIMATE_THETA, sep='_')),
      envir = environment()
    ) 
  }

} 


## ------------------------------------------------------------------------- ##
## Autocorrelation of the state-component estimates
## ------------------------------------------------------------------------- ##

if (RUN_AUTOCORRELATIONS) {

  nIterations  <- 10000 ## NOTE: we need to make this much larger because we need to "speed" up the algorithm linearly in the dimension
  nSimulations <- 100
  nParticles   <- 31

  initialiseStatesFromStationarity <- TRUE # should the CSMC-type algorithms initialise the state sequence from stationarity?
  storeParentIndices   <- FALSE # should the CSMC-type algorithms all their parent indices?
  storeParticleIndices <- FALSE # should the CSMC-type algorithms store the indices of the (input and output) reference particle lineage?
  TIMES      <- c(0:(nObservations-1)) # time steps at which components of the latent states are to be stored
  COMPONENTS <- rep(0, times=length(TIMES)) # components which are to be stored

  MM <- nSimulations  # number of independent replicates
  
  outputStates          <- array(NA, c(length(TIMES), nIterations, LL, DD, MM))
  marginalObservations  <- matrix(NA, nObservations, MM)
  smoothedMeansTrue     <- matrix(NA, nObservations, MM)
  smoothedVariancesTrue <- matrix(NA, nObservations, MM)

  for (mm in 1:MM) { 
  
    hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, max(DIMENSIONS), max(DIMENSIONS), nOffDiagonals)
    DATA <- simulateDataCpp(nObservations, hyperparameters, thetaTrue, nCores)
  
    if (max(DIMENSIONS) == 1) {
      observations <- as.numeric(DATA$y) # simulated observations
    } else {
      observations <- DATA$y[1,] # simulated observations
    }
  
    aux <- runKalmanSmootherUnivariateCpp(
      thetaTrue[1], # transition "matrix" of the system equation
      thetaTrue[2]^2, # variance of the system equation
      1, # transition "matrix" of the observation equation
      thetaTrue[3]^2, # variance of the observation equation
      0, # mean of the initial state
      1, # variance of the initial state
      observations
    )

    marginalObservations[,mm]  <- observations
    smoothedMeansTrue[,mm]     <- aux$smoothedMeans
    smoothedVariancesTrue[,mm] <- aux$smoothedVariances
  
    for (dd in 1:DD) {

      dimX <- DIMENSIONS[dd]
      dimY <- dimX
      
      hyperparameters <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)

      if (dimX == 1) {
        observations <- t(as.matrix(DATA$y[1:dimX,])) # simulated observations
      } else {
        observations <- as.matrix(DATA$y[1:dimX,]) # simulated observations
      }
      
      hyperparameters       <- c(shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD, dimX, dimY, nOffDiagonals)
      smcParameters         <- rep(sqrt(1.0/dimX), times=nObservations); # additional parameters for the SMC algorithms
      ensembleOldParameters <- c(rep(0, dimX), 1.0/dimX, 0.5); # additional parameters for the original EHMM algorithms
      ensembleNewParameters <- c(sqrt(1.0/dimX), sqrt(1-1.0/dimX), 1.0, 1.0) # additional parameters for the alternative EHMM algorithms: c("scale for random-walk proposals", "correlation parameter for autoregressive updates)
      rwmhSd                <- sqrt(rep(1, times=dimTheta)/(100*dimTheta*dimX*nObservations))

      for (ll in 1:LL) {
      
        print(paste("dd: ", dd, "; mm: ", mm, "; ll: ", ll, sep=''))
    
        aux <- runMcmcCpp(1, LOWER[ll], dimTheta, hyperparameters, support, observations, nIterations, kern, useGradients, rwmhSd, fixedLagSmoothingOrder, PROP[ll], resampleType, nThetaUpdates, nSteps, nParticles, 0, essResamplingThreshold, BACK[ll], useNonCentredParametrisation, nonCentringProbability, smcParameters, ensembleOldParameters, ensembleNewParameters, LOCAL[ll], ESTIMATE_THETA, thetaTrue, TIMES, COMPONENTS,
        initialiseStatesFromStationarity, storeParentIndices, storeParticleIndices, nCores)
        
        outputStates[,,ll,dd,mm] <- matrix(unlist(aux$states), length(TIMES), nIterations)
        
        rm(aux)
      }
    }
    
    save(
      list  = ls(envir = environment(), all.names = TRUE), 
      file  = file.path(pathToResults, paste("AUTOCORRELATIONS_nIterations", nIterations, "nObservations", nObservations, "nSimulations", nSimulations, "nParticles", nParticles, "estimateTheta", ESTIMATE_THETA, sep='_')),
      envir = environment()
    ) 
  }
}


## ========================================================================= ##
## PLOT THE OUTPUT OF THE SIMULATION STUDIES
## ========================================================================= ##

## ------------------------------------------------------------------------- ##
## Global graphics parameters
## ------------------------------------------------------------------------- ##

library(scales) # provides alpha(); for transparent lines
library(tikzDevice)
options( 
  tikzDocumentDeclaration = c(
    "\\documentclass[11pt]{article}",
    "\\usepackage{amssymb,amsmath,graphicx,mathtools,mathdots,stmaryrd}",
    "\\usepackage{tikz}" 
  )
)

pathToPaperFigures <- "/home/axel/Dropbox/AF-AHT/scaling_analysis_of_local_conditional_smc/fig"
imageType     <- "tex" # either "pdf" or "tex" (the latter uses tikz)

myJitter <- function(x, factor) {
  x + factor * runif(n=length(x), min=-1, max=1)
}

propNames <- c("conventional CSMC", "", "", "", "", "", "random-walk CSMC")
backwardSamplingTypeNames <- c("without backward sampling" ,"with backward sampling", "with ancestor sampling")

jitterFactor <- 0.1
lineColour <- alpha(rgb(0,0,0), 0.15)

#### dimensionCol   <- rev(heat.colors(DD+2))[3:(DD+2)] #####################

library(RColorBrewer)
dimensionCol <- brewer.pal(8, "Blues") ### alpha("darkblue", c(0.15,0.32,0.47,0.62,0.80, 1))
dimensionCol <- dimensionCol[3:8]

# library("RColorBrewer")
# dimensionCol = list(color = brewer.pal(DD, "YlOrRd"))

dimensionLwd   <- rep(1.5, times=DD)
dimensionNames <- c()
for (dd in 1:DD) {
  dimensionNames <- c(dimensionNames, paste("$D =", DIMENSIONS[dd], "$", sep=''))
}

cexLegend <- 0.85
cexAxis   <- 0.85
cexLab    <- 0.85
cexMtext  <- 0.85 ## 0.8
# cexMain   <- cexMtext
cex       <- 0.85

width  <- 3
height <- 3

tckLocalX <- -0.025
mgpLocalX <- c(3, 0.05, 0)
tckLocalY <- -0.025
mgpLocalY <- c(3, 0.3, 0)

## ------------------------------------------------------------------------- ##
## Acceptance rates
## ------------------------------------------------------------------------- ##

if (PLOT_ACCEPTANCE_RATES) {

#   load(file.path(pathToResults, "ACCEPTANCE_RATES_nIterations_10_nObservations_25_nSimulations_100_nParticles_31_estimateTheta_FALSE"))
  load(file.path(pathToResults, "ACCEPTANCE_RATES_nIterations_100_nObservations_25_nSimulations_200_nParticles_31_estimateTheta_FALSE"))
  
  
  acceptanceRates <- array(NA, c(nObservations, LL, DD))
  for (dd in 1:DD) {
    for (ll in 1:LL) {
      for (pp in 1:nObservations) {
        acceptanceRates[pp, ll, dd] <- sum(outputParticleIndicesOut[pp,,ll,dd,1:mm] != 0) / length(c(outputParticleIndicesOut[pp,,ll,dd,1:mm]))
      }
    }
  }

  op <- par(oma=c(0,0,0,0), mar=c(0,0,0,0)) # bottom, left, top, right

  for (ll in 1:LL) {
  
    figureName <- paste("ACCEPTANCE_RATES_PROP", PROP[ll], "BACK", BACK[ll] , sep='_')

    if (imageType == "tex") {
      tikz(file.path(pathToPaperFigures, paste(figureName, ".tex", sep='')),  width=width, height=height)
    } else if (imageType == "pdf") {
      pdf( file.path(pathToPaperFigures, paste(figureName, ".pdf", sep='')),  width=width, height=height)
    }

    plot(1:nSteps, acceptanceRates[,ll,dd], type='l', ylab='', xlab='', ylim=c(0,1), col="white", axes=FALSE, xaxs='i', xaxt="n", yaxt="n", cex=cex, main='') 
     
    for (dd in 1:DD) {
      lines(1:nSteps, acceptanceRates[,ll,dd], type='l', col=dimensionCol[dd], lwd=dimensionLwd[dd])
    }
   
    tickPositionsY <- c(0,1)
    tickPositionsX <- c(seq(from=5, to=nObservations, by=5))
      
#     if (ll == 1 || ll == 3) {
#       mtext(text="Acceptance rate $\\alpha_{n, d}^N(p|\\mathbf{x}_{1:n})$", las=2, side=2, line=0.8, outer=FALSE, cex=cexMtext)
#     } else {
#       mtext(text="Acceptance rate $\\bar{\\alpha}_{n, d}^N(p|\\mathbf{x}_{1:n})$", las=1, side=2, line=0.8, outer=FALSE, cex=cexMtext)
#     }

#     text(x=10, y=0.5, labels="Acceptance rate", srt=90, pos=2, cex=cexMtext)


    if (ll == 1 || ll == 3) {
      mtext(text="Acceptance rate", side=2, line=0.8, outer=FALSE, cex=cexMtext)
    } else {
      mtext(text="Acceptance rate", side=2, line=0.8, outer=FALSE, cex=cexMtext)
    }
    
    
    mtext(text="Time $t$", side=1, line=1.1, outer=FALSE, cex=cexMtext)
    axis(side=2, at=tickPositionsY, labels=tickPositionsY, las=1, cex.axis=cexAxis, mgp=mgpLocalY, tck=tckLocalY)
    axis(side=1, at=tickPositionsX, labels=tickPositionsX, las=1, cex.axis=cexAxis, mgp=mgpLocalX, tck=tckLocalX)

    legend("topleft", legend=dimensionNames, col=dimensionCol, lty=1, cex=cexLegend, bty='n')

    box()
    dev.off()
    
  }
  
  par(op)

  for (i in 1:1000) { dev.off() } #

}

## ------------------------------------------------------------------------- ##
## Lag-d autocorrelations
## ------------------------------------------------------------------------- ##

if (PLOT_AUTOCORRELATIONS) {

 
  load(file.path(pathToResults, "AUTOCORRELATIONS_nIterations_10000_nObservations_25_nSimulations_50_nParticles_11_estimateTheta_FALSE"))
#   
#   load(file.path(pathToResults, "AUTOCORRELATIONS_nIterations_10000_nObservations_25_nSimulations_100_nParticles_31_estimateTheta_FALSE"))
# 
#   
#   ######################
#     library("corrplot")
#     myDim <- DD
#     myAlg <- LL
#     myCor <- matrix(0, nObservations, nObservations)
#     myCov <- matrix(0, nObservations, nObservations)
#     for (mm in 1:MM) {
#       myCor <- myCor + cor(t(outputStates[,2:nIterations,myAlg,myDim,mm] - outputStates[,1:(nIterations-1),myAlg,myDim,mm]))
#       myCov <- myCov + cov(t(outputStates[,2:nIterations,myAlg,myDim,mm] - outputStates[,1:(nIterations-1),myAlg,myDim,mm]))
#     }
#     myCor <- myCor/MM
#     myCov <- DIMENSIONS[myDim]*myCov/MM
#     print(myCov, digits=1, scipen=-2)
#     print(myCor, digits=1, scipen=-2)
#     corrplot(myCor, method="number", number.digits=3, number.cex=0.5, col="black", cl.pos='n')
#   #####################

  lagDAutocorrelations <- array(NA, c(nObservations, LL, DD))
  for (dd in 1:DD) {
    for (ll in 1:LL) {
      for (pp in 1:nObservations) {
        aux <- acf(outputStates[pp,,ll,dd,mm], lag.max=DIMENSIONS[dd], plot=FALSE)$acf
        aux[is.nan(aux)] <- 1
        lagDAutocorrelations[pp, ll, dd] <- aux[length(aux)]
        
      }
    }
  }
  

  

#   
#   rm(outputStates)
#   save(lagDAutocorrelations, file=file.path(pathToResults,"AUTOCORRELATIONS_nIterations_10000_nObservations_25_nSimulations_100_nParticles_31_estimateTheta_FALSE_extract"))
#   
#   save(
#     list  = ls(envir = environment(), all.names = TRUE), 
#     file  = file.path(pathToResults, "AUTOCORRELATIONS_nIterations_10000_nObservations_25_nSimulations_100_nParticles_31_estimateTheta_FALSE_extract"),
#     envir = environment()
#   ) 
#  load(file.path(pathToResults, "AUTOCORRELATIONS_nIterations_10000_nObservations_25_nSimulations_100_nParticles_31_estimateTheta_FALSE_extract"))
# 

  
  

  
  op <- par(oma=c(0,0,0,0), mar=c(0,0,0,0)) # bottom, left, top, right

  for (ll in 1:LL) {
  
    figureName <- paste("AUTOCORRELATIONS_PROP", PROP[ll], "BACK", BACK[ll] , sep='_')

    if (imageType == "tex") {
      tikz(file.path(pathToPaperFigures, paste(figureName, ".tex", sep='')),  width=width, height=height)
    } else if (imageType == "pdf") {
      pdf( file.path(pathToPaperFigures, paste(figureName, ".pdf", sep='')),  width=width, height=height)
    }

    plot(1:nSteps, lagDAutocorrelations[,ll,dd], type='l', ylab='', xlab='', ylim=c(0,1), col="white", axes=FALSE, xaxs='i', xaxt="n", yaxt="n", cex=cex, main='') 
     
    for (dd in 1:DD) {
      lines(1:nSteps, lagDAutocorrelations[,ll,dd], type='l', col=dimensionCol[dd], lwd=dimensionLwd[dd])
    }
   
    tickPositionsY <- c(0,1)
    tickPositionsX <- c(seq(from=5, to=nObservations, by=5))
      
#     if (ll == 1 || ll == 3) {
#       mtext(text="Acceptance rate $\\alpha_{n, d}^N(p|\\mathbf{x}_{1:n})$", las=2, side=2, line=0.8, outer=FALSE, cex=cexMtext)
#     } else {
#       mtext(text="Acceptance rate $\\bar{\\alpha}_{n, d}^N(p|\\mathbf{x}_{1:n})$", las=2, side=2, line=0.8, outer=FALSE, cex=cexMtext)
#     }

   if (ll == 1 || ll == 3) {
      mtext(text="Lag-$D$ autocorrelation", side=2, line=0.8, outer=FALSE, cex=cexMtext)
    } else {
      mtext(text="Lag-$D$ autocorrelation", side=2, line=0.8, outer=FALSE, cex=cexMtext)
    }
    mtext(text="Time $t$", side=1, line=1.1, outer=FALSE, cex=cexMtext)
    axis(side=2, at=tickPositionsY, labels=tickPositionsY, las=1, cex.axis=cexAxis, mgp=mgpLocalY, tck=tckLocalY)
    axis(side=1, at=tickPositionsX, labels=tickPositionsX, las=1, cex.axis=cexAxis, mgp=mgpLocalX, tck=tckLocalX)

    legend("topleft", legend=dimensionNames, col=dimensionCol, lty=1, cex=cexLegend, bty='n')

    box()
    dev.off()
    
  }
  
  par(op)

  for (i in 1:1000) { dev.off() } #

}


## Figures for use in the paper

## Lag-d autocorrelations
grid <- seq(from=-5, to=5, length=1000)
# lagMax <- 10
# op <- par(mfrow=c(DD,LL))
# for (dd in 1:DD) {
#   for (ll in 1:LL) {
#     plot(c(0,1), c(0,1), type='l', col="white", xlim=c(1,nObservations), ylim=c(-0.1,1), xaxs='i', yaxs='i')
#     for (mm in 1:MM) {
#       dthOrderAcf <- rep(NA, times=nObservations)
#       for (pp in 1:nObservations) {
#         aux <- acf(outputStates[pp,,ll,dd,mm], lag.max=lagMax*DIMENSIONS[dd], plot=FALSE)$acf
#         dthOrderAcf[pp] <- aux[length(aux)]
#       }
#       lines(1:nObservations, dthOrderAcf, col=lineColour)
#     }
#   }
# }


## Checking the approximation of the marginals

pp <- nObservations
op <- par(mfrow=c(DD,LL))
for (dd in 1:DD) {
  for (ll in 1:LL) {
    plot(grid, dnorm(grid), col="red", lty=2, type='l', xlim=c(-2,2))
    for (mm in 1:MM) {
      lines(density((outputStates[pp,,ll,dd,mm] - smoothedMeansTrue[pp,mm])/sqrt(smoothedVariancesTrue[pp,mm])), col=lineColour)
    }
  }
}
par(op)



## ------------------------------------------------------------------------- ##
## PLOT THE OUTPUT
## ------------------------------------------------------------------------- ##





# 
# 
# if (PLOT_LINEAGES) {
# 
#   width  <- 3.35
#   height <- 2.5
# 
#   cexLegend <- 0.8
#   cexAxis   <- 0.8
#   cexLab    <- 0.8
#   cexMtext  <- 1.0 ## 0.8
#   # cexMain   <- cexMtext
#   cex       <- 0.8
# 
#   tckLocalX <- -0.025
#   mgpLocalX <- c(3, 0.05, 0)
#   tckLocalY <- -0.025
#   mgpLocalY <- c(3, 0.3, 0)
# 
# 
#   op <- par(oma=c(0,0,0,0), mar=c(0,0,0,0)) # bottom, left, top, right
# 
#   for (dd in 1:DD) {
#     for (ll in 1:LL) {
#     
#       figureName <- paste("DIM", DIMENSIONS[dd], "PROP", PROP[ll], "BACK", BACK[ll] , sep='_')
# 
#       if (imageType == "tex") {
#         tikz(file.path(pathToPaperFigures, paste(figureName, ".tex", sep='')),  width=width, height=height)
#       } else if (imageType == "pdf") {
#         pdf( file.path(pathToPaperFigures, paste(figureName, ".pdf", sep='')),  width=width, height=height)
#       }
#  
# 
#       plot(outputParticleIndicesOut[,1,ll,dd,1], type='l', ylab='', xlab='', ylim=c(0,nParticles-1), col="white", axes=FALSE, xaxs='i', xaxt="n", yaxt="n", cex=cex, main='') #paste(propNames[PROP[ll]+1], "; d = ", DIMENSIONS[dd], sep=''))
#       for (mm in 1:MM) {
#         for (gg in 1:nIterations) {
#           lines(myJitter(outputParticleIndicesOut[,gg,ll,dd,mm],factor=jitterFactor), type='l', col=lineColour)
#         }
#       }
#       
# 
#    
# #       text(x=1,y=nParticles*0.8, labels=paste("$d = ", DIMENSIONS[dd], "$", sep=''), cex=cexMtext, pos=4)
# 
#       if (dd == 1) {
#           mtext(text=propNames[PROP[ll]+1], side=3, line=0.8, outer=FALSE, cex=cexMtext)
#         if (BACK[ll] == 0) {
#           mtext(text="(without backward sampling)", side=3, line=0.1, outer=FALSE, cex=0.75*cexMtext)
#         } else if (BACK[ll] == 1) {
#           mtext(text="(with backward sampling)", side=3, line=0.1, outer=FALSE, cex=0.75*cexMtext)
#         }
#       }
# 
# #     mtext(text=paste("$\\alpha = ", ALPHA[ii], "$", sep=''), side=3, line=0, outer=FALSE, cex=cexMtext)
#       
#     tickPositionsY <- seq(from=0, to=(nParticles-1), by=2)
#     tickPositionsX <- c(seq(from=5, to=nObservations, by=5))
#       
#     if (ll == 1 || ll == 3) {
#     
#       if (dd == floor(DD/2)+1) {
#         if (PLOT_ACCEPTANCE_RATES) {
#           mtext(text="acceptance rate", side=2, line=1.05, outer=FALSE, cex=cexMtext, las=2)
#         } else {
#           mtext(text="$k_p$", side=2, line=1.05, outer=FALSE, cex=cexMtext, las=2)
#         }
#       }
#       axis(side=2, at=tickPositionsY, labels=tickPositionsY, las=1, cex.axis=cexAxis, mgp=mgpLocalY, tck=tckLocalY)
#     } else {
# #       axis(side=2, at=tickPositionsY, labels=tickPositionsY, las=1, cex.axis=cexAxis, mgp=mgpLocalY, tck=tckLocalY)
#       axis(side=2, at=tickPositionsY, labels=rep('', times=length(tickPositionsY)), las=1, cex.axis=cexAxis, mgp=mgpLocalY, tck=tckLocalY) 
#     }
# 
#     if (dd == DD) {
#       mtext(text="$p$", side=1, line=0.9, outer=FALSE, cex=cexMtext)
#       axis(side=1, at=tickPositionsX, labels=tickPositionsX, las=1, cex.axis=cexAxis, mgp=mgpLocalX, tck=tckLocalX)
#     } else {
# #       axis(side=1, at=tickPositionsX, labels=tickPositionsX, las=1, cex.axis=cexAxis, mgp=mgpLocalX, tck=tckLocalX)
#       axis(side=1, at=tickPositionsX, labels=rep('', times=length(tickPositionsX)), las=1, cex.axis=cexAxis, mgp=mgpLocalX, tck=tckLocalX)
#     }
#     
#     if (DIMENSIONS[dd] < 10) {
#       legend("topleft", legend=paste("$\\!\\!d = ", DIMENSIONS[dd], "$", sep=''), bg="white", col="white", lty=1, cex=cexLegend, seg.len=-1, text.width=2.6, bty='o')
#     } else {
#       legend("topleft", legend=paste("$\\!\\!\\!\\!d = ", DIMENSIONS[dd], "$", sep=''), bg="white", col="white", lty=1, cex=cexLegend, seg.len=-1, text.width=2.6, bty='o')
#     }
#     box()
#     dev.off()
#     
#     }
#   }
#   par(op)
# 
#   for (i in 1:1000) { dev.off() } #
# }
# 
