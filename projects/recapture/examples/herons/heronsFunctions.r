## Some additional R functions for use with the integrated model 
## for the herons

###############################################################################
## Returns a number of (known) auxiliary model variables.
###############################################################################
getAuxiliaryModelParameters <- function(modelType, nAgeGroups, nLevels, miscParameters) {
  
  ## ----------------------------------------------------------------------- ##
  ## Loading Data
  ## ----------------------------------------------------------------------- ##
  
  ## Count data (from time 1 to time nObservationsCount):
  countAux <- data.matrix(read.table(file.path(miscParameters$pathToData, "count.dat")))
  count    <- countAux[,2]

  ## Ring-recovery data (from time 1 <= t1 to time t2 < nObservationsCount, 
  ## where t1 = t1Ring - t1Count + 1 and t2 = t2Ring - t1Count +1):
  ringRecoveryAux <- data.matrix(read.table(file.path(miscParameters$pathToData, "ringRecovery.dat")))
  ringRecovery    <- ringRecoveryAux[,c(1:(dim(ringRecoveryAux)[2]-1))]

  t1Count <- countAux[1,1]
  t2Count <- countAux[dim(countAux)[1],1]
  t1Ring  <- ringRecoveryAux[1,dim(ringRecoveryAux)[2]]
  t2Ring  <- ringRecoveryAux[dim(ringRecoveryAux)[1],dim(ringRecoveryAux)[2]]
  
  
  ## ----------------------------------------------------------------------- ##
  ## Model Index
  ## ----------------------------------------------------------------------- ##
  
  initialDistributionType <- 1 # prior on the initial latent state; 0: Poisson; 1: negative binomial; 2: discret uniform
  observationEquationType <- 1 # type of observation equation; 0: Poisson; 1: negative binomial
  
  nModelIndependentParameters <- 2*(1+nAgeGroups)
  if (initialDistributionType == 0) # i.e. if the prior on the initial state is Poisson
  {
    nModelIndependentParameters <- nModelIndependentParameters + 2
  } 
  if (observationEquationType == 1) # i.e. if the observation equation uses a negative-binomial distribution
  {
    nModelIndependentParameters <- nModelIndependentParameters + 1
  }

  ## Number of unknown parameters, i.e. the length of the vector theta:
 
  if (modelType == 0) {
    dimTheta <- nModelIndependentParameters + 1
    dimLatentVariable <- nAgeGroups
  } else if (modelType == 1) {
    dimTheta <- nModelIndependentParameters + 2
    dimLatentVariable <- nAgeGroups
  } else if (modelType == 2) {
    dimTheta <- nModelIndependentParameters + 2
    dimLatentVariable <- nAgeGroups
  } else if (modelType == 3) {
    dimTheta <- nModelIndependentParameters + 2*(nLevels-1) + 2
    dimLatentVariable <- nAgeGroups
  } else if (modelType == 4) {
    dimTheta <- nModelIndependentParameters + 2*(nLevels-1) + 1
    dimLatentVariable <- nAgeGroups
  } else if (modelType == 5) {
    dimTheta <- nModelIndependentParameters + nLevels + nLevels^2
    dimLatentVariable <- nAgeGroups + 1
  } 
  
  
  fDaysCovarAux <- data.matrix(read.table(file.path(pathToCovar, "fDaysCovar.dat")))
  fDaysCovar    <- fDaysCovarAux[,2]
  
  
  ## ----------------------------------------------------------------------- ##
  ## Loading Known Covariates
  ## ----------------------------------------------------------------------- ##
  
  ## fDays (from time 0 to time nObservationsCount-1):
  fDaysCovarAux <- data.matrix(read.table(file.path(miscParameters$pathToCovar, "fDaysCovar.dat")))
  fDaysCovar    <- fDaysCovarAux[,2]
  ## timeNorm (from time 1 to time nObservationsCount-1):
  timeNormCovarAux <- data.matrix(read.table(file.path(miscParameters$pathToCovar, "timeNormCovar.dat")))
  timeNormCovar <- timeNormCovarAux[,2]
  
  ## ----------------------------------------------------------------------- ##
  ## Specifying Hyperparameters
  ## ----------------------------------------------------------------------- ##
  
  # minimum and maximum of the support of the first state if 
  # we assume a discrete uniform distribution
  minHyperInit0 <- 0
  minHyperInit1 <- 0
  maxHyperInit0 <- 2000
  maxHyperInit1 <- 6000
  
  # parameters for a potential negative binomial prior on the 
  # initial latent state
  mean0 <- 5000 / 5 # assuming that the herons live for around 5 years
  mean1 <- 5000 * (1 - (nAgeGroups-1) / 5)
  negativeBinomialProbHyperInit0 <- 1/100
  negativeBinomialProbHyperInit1 <- 1/100
  negativeBinomialSizeHyperInit0 <- negativeBinomialProbHyperInit0 / (1 -negativeBinomialProbHyperInit0) * mean0
  negativeBinomialSizeHyperInit1 <- negativeBinomialProbHyperInit1 / (1 -negativeBinomialProbHyperInit1) * mean1
  
  meanHyper <- rep(0, dimTheta)
  sdHyper   <- rep(1, dimTheta)
  
  # Parameters $\alpha_{0}, \beta_{0}:
  meanHyper[1:2] <- 0
  sdHyper[1:2]   <- 1
  
  # Parameters $\alpha_{1:A}, \beta_{1:A}:
  meanHyper[3:(2*(nAgeGroups+1))] <- 0
  sdHyper[3:(2*(nAgeGroups+1))]   <- 1
  
  if (observationEquationType == 1) {
    # Parameter $\omega$:
    meanHyper[2*(nAgeGroups+1)+1] <- -4
    sdHyper[2*(nAgeGroups+1)+1]   <- 2
  }
  if (initialDistributionType == 0) {
    # Parameters $\delta_{0:1}$:
    meanHyper[(nModelIndependentParameters-1):nModelIndependentParameters] <- c(6, 7)
    sdHyper[(nModelIndependentParameters-1):nModelIndependentParameters]   <- c(2, 2)
  }
  
  if (modelType == 0) {
    # parameter $\psi$
    meanHyper[nModelIndependentParameters+1] <- 0
    sdHyper[nModelIndependentParameters+1]   <- 1
  } else if (modelType == 1) {
    # Parameters $\gamma_{0:1}$:
    meanHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+2)] <- c(0,0)
    sdHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+2)]   <- c(1,1)
  } else if (modelType == 2) {
    # Parameters $\varepsilon_{0:1}$:
    meanHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+2)] <- c(0,0)
    sdHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+2)]   <- c(1,1)
  } else if (modelType == 3) {
    # Parameters $\zeta_{0:K}$:
    meanHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+1+nLevels-1)] <- 0
    sdHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+1+nLevels-1)]   <- 1
    # Parameters $\eta_{1:K}$:
    meanHyper[(nModelIndependentParameters+1+1+nLevels-1):(nModelIndependentParameters+1+2*(nLevels-1)+1)] <- 0
    sdHyper[(nModelIndependentParameters+1+1+nLevels-1):(nModelIndependentParameters+1+2*(nLevels-1)+1)]   <- 1
  } else if (modelType == 4) {
    # Parameters $\zeta_{0:K}$:
    meanHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+1+nLevels-1)] <- 0
    sdHyper[(nModelIndependentParameters+1):(nModelIndependentParameters+1+nLevels-1)]   <- 1
    # Parameters $\eta_{1:K}$:
    meanHyper[(nModelIndependentParameters+1+1+nLevels-1):(nModelIndependentParameters+1+2*(nLevels-1))] <- 7
    sdHyper[(nModelIndependentParameters+1+1+nLevels-1):(nModelIndependentParameters+1+2*(nLevels-1))]   <- 3
  } else if (modelType == 5) {
    ## Empty! (i.e. means are 0 and sd's are 1)
  }

  ## Hyperparameters to be passed to C++
   
  # Collecting all the hyper parameters in a vector:
  hyperParameters <- c(
    modelType,
    t1Count, t2Count, t1Ring, t2Ring,
    dimTheta,
    meanHyper, sdHyper, 
    nAgeGroups,
    nLevels,
    initialDistributionType,
    observationEquationType,
    minHyperInit0,
    minHyperInit1,
    maxHyperInit0,
    maxHyperInit1,
    negativeBinomialSizeHyperInit0,
    negativeBinomialSizeHyperInit1,
    negativeBinomialProbHyperInit0,
    negativeBinomialProbHyperInit1,
    timeNormCovar, fDaysCovar,
    count
  )

  
  ## ----------------------------------------------------------------------- ##
  ## Names of the Parameters (only used for graphical output)
  ## ----------------------------------------------------------------------- ##
  
  thetaNames <- c("alpha_0", "beta_0")
  for (ii in 1:nAgeGroups) {
    thetaNames <- c(thetaNames, paste("alpha_", ii, sep=''))
  }
  for (ii in 1:nAgeGroups) {
    thetaNames <- c(thetaNames, paste("beta_", ii, sep=''))
  }
  if (observationEquationType == 1) {
    thetaNames <- c(thetaNames, "omega")
  }
  if (initialDistributionType == 0) {
    thetaNames <- c(thetaNames, "delta_0", "delta_1")
  }
  
  if (modelType == 1) {
    thetaNames <- c(thetaNames, "gamma_0", "gamma_1")
  } else if (modelType == 2) {
    thetaNames <- c(thetaNames, "epsilon_0", "epsilon_1")
  } else if (modelType == 3) {
    for (ii in 1:nLevels) {
      thetaNames <- c(thetaNames, paste("zeta_", ii, sep=''))
    }
    for (ii in 1:(nLevels)) {
      thetaNames <- c(thetaNames, paste("eta_", ii, sep=''))
    }
  } else if (modelType == 4) {
    for (ii in 1:nLevels) {
      thetaNames <- c(thetaNames, paste("zeta_", ii, sep=''))
    }
    for (ii in 1:(nLevels-1)) {
      thetaNames <- c(thetaNames, paste("eta_", ii, sep=''))
    }
  } else if (modelType == 5) {
    for (ii in 1:nLevels) {
      thetaNames <- c(thetaNames, paste("zeta_", ii, sep=''))
    }
    for (ii in 1:nLevels) {
      for (jj in 1:nLevels) {
        thetaNames <- c(thetaNames, paste("varpi_{", ii, ",", jj, "}", sep=''))
      }
    }
  } else if (modelType == 0) {
    thetaNames <- c(thetaNames, "psi")
  }
  
  
  ## ----------------------------------------------------------------------- ##
  ## Support of the Unknown Parameters
  ## ----------------------------------------------------------------------- ##
  
  supportMin <- rep(-Inf, times=dimTheta) # minimum
  supportMax <- rep( Inf, times=dimTheta) # maximum
  support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

  return(list(modelType=modelType, dimTheta=dimTheta, dimLatentVariable=dimLatentVariable, hyperParameters=hyperParameters, support=support, thetaNames=thetaNames, meanHyper=meanHyper, sdHyper=sdHyper, timeNormCovar=timeNormCovar, fDaysCovar=fDaysCovar,
  nObservationsCount=t2Count-t1Count+1, nObservationsRing=t2Ring-t1Ring+1, count=count, ringRecovery=ringRecovery, t1Count=t1Count, t2Count=t2Count, t1Ring=t1Ring, t2Ring=t2Ring, countTrue=numeric(0), thetaTrue=numeric(0), rhoTrue=numeric(0), lambdaTrue=numeric(0), phiTrue=numeric(0)))
}


