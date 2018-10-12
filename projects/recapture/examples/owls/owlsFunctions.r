## Some additional R functions for use with the integrated model 
## for the little owls

###############################################################################
## Returns a number of (known) auxiliary model variables.
###############################################################################
getAuxiliaryModelParameters <- function(modelType, miscParameters) {

  ## ----------------------------------------------------------------------- ##
  ## Loading Data
  ## ----------------------------------------------------------------------- ##

  ## Fecundity data:
  fecundity <- data.matrix(read.table(file.path(miscParameters$pathToData,"fecundity.dat")))

  ## Count data:
  count <- as.numeric(unlist(read.table(file.path(miscParameters$pathToData, "count.dat"))))
  nObservations <- length(count) # number of (count-data) observations

  ## Capture-recapture data:
  capRecapFemaleFirst <- data.matrix(read.table(file.path(miscParameters$pathToData, "capRecapFemaleFirst.dat")))
  capRecapFemaleAdult <- data.matrix(read.table(file.path(miscParameters$pathToData, "capRecapFemaleAdult.dat"))) 
  capRecapMaleFirst   <- data.matrix(read.table(file.path(miscParameters$pathToData, "capRecapMaleFirst.dat"))) 
  capRecapMaleAdult   <- data.matrix(read.table(file.path(miscParameters$pathToData, "capRecapMaleAdult.dat"))) 

  
  ## ----------------------------------------------------------------------- ##
  ## Model Index
  ## ----------------------------------------------------------------------- ##
  
  # Model to be estimated (needed for model comparison):
  #
  #  0: time-dependent capture probabilities; time-dependent productivity rates
  #  1: ... with $\delta_1 = 0$
  #  2: time-independent capture probabilities; time-dependent productivity rates
  #  3: ... with $\delta_1 = 0$
  #  4: time-dependent capture probabilities; time-independent productivity rates
  #  5: ... with $\delta_1 = 0$
  #  6: ... with $\alpha_3 = 0$
  #  7: ... with $\alpha_3 = \delta_1 = 0$
  #  8: ... with $\alpha_1 = \alpha_3 = 0$
  #  9  ... with $\alpha_1 = \beta_1 = \delta_1 = 0$
  # 10: time-independent capture probabilities; time-independent productivity rates
  # 11: ... with $\delta_1 = 0$
  # 12: ... with $\alpha_3 = 0$
  # 13: ... with $\alpha_3 = \delta_1 = 0$
  # 14: ... with $\alpha_1 = \alpha_3 = 0$
  # 15  ... with $\alpha_1 = \beta_1 = \delta_1 = 0$
  
  ## Number of unknown parameters, i.e. the length of the vector theta:
  if (modelType == 0) {
    dimTheta <- 6 + 2*nObservations 
  } else if (modelType == 1) {
    dimTheta <- 5 + 2*nObservations
  } else if (modelType == 2) {
    dimTheta <- 8 + nObservations    
  } else if (modelType == 3) {
    dimTheta <- 7 + nObservations
  } else if (modelType == 4) {
    dimTheta <- 7 + nObservations
  } else if (modelType == 5) {
    dimTheta <- 6 + nObservations
  } else if (modelType == 6) {
    dimTheta <- 6 + nObservations
  } else if (modelType == 7) {
    dimTheta <- 5 + nObservations
  } else if (modelType == 8) {
    dimTheta <- 5 + nObservations
  } else if (modelType == 9) {
    dimTheta <- 4 + nObservations
  } else if (modelType == 10) {
    dimTheta <- 7 + 1 + 1
  } else if (modelType == 11) {
    dimTheta <- 6 + 1 + 1
  } else if (modelType == 12) {
    dimTheta <- 6 + 1 + 1
  } else if (modelType == 13) {
    dimTheta <- 5 + 1 + 1
  } else if (modelType == 14) {
    dimTheta <- 5 + 1 + 1
  } else if (modelType == 15) {
    dimTheta <- 4 + 1 + 1
  }
  
  modelType <- modelType
  
  
  
  ## ----------------------------------------------------------------------- ##
  ## Loading Known Covariates
  ## ----------------------------------------------------------------------- ##
  
  voleCovar <- as.numeric(unlist(read.table(file.path(miscParameters$pathToCovar, "voleCovar.dat"))))
  timeNormCovar <- as.numeric(unlist(read.table(file.path(miscParameters$pathToCovar, "timeNormCovar.dat"))))
  
  
  ## ----------------------------------------------------------------------- ##
  ## Specifying Hyperparameters
  ## ----------------------------------------------------------------------- ##
  
  ## Hyperparameters and other known covariates
  meanHyper    <- rep(0, dimTheta)
  sdHyper      <- rep(sqrt(2), dimTheta) ################# NOTE: changed this from 1 to 2
  minHyperInit <- 0
  maxHyperInit <- 50
  
  if (modelType <= 3) {
    meanHyper[6] <- -2
  } else if (modelType %in% c(6:7,12:13)) {
    meanHyper[5] <- -2
  } else if (modelType %in% c(8:9,14:15)) {
    meanHyper[4] <- -2
  }

  # Collecting all the hyper parameters in a vector:
  hyperParameters <- c(
    modelType,
    nObservations,
    dimTheta,
    meanHyper, sdHyper, 
    minHyperInit, maxHyperInit, 
    timeNormCovar, voleCovar
  )

  
  ## ----------------------------------------------------------------------- ##
  ## Names of the Parameters (only used for graphical output)
  ## ----------------------------------------------------------------------- ##
  
  if (modelType %in% c(0:5,10:11)) {
    thetaNames <- c("alpha_0", "alpha_1", "alpha_2", "alpha_3", "beta_1", "delta_0")
  } else if (modelType %in% c(6:7,12:13)) {
    thetaNames <- c("alpha_0", "alpha_1", "alpha_2", "beta_1", "delta_0")
  } else if (modelType %in% c(8:9,14:15)) {
    thetaNames <- c("alpha_0", "alpha_2", "alpha_3", "delta_0")
  }

  if (modelType %% 2 == 0) {
    thetaNames <- c(thetaNames, "delta_1")
  }
  if (modelType %in% c(0,1,2,3)) {
    for (t in 1:nObservations) {
      thetaNames <- c(thetaNames, paste("gamma_{", t, "}", sep=''))
    }
  } else {
    thetaNames <- c(thetaNames, "gamma")
  }
  if (modelType %in% c(0:1,4:9)) {
    for (t in 2:nObservations) {
      thetaNames <- c(thetaNames, paste("beta_{", t, "}", sep=''))
    }
  } else {
    thetaNames <- c(thetaNames, "beta")
  }

  
  ## ----------------------------------------------------------------------- ##
  ## Support of the Unknown Parameters
  ## ----------------------------------------------------------------------- ##
  
  supportMin <- rep(-Inf, times=dimTheta) # minimum
  supportMax <- rep( Inf, times=dimTheta) # maximum
  support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

  return(list(modelType=modelType, dimTheta=dimTheta, hyperParameters=hyperParameters, support=support, thetaNames=thetaNames, meanHyper=meanHyper, sdHyper=sdHyper, voleCovar=voleCovar, timeNormCovar=timeNormCovar,nObservations=nObservations, nObservationsCount=nObservations, count=count, fecundity=fecundity, capRecapFemaleFirst=capRecapFemaleFirst, capRecapFemaleAdult=capRecapFemaleAdult, capRecapMaleFirst=capRecapMaleFirst, capRecapMaleAdult=capRecapMaleAdult, dimLatentVariable=2))
}