## Some additional R functions for the copula stochastic volatility model

###############################################################################
## Returns a number of (known) auxiliary model variables.
###############################################################################
getAuxiliaryModelParameters <- function(modelType, copulaTypeH, copulaTypeZ, copulaTypeHZ, miscParameters, dataSetName) {
  
  ## ----------------------------------------------------------------------- ##
  ## Loading Data
  ## ----------------------------------------------------------------------- ##
  
  logExchangeRates        <- data.matrix(read.table(file.path(miscParameters$pathToData, dataSetName, "logExchangeRates.dat")))
  logVolatilities         <- data.matrix(read.table(file.path(miscParameters$pathToData, dataSetName, "logVolatilities.dat")))
  initialLogExchangeRates <- data.matrix(read.table(file.path(miscParameters$pathToData, dataSetName, "initialLogExchangeRates.dat")))
  initialLogVolatilities  <- data.matrix(read.table(file.path(miscParameters$pathToData, dataSetName, "initialLogVolatilities.dat")))

  nObservations  <- dim(logExchangeRates)[2]
  nExchangeRates <- dim(logExchangeRates)[1]
  
  ## ----------------------------------------------------------------------- ##
  ## Model Index
  ## ----------------------------------------------------------------------- ##

  ## Number of unknown parameters, i.e. the length of the vector theta:
 
  if (modelType == 0) {
    dimTheta <- 7 * nExchangeRates + 1
    dimLatentVariable <- 2
  } else if (modelType == 1) {
    dimTheta <- 7 * nExchangeRates + 2
    dimLatentVariable <- 2 + nExchangeRates
  }
  
  ## ----------------------------------------------------------------------- ##
  ## Specifying Hyperparameters
  ## ----------------------------------------------------------------------- ##
  
  hyperParametersList <- setHyperParameters(modelType, copulaTypeH, copulaTypeZ, copulaTypeHZ, nExchangeRates, dimTheta)
   
#   delta <- 1/252 # the mesh size/time between observations
#   
#   ## Parameters of the normal distribution for the initial log-volatility
#   ## TODO: is it OK to assume that these are known?
#   meanInitialLogVolatility <- rep(-log(100), times=nExchangeRates)
#   sdInitialLogVolatility   <- rep(2, times=nExchangeRates)
#   
#   #### TODO: find suitable prior means/variances!
#   meanHyperAux <- rep(c(0, 1/2, 2.5, 1/100, 1/10, 0, 0), each=nExchangeRates)
#   meanHyper <- c(meanHyperAux, 0)
#   if (modelType == 1) { # adding the prior mean for log(omega)
#     meanHyper <- c(meanHyper, 0)
#   }
#   sdHyper <- rep(1, dimTheta) 
#  
#   ## Hyperparameters to be passed to C++:
#   # Collecting all the hyper parameters in a vector:
#   hyperParameters <- c(
#     modelType,
#     copulaTypeH, copulaTypeZ, copulaTypeHZ,
#     nExchangeRates,
#     dimTheta,
#     delta,
#     meanInitialLogVolatility,
#     sdInitialLogVolatility,
#     meanHyper, sdHyper
#   )
 
  ## ----------------------------------------------------------------------- ##
  ## Names of the Parameters (only used for graphical output)
  ## ----------------------------------------------------------------------- ##
  
  thetaNames <- c()
  for (ii in 1:nExchangeRates) {
    thetaNames <- c(thetaNames, paste("alpha_", ii, sep=''))
  }
  for (ii in 1:nExchangeRates) {
    thetaNames <- c(thetaNames, paste("beta_", ii, sep=''))
  }
  for (ii in 1:nExchangeRates) {
    thetaNames <- c(thetaNames, paste("log kappa_", ii, sep=''))
  }
  for (ii in 1:nExchangeRates) {
    thetaNames <- c(thetaNames, paste("log(2*kappa*mu - sigma^2)_", ii, sep=''))
  }
  for (ii in 1:nExchangeRates) {
    thetaNames <- c(thetaNames, paste("log sigma_", ii, sep=''))
  }
  for (ii in 1:nExchangeRates) {
    thetaNames <- c(thetaNames, paste("h_C(vartheta^H)_", ii, sep=''))
  }
  for (ii in 1:nExchangeRates) {
    thetaNames <- c(thetaNames, paste("h_C(vartheta^Z)_", ii, sep=''))
  }
  thetaNames <- c(thetaNames, "h_C(vartheta^{HZ})")
  if (modelType == 1) {
    thetaNames <- c(thetaNames, "log omega")
  } 
  
  ## ----------------------------------------------------------------------- ##
  ## Support of the Unknown Parameters
  ## ----------------------------------------------------------------------- ##
  
  supportMin <- rep(-Inf, times=dimTheta) # minimum
  supportMax <- rep( Inf, times=dimTheta) # maximum
  support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

  return(list(modelType=modelType, copulaTypeH=copulaTypeH, copulaTypeZ=copulaTypeZ, copulaTypeHZ=copulaTypeHZ, nExchangeRates=nExchangeRates, dimTheta=dimTheta, dimLatentVariable=dimLatentVariable, hyperParameters=hyperParametersList$hyperParameters, support=support, thetaNames=thetaNames, meanHyper=hyperParametersList$meanHyper, sdHyper=hyperParametersList$sdHyper, delta=hyperParametersList$delta, meanInitialLogVolatility=hyperParametersList$meanInitialLogVolatility, sdInitialLogVolatility=hyperParametersList$sdInitialLogVolatility,
  nObservations=nObservations, initialLogExchangeRates=initialLogExchangeRates, initialLogVolatilities=initialLogVolatilities, logExchangeRates=logExchangeRates, logVolatilities=logVolatilities))
}

###############################################################################
## Specifies the (known) model parameters
###############################################################################
setHyperParameters <- function(modelType, copulaTypeH, copulaTypeZ, copulaTypeHZ, nExchangeRates, dimTheta) {

  delta <- 1/252 # the mesh size/time between observations

  ## Parameters of the normal distribution for the initial log-volatility
  ## TODO: is it OK to assume that these are known?
  meanInitialLogVolatility <- rep(-log(100), times=nExchangeRates)
  sdInitialLogVolatility   <- rep(1, times=nExchangeRates)
#   meanInitialLogVolatility <- rep(0, times=nExchangeRates)
#   sdInitialLogVolatility   <- rep(0.1, times=nExchangeRates)
  
  #### TODO: find suitable prior means/variances!
#   meanHyperAux <- rep(c(0, 1/2, 2.5, 1/100, 1/10, 0, 0), each=nExchangeRates)
  meanHyperAux <- rep(c(0, 1/2, 2.5, -1.7, -2.5, 0, 0), each=nExchangeRates)
  meanHyper <- c(meanHyperAux, 0)
  if (modelType == 1) { # adding the prior mean for log(omega)
    meanHyper <- c(meanHyper, 0)
  }
  sdHyper <- rep(1, dimTheta) 

  # Collecting all the hyper parameters in a vector:
  hyperParameters <- c(
    modelType,
    copulaTypeH, copulaTypeZ, copulaTypeHZ,
    nExchangeRates,
    dimTheta,
    delta,
    meanInitialLogVolatility,
    sdInitialLogVolatility,
    meanHyper, sdHyper
  )
  
  return(list(
    modelType=modelType, 
    copulaTypeH=copulaTypeH, 
    copulaTypeZ=copulaTypeZ, 
    copulaTypeHZ=copulaTypeHZ,
    nExchangeRates=nExchangeRates,
    dimTheta=dimTheta,
    delta=delta,
    meanInitialLogVolatility=meanInitialLogVolatility,
    sdInitialLogVolatility=sdInitialLogVolatility,
    meanHyper=meanHyper,
    sdHyper=sdHyper,
    hyperParameters=hyperParameters
  ))
}

###############################################################################
## Simulates and stores a data set given a set of parameter values
###############################################################################
simulateData <- function(nObservations, nExchangeRates, modelType, copulaTypeH, copulaTypeZ, copulaTypeHZ, dataSetName, thetaTrue, miscParameters, nCores) {

  dataSetNameFull <- paste(dataSetName, "nObservations", nObservations, "nExchangeRates", nExchangeRates, "modelType", modelType, "copulaTypeH", copulaTypeH, "copulaTypeZ", copulaTypeZ, "copulaTypeHZ", copulaTypeHZ, sep="_")
  
  dir.create(file.path(miscParameters$pathToData, dataSetNameFull), showWarnings = FALSE) 
  
  hyperParametersList <- setHyperParameters(modelType, copulaTypeH, copulaTypeZ, copulaTypeHZ, nExchangeRates, length(thetaTrue))
  parTrueAux <- simulateDataCpp(nObservations, hyperParametersList$hyperParameters, thetaTrue, nCores)
  
  write.table(parTrueAux$logExchangeRates, file.path(miscParameters$pathToData, dataSetNameFull, "logExchangeRates.dat"))
  write.table(parTrueAux$logVolatilities,  file.path(miscParameters$pathToData, dataSetNameFull, "logVolatilities.dat"))
  write.table(parTrueAux$initialLogExchangeRates, file.path(miscParameters$pathToData, dataSetNameFull, "initialLogExchangeRates.dat"))
  write.table(parTrueAux$initialLogVolatilities,  file.path(miscParameters$pathToData, dataSetNameFull,"initialLogVolatilities.dat"))
  write.table(thetaTrue,  file.path(miscParameters$pathToData, dataSetNameFull, "thetaTrue.dat")) # true parameter values
  write.table(parTrueAux$latentTrue, file.path(miscParameters$pathToData, dataSetNameFull, "latentTrue.dat")) # true latent variables

}
