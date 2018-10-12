# Applies the logit transformation.
logit <- function(p) {-log(1/p-1)}
# Applies the inverse of the logit transformation.
inverseLogit <- function(x) {1/(1+exp(-x))}
# Returns the names of the copula types.
getCopulaName <- function(copulaType) {
  if (copulaType == 0) {
    copulaName <- "gaussian"
  } else if (copulaType == 1) {
    copulaName <- "90"
  } else if (copulaType == 2) {
    copulaName <- "270"
  }
  return(copulaName)
}
# Converts the reparametrised (scalar) copula parameter
# to the original copula parameter.
transformThetaToLambda <- function(theta, copulaType) {
  if (copulaType == 0) {        # i.e. Gaussian copula
    lambda <- 2.0 * inverseLogit(theta) - 1
  } else if (copulaType == 1) { # i.e. 90-degree rotated copula
    lambda <- exp(theta)
  } else if (copulaType == 2) { # i.e. 270-degree rotated copula
    lambda <- exp(theta)
  }
  return(lambda)
}
# Converts the original (scalar) copula parameter
# to the transformed (scalar) copula parameter.
transformLambdaToTheta <- function(lambda, copulaType) {
  if (copulaType == 0) {        # i.e. Gaussian copula
    theta <- logit((lambda+1)/2)
  } else if (copulaType == 1) { # i.e. 90-degree rotated copula
    theta <- log(lambda)
  } else if (copulaType == 2) { # i.e. 270-degree rotated copula
    theta <- log(lambda)
  }
  return(theta)
}
# Converts the reparametrised model parameters theta to 
# a vector of the orignal model parameters.
transformThetaToParameters <- function(theta, copulaTypeH, copulaTypeZ, copulaTypeHZ, modelType) {
  
  if (modelType == 0) {
    nExchangeRates <- (length(theta)-1)/7
  } else if (modelType == 1) {
    nExchangeRates <- (length(theta)-2)/7
  }
  K <- nExchangeRates
  
  alpha <- theta[1:K]
  beta  <- theta[(1*K+1):(2*K)]
  kappa <- exp(theta[(2*K+1):(3*K)])
  sigma <- exp(theta[(4*K+1):(5*K)])
  mu    <- (exp(theta[(3*K+1):(4*K)]) + sigma^2)/(2*kappa)
  lambdaH  <- transformThetaToLambda(theta[(5*K+1):(6*K)], copulaTypeH)
  lambdaZ  <- transformThetaToLambda(theta[(6*K+1):(7*K)], copulaTypeZ)
  lambdaHZ <- transformThetaToLambda(theta[7*K+1], copulaTypeHZ)
  
  if (modelType == 1) {
    omega <- theta[7*K+2]
    return(list(c(alpha=alpha, beta=beta, kappa=kappa, mu=mu, sigma=sigma, lambdaH=lambdaH, lambdaZ=lambdaZ, lambdaHZ=lambdaHZ, omega=omega)))
  } else {
    return(list(c(alpha=alpha, beta=beta, kappa=kappa, mu=mu, sigma=sigma, lambdaH=lambdaH, lambdaZ=lambdaZ, lambdaHZ=lambdaHZ)))
  }
}
# Converts the vector of original model parameters to the 
# reparametrised version theta.
transformParametersToTheta <- function(alpha, beta, kappa, mu, sigma, lambdaH, lambdaZ, lambdaHZ, omega, copulaTypeH, copulaTypeZ, copulaTypeHZ, modelType) {
  
  K <- length(alpha) # number of exchange rates
  theta <- c(alpha, beta, log(kappa), log(2*kappa*mu - sigma^2), log(sigma), 
            transformLambdaToTheta(lambdaH, copulaTypeH), transformLambdaToTheta(lambdaZ, copulaTypeZ), transformLambdaToTheta(lambdaHZ, copulaTypeHZ))
            
  if (modelType == 1) {
    theta <- c(theta, log(omega))
  }
  return(theta)
}


concatenateToMatrixOfArrays <- function(name, pathToResults, pathToProcessed, idxConfigurations, idxReplicates) {
  files <- list.files(path=pathToResults)
  
  in0 <- readRDS(paste(pathToResults, "/", name, "_", 1, "_", 1, ".rds", sep=''))
  
  if (is.vector(in0)) { # also for scalar
    dimIn0 <- length(in0)
  } else if (is.array(in0)) {
    dimIn0 <- dim(in0)
  }
  nDimIn0 <- length(dimIn0)
  if (nDimIn0 == 1 && length(in0) == 1) {
    out0 <- array(NA, c(length(idxConfigurations), length(idxReplicates)))
  } else {
    out0 <- array(NA, c(dimIn0, length(idxConfigurations), length(idxReplicates)))
  }

 
  for (jj in 1:length(idxReplicates)) {
    for (ii in 1:length(idxConfigurations)) {
      if (nDimIn0 == 1 && length(in0) == 1) {
        out0[ii,jj]    <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", idxReplicates[jj], ".rds", sep=''))
      } else if (nDimIn0 == 1 && length(in0) > 1) {
        out0[,ii,jj]   <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", idxReplicates[jj], ".rds", sep=''))
      } else if (nDimIn0 == 2) {
        out0[,,ii,jj]  <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", idxReplicates[jj], ".rds", sep=''))
      } else if (nDimIn0 == 3) {
        out0[,,,ii,jj] <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", idxReplicates[jj], ".rds", sep=''))
      }
    }
  }
  saveRDS(out0, file.path(pathToProcessed, paste(name, ".rds", sep='')))
}

concatenateToListOfArrays <- function(name, pathToResults, pathToProcessed, idxConfigurations, idxReplicates) {
  files <- list.files(path=pathToResults)
  
  out0 <- list()
  
  for (ii in 1:length(idxConfigurations)) {
  
    in1 <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", 1, ".rds", sep=''))
    if (is.vector(in1)) { # also for scalar
      dimIn1 <- length(in1)
    } else if (is.array(in1)) {
      dimIn1 <- dim(in1)
    }
    out1 <- array(NA, c(dimIn1, length(idxReplicates)))
    nDimIn1 <- length(dimIn1)
  
    for (jj in 1:length(idxReplicates)) {
      if (nDimIn1 == 1) {
        out1[,jj]   <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", idxReplicates[jj], ".rds", sep=''))
      } else if (nDimIn1 == 2) {
        out1[,,jj]  <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", idxReplicates[jj], ".rds", sep=''))
      } else if (nDimIn1 == 3) {
        out1[,,,jj] <- readRDS(paste(pathToResults, "/", name, "_", idxConfigurations[ii], "_", idxReplicates[jj], ".rds", sep=''))
      }
    }
    print(dim(out1))
    out0 <- c(out0, list(out1))
  }
  saveRDS(out0, file.path(pathToProcessed, paste(name, ".rds", sep='')))
}

processOutputSmc <- function(pathToResults, pathToProcessed, idxConfigurations, idxReplicates) {

  concatenateToMatrixOfArrays("standardLogEvidenceEstimate", pathToResults=pathToResults, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicates)
  concatenateToMatrixOfArrays("cpuTime", pathToResults=pathToResults, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicates)
#   concatenateToListOfArrays("finalSelfNormalisedWeights", pathToResults=pathToResults, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicates)
#   concatenateToListOfArrays("finalParameters", pathToResults=pathToResults, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicates)

}

# processOutputMcmc <- function() {
# 
# }
