## Some additional R functions for use with the random-effects model

###############################################################################
## Returns a number of (known) auxiliary model variables.
###############################################################################
getAuxiliaryModelParameters <- function(modelType, miscParameters) {
  
  
  ## ----------------------------------------------------------------------- ##
  ## Loading Data
  ## ----------------------------------------------------------------------- ##
  
  ## Note: we assume here that modelType cannot take value "0"!
  nObs          <- c(1,2,3,4,5,10,15,25,50,75,100,250,500,750,1000) # observation sequences which are available
  observations  <- as.numeric(unlist(read.table(file.path(miscParameters$pathToData, paste("observations_", nObs[modelType], sep='')))))
  nObservations <- length(observations) # number of time steps/observations
  
  ## ----------------------------------------------------------------------- ##
  ## Specifying Hyperparameters
  ## ----------------------------------------------------------------------- ##
  
  dimTheta   <- 3 # length of the parameter vector
  
  supportMin <- c(-Inf, 0, 0) # minimum of the support for the mean and the variance-parameters
  supportMax <- c(Inf, Inf, Inf) # maximum of the support for the mean and the variance-parameters
  support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

  meanHyperA  = 0
  varHyperA   = 1
  shapeHyperB = 3
  scaleHyperB = 2
  shapeHyperD = 3
  scaleHyperD = 2
  hyperParameters <- c(meanHyperA, varHyperA, shapeHyperB, scaleHyperB, shapeHyperD, scaleHyperD)

  aTrue <- 0
  bTrue <- 1
  dTrue <- 0.1
  thetaTrue     <- c(aTrue, bTrue, dTrue) # "true" parameter values used for generating the data
  thetaNames    <- c("a", "b", "d") # names of the parameters
  
  
  ## ----------------------------------------------------------------------- ##
  ## Simulate data and calculate importance-sampling approximation
  ## of marginal likelihood
  ## ----------------------------------------------------------------------- ##
#   
#   testFun <- function(x) {(x^3)*(x > 0.25) + 0.01*(x <= 0.25)}
# #   
# #   simpleImportanceSampling2 <- function(nSamples, observations) {
# #     theta     <- matrix(NA, nrow=dimTheta, ncol=nSamples)
# #     theta[1,] <- rnorm(nSamples, mean=meanHyperA, sd=sqrt(varHyperA))
# #     theta[2,] <- 1/rgamma(nSamples, shape=shapeHyperB, scale=1/scaleHyperB)
# #     theta[3,] <- 1/rgamma(nSamples, shape=shapeHyperD, scale=1/scaleHyperD)
# #     
# #     logW <- rep(0, times=nSamples)
# #     
# #     for (tt in 1:nObservations) {
# #       x    <- rnorm(nSamples, mean=theta[1,], sd=theta[2,])
# #       logW <- logW + dnorm(rep(observations[tt], times=nSamples), mean=x, sd=theta[3,], log=TRUE)
# #     }
# # 
# #     logEvidence <- log(sum(exp(logW))) - log(nSamples)
# #     
# #     moment1 <- rep(NA, times=dimTheta)
# #     moment2 <- matrix(NA, nrow=dimTheta, ncol=dimTheta)
# #     
# #     w <- exp(logW - max(logW)) 
# #     
# #     for (ii in 1:dimTheta) {
# #       moment1[ii] <- (w %*% testFun(theta[ii,])) / sum(w)
# #       for (jj in 1:dimTheta) {
# #         moment2[ii,jj] <- sum(w * theta[ii,] * theta[jj,]) / sum(w)
# #       }
# #     }
# #     
# #     return(list(logEvidence=logEvidence, moment1=moment1, moment2=moment2))
# #   }
# 
#   simpleImportanceSampling <- function(nSamples, observations) {
#     theta     <- matrix(NA, nrow=dimTheta, ncol=nSamples)
#     theta[1,] <- rnorm(nSamples, mean=meanHyperA, sd=sqrt(varHyperA))
#     theta[2,] <- 1/rgamma(nSamples, shape=shapeHyperB, scale=1/scaleHyperB)
#     theta[3,] <- 1/rgamma(nSamples, shape=shapeHyperD, scale=1/scaleHyperD)
# 
#     logW <- rep(0, times=nSamples)
#     for (tt in 1:nObservations) {
#       logW <- logW + dnorm(rep(observations[tt], times=nSamples), mean=theta[1,], sd=sqrt(theta[2,]^2 + theta[3,]^2), log=TRUE)
#     }
# 
#     logEvidence <- log(sum(exp(logW))) - log(nSamples)
#     
#     moment1 <- rep(NA, times=dimTheta)
#     moment2 <- matrix(NA, nrow=dimTheta, ncol=dimTheta)
#     
#     w <- exp(logW - max(logW)) 
#     
#     for (ii in 1:dimTheta) {
#       moment1[ii] <- (w %*% testFun(theta[ii,])) / sum(w)
#       for (jj in 1:dimTheta) {
#         moment2[ii,jj] <- sum(w * theta[ii,] * theta[jj,]) / sum(w)
#       }
#     }
#     return(list(logEvidence=logEvidence, moment1=moment1, moment2=moment2))
#   }
#   
#   for (ii in 1:length(nObs)) {
#     nObservations <- nObs[ii]
#     observationTimes <- seq(from=1, to=nObservations, by=1)
#     DATA <- simulateDataCpp(nObservations, hyperParameters, thetaTrue, numeric(0), 1)
#     observations  <- DATA$y # simulated observations
#     write(observations, file = file.path(getwd(), "data", paste("observations_", nObservations, sep='')))
#    
#     IIMax <- 10
#     logEvidenceTrue <- rep(NA, times=IIMax)
#     for (ii in 1:IIMax) {
#       aux <- simpleImportanceSampling(nSamples=1000000, observations=observations) 
#       logEvidenceTrue[ii] <- aux$logEvidence
#     }
#     write(logEvidenceTrue, file = file.path(getwd(), "data", paste("logEvidenceTrue_", nObservations, sep='')))
#     
#     print(logEvidenceTrue)
# #     aux2 <- simpleImportanceSampling2(nSamples=1000000, observations=observations) 
# #     logEvidenceTrue2 <- aux2$logEvidence
#     
# #     print(aux$logEvidence - logEvidenceTrue2)
# #     write(logEvidenceTrue2, file = file.path(getwd(), "data", paste("logEvidenceTrue_", nObservations, sep='')))
#   }
#   
# 
# 
#   
#   
  
  
  
  
  
  return(list(observations=observations, dimTheta=dimTheta, hyperParameters=hyperParameters, support=support, nObservations=nObservations, thetaNames=thetaNames, thetaTrue=thetaTrue))
}
