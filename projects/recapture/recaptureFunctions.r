
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