################################################################################
## Calculate block borders for blocked SMC filtering and smoothing schemes
################################################################################

setBlocks <- function(blockSizeInn, blockSizeExt, V)
{
  # Partition of the state space:
  blockInn <- seq(from=min(blockSizeInn,V), to=V, by=blockSizeInn)
  blockInn <- cbind(blockInn-blockSizeInn+1, blockInn)
  blockInn[blockInn < 1] <- 1
  
  if (max(blockInn) < V) {
    blockInn <- rbind(blockInn, c(max(blockInn)+1,V))
  }

  # Extension to obtain overlapping blocks
  blockOut <- blockInn
  for (j in 1:dim(blockInn)[1]) {
    blockOut[j,1] <- blockOut[j,1] - blockSizeExt
    blockOut[j,2] <- blockOut[j,2] + blockSizeExt
  }
  blockOut[blockOut < 1] <- 1
  blockOut[blockOut > V] <- V
  
  # Enlarging the last block in case it doesn't reach 
  # all the way to the upper boundary of the state space:
#   blockInn[dim(blockInn)[1],2] <- V
#   blockOut[dim(blockOut)[1],2] <- V

  blockInn <- t(blockInn) - 1 ## start counting from 0
  blockOut <- t(blockOut) - 1 ## start counting from 0
  
  return(list(blockInn=blockInn, blockOut=blockOut))
}

