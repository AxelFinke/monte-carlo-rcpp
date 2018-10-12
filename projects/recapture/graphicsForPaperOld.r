rm(list = ls())
set.seed(123)
library(abind)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"

exampleName       <- "herons"
# exampleName       <- "owls"
projectName       <- "recapture"
# jobName           <- "smc_sge_array_debug"
jobName           <- "smc_sge_array_2017-06-29"

source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study

MAX_SIM_SMC <- 4
idxReplicates     <- (1:MAX_SIM_SMC) # indices of independent replicates used by the simulation study
idxConfigurations <- (1:N_CONFIGS)[-c(7:9,16:18,25:27)] ### 1:N_CONFIGS ## 1:36 # indices of the configurations to be plotted
modelNames        <- MODEL_NAMES[idxConfigurations]

imageType <- "pdf"  # either "pdf" or "tex" (the latter uses tikz)
paperType <- "none" # either "tsp", "arXiv" or "none"

WIDTH  <- 5
HEIGHT <- 8

WIDTH_BOXPLOT  <- 8
HEIGHT_BOXPLOT <- 7

PAR_LIM <- c(-8, 8)
KDE_LIM <- c(0, 10)

# Miscellaneous parameters:
miscParameters <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)


## ========================================================================= ##
## Extract log-evidence estimates
## ========================================================================= ##

concatenateToMatrixOfArrays <- function(name) {
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


concatenateToListOfArrays <- function(name) {
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


concatenateToMatrixOfArrays("standardLogEvidenceEstimate")
# concatenateToMatrixOfArrays("alternateLogEvidenceEstimate")
# concatenateToMatrixOfArrays("cpuTime")
concatenateToListOfArrays("finalSelfNormalisedWeights")
concatenateToListOfArrays("finalParameters")

logEvidenceEstimates         <- readRDS(file.path(pathToProcessed, paste("standardLogEvidenceEstimate", ".rds", sep='')))
# alternateLogEvidenceEstimate <- readRDS(file.path(pathToProcessed, paste("alternateLogEvidenceEstimate", ".rds", sep='')))
# cpuTime                      <- readRDS(file.path(pathToProcessed, paste("cpuTime", ".rds", sep='')))


finalSelfNormalisedWeights   <- readRDS(file.path(pathToProcessed, paste("finalSelfNormalisedWeights", ".rds", sep='')))
finalParameters              <- readRDS(file.path(pathToProcessed, paste("finalParameters", ".rds", sep='')))



## ========================================================================= ##
## Plot log-evidence estimates from multiple SMC runs for multiple models:
## ========================================================================= ##

## Herons
## ------------------------------------------------------------------------- ##
if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToFigures, "logEvidenceEstimates"), ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToFigures, "logEvidenceEstimates"), ".tex", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "none") {
  # EMPTY
}

op <- par(mfrow=c(1,length(N_AGE_GROUPS_AUX)), mar=c(7,2,1,0.1), oma=c(7,2,1,0.1))
for (ii in 1:length(N_AGE_GROUPS_AUX)) {

  auxIdx <- which(N_AGE_GROUPS[idxConfigurations] == N_AGE_GROUPS_AUX[ii])
  
  horizontalLines <- seq(from=-1300, to=-1150, by=10)
  yAxisTickmarks <- horizontalLines
  yAxisTickmarks[8:9] <- ""
  
  logZ <- logEvidenceEstimates
  
  logZ[logEvidenceEstimates == -Inf] <- - 10^10
  
  boxplot(t(logZ[auxIdx,]), 
    ylab="",
    xlab="",
    range=0,
    las=2,
    axes=FALSE,
    names=modelNames[auxIdx],
    ylim = range(logZ[logEvidenceEstimates != -Inf]),
    main=""
  )
  if (ii == 1) {
    mtext(text="Log-evidence", side=2, line=2, outer=FALSE)
    axis(side=2, at=horizontalLines, labels=yAxisTickmarks, las=1)
  }
  if (ii == 2) {
    mtext(text="Model for the productivity rate", side=1, line=13, outer=FALSE)
  }
  mtext(text=paste("Number of age groups: ", N_AGE_GROUPS_AUX[ii], sep=''), side=3, line=0, outer=FALSE)
  axis(side=1, at=1:length(auxIdx), labels=modelNames[auxIdx], las=2)
  box()
  
  abline(h=horizontalLines, lty=3)

}
if (imageType != "none") {
  dev.off()
}
par(op)



## Owls
## ------------------------------------------------------------------------- ##
replicateColours <- c("red", rep("black", times=MAX_SIM_SMC-1)) 

if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToFigures, "logEvidenceEstimates"), ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToFigures, "logEvidenceEstimates"), ".tex", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "none") {
  # EMPTY
}
op <- par(mar=c(3,2,1,0.1), oma=c(3,2,1,0.1))

  boxplot(t(logEvidenceEstimates[,]), 
    ylab="",
    xlab="",
    range=0,
    las=2,
    axes=FALSE,
    names=MODEL_NAMES,
    ylim = c(-350,-310), ### range(logEvidenceEstimates),
    main=""
  )
#   for (kk in 1:N_CONFIGS) {
#     for (mm in 1:MAX_SIM_SMC) {
#       points(kk, logEvidenceEstimates[kk,mm], col=replicateColours[mm])
#     }
#   }

  mtext(text="Log-evidence", side=2, line=2, outer=FALSE)
  axis(side=2, at=c(-350, -340,-320,-310), labels=c(-350, -340,-320,-310), las=1)
  mtext(text="Model", side=1, line=4, outer=FALSE)
  axis(side=1, at=1:N_CONFIGS, labels=1:N_CONFIGS, las=1)
  box()


  abline(h=-310, lty=3)
  abline(h=-320, lty=3)
  abline(h=-330, lty=3)
  abline(h=-340, lty=3)
  abline(h=-350, lty=3)

  if (imageType != "none") {
    dev.off()
  }
par(op)


## ========================================================================= ##
## Plot parameter estimates from multiple SMC runs for multiple models
## ========================================================================= ##

## Herons
## ------------------------------------------------------------------------- ##


for (ii in 1:length(idxConfigurations)) {
# for (ii in 11:length(idxConfigurations)) {

  print(ii)

  if (imageType == "pdf") {
    pdf(file=paste(file.path(pathToFigures, "finalParameters_"), idxConfigurations[ii], ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
  } else if (imageType == "tex") {
    tikz(file=paste(file.path(pathToFigures, "finalParameters_"), idxConfigurations[ii], ".tex", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
  } else if (imageType == "none") {
    # EMPTY
  }
  
  if (exampleName == "herons") {
    modelParameters <- getAuxiliaryModelParameters(MODELS[idxConfigurations[ii]], N_AGE_GROUPS[idxConfigurations[ii]], N_LEVELS[idxConfigurations[ii]], miscParameters) # for herons
  } else if (exampleName == "owls") {
    modelParameters <- getAuxiliaryModelParameters(MODELS[idxConfigurations[ii]], miscParameters) # for owls
  } 
  
  dimTheta        <- modelParameters$dimTheta
  thetaNames      <- modelParameters$thetaNames
  ###theta           <- finalParameters[[ii]]
  
  op <- par(mfrow=c(1, 1))
  for (jj in 1:dimTheta) {
  
    thetaAux  <- readRDS(paste(pathToResults, "/", "finalParameters", "_", idxConfigurations[ii], "_", 1, ".rds", sep=''))
    weightAux <- readRDS(paste(pathToResults, "/", "finalSelfNormalisedWeights", "_", idxConfigurations[ii], "_", 1, ".rds", sep=''))
    plot(density(thetaAux[jj,], weights=weightAux), type='l', col="white", xlab=thetaNames[jj], ylab="Density", xlim=PAR_LIM, ylim=KDE_LIM, main='')
    for (mm in 1:length(idxReplicates)) {
      thetaAux  <- readRDS(paste(pathToResults, "/", "finalParameters", "_", idxConfigurations[ii], "_", idxReplicates[mm], ".rds", sep=''))
      weightAux <- readRDS(paste(pathToResults, "/", "finalSelfNormalisedWeights", "_", idxConfigurations[ii], "_", idxReplicates[mm], ".rds", sep=''))
      lines(density(thetaAux[jj,], weights=weightAux), type='l', col="black")
    }
  
  
##     plot(density(theta[jj,,1]), type='l', col="white", xlab=thetaNames[jj], ylab="Density", xlim=PAR_LIM, ylim=KDE_LIM, main='')
##     for (mm in 1:MAX_SIM_SMC) {
##        lines(density(theta[jj,,mm]), type='l')
##     }
    
    
    grid <- seq(from=min(PAR_LIM), to=max(PAR_LIM), length=10000)
    lines(grid, dnorm(grid, mean=modelParameters$meanHyper[jj], sd=modelParameters$sdHyper[jj]), col="black", lty=2)
    legend("topleft", legend=c("Estimated posterior density", "Prior density"), col=c("black", "black"), bty='n', lty=c(1,2))
  }
   

  par(op)
  if (imageType != "none") {
    dev.off()
  } 
}



## Owls
## ------------------------------------------------------------------------- ##

PAR_LIM_OWLS <- c(-4, 4)
KDE_LIM_OWLS <- c(0,3.5)
COL_SMC      <- "gray"
COL_MCMC     <- "black"
COL_PRIOR    <- "black"
LTY_SMC      <- 1
LTY_MCMC     <- 1
LTY_PRIOR    <- 2




pathToMcmcResults <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls/mcmc_sge_array_2017-06-07/results/"
MAX_SIM_MCMC <- 2


for (ii in 1:N_CONFIGS) {

  if (imageType == "pdf") {
    pdf(file=paste(file.path(pathToFigures, "finalParameters_"), ii, ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
  } else if (imageType == "tex") {
    tikz(file=paste(file.path(pathToFigures, "finalParameters_"), ii, ".tex", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
  } else if (imageType == "none") {
    # EMPTY
  }
  
  modelParameters <- getAuxiliaryModelParameters(MODELS[ii], miscParameters)
  dimTheta        <- modelParameters$dimTheta
  thetaNames      <- modelParameters$thetaNames

  op <- par(mfrow=c(1, 1))
  for (jj in 1:dimTheta) {
  
    thetaAux  <- readRDS(paste(pathToResults, "/", "finalParameters", "_", ii, "_", 1, ".rds", sep=''))
    weightAux <- readRDS(paste(pathToResults, "/", "finalSelfNormalisedWeights", "_", ii, "_", 1, ".rds", sep=''))
    plot(density(thetaAux[jj,], weights=weightAux), type='l', col="white", xlab=thetaNames[jj], ylab="Density", xlim=PAR_LIM_OWLS, ylim=KDE_LIM_OWLS, main='')
    for (mm in 1:MAX_SIM_SMC) {
      thetaAux  <- readRDS(paste(pathToResults, "/", "finalParameters", "_", ii, "_", mm, ".rds", sep=''))
      weightAux <- readRDS(paste(pathToResults, "/", "finalSelfNormalisedWeights", "_", ii, "_", mm, ".rds", sep=''))
      lines(density(thetaAux[jj,], weights=weightAux), type='l', col="black")
    }
    
    for (mm in 1:MAX_SIM_MCMC) {
      thetaAux  <- readRDS(paste(pathToResults, "/", "finalParameters", "_", ii, "_", mm, ".rds", sep=''))
      weightAux <- readRDS(paste(pathToResults, "/", "finalSelfNormalisedWeights", "_", ii, "_", mm, ".rds", sep=''))
      lines(density(thetaAux[jj,], weights=weightAux), type='l', col="black")
    }
  
  
##     plot(density(theta[jj,,1]), type='l', col="white", xlab=thetaNames[jj], ylab="Density", xlim=PAR_LIM, ylim=KDE_LIM, main='')
##     for (mm in 1:MAX_SIM_SMC) {
##        lines(density(theta[jj,,mm]), type='l')
##     }
    
    
    grid <- seq(from=min(PAR_LIM_OWLS)-1, to=max(PAR_LIM_OWLS)+1, length=10000)
    lines(grid, dnorm(grid, mean=modelParameters$meanHyper[jj], sd=modelParameters$sdHyper[jj]), col=COL_PRIOR, lty=LTY_PRIOR)
    legend("topleft", legend=c("Estimated posterior density (MCMC)", "Estimated posterior density (SMC)",  "Prior density"), col=c(COL_MCMC, COL_SMC, COL_PRIOR), bty='n', lty=c(LTY_MCMC, LTY_SMC, LTY_PRIOR))
  }
   

  par(op)
  if (imageType != "none") {
    dev.off()
  } 
}




ii <- 3

## Plotting the parameter estimates for the best-performing model
if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToFigures, "bestModel_"), ii, ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToFigures, "bestModel_"), ii, ".tex", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "none") {
  # EMPTY
}

modelParameters <- getAuxiliaryModelParameters(MODELS[ii], miscParameters)
dimTheta        <- modelParameters$dimTheta
thetaNames      <- modelParameters$thetaNames

op <- par(mfrow=c(1, 1))
for (jj in 1:dimTheta) {

  thetaAux  <- readRDS(paste(pathToResults, "/", "finalParameters", "_", ii, "_", 1, ".rds", sep=''))
  weightAux <- readRDS(paste(pathToResults, "/", "finalSelfNormalisedWeights", "_", ii, "_", 1, ".rds", sep=''))
  plot(density(thetaAux[jj,], weights=weightAux), type='l', col="white", xlab=thetaNames[jj], ylab="Density", xlim=PAR_LIM_OWLS, ylim=KDE_LIM_OWLS, main='')
  for (mm in 1:MAX_SIM_SMC) {
    thetaAux  <- readRDS(paste(pathToResults, "/", "finalParameters", "_", ii, "_", mm, ".rds", sep=''))
    weightAux <- readRDS(paste(pathToResults, "/", "finalSelfNormalisedWeights", "_", ii, "_", mm, ".rds", sep=''))
    lines(density(thetaAux[jj,], weights=weightAux), type='l', col=COL_SMC, lty=LTY_SMC)
  }
  
  for (mm in 1:MAX_SIM_MCMC) {
    thetaAux  <- readRDS(paste(pathToMcmcResults, "/", "parameters", "_", 2, "_", mm, ".rds", sep=''))
    lines(density(thetaAux[jj,]), type='l', col=COL_MCMC, lty=LTY_MCMC)
  }

  grid <- seq(from=min(PAR_LIM_OWLS)-1, to=max(PAR_LIM_OWLS)+1, length=10000)
  lines(grid, dnorm(grid, mean=modelParameters$meanHyper[jj], sd=modelParameters$sdHyper[jj]), col=COL_PRIOR, lty=LTY_PRIOR)
  legend("topleft", legend=c("Estimated posterior density (MCMC)", "Estimated posterior density (SMC)",  "Prior density"), col=c(COL_MCMC, COL_SMC, COL_PRIOR), bty='n', lty=c(LTY_MCMC, LTY_SMC, LTY_PRIOR))
}
  

par(op)
if (imageType != "none") {
  dev.off()
} 


## Plotting the ACFs (normalised by CPU time) for the best performing model

ii <- 3
maxLag  <- 5000
maxPlot <- 10
grid <- 0:maxLag
nIterations <- 10000000

LTY_MCMC_NON_DA <- 2
COL_MCMC_NON_DA <- "black"

if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToFigures, "acf_"), ii, ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToFigures, "acf_"), ii, ".tex", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "none") {
  # EMPTY
}

modelParameters <- getAuxiliaryModelParameters(MODELS[ii], miscParameters)
dimTheta        <- modelParameters$dimTheta
thetaNames      <- modelParameters$thetaNames

# average cpu times per iteration
cpuTimeNonDa <- c()
cpuTime      <- c()

for (mm in 1:MAX_SIM_MCMC) {
  cpuTimeNonDa  <- c(cpuTimeNonDa, readRDS(paste(pathToMcmcResults, "/", "cpuTime", "_", 1, "_", mm, ".rds", sep='')))
  cpuTime       <- c(cpuTime, readRDS(paste(pathToMcmcResults, "/", "cpuTime", "_", 2, "_", mm, ".rds", sep='')))
}
# factor by which delayed acceptance decreases the CPU times
cpuTimeFactor <- mean(cpuTime)/mean(cpuTimeNonDa)
cpuTimePerIteration      <- mean(cpuTime) / nIterations
cpuTimePerIterationNonDa <- mean(cpuTimeNonDa) / nIterations

op <- par(mfrow=c(1, 1))
for (jj in 1:dimTheta) {

  acf  <- readRDS(paste(pathToMcmcResults, "/", "acf", "_", 1, "_", 1, ".rds", sep=''))
  plot(grid, acf[jj,1:(maxLag+1)], type='l', col="white", xlab="", ylab="", ylim=c(0,1), xlim=c(0,maxPlot), xaxs='i', yaxs='i')
  for (mm in 1:MAX_SIM_MCMC) {
    acf  <- readRDS(paste(pathToMcmcResults, "/", "acf", "_", 1, "_", mm, ".rds", sep=''))
    lines(grid*cpuTimePerIterationNonDa, acf[jj,1:(maxLag+1)], type='l', col=COL_MCMC_NON_DA, lty=LTY_MCMC_NON_DA)
    acf  <- readRDS(paste(pathToMcmcResults, "/", "acf", "_", 2, "_", mm, ".rds", sep=''))
    lines(grid*cpuTimePerIteration, acf[jj,1:(maxLag+1)], type='l', col=COL_MCMC, lty=LTY_MCMC)
  }
  mtext(text="Autocorrelation", side=2, line=2.5, outer=FALSE)
  mtext(text="Lag x CPU time per iteration [in seconds]", side=1, line=2.5, outer=FALSE)
  mtext(text=paste(thetaNames[jj], sep=''), side=3, line=1, outer=FALSE)
  
  legend("topright", legend=c("without delayed acceptance", "with delayed acceptance"), col=c(COL_MCMC_NON_DA, COL_MCMC), bty='n', lty=c(LTY_MCMC_NON_DA, LTY_MCMC))
}
  

par(op)
if (imageType != "none") {
  dev.off()
} 





#############
## checking computation times in particular folder
#############

FILES <- list.files(path=file.path(pathToOutput, "results"), pattern='cpuTime*', recursive=TRUE)

cpuTime <- c()
for (ii in 1:length(FILES)) {
  cpuTime <- c(cpuTime, readRDS(file.path(pathToOutput, "results", FILES[ii])))
}

print(ceiling(cpuTime/3600))


