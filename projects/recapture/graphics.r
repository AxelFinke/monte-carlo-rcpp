rm(list = ls())
set.seed(123)
library(abind)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"

exampleName       <- "herons"
# exampleName       <- "owls"
projectName       <- "recapture"
# jobName           <- "smc_sge_array_debug"
jobName           <- "smc_sge_array_2017-06-07"

source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study

MAX_SIM_SMC <- 8
idxReplicates     <- 1:MAX_SIM_SMC # indices of independent replicates used by the simulation study
idxConfigurations <- (1:N_CONFIGS)[-c(22,32)] ### 1:N_CONFIGS ## 1:36 # indices of the configurations to be plotted
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
  
  boxplot(t(logEvidenceEstimates[auxIdx,]), 
    ylab="",
    xlab="",
    range=0,
    las=2,
    axes=FALSE,
    names=modelNames[auxIdx],
    ylim = range(logEvidenceEstimates),
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




pathToMcmcResults <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls/mcmc_sge_array_2017-06-04/results/"
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



## Comparing parameter estimates from single and double tempering and from a benchmark SMC sampler with a large number of particles.
## Plotting the parameter estimates for (one of) the best-performing model(s)
## with delta_1 != 0

PAR_LIM_OWLS <- c(-4, 4)
KDE_LIM_OWLS <- c(0,3.5)

ltySmc   <- 1
ltyMcmc  <- 1
ltyPrior <- 2

colSmcSingle    <- "blue"
colSmcDouble    <- "green"
colSmcBenchmark <- "red"
# colMcmc         <- "black"
colPrior        <- "black"

# pathToMcmcResults         <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls/mcmc_sge_array_2017-06-04/results"
pathToSmcSingleResults    <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls/smc_sge_array_2017-08-01/results"
pathToSmcDoubleResults    <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls/smc_sge_array_2017-08-02/results"
pathToSmcBenchmarkResults <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls/smc_sge_array_2017-08-03/results"

# idxReplicatesMcmc         <- c(1,2)
idxReplicatesSmcSingle    <- (1:100)[-c(1,82)]
idxReplicatesSmcDouble    <- (1:60)[-c(13,33,34,55,59,53)]
idxReplicatesSmcBenchmark <- c(1:3, 5:7)

# idxConfigurationMcmc <- 2
# idxConfigurationSmc  <- 3

imageType <- "pdf"
pathToFiguresAux <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls"

exampleName       <- "owls"
projectName       <- "recapture"
jobName           <- "smc_sge_array_2017-08-03"

source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study
# Miscellaneous parameters:
miscParameters <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)

for (ii in 1:16) {

idxConfigurationSmc  <- ii

if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToFiguresAux, "bestModel_"), ii, ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToFiguresAux, "bestModel_"), ii, ".tex", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "none") {
  # EMPTY
}

modelParameters <- getAuxiliaryModelParameters(MODELS[idxConfigurationSmc], miscParameters)
dimTheta        <- modelParameters$dimTheta
thetaNames      <- modelParameters$thetaNames

op <- par(mfrow=c(1, 1))
for (jj in 1:dimTheta) {

  plot(c(1,1), c(1,1), type='l', col="white", xlab=thetaNames[jj], ylab="Density", xlim=PAR_LIM_OWLS, ylim=KDE_LIM_OWLS, main='')
  
  for (mm in 1:length(idxReplicatesSmcSingle)) {
    thetaAux  <- readRDS(paste(pathToSmcSingleResults, "/", "finalParameters", "_", idxConfigurationSmc, "_", idxReplicatesSmcSingle[mm], ".rds", sep=''))
    weightAux <- readRDS(paste(pathToSmcSingleResults, "/", "finalSelfNormalisedWeights", "_", idxConfigurationSmc, "_", idxReplicatesSmcSingle[mm], ".rds", sep=''))
    lines(density(thetaAux[jj,], weights=weightAux), type='l', col=colSmcSingle, lty=ltySmc)
  }
  
  for (mm in 1:length(idxReplicatesSmcDouble)) {
    thetaAux  <- readRDS(paste(pathToSmcDoubleResults, "/", "finalParameters", "_", idxConfigurationSmc, "_", idxReplicatesSmcDouble[mm], ".rds", sep=''))
    weightAux <- readRDS(paste(pathToSmcDoubleResults, "/", "finalSelfNormalisedWeights", "_", idxConfigurationSmc, "_", idxReplicatesSmcDouble[mm], ".rds", sep=''))
    lines(density(thetaAux[jj,], weights=weightAux), type='l', col=colSmcDouble, lty=ltySmc)
  }
  
  for (mm in 1:length(idxReplicatesSmcBenchmark)) {
    thetaAux  <- readRDS(paste(pathToSmcBenchmarkResults, "/", "finalParameters", "_", idxConfigurationSmc, "_", idxReplicatesSmcBenchmark[mm], ".rds", sep=''))
    weightAux <- readRDS(paste(pathToSmcBenchmarkResults, "/", "finalSelfNormalisedWeights", "_", idxConfigurationSmc, "_", idxReplicatesSmcBenchmark[mm], ".rds", sep=''))
    lines(density(thetaAux[jj,], weights=weightAux), type='l', col=colSmcBenchmark, lty=ltySmc)
  }
  
#   for (mm in 1:length(idxReplicatesMcmc)) {
#     thetaAux  <- readRDS(paste(pathToMcmcResults, "/", "parameters", "_", idxConfigurationMcmc, "_", idxReplicatesMcmc[mm], ".rds", sep=''))
#     lines(density(thetaAux[jj,]), type='l', col=colMcmc, lty=ltyMcmc)
#   }

  grid <- seq(from=min(PAR_LIM_OWLS)-1, to=max(PAR_LIM_OWLS)+1, length=10000)
  lines(grid, dnorm(grid, mean=modelParameters$meanHyper[jj], sd=modelParameters$sdHyper[jj]), col=colPrior, lty=ltyPrior)
  legend("topleft", legend=c("SMC (standard tempering)", "SMC (refined tempering)", "SMC (benchmark)",  "Prior"), col=c(colSmcSingle, colSmcDouble, colSmcBenchmark, colPrior), bty='n', lty=c(ltySmc,ltySmc,ltySmc, ltyPrior))
}
  

par(op)
if (imageType != "none") {
  dev.off()
} 

}












# Shell script for renaming the files in the herons example:
# ----------------------------------------------
# cd /home/axel/from_legion/Scratch/output/recapture/herons/
# for file in $(ls herons_smc_*_task_id_*); do 
#   mv $file ${file/${file:0:102}/}; 
# done
# ----------------------------------------------

## owls:
II <- 8 # no. of configs
JJ <- 6 # no. of replicates


modelCol <- c("black", "gray", "red", "darkblue", "lightblue", "darkgreen", "lightgreen", "blue")
# MODELS_AUX               <- 0:1
# MODELS                   <- rep(MODELS_AUX, times=2)
# mycol <- rep(modelCol, times=2)
# mylty <- rep(c(1,2), each=length(MODELS_AUX))

mycol <- modelCol
mylty <- rep(1, times=N_CONFIGS)

pathToOutputOld <- "/home/axel/Dropbox/research/output/cpp/mc/recapture/owls/smc_sge_array_2017-05-29"
op <- par(mfrow=c(1,2))
plot(1:2000, rep(1,2000), col="white", ylim=c(0,1))
for (jj in 1:JJ) {
  for (ii in 1:II) {
#     print(c(ii, jj))
    aux <- readRDS(paste(pathToOutput, "/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
#     print(length(aux))
    lines(c(aux), type='l', lty=mylty[ii], col=mycol[ii])
  }
}
plot(1:2000, rep(1,2000), col="white", ylim=c(0,1))
for (jj in 1:JJ) {
  for (ii in 1:II) {
#     print(c(ii, jj))
    aux <- readRDS(paste(pathToOutputOld, "/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
#     print(length(aux))
    lines(x=1:length(c(aux)), y=c(aux), type='l', lty=mylty[ii], col=mycol[ii])
  }
}
par(op)


plot(1:600, rep(1,600), col="white", xlim=c(0,1), ylim=c(0.95,1), xlab="inverse temperature", ylab="autocorrelation (pre vs. post mutation)")
for (jj in 1:JJ) {
  for (ii in 1:II) {
#     print(c(ii, jj))
    aux1 <- readRDS(paste(pathToOutput, "/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
    aux2 <- readRDS(paste(pathToOutput, "/results/maxParticleAutocorrelations_", ii, "_", jj, ".rds", sep=''))
    print(length(aux))
    lines(c(aux1[-1]), c(aux2), type='l', lty=mylty[ii], col=mycol[ii])
  }
}





logZ <- matrix(NA, II, JJ)
for (ii in 1:II) {
  for (jj in 1:JJ) {
    logZ[ii,jj] <- readRDS(paste(pathToOutput,"/results/standardLogEvidenceEstimate_", ii, "_", jj, ".rds", sep=''))
  }
}
print(logZ)
boxplot(t(logZ), ylim=c(-380,-300),range=0,las=2)






## herons:
modelCol <- c("black", "darkblue", "lightblue", "darkgreen", "lightgreen")
MODELS_AUX               <- 0:1
MODELS                   <- rep(MODELS_AUX, times=2)
mycol <- rep(modelCol, times=2)
mylty <- rep(c(1,2), each=length(MODELS_AUX))

plot(1:600, rep(1,600), col="white", ylim=c(0,1))
for (jj in 1:1) {
  for (ii in 1:4) {
#     print(c(ii, jj))
    aux <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recapture/herons/smc_sge_array_debug/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
    print(length(aux))
    lines(c(aux), type='l', lty=mylty[ii], col=mycol[ii])
  }
}

II <- 3 # no. of configs
JJ <- 1 # no. of replicates
logZ <- matrix(NA, II, JJ)
for (ii in 1:II) {
  for (jj in 1:JJ) {
    logZ[ii,jj] <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recapture/herons/smc_sge_array_debug/results/standardLogEvidenceEstimate_", ii, "_", jj, ".rds", sep=''))
  }
}
print(logZ)
boxplot(t(logZ),range=0,las=2)



II <- 4
JJ <- 1
ii <- 3
param   <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recapture/herons/smc_sge_array_debug/raw/parameters_", ii, "_", jj, ".rds", sep=''))
dimTheta <- dim(param)[1]

op <- par(mfrow=c(ceiling(sqrt(dimTheta)),ceiling(sqrt(dimTheta))))
for (kk in 1:dimTheta) {
  plot(1:10, rep(1,10), col="white", ylim=c(0,10), xlim=c(-5,10))
  
  for (jj in 1:JJ) {
  weights <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recapture/herons/smc_sge_array_debug/raw/selfNormalisedWeights_", ii, "_", jj, ".rds", sep=''))
  param   <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recapture/herons/smc_sge_array_debug/raw/parameters_", ii, "_", jj, ".rds", sep=''))
  invTemp <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recapture/herons/smc_sge_array_debug/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
  nSteps <- length(invTemp)
    for (ll in seq(from=1, to=nSteps, length=10)) {
      lines(density(param[kk,,ll], weights=weights[,ll]), type='l', lty=mylty[ii], col=mycol[ii])
    }
  }
}
par(op)







## random:
II <- 10 # no. of configs
JJ <- 100 # no. of replicates


modelCol <- c("black", "gray", "blue", "green", "red")

mycol <- rep(modelCol, times=2)
mylty <- rep(c(1,2), each=length(MODELS_AUX))



op <- par(mfrow=c(1,2))
plot(1:600, rep(1,600), col="white", xlim=c(0,1), ylim=c(0.7,1), xlab="inverse temperature", ylab="autocorrelation (pre vs. post mutation)")
for (jj in 1:JJ) {
  for (ii in 1:(II/2)) {
#     print(c(ii, jj))
    aux1 <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
    aux2 <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/maxParticleAutocorrelations_", ii, "_", jj, ".rds", sep=''))
    print(length(aux1))
    lines(c(aux1[-1]), c(aux2), type='l', lty=mylty[ii], col=mycol[ii])
  }
}
plot(1:600, rep(1,600), col="white", xlim=c(0,1), ylim=c(0.7,1), xlab="inverse temperature", ylab="autocorrelation (pre vs. post mutation)")
for (jj in 1:JJ) {
  for (ii in (II/2+1):II) {
#     print(c(ii, jj))
    aux1 <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
    aux2 <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/maxParticleAutocorrelations_", ii, "_", jj, ".rds", sep=''))
    print(length(aux1))
    lines(c(aux1[-1]), c(aux2), type='l', lty=mylty[ii], col=mycol[ii])
  }
}
par(op)




nObs <- c(1,2,3,4,5,10,15,25,50,75,100,250,500,750,1000,2500,5000,7500,10000)
logEvidenceTrue <- as.numeric(unlist(read.table(file.path(miscParameters$pathToData, paste("logEvidenceTrue_", nObs[MODELS[1]], sep='')))))

# logZ    <- matrix(NA, II, JJ)
namesAll  <- c("standard estimate", "importance tempering without resampling", "importance tempering with resampling")
logZAll <- array(NA, c(length(namesAll), II, JJ))
nSteps  <- matrix(NA, II, JJ)

logZMse                <- matrix(NA, length(namesAll), II)
logZMseNormActual      <- matrix(NA, length(namesAll), II)
logZMseNormTheoretical <- matrix(NA, length(namesAll), II)

zMse                <- matrix(NA, length(namesAll), II)
zMseNormActual      <- matrix(NA, length(namesAll), II)
zMseNormTheoretical <- matrix(NA, length(namesAll), II)

cpuTimeTheoretical     <- matrix(NA, II, JJ)
cpuTimeActual          <- matrix(NA, II, JJ)
  
for (ii in 1:II) {
  for (jj in 1:JJ) {
#     logZAll[,ii,jj]      <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/alternateLogEvidenceEstimate_", ii, "_", jj, ".rds", sep=''))

    logZAll[,ii,jj]      <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/standardLogEvidenceEstimate_", ii, "_", jj, ".rds", sep=''))

    nSteps[ii,jj]        <- length(readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/inverseTemperatures_", ii, "_", jj, ".rds", sep='')))
    cpuTimeActual[ii,jj] <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/cpuTime_", ii, "_", jj, ".rds", sep=''))
  }
    
  cpuTimeTheoretical[ii,] <- nSteps[ii,] * N_METROPOLIS_HASTINGS_UPDATES[ii] * N_PARTICLES_UPPER[ii]
  
  for (rr in 1:length(namesAll)) {
    logZMse[rr,ii]                <- mean((logZAll[rr,ii,] - mean(logEvidenceTrue))^2)
    logZMseNormActual[rr,ii]      <- logZMse[rr,ii] * mean(cpuTimeActual[ii,]) 
    logZMseNormTheoretical[rr,ii] <- logZMse[rr,ii] * mean(cpuTimeTheoretical[ii,])
   
    zMse[rr,ii]                <- mean((exp(logZAll[rr,ii,]) - mean(exp(logEvidenceTrue)))^2)
    zMseNormActual[rr,ii]      <- zMse[rr,ii] * mean(cpuTimeActual[ii,]) 
    zMseNormTheoretical[rr,ii] <- zMse[rr,ii] * mean(cpuTimeTheoretical[ii,])
  }
}


rrIdx <- 1
op <- par(mfcol=c(4,length(rrIdx)))
  for (rr in 1:length(rrIdx)) {
    boxplot(t(logZAll[rr,,]), range=0,las=2, ylab="logZEst", main="")
    abline(h=logEvidenceTrue, col="red")
    boxplot(t(log(zMse[rr,])), range=0,las=2, ylab="log of MSE of zEst", main=namesAll[rr])
    boxplot(t(log(zMseNormActual[rr,])), range=0,las=2, ylab="log of MSE of zEst times average cpuTime", main=namesAll[rr])
    boxplot(t(log(zMseNormTheoretical[rr,])), range=0,las=2, ylab="log of MSE of zEst times theroretical computational complexity", main=namesAll[rr])
  }
par(op)

print(rowMeans(nSteps))








# 
# 
# 
# 
# 
# 
# op <- par(mfrow=c(2,1))
# boxplot(t(logZ), range=0,las=2, main="log-marginal likelihood estimates")
# abline(h=logEvidenceTrue, col="red")
# # boxplot(t(nSteps), range=0,las=2, main="number of SMC steps",ylim=c(0,max(nSteps)))
# boxplot(t(nStepsNorm), range=0,las=2, main="no. of SMC steps x no. of MH updates per step x no. of particles",ylim=c(0,max(nStepsNorm)))
# par(op)
# 
# op <- par(mfcol=c(2,3))
# for (ii in 1:3) {
# boxplot(t(logZAll[ii,,]), range=0,las=2, main="log-marginal likelihood estimates")
# abline(h=logEvidenceTrue, col="red")
# # boxplot(t(nSteps), range=0,las=2, main="number of SMC steps",ylim=c(0,max(nSteps)))
# boxplot(t(nStepsNorm), range=0,las=2, main="no. of SMC steps x no. of MH updates per step x no. of particles",ylim=c(0,max(nStepsNorm)))
# }
# par(op)
# 
# 
# op <- par(mfcol=c(3,length(namesAll)))
#   for (rr in 1:length(namesAll)) {
#     boxplot(t(logZMse[rr,]), range=0,las=2, ylab="MSE of logZEst", main=namesAll[rr])
#     boxplot(t(logZMseNormActual[rr,]), range=0,las=2, ylab="MSE of logZEst times average cpuTime", main=namesAll[rr])
#     boxplot(t(logZMseNormTheoretical[rr,]), range=0,las=2, ylab="MSE of logZEst times theroretical computational complexity", main=namesAll[rr])
#   }
# par(op)
# 
# 





# ii <- 10
# jj <- 5
# 
# 
# 
# op <- par(mfrow=c(3,1))
# 
# for (kk in 1:3) {
#   plot(1:10, rep(1,10), col="white", ylim=c(0,10), xlim=c(-5,20))
#   
#   for (jj in 1:JJ) {
#   weights <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/raw/selfNormalisedWeights_", ii, "_", jj, ".rds", sep=''))
#   param   <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/raw/parameters_", ii, "_", jj, ".rds", sep=''))
#   invTemp <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
#   nSteps <- length(invTemp)
#     for (ll in seq(from=1, to=nSteps, length=10)) {
#       lines(density(param[kk,,ll], weights=weights[,ll]), type='l', lty=mylty[ii], col=mycol[ii])
#     }
#   }
# }
# par(op)
# 
# 
# 
# plot(1:300, rep(1,300), col="white", ylim=c(0,1))
# for (jj in 1:JJ) {
#   for (ii in 1:II) {
#     print(c(ii, jj))
#     aux <- readRDS(paste("/home/axel/Dropbox/research/output/cpp/mc/recycling/random/smc_sge_array_debug/results/inverseTemperatures_", ii, "_", jj, ".rds", sep=''))
#     print(length(aux))
#     lines(c(aux), type='l', lty=mylty[ii], col=mycol[ii])
#   }
# }





################################
## GRAPHICS FOR THE PAPER
