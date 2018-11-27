rm(list = ls())
set.seed(123)


library(tikzDevice)
options( 
  tikzDocumentDeclaration = c(
    "\\documentclass[12pt]{beamer}",
    "\\usepackage{amssymb,amsmath,graphicx,mathtools,mathdots,stmaryrd}",
    "\\usepackage{tikz}" 
  )
)

## for graphs


# cd ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-19/temp
# 
# pdflatex graphics.tex
# 
# mv ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-19/temp/graphics.pdf ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-19/fig
# 
# cd ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-19/fig
# 
# pdftk graphics.pdf burst output %02d.pdf
# 
# cd ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-19/
# 
# pdflatex main.tex
# 
# pdflatex main.tex



# for manuscript
# cd ~/Dropbox/AF-SSS/blocked_smoothing/tsp_submission/
# pdftk block.pdf cat 1-12 output block_main.pdf
# pdftk block.pdf cat 13-17 output block_supp.pdf



## ========================================================================= ##
##
## GLOBAL PARAMETERS
##
## ========================================================================= ##

imageType <- "tex"  # either "pdf" or "tex" (the latter uses tikz)
paperType <- "biom" # either "biom", "arXiv" or "none"

WIDTH  <- 5
HEIGHT <- 8

WIDTH_BOXPLOT  <- 8
HEIGHT_BOXPLOT <- 7

PAR_LIM <- c(-8, 8)
KDE_LIM <- c(0, 10)

cexLegend <- 0.8
cexMtext  <- 1.0
cexAxis   <- 0.8
cexLab    <- 0.8
cex       <- 0.8
tck       <- -0.03
mgp       <- c(3, 0.5, 0)

alpha <- 0.15
col1 <- "black"
col2 <- "forestgreen"
col3 <- "magenta3"
col4 <- "darkblue"
col1Numeric  <- as.numeric(col2rgb(col1))/256 
col2Numeric  <- as.numeric(col2rgb(col2))/256
col3Numeric  <- as.numeric(col2rgb(col3))/256
col4Numeric  <- as.numeric(col2rgb(col4))/256
col1Shade <- rgb(col1Numeric[1], col1Numeric[2], col1Numeric[3], alpha)
col2Shade <- rgb(col2Numeric[1], col2Numeric[2], col2Numeric[3], alpha)
col3Shade <- rgb(col3Numeric[1], col3Numeric[2], col3Numeric[3], alpha)
col4Shade <- rgb(col4Numeric[1], col4Numeric[2], col4Numeric[3], alpha)

projectName       <- "recapture"
pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp" # put the path to the monte-carlo-rcpp directory here
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp" # put the path to the folder which whill contain the simulation output here

pathToPaper <- "/home/axel/Dropbox/ATI - SSmodel/manuscript_backup/2017-07-27"
pathToPaperFigures <- file.path(pathToPaper, "temp")



## ========================================================================= ##
##
## LITTLE OWLS
##
## ========================================================================= ##

## ------------------------------------------------------------------------- ##
## Model Evidence
## ------------------------------------------------------------------------- ##


exampleName <- "owls"
# jobName <- "smc_sge_array_2017-07-29"
jobName <- "smc_sge_array_2017-08-03"
# jobName <- "smc_sge_array_debug"
source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study
source(file="/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp/projects/recapture/recaptureFunctions.r")
# 
# jobName <- "smc_sge_array_2017-08-04"
# idxReplicates <- (1:20)[-c(9,17,2)]

idxReplicates <- (1:20)[-c(8,10,19)] # indices of independent replicates used by the simulation study
MAX_SIM_SMC       <- length(idxReplicates)
idxConfigurations <- 1:16

processOutputSmc(pathToResults=pathToResults, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicates)

logEvidenceEstimates         <- readRDS(file.path(pathToProcessed, paste("standardLogEvidenceEstimate", ".rds", sep='')))
cpuTime                      <- readRDS(file.path(pathToProcessed, paste("cpuTime", ".rds", sep='')))
# finalSelfNormalisedWeights   <- readRDS(file.path(pathToProcessed, paste("finalSelfNormalisedWeights", ".rds", sep='')))
# finalParameters              <- readRDS(file.path(pathToProcessed, paste("finalParameters", ".rds", sep='')))

if (paperType=="biom") {
  ## For inclusion in supplementary materials:
  WIDTH  <- 5
  HEIGHT <- 4
  ## For inclusion in main body of the paper:
#   WIDTH  <- 3
#   HEIGHT <- 3
} else {
  WIDTH  <- 4
  HEIGHT <- 3
}

tckLocalX       <- -0.02
mgpLocalX       <- c(3, 0.3, 0)
tckLocalY       <- -0.02
mgpLocalY       <- c(3, 0.5, 0)

N_CONFIGS <- 16
atPlot    <- c(1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23)
atLabels  <- c()
for (ii in 1:(N_CONFIGS/2)) {
  atLabels <- c(atLabels, atPlot[2*ii]-0.5)
}
 
 
if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToPaperFigures, "owls_evidence_boxplot"), ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToPaperFigures, "owls_evidence_boxplot"), ".tex", sep=''), width=WIDTH, height=HEIGHT)
} else if (imageType == "none") {
  # EMPTY
}
op <- par(mar=c(1.5,1.6,0.1,0.1), oma=c(1.6,1.6,0.1,0.1))

  boxplot(t(logEvidenceEstimates), 
    ylab="",
    xlab="",
    at=atPlot,
    range=0,
    las=2,
    axes=FALSE,
    names="",
    ylim = c(-340,-310),
    yaxs='i',
    main="",
    col=rep(c(col3Shade, col2Shade), times=5),
    border=rep(c(col3, col2), times=5),
    cex=cex
  )
  
  mtext(text="Log-evidence", side=2, line=2.2, outer=FALSE, cex=cexMtext)
  axis(side=2, at=c(-340,-330,-320,-310), labels=c(-340,-330,-320,-310), las=1, cex.axis=cexAxis, mgp=mgpLocalY, tck=tckLocalY)
  mtext(text="Model", side=1, line=1.85, outer=FALSE, cex=cexMtext)
  axis(side=1, at=atLabels, labels=c("$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"), las=1, cex.axis=1.0, mgp=mgpLocalX, tck=tckLocalX)
  box()
  legend("topleft", legend=c("immigration depends on vole abundance", "immigration independent of vole abundance"), border=c(col3, col2), bty='n', fill=c(col3Shade, col2Shade), cex=cexLegend*0.95, seg.len=1.5, xjust=1, yjust=0)
#   legend("bottomright", legend=c("$\\delta_1 \\neq 0$", "$\\delta_1 = 0$"), border=c(col1, col2), bty='n', fill=c(col1Shade, col2Shade), cex=cexLegend, seg.len=1.5, xjust=1, yjust=0)
  
#   abline(h=-310, lty=3)
  abline(h=-320, lty=3)
  abline(h=-330, lty=3)
#   abline(h=-340, lty=3)
  
  if (imageType != "none") {
    dev.off()
  }
par(op)




## ------------------------------------------------------------------------- ##
## Investigating the advantage of "double" tempering
## ------------------------------------------------------------------------- ##

exampleName <- "owls"
idxConfigurations <- 1:16

## Benchmark results:

pathToResultsBenchmark <- "~/Dropbox/research/output/cpp/monte-carlo-rcpp/projects/recapture/owls/mcmc_sge_array_2017-08-05/results"

# jobName <- "smc_sge_array_2017-08-03" 
# source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study
# source(file="/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp/projects/recapture/recaptureFunctions.r")
# 
# idxReplicatesBenchmark <-(1:20)[-c(8,10,19)] # indices of independent replicates used by the simulation study
# pathToResultsBenchmark <- pathToResults
# processOutputSmc(pathToResults=pathToResultsBenchmark, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicatesBenchmark)
# cpuTimeBenchmark              <- readRDS(file.path(pathToProcessed, paste("cpuTime", ".rds", sep='')))
# logEvidenceEstimatesBenchmark <- readRDS(file.path(pathToProcessed, paste("standardLogEvidenceEstimate", ".rds", sep='')))

## "Single" tempering:
# jobName <- "smc_sge_array_2017-08-01"
# idxReplicatesSingle <- (1:100)
# jobName <- "smc_sge_array_2017-08-04"
# idxReplicatesSingle <- (1:20)
jobName <- "smc_sge_array_2017-08-06"
idxReplicatesSingle <- (1:89)[-c(40,60,64,70,71,74,75,76,85,86,87,89)]

source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study
source(file="/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp/projects/recapture/recaptureFunctions.r")


pathToResultsSingle <- pathToResults
processOutputSmc(pathToResults=pathToResultsSingle, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicatesSingle)
cpuTimeSingle              <- readRDS(file.path(pathToProcessed, paste("cpuTime", ".rds", sep='')))
logEvidenceEstimatesSingle <- readRDS(file.path(pathToProcessed, paste("standardLogEvidenceEstimate", ".rds", sep='')))

## "Double" tempering:
# jobName <- "smc_sge_array_2017-08-02"
# idxReplicatesDouble <- (1:100) # indices of independent replicates used by the simulation study
# jobName <- "smc_sge_array_2017-08-03"
# idxReplicatesDouble <- (1:20)[-c(8,10,19)] # indices of independent replicates used by the simulation study
jobName <- "smc_sge_array_2017-08-07"
idxReplicatesDouble <- (1:100)[-c(2,3,24,26,27,29,32,34,35,38,39,42,43,46,47,50,52,53,54,55,56,59,60,63,64,67,73,74,77)] # indices of independent replicates used by the simulation study
source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study
source(file="/home/axel/Dropbox/research/code/cpp/monte-carlo-rcpp/projects/recapture/recaptureFunctions.r")

pathToResultsDouble <- pathToResults
processOutputSmc(pathToResults=pathToResultsDouble, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicatesDouble)
cpuTimeDouble              <- readRDS(file.path(pathToProcessed, paste("cpuTime", ".rds", sep='')))
logEvidenceEstimatesDouble <- readRDS(file.path(pathToProcessed, paste("standardLogEvidenceEstimate", ".rds", sep='')))

meanCpuTimeSingle <- rowMeans(cpuTimeSingle)
meanCpuTimeDouble <- rowMeans(cpuTimeDouble)
meanCpuTimeBenchmark <- rowMeans(cpuTimeBenchmark)



## Computes the MSE of various posterior moments:
computeMse <- function(idxConfigurations, pathToResults, idxReplicates, pathToResultsBenchmark) 
{
  avgMseMoment1  <- rep(0, times=length(idxConfigurations))
  avgMseMoment2  <- rep(0, times=length(idxConfigurations))
  avgMseVar      <- rep(0, times=length(idxConfigurations))

  for (ii in 1:length(idxConfigurations)) {
   
      X <- readRDS(file.path(pathToResultsBenchmark, paste("parameters_", idxConfigurations[ii], "_", 1, ".rds", sep='')))
  
      moment1Benchmark <- rowMeans(X)
      moment2Benchmark <- rowMeans(X^2)
      varBenchmark     <- rowMeans(X^2) - rowMeans(X)^2
      
      print(moment1Benchmark)
      
      dimTheta <- dim(X)[1] 
  
#     X <- readRDS(file.path(pathToResultsBenchmark, paste("finalParameters_", idxConfigurations[ii], "_", idxReplicates[1], ".rds", sep='')))
#     
#     dimTheta   <- dim(X)[1] 
#     
#     moment1Benchmark <- rep(0, times=dimTheta)
#     moment2Benchmark <- rep(0, times=dimTheta)
#     varBenchmark     <- rep(0, times=dimTheta)
#     
#     ## Generating "true", i.e. benchmark, estimates
#     for (mm in 1:length(idxReplicatesBenchmark)) {
#       
#       X <- readRDS(file.path(pathToResultsBenchmark, paste("finalParameters_", idxConfigurations[ii], "_", idxReplicatesBenchmark[mm], ".rds", sep='')))
#       W <- readRDS(file.path(pathToResultsBenchmark, paste("finalSelfNormalisedWeights_", idxConfigurations[ii], "_", idxReplicatesBenchmark[mm], ".rds", sep='')))
#       
#       # computing average (over all parameters) posterior moments:
#       
#       moment1Benchmark <- moment1Benchmark + X %*% as.matrix(W) / length(idxReplicatesBenchmark)
#       moment2Benchmark <- moment2Benchmark + X^2 %*% as.matrix(W) / length(idxReplicatesBenchmark)
#       varBenchmark     <- varBenchmark     + (X^2 %*% as.matrix(W) - (X %*% as.matrix(W))^2) / length(idxReplicatesBenchmark)
#     }
   
   
    moment1Estimates <- matrix(NA, dimTheta, length(idxReplicates))
    moment2Estimates <- matrix(NA, dimTheta, length(idxReplicates))
    varEstimates     <- matrix(NA, dimTheta, length(idxReplicates))
    
    ## Generating estimates:
    for (mm in 1:length(idxReplicates)) {
    
      X <- readRDS(file.path(pathToResults, paste("finalParameters_", idxConfigurations[ii], "_", idxReplicates[mm], ".rds", sep='')))
      W <- readRDS(file.path(pathToResults, paste("finalSelfNormalisedWeights_", idxConfigurations[ii], "_", idxReplicates[mm], ".rds", sep='')))
      
      moment1Estimates[,mm] <- X %*% as.matrix(W)
      moment2Estimates[,mm] <- X^2 %*% as.matrix(W)
      varEstimates[,mm]     <- X^2 %*% as.matrix(W) - (X %*% as.matrix(W))^2
    }
    
    ## Computing the average (over all parameters) MSEs:
    for (jj in 1:dimTheta) {
      avgMseMoment1[ii] <- avgMseMoment1[ii] + mean((moment1Estimates[jj,] - moment1Benchmark[jj])^2) / dimTheta
      avgMseMoment2[ii] <- avgMseMoment2[ii] + mean((moment2Estimates[jj,] - moment2Benchmark[jj])^2) / dimTheta
      avgMseVar[ii]     <- avgMseVar[ii]     + mean((varEstimates[jj,] - varBenchmark[jj])^2) / dimTheta
    }
  }
  return(list(avgMseMoment1=avgMseMoment1, avgMseMoment2=avgMseMoment2, avgMseVar=avgMseVar))
}

## MSEs of posterior moments 
aux <- computeMse(idxConfigurations=idxConfigurations, pathToResults=pathToResultsSingle, idxReplicates=idxReplicatesSingle, pathToResultsBenchmark=pathToResultsBenchmark) 
avgMseMoment1Single <- aux$avgMseMoment1
avgMseMoment2Single <- aux$avgMseMoment2
avgMseVarSingle     <- aux$avgMseVar

aux <- computeMse(idxConfigurations=idxConfigurations, pathToResults=pathToResultsDouble, idxReplicates=idxReplicatesDouble, pathToResultsBenchmark=pathToResultsBenchmark) 
avgMseMoment1Double <- aux$avgMseMoment1
avgMseMoment2Double <- aux$avgMseMoment2
avgMseVarDouble     <- aux$avgMseVar



(avgMseMoment1Single * meanCpuTimeSingle)/(avgMseMoment1Double * meanCpuTimeDouble)
(avgMseMoment2Single * meanCpuTimeSingle)/(avgMseMoment2Double * meanCpuTimeDouble)
(avgMseVarSingle * meanCpuTimeSingle)/(avgMseVarDouble * meanCpuTimeDouble)



varrho <- (avgMseMoment1Single * meanCpuTimeSingle)/(avgMseMoment1Double * meanCpuTimeDouble)
print(varrho[seq(from=1,to=15, by=2)], digits=1)


## Variance/MSE of the likelihood estimate multiplied computation time

## Variance of the evidence estimates
relVarSingle <- diag(var(t(exp(logEvidenceEstimatesSingle)))) * meanCpuTimeSingle
relVarDouble <- diag(var(t(exp(logEvidenceEstimatesDouble)))) * meanCpuTimeDouble
relVarSingle/relVarDouble

## Variance of the log of the evidence estimates
relVarSingle <- diag(var(t(logEvidenceEstimatesSingle))) * meanCpuTimeSingle
relVarDouble <- diag(var(t(logEvidenceEstimatesDouble))) * meanCpuTimeDouble
relVarSingle/relVarDouble

## MSE of the evidence estimates

truth      <- as.numeric(rowMeans(exp(logEvidenceEstimatesBenchmark))) ##as.numeric(rowMeans(exp(logEvidenceEstimatesDouble)))
meanSingle <- as.numeric(rowMeans(exp(logEvidenceEstimatesSingle)))
meanDouble <- as.numeric(rowMeans(exp(logEvidenceEstimatesDouble)))
varSingle  <- diag(var(t(exp(logEvidenceEstimatesSingle))))
varDouble  <- diag(var(t(exp(logEvidenceEstimatesDouble))))
biasSingle <- meanSingle - truth
biasDouble <- meanDouble - truth

relMseSingle <- (varSingle + biasSingle^2) * meanCpuTimeSingle
relMseDouble <- (varDouble + biasDouble^2) * meanCpuTimeDouble
relMseSingle/relMseDouble


# plot(meanSingle * (1000), type='l', col="blue")
# lines(meanDouble * (500), type='l', col="green")
# lines(truth * (20000), type='l', col="red")
# 
# plot(meanSingle, type='l', col="blue")
# lines(meanDouble, type='l', col="green")
# lines(truth, type='l', col="red")








## MSE of the log of the evidence estimates

truth      <- as.numeric(rowMeans(logEvidenceEstimatesBenchmark)) ##as.numeric(rowMeans(logEvidenceEstimatesDouble))
meanSingle <- as.numeric(rowMeans(logEvidenceEstimatesSingle))
meanDouble <- as.numeric(rowMeans(logEvidenceEstimatesDouble))
varSingle  <- diag(var(t(logEvidenceEstimatesSingle)))
varDouble  <- diag(var(t(logEvidenceEstimatesDouble)))
biasSingle <- meanSingle - truth
biasDouble <- meanDouble - truth

relMseSingle <- (varSingle + biasSingle^2) * meanCpuTimeSingle
relMseDouble <- (varDouble + biasDouble^2) * meanCpuTimeDouble

round(relMseSingle/relMseDouble, digits=1)


forTable <- round(relMseSingle/relMseDouble, digits=1)
print(forTable[c(1,3,5,7,9,11,13,15)])
print(forTable[c(2,4,6,8,10,12,14,16)])



op <- par(mfrow=c(3,1))
YLIM <- range(cbind(logEvidenceEstimatesSingle, logEvidenceEstimatesDouble))
boxplot(t(logEvidenceEstimatesSingle), range=0, ylim=YLIM)
boxplot(t(logEvidenceEstimatesDouble), range=0, ylim=YLIM)
boxplot(t(logEvidenceEstimatesBenchmark), range=0, ylim=YLIM)
par(op)

meanCpuTimeSingle/meanCpuTimeDouble


plot(meanSingle, type='l', col="blue")
lines(meanDouble, type='l', col="green")
lines(truth, type='l', col="red")

## ------------------------------------------------------------------------- ##
## Autocorrelation
## ------------------------------------------------------------------------- ##


exampleName <- "owls"
# jobName <- "mcmc_sge_array_2017-06-07"
jobName <- "mcmc_sge_array_2017-07-30"
source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study
# Miscellaneous parameters:
miscParameters <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)


# pathToMcmcResults <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp/projects/recapture/owls/mcmc_sge_array_2017-06-07/results/"
pathToMcmcResults <- "/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp/projects/recapture/owls/mcmc_sge_array_2017-07-30/results/"
MAX_SIM_MCMC <- 2

ii <- 1
maxLag  <- 5000
maxPlot <- 10
grid <- 0:maxLag
nIterations <- 10000000

tckLocalX       <- -0.04
mgpLocalX       <- c(3, 0.1, 0)
tckLocalY       <- -0.04
mgpLocalY       <- c(3, 0.5, 0)


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
cpuTimeFactor            <- mean(cpuTime)/mean(cpuTimeNonDa)
cpuTimePerIteration      <- mean(cpuTime) / nIterations
cpuTimePerIterationNonDa <- mean(cpuTimeNonDa) / nIterations


parIndices <- c(1,5) ## indices of the parameters for which we 
parNames   <- c("$\\alpha_0$", "$\\beta_1$")


if (paperType=="biom") {
  WIDTH  <- 3.3
  HEIGHT <- 2.0
} else {
  WIDTH  <- 4.7
  HEIGHT <- 3
}

if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToPaperFigures, "owls_acf"), ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToPaperFigures, "owls_acf"),  ".tex", sep=''), width=WIDTH, height=HEIGHT)
} else if (imageType == "none") {
  # EMPTY
}

op <- par(mfrow=c(1,2), mar=c(1.2,1.2,0.5,0.1), oma=c(1.2,1.2,0.5,0.2))

for (jj in 1:length(parIndices)) {

  acf  <- readRDS(paste(pathToMcmcResults, "/", "acf", "_", 1, "_", 1, ".rds", sep=''))
  plot(grid, acf[parIndices[jj],1:(maxLag+1)], type='l', col="white", xlab="", ylab="", ylim=c(0,1), xlim=c(0,maxPlot), xaxs='i', yaxs='i', axes=FALSE, cex=cex)
  for (mm in 1:MAX_SIM_MCMC) {
    acf  <- readRDS(paste(pathToMcmcResults, "/", "acf", "_", 1, "_", mm, ".rds", sep=''))
    lines(grid*cpuTimePerIterationNonDa, acf[parIndices[jj],1:(maxLag+1)], type='l', col=col3, lty=1)
    acf  <- readRDS(paste(pathToMcmcResults, "/", "acf", "_", 2, "_", mm, ".rds", sep=''))
    lines(grid*cpuTimePerIteration, acf[parIndices[jj],1:(maxLag+1)], type='l', col=col2, lty=1)
  }
  box()
  
  if (jj == 1){
    mtext(text="Autocorrelation", side=2, line=1.5, outer=FALSE, cex=cexMtext)
    mtext(text="Lag $\\times$ CPU time per iteration [in seconds]", side=1, line=0.07, outer=TRUE, cex=cexMtext)
    axis(side=2, las=1, cex.axis=cexAxis, tck=tckLocalY, mgp=mgpLocalY)
  }
  
  if (jj == 2){
    legend(x=2, y=1.04, legend=c("without DA", "with DA"), col=c(col3, col2), bty='n', lty=c(1,1), cex=cexLegend, seg.len=1.5)
  }
  axis(side=1, las=1, cex.axis=cexAxis, tck=tckLocalX, mgp=mgpLocalX)
  mtext(text=paste("Parameter: ", parNames[jj], sep=''), side=3, line=0.25, outer=FALSE, cex=cexMtext)
  

}

par(op)
if (imageType != "none") {
  dev.off()
}

  


cpuTime <- readRDS("/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp/projects/recapture/owls/smc_sge_array_2017-07-29/processed/cpuTime.rds")
print(rowMeans(cpuTime / 3600)*60, digits=1)


# print(matrix(rowMeans(cpuTime / 3600), 9, 3), digits=1)









## ========================================================================= ##
##
## GREY HERONS 
##
## ========================================================================= ##


exampleName <- "herons"

## ------------------------------------------------------------------------- ##
## Model Evidence
## ------------------------------------------------------------------------- ##

jobName <- "smc_sge_array_2017-06-29"
source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study

tckLocal       <- -0.03
mgpLocal       <- c(3, 0.9, 0)


MODEL_TYPE_NAMES         <- c(
  "Constant",
  "Regressed on number of frost days", 
  "Direct-density dependence", 
  "Threshold dependence", 
  "Threshold dependence (true counts)", 
  "Regime switching"
)

MODEL_NAMES <- c()
for (ii in 1:N_CONFIGS) {
  if (MODELS[ii] %in% c(3:5) && N_LEVELS[ii] == 3) {
    MODEL_NAMES <- c(MODEL_NAMES, paste(MODEL_TYPE_NAMES[MODELS[ii]+1], "~~ $K=", N_LEVELS[ii], " $", sep=''))
  } else if (MODELS[ii] %in% c(3:5) && N_LEVELS[ii] != 3) {
    MODEL_NAMES <- c(MODEL_NAMES, paste("$K=", N_LEVELS[ii], " $", sep=''))
  } else {
    MODEL_NAMES <- c(MODEL_NAMES, paste(MODEL_TYPE_NAMES[MODELS[ii]+1], sep=''))
  }
}








modelNames <- MODEL_NAMES

MAX_SIM_SMC       <- 10
idxReplicates     <- 1:MAX_SIM_SMC # indices of independent replicates used by the simulation study
idxConfigurations <- 28:54

processOutputSmc(pathToResults=pathToResults, pathToProcessed=pathToProcessed, idxConfigurations=idxConfigurations, idxReplicates=idxReplicates)

logEvidenceEstimates         <- readRDS(file.path(pathToProcessed, paste("standardLogEvidenceEstimate", ".rds", sep='')))
# cpuTime                      <- readRDS(file.path(pathToProcessed, paste("cpuTime", ".rds", sep='')))
# finalSelfNormalisedWeights   <- readRDS(file.path(pathToProcessed, paste("finalSelfNormalisedWeights", ".rds", sep='')))
# finalParameters              <- readRDS(file.path(pathToProcessed, paste("finalParameters", ".rds", sep='')))


if (paperType=="biom") {
  WIDTH  <- 6.75
  HEIGHT <- 5
} else {
  WIDTH  <- 4.7
  HEIGHT <- 4.5
}


if (imageType == "pdf") {
  pdf(file=paste(file.path(pathToPaperFigures, "herons_evidence_boxplot"), ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
} else if (imageType == "tex") {
  tikz(file=paste(file.path(pathToPaperFigures, "herons_evidence_boxplot"), ".tex", sep=''), width=WIDTH, height=HEIGHT)
} else if (imageType == "none") {
  # EMPTY
}

atPlot    <- c(1,2.5,4,5.5,6.5,7.5,9,10,11)
atLabels  <- atPlot
 
op <- par(mfrow=c(1,length(N_AGE_GROUPS_AUX)), mar=c(9,2,0.8,0.1), oma=c(9,2,0.8,0.1))
for (ii in 1:length(N_AGE_GROUPS_AUX)) {

  auxIdx <- which(N_AGE_GROUPS[idxConfigurations] == N_AGE_GROUPS_AUX[ii])
  
  horizontalLines <- seq(from=-1300, to=-1150, by=10)
  yAxisTickmarks <- horizontalLines
  
  logZ <- logEvidenceEstimates
  
  logZ[logEvidenceEstimates == -Inf] <- - 10^10
  
  boxplot(t(logZ[auxIdx,]), 
    ylab="",
    xlab="",
    range=0,
    at=atPlot,
    las=2,
    axes=FALSE,
    names="",
#     names=modelNames[auxIdx],
    ylim=range(logZ[logEvidenceEstimates != -Inf]),
    col=rep(c(col4Shade, col3Shade, col2Shade), each=3),
    border=rep(c(col4, col3, col2), each=3),
    main="",
    cex=1.0
  )
  if (ii == 1) {
    mtext(text="Log-evidence", side=2, line=3, outer=FALSE, cex=cexMtext)
    axis(side=2, at=horizontalLines, labels=yAxisTickmarks, las=1, cex.axis=1.0, tck=tckLocal, mgp=mgpLocal)
  }
  if (ii == 2) {
    mtext(text="Model for the productivity rate", side=1, line=16.5, outer=FALSE, cex=cexMtext)
  }
  mtext(text=paste("No.\\ of age groups: ", N_AGE_GROUPS_AUX[ii], sep=''), side=3, line=0.5, outer=FALSE, cex=cexMtext)
  axis(side=1, at=atLabels, labels=modelNames[auxIdx], las=2,  cex.axis=1.5, tck=tckLocal, mgp=mgpLocal)

#   axis(side=1, at=1:length(auxIdx), labels=modelNames[auxIdx], las=2,  cex.axis=1.5, tck=tckLocal, mgp=mgpLocal)
  box()
  
  y1 <- -1262.5
  y0 <- -1264.5
  
  x0 <- atPlot[4]-0.45
  x1 <- atPlot[6]+0.65
  abline(h=horizontalLines, lty=3)
  lines(c(x0,x1), c(y0,y0), xpd=TRUE, type='l') 
  lines(c(x0,x0), c(y0,y1), xpd=TRUE, type='l')
  lines(c(x1,x1), c(y0,y1), xpd=TRUE, type='l')
  
  x0 <- atPlot[7]-0.45
  x1 <- atPlot[9]+0.65
  lines(c(x0,x1), c(y0,y0), xpd=TRUE, type='l') 
  lines(c(x0,x0), c(y0,y1), xpd=TRUE, type='l')
  lines(c(x1,x1), c(y0,y1), xpd=TRUE, type='l')

}
if (imageType != "none") {
  dev.off()
}
par(op)




## ------------------------------------------------------------------------- ##
## Overview
## ------------------------------------------------------------------------- ##

jobName <- "smc_sge_array_2017-06-29"
source(file=file.path(pathToInputBase, "setupRCpp.r")) # load parameters used for the simulation study

modelNames <- MODEL_NAMES
idxReplicates     <- 1:10
idxConfigurations <- c(46,47,48,51,54)

# Miscellaneous parameters:
miscParameters      <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)
# Auxiliary (known) model parameters and data:


addShaded <- function(x, probs=c(1,0.9), alpha=c(0.18,0.3), col="magenta3", lwd=1) {
  # x: (N,T)-matrix where we average over N samples and display the results for T time steps.
  # probs: a vector of coverage probabilities.
  # col: a single string indicating the colour.
  # lwd: a single value indicating the width of the line representing the median.
  # alpha: transparency parameters (must be the same length as probs).

  T <- dim(x)[2]
  colNumeric <- as.numeric(col2rgb(col))/256
  nProbs  <- length(probs)
  xProbs  <- array(NA, c(T, 2, nProbs))
  xMedian <- rep(NA, times=T)
  
  for (t in 1:T) {
    xMedian[t] <- median(x[,t])
    for (j in 1:nProbs) {
      xProbs[t,1,j] <- quantile(x[,t], probs=(1-probs[j])/2)
      xProbs[t,2,j] <- quantile(x[,t], probs=1-(1-probs[j])/2)
    }
  }

  for (j in 1:nProbs) {
    polygon(c(1:T, T:1), c(xProbs[1:T,2,j], xProbs[T:1,1,j]), border=NA, col=rgb(colNumeric[1], colNumeric[2], colNumeric[3], alpha[j]))
  }
  lines(1:T, xMedian, col=col, type='l')
}


col1    <- "forestgreen"
colTrue <- "black"

tckLocalX       <- -0.04
mgpLocalX       <- c(3, 0.1, 0)
tckLocalY       <- -0.04
mgpLocalY       <- c(3, 0.3, 0)


if (paperType=="biom") {
  WIDTH  <- 5.5
  HEIGHT <- 2.8
} else {
  WIDTH  <- 4.5
  HEIGHT <- 2.8
}


for (ii in 1:length(idxConfigurations)) {

  nAgeGroups <- N_AGE_GROUPS[idxConfigurations[ii]]
  nLevels    <- N_LEVELS[idxConfigurations[ii]]
  modelType  <- MODELS[idxConfigurations[ii]]
  lower      <- LOWER[idxConfigurations[ii]]
  
  modelParameters   <- getAuxiliaryModelParameters(modelType, nAgeGroups, nLevels, miscParameters)
  
  productivityRates <- readRDS(file.path(pathToResults, paste("productivityRates_", idxConfigurations[ii], "_", idxReplicates[1], ".rds", sep='')))
  N <- dim(productivityRates)[2]
  K <- length(idxReplicates)
  for (kk in 2:K) {
    productivityRates <- cbind(productivityRates, readRDS(file.path(pathToResults, paste("productivityRates_", idxConfigurations[ii], "_", idxReplicates[kk], ".rds", sep=''))))
  }
  
  T <- modelParameters$nObservationsCount
  
  trueCounts <- array(NA, c(nAgeGroups, T, N*K))
  smoothedMeans <- array(NA, c(nAgeGroups, T, N*K))
  smoothedVariances <- array(NA, c(nAgeGroups, T, N*K))
  if (lower == 3) {
    for (kk in 1:K) {
      smoothedMeans[,,(N*(kk-1)+1):(N*kk)]     <- readRDS(file.path(pathToResults, paste("smoothedMeans_", idxConfigurations[ii], "_", idxReplicates[kk], ".rds", sep='')))
      smoothedVariances[,,(N*(kk-1)+1):(N*kk)] <- readRDS(file.path(pathToResults, paste("smoothedVariances_", idxConfigurations[ii], "_", idxReplicates[kk], ".rds", sep='')))
    }
    for (t in 1:T) {
      for (n in 1:(N*K)) {
        trueCounts[,t,n] <- rnorm(nAgeGroups, smoothedMeans[,t,n], sqrt(smoothedVariances[,t,n]))
      }
    }
  } else {
    for (kk in 1:K) {
      trueCounts[,,(N*(kk-1)+1):(N*kk)] <- readRDS(file.path(pathToResults, paste("trueCounts_", idxConfigurations[ii], "_", idxReplicates[kk], ".rds", sep='')))
    }
  }

  adultCounts <- matrix(NA, T, N*K)
  for (t in 1:T) {
    for (n in 1:(N*K)) {
      adultCounts[t,n] <- sum(trueCounts[2:nAgeGroups,t,n])
    }
  }

  if (imageType == "pdf") {
    pdf(file=paste(file.path(pathToPaperFigures, "herons_overview_"), ii, ".pdf", sep=''), width=WIDTH_BOXPLOT, height=HEIGHT_BOXPLOT)
  } else if (imageType == "tex") {
    tikz(file=paste(file.path(pathToPaperFigures, "herons_overview_"), ii, ".tex", sep=''), width=WIDTH, height=HEIGHT)
  } else if (imageType == "none") {
    # EMPTY
  }
  op <- par(mfrow=c(2,1), mar=c(0.5,1.4,0.2,0.1), oma=c(0.5,1.4,0.1,0.1))
  
  ## Counts:
  plot(1:T, rep(1,T), col="white", xlab='', ylab='', ylim=c(2000,max(modelParameters$count)), yaxs='i', xaxs='i', xaxt="n", yaxt="n", cex=cex)
  addShaded(t(adultCounts), col=col1)
  lines(1:T, modelParameters$count, type='p', pch=1, cex=0.5, col=colTrue) # observed counts

  axis(side=2, cex.axis=cexAxis, tck=tckLocalY, mgp=mgpLocalY)
#   axis(side=1, seq(from=3, to=70, by=10), labels=seq(from=1930, to=1990, by=10), cex.axis=cexAxis, mgp=mgp)
  mtext("Population (age $>$ 1)", side=2, outer=FALSE, line=1.5, cex=cexMtext)
  legend(x=2, y=7150, legend=c("estimated counts  (posterior median)", "observed count"), col=c(col1, colTrue), pch=c(-1,1), pt.cex=0.5, lty=c(1,-1), bty='n', cex=cexLegend)

  ## Productivity Rates:
  plot(1:T, rep(1,T), col="white", xlab='', ylab='', ylim=c(0,2.3), yaxs='i', xaxs='i', xaxt="n", yaxt="n", cex=cex)
  addShaded(t(productivityRates), col=col1)
  axis(side=2, at=c(0,1,2), labels=c(0,1,2), cex.axis=cexAxis, cex.axis=cexAxis, tck=tckLocalY, mgp=mgpLocalY)
  axis(side=1, seq(from=3, to=70, by=10), labels=seq(from=1930, to=1990, by=10), cex.axis=cexAxis, tck=tckLocalX, mgp=mgpLocalX)
  mtext("Productivity", side=2, outer=FALSE, line=1.5, cex=cexMtext)
  legend(x=1, y=2.4, legend=c("estimated productivity rate (posterior median)"), col=c(col1), lty=c(1), bty='n', cex=cexLegend)
  
  if (imageType != "none") {
    dev.off()
  }
  par(op)

}


cpuTime <- readRDS("/home/axel/Dropbox/research/output/cpp/monte-carlo-rcpp/projects/recapture/herons/smc_sge_array_2017-06-29/processed/cpuTime.rds")


print(matrix(rowMeans(cpuTime / 3600), 9, 3), digits=1)



mkdir ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-27/temp
cd ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-27/temp

pdflatex biometGraphics.tex

mv ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-27/temp/biometGraphics.pdf ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-27/fig

cd ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-27/fig

pdftk biometGraphics.pdf burst output %02d.pdf

cd ~/Dropbox/"ATI - SSmodel"/manuscript_backup/2017-07-27/

pdflatex main.tex

pdflatex main.tex

pdflatex supplementary_materials.tex

pdflatex supplementary_materials.tex





