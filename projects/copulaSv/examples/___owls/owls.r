## PMMH with delayed acceptance in the "little-owls" capture-recapture model

rm(list = ls())
set.seed(123)

pathToInputBase   <- "/home/axel/Dropbox/research/code/cpp/mc"
pathToOutputBase  <- "/home/axel/Dropbox/research/output/cpp/mc"

exampleName       <- "owls"
projectName       <- "recapture"
jobName           <- "debug"

source(file=file.path(pathToInputBase, "setupRCpp.r"))

## ========================================================================= ##
## MODEL
## ========================================================================= ##

## Selecting the model:
modelType <- 3
# Model to be estimated (needed for model comparison):
# 0: time-dependent capture probabilities; time-dependent productivity rates
# 1: time-independent capture probabilities; time-dependent productivity rates
# 2: time-dependent capture probabilities; time-independent productivity rates
# 3: time-independent capture probabilities; time-independent productivity rates
# 4: same as Model 3 but with alpha_3 = 0
# 5: same as Model 3 but with delta_1 = alpha_3 = 0
# 6: same as Model 3 but with alpha_1 = alpha_3 = 0
# 7: same as Model 3 but with alpha_1 = alpha_3 = delta_1 = 0

smcParameters  <- numeric(0) # additional parameters to be passed to the particle filter 
mcmcParameters <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates

## Miscellaneous parameters:
MISC_PAR      <- list(pathToCovar=pathToCovar, pathToData=pathToData, pathToFigures=pathToFigures, pathToInputBase=pathToInputBase, pathToOutputBase=pathToOutputBase)
## Auxiliary (known) model parameters and data:
MODEL_PAR     <- getAuxiliaryModelParameters(modelType, MISC_PAR)
## Auxiliary parameters for the SMC and MCMC algorithms:
ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(MODEL_PAR$dimTheta, MODEL_PAR$nObservations)

## ========================================================================= ##
## MCMC algorithms
## ========================================================================= ##
# 
# nIterations <- 200000
# nSimulations <- 1
# 
# USE_DELAYED_ACCEPTANCE <- c(0,1,0,1) # should delayed acceptance be used?
# ADAPT_PROPOSAL <- c(0,0,1,1) # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
# N_PARTICLES <- c(2000)
# 
# MM <- nSimulations # number of independent replicates
# LL <- length(USE_DELAYED_ACCEPTANCE)
# NN <- length(N_PARTICLES)
# 
# 
# outputTheta  <- array(NA, c(MODEL_PAR$dimTheta, nIterations, LL, NN, MM))
# outputCpuTime <- array(NA, c(LL, NN, MM))
# outputAcceptanceRateStage1 <- array(NA, c(LL, NN, MM))
# outputAcceptanceRateStage2 <- array(NA, c(LL, NN, MM))
# 
# for (mm in 1:MM) {
# 
#   thetaInit <- sampleFromPriorCpp(MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nCores); # making sure that all algorithms are initialised in the same way
#   
#  
#   for (nn in 1:NN) { 
#     for (ll in 1:LL) {
#     
#       aux <- runPmmhCpp(
#         MODEL_PAR$fecundity, MODEL_PAR$count, MODEL_PAR$capRecapFemaleFirst, MODEL_PAR$capRecapMaleFirst, MODEL_PAR$capRecapFemaleAdult, MODEL_PAR$capRecapMaleAdult,
#         MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nIterations, N_PARTICLES[nn], ALGORITHM_PAR$essResamplingThresholdLower, smcParameters, 
#         mcmcParameters, USE_DELAYED_ACCEPTANCE[ll], ADAPT_PROPOSAL[ll], ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, thetaInit, ALGORITHM_PAR$burninPercentage, nCores 
#       )
#       
#       outputTheta[,,ll,nn,mm] <- matrix(unlist(aux$theta), MODEL_PAR$dimTheta, nIterations)
#       outputCpuTime[ll,nn,mm] <- unlist(aux$cpuTime)
#       outputAcceptanceRateStage1[ll,nn,mm] <- unlist(aux$acceptanceRateStage1)
#       outputAcceptanceRateStage2[ll,nn,mm] <- unlist(aux$acceptanceRateStage2)
#         
#     }
#   }
#   
#   save(
#     list  = ls(envir = environment(), all.names = TRUE), 
#     file  = paste(pathToOutputBase, "pmmh_nIterations_", nIterations, "_nSimulations_", nSimulations, "_modelType_", modelType, sep=''), ## TODO
#     envir = environment()
#   ) 
# }
# 




## ========================================================================= ##
## PLOT RESULTS
## ========================================================================= ##

# load(paste(pathToOutputBase, "pmmh_nIterations_1e+05_nSimulations_1_modelType_0", sep=''))

#########################
## Summary statistics
# X <- outputTheta[,50000:100000,3,1,1]
# Z <- summary(t(X))
# postMean <- round(apply(X,1,mean), digits=2)
# postSd <- round(apply(X,1,sd), digits=2)
# 
# 
# TAB <- data.frame(postMean, postSd)
# rownames(TAB) <- thetaNames
# colnames(TAB) <- c("mean", "sd")
#########################


COL_LL  <- c("royalblue4", "red4", "royalblue", "red2")
# LTY_NN  <- c(2,1)
LTY_NN  <- c(1)
LAG_MAX <- 500
PAR_LIM <- cbind(rep(-5, times=MODEL_PAR$dimTheta), rep(5, times=MODEL_PAR$dimTheta))
KDE_LIM <- rep(5, times=MODEL_PAR$dimTheta)
CEX_LEGEND <- 0.9
ALGORITHMS <- c("Standard PMMH", "Standard PMMH with Delayed Acceptance", "Adaptive PMMH", "Adaptive PMMH with Delayed Acceptance")

WIDTH  <- 15
HEIGHT <- 3.5


plotThetaMcmc <- function(title, theta, idx, burnin, KDE=TRUE, ACF=TRUE, TRACE=TRUE)
{
  II <- length(idx)
  JJ <- sum(KDE, ACF, TRACE)
  
  GG <- dim(theta)[2] # number of iterations
  LL <- dim(theta)[3] # number of different algorithms to compare
  NN <- dim(theta)[4] # number of different numbers of particles to compare
  MM <- dim(theta)[5] # number of independent replicates
  
  pdf(file=file.path(pathToFigures, paste(title, ".pdf", sep='')), width=WIDTH, height=HEIGHT)
      
  for (ii in 1:II) {
    op <- par(mfrow=c(1, JJ))
    if (KDE) {
      plot(density(theta[ii,burnin:GG,LL,nn,mm]), type='l', col="white", xlab=MODEL_PAR$thetaNames[idx[ii]], ylab="Density", xlim=PAR_LIM[idx[ii],], ylim=c(0, KDE_LIM[idx[ii]]), main='')
      for (ll in 1:LL) {
        for (nn in 1:NN) {
          for (mm in 1:MM) {
            lines(density(theta[ii,burnin:GG,ll,nn,mm]), type='l', col=COL_LL[ll], lty=LTY_NN[nn], main='')
          }
        }
      }
      grid <- seq(from=min(PAR_LIM[idx[ii],]), to=max(PAR_LIM[idx[ii],]), length=10000)
      lines(grid, dnorm(grid, mean=MODEL_PAR$meanHyper[idx[ii]], sd=MODEL_PAR$sdHyper[idx[ii]]), col="black", lty=2)
      legend("topleft", legend=c(ALGORITHMS, "Prior"), col=c(rep(COL_LL,2), "black"), bty='n', lty=c(rep(LTY_NN, each=LL), 2), cex=CEX_LEGEND)
    }
    if (ACF) {
      plot(0:LAG_MAX, rep(1, times=LAG_MAX+1), type='l', col="white", xlab=paste("Lag (",MODEL_PAR$thetaNames[idx[ii]],")", sep=''), ylab="ACF", ylim=c(0,1), xlim=c(0,LAG_MAX))
      mtext(paste("No. of iterations: ", GG, "; of which burn-in: ", burnin, sep=''), side = 3, line = 1, outer = FALSE)
      for (ll in 1:LL) {
        for (nn in 1:NN) {
          for (mm in 1:MM) {    
            ACF_PLOT <-as.numeric(acf(theta[ii,burnin:GG,ll,nn,mm], lag.max=LAG_MAX, plot=FALSE)$acf)
            lines(0:LAG_MAX, ACF_PLOT, type='l', col=COL_LL[ll], lty=LTY_NN[nn])
          }
        }
      }
    }
    if (TRACE) {
      plot(1:GG, rep(1, times=GG), type='l', col="white", ylab=MODEL_PAR$thetaNames[idx[ii]], xlab="Iteration", ylim=PAR_LIM[idx[ii],], xlim=c(1,GG))
      for (ll in 1:LL) {
        for (nn in 1:NN) {
          for (mm in 1:MM) {    
            lines(1:GG, theta[ii,,ll,nn,mm], type='l', col=COL_LL[ll], lty=LTY_NN[nn])
          }
        }
      }
    }
    par(op)
  }
  dev.off()

}


# plotThetaMcmc(paste("mcmc_theta_modelType_", modelType, sep=''), outputTheta, 1:MODEL_PAR$dimTheta, burnin=ceiling(nIterations * ALGORITHM_PAR$burninPercentage))
# 
# ## Plots an overview of the posterior correlations
# 
# # load(paste(pathToOutputBase, "pmmh_nIterations_1e+05_nSimulations_1_modelType_1", sep=''))
# 
# pdf(file=paste(pathToFigures, "mcmc_estimated_posterior_correlations_in_model_", modelType, ".pdf", sep=''), width=WIDTH, height=WIDTH)
# op <- par(oma=c(1,4,4,1), mar=c(1,4,4,1))
# corrMat <- cor(t(outputTheta[,,3,1,1]))
# corrMat <- corrMat[MODEL_PAR$dimTheta:1,]
# image(corrMat, main="", xaxt='n', yaxt='n', col=terrain.colors(500))
# for (i in 1:MODEL_PAR$dimTheta) {
#   for (j in (MODEL_PAR$dimTheta-i+1):MODEL_PAR$dimTheta) {
#     text(x=(i-1)/(MODEL_PAR$dimTheta-1), y=(j-1)/(MODEL_PAR$dimTheta-1), labels=round(corrMat[i,j], digits=2), cex=0.5)
#   }
# }
# axis(3, at=seq(from=0, to=1, length=MODEL_PAR$dimTheta), labels=MODEL_PAR$thetaNames, las=2)
# axis(2, at=seq(from=0, to=1, length=MODEL_PAR$dimTheta), labels=MODEL_PAR$thetaNames[MODEL_PAR$dimTheta:1], las=1)
# par(op)
# dev.off()


## ========================================================================= ##
## SMC Samplers for parameter estimation
## Testing importance tempering!
## ========================================================================= ##
# 
# nParticlesUpper <- 100
# nParticlesLower <- 1000
# useImportanceTempering <- TRUE
# nSteps <- 50
# nSimulations <- 10
# 
# USE_DELAYED_ACCEPTANCE <- c(0,1,0,1) # should delayed acceptance be used?
# USE_ADAPTIVE_TEMPERING <- c(0,0,1,1)
# ADAPT_PROPOSAL <- c(1,1,1,1) # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
# ALPHA <- seq(from=0, to=1, length=nSteps)
# 
# MM <- nSimulations # number of independent replicates
# LL <- length(USE_DELAYED_ACCEPTANCE)
# ll <- 4
# 
# outputLogEvidenceEstimate <- rep(NA, MM)
# posteriorMeans1 <- matrix(NA, MODEL_PAR$dimTheta, MM)
# posteriorMeans2 <- matrix(NA, MODEL_PAR$dimTheta, MM)
# 
# for (mm in 1:MM) {
# 
#     aux <- runSmcSamplerCpp(
#           MODEL_PAR$fecundity, MODEL_PAR$count, MODEL_PAR$capRecapFemaleFirst, MODEL_PAR$capRecapMaleFirst, MODEL_PAR$capRecapFemaleAdult, MODEL_PAR$capRecapMaleAdult,
#           MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, nParticlesUpper, nParticlesLower, ALGORITHM_PAR$essResamplingThresholdUpper, ALGORITHM_PAR$essResamplingThresholdLower, 
#           smcParameters, mcmcParameters, 
#           USE_ADAPTIVE_TEMPERING[ll], useImportanceTempering, USE_DELAYED_ACCEPTANCE[ll], ADAPT_PROPOSAL[ll], ALGORITHM_PAR$cessTarget, ALPHA, ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, nCores
#         )
#          
#     outputLogEvidenceEstimate[mm] <- aux$logEvidenceEstimate
#     
#     theta <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, length(aux$inverseTemperatures), nParticlesUpper))
#     posteriorMeans1[,mm] <- rowMeans(theta[,length(aux$inverseTemperatures),])
#     for (ii in 1:MODEL_PAR$dimTheta) {
#       posteriorMeans2[ii,mm] <- sum(t(aux$W) * theta[ii,,])
#     }
# 
#     save(
#         list  = ls(envir = environment(), all.names = TRUE), 
#         file  = paste(pathToOutputBase, "smc_estimating_model_evidence_nParticlesUpper_", nParticlesUpper, "nParticlesLowerr_", nParticlesLower, "_nSimulations_", nSimulations, "_modelType_", modelType, sep=''), ## TODO
#         envir = environment()
#     ) 
# }
#     
# 
# op <- par(mfrow=c(MODEL_PAR$dimTheta, 1))
# for (ii in 1:(MODEL_PAR$dimTheta)) {
# 
#   boxplot(t(rbind(posteriorMeans1[ii,], posteriorMeans2[ii,])),
#       ylab="Estimate of posterior Mean",
#       xlab=MODEL_PAR$thetaName[ii],
#       range=0,
#       las=1, 
#       names=c("Final-Time samples", "General Importance Tempering"),
#       main=""
#     )
# }
# par(op) 
    
    
    
    
    
    
# print(paste("Log of stimated evidence for model ", modelType, ": ", aux$logEvidenceEstimate, sep=''))
#       
# theta  <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nSteps, nParticlesUpper))
# theta <- theta[,nSteps,]
# 
# WIDTH  <- 10
# pathToFigures <- "/home/axel/Dropbox/ATI - SSmodel/results/owls/figures/"
# 
# 
# plotThetaSmc <- function(title, theta, idx)
# {
#   II <- length(idx)
# 
#   pdf(file=paste(pathToFigures, title, ".pdf", sep=''), width=WIDTH, height=HEIGHT)
#       
#   for (ii in 1:II) {
#     plot(density(theta[ii,]), type='l', col="white", xlab=MODEL_PAR$thetaNames[idx[ii]], ylab="Density", xlim=PAR_LIM[idx[ii],], ylim=c(0, KDE_LIM[idx[ii]]), main='')
#     lines(density(theta[ii,]), type='l', col="red", lty=1, main='')
#     grid <- seq(from=min(PAR_LIM[idx[ii],]), to=max(PAR_LIM[idx[ii],]), length=10000)
#     lines(grid, dnorm(grid, mean=MODEL_PAR$meanHyper[idx[ii]], sd=MODEL_PAR$sdHyper[idx[ii]]), col="black", lty=2)
#     legend("topleft", legend=c("SMC sampler", "Prior"), col=c("red", "black"), bty='n', lty=c(1, 2), cex=CEX_LEGEND)
#   }
#   dev.off()
# }
# 
# 
# plotThetaSmc(paste("smc_theta_modelType_", modelType, sep=''), theta, 1:MODEL_PAR$dimTheta)
# 
# 
# 
# pdf(file=paste(pathToFigures, "smc_estimated_posterior_correlations_in_model_", modelType, ".pdf", sep=''), width=WIDTH, height=WIDTH)
# op <- par(oma=c(1,4,4,1), mar=c(1,4,4,1))
# corrMat <- cor(t(theta))
# corrMat <- corrMat[MODEL_PAR$dimTheta:1,]
# image(corrMat, main="", xaxt='n', yaxt='n', col=terrain.colors(500))
# for (i in 1:MODEL_PAR$dimTheta) {
#   for (j in (MODEL_PAR$dimTheta-i+1):MODEL_PAR$dimTheta) {
#     text(x=(i-1)/(MODEL_PAR$dimTheta-1), y=(j-1)/(MODEL_PAR$dimTheta-1), labels=round(corrMat[i,j], digits=2), cex=0.5)
#   }
# }
# axis(3, at=seq(from=0, to=1, length=MODEL_PAR$dimTheta), labels=MODEL_PAR$thetaNames, las=2)
# axis(2, at=seq(from=0, to=1, length=MODEL_PAR$dimTheta), labels=MODEL_PAR$thetaNames[MODEL_PAR$dimTheta:1], las=1)
# par(op)
# dev.off()




## ========================================================================= ##
## Estimating the model evidence
## ========================================================================= ##

nParticlesUpper <- 5000
nParticlesLower <- 1000
useImportanceTempering <- TRUE
nSteps <- 100
nSimulations <- 100
nMetropolisHastingsUpdates <- 1
nMetropolisHastingsUpdatesFirst <- 2

ALPHA                  <- seq(from=0, to=1, length=nSteps)
USE_DELAYED_ACCEPTANCE <- c(1) # should delayed acceptance be used?
USE_ADAPTIVE_TEMPERING <- c(1)
ADAPT_PROPOSAL         <- c(1) # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
USE_DOUBLE_TEMPERING   <- c(0)

MM <- nSimulations # number of independent replicates
LL <- length(USE_DELAYED_ACCEPTANCE)

MODELS <- c(7)

outputLogEvidenceEstimate <- array(NA, c(length(MODELS), LL, 5, MM))


for (mm in 1:MM) {

  for (ll in 1:LL) {

    for (ii in 1:length(MODELS)) {
    
      print(paste("Owls model selection: ", mm , ", " , ii, sep=''))
    
      ## Auxiliary (known) model parameters:
      MODEL_PAR     <- getAuxiliaryModelParameters(MODELS[ii], MISC_PAR)
      ## Auxiliary parameters for the SMC and MCMC algorithms:
      ALGORITHM_PAR <- getAuxiliaryAlgorithmParameters(MODEL_PAR$dimTheta, MODEL_PAR$nObservations)

      aux <- runSmcSamplerCpp(
            MODEL_PAR$fecundity, MODEL_PAR$count, MODEL_PAR$capRecapFemaleFirst, MODEL_PAR$capRecapMaleFirst, MODEL_PAR$capRecapFemaleAdult, MODEL_PAR$capRecapMaleAdult,
            MODEL_PAR$dimTheta, MODEL_PAR$hyperParameters, MODEL_PAR$support, 0, nParticlesUpper, nParticlesLower, nMetropolisHastingsUpdates, nMetropolisHastingsUpdatesFirst, ALGORITHM_PAR$essResamplingThresholdUpper, ALGORITHM_PAR$essResamplingThresholdLower, 
            smcParameters, mcmcParameters, 
            USE_ADAPTIVE_TEMPERING[ll], useImportanceTempering, USE_DELAYED_ACCEPTANCE[ll], ADAPT_PROPOSAL[ll], USE_DOUBLE_TEMPERING[ll], ALGORITHM_PAR$cessTarget, ALGORITHM_PAR$cessTargetFirst, ALPHA, ALGORITHM_PAR$adaptiveProposalParameters, ALGORITHM_PAR$rwmhSd, nCores
          )
          
      outputLogEvidenceEstimate[ii,ll,1,mm] <- aux$logEvidenceEstimate
      outputLogEvidenceEstimate[ii,ll,2,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeights) ## uses importance-tempering weights proportional to the generalised ESS 
      outputLogEvidenceEstimate[ii,ll,3,mm] <- aux$logEvidenceEstimateEssAlternate ## uses importance-tempering weights proportional to the ESS 
      outputLogEvidenceEstimate[ii,ll,4,mm] <- computeLogZAlternate(aux$logUnnormalisedReweightedWeightsResampled)
      outputLogEvidenceEstimate[ii,ll,5,mm] <- aux$logEvidenceEstimateEssResampledAlternate ## based importance-tempering weights proportional to the ESS 
      
      ## Checking calculations (results should be numerically 0)
      print(computeLogZEss(aux$logUnnormalisedReweightedWeights) - aux$logEvidenceEstimateEssAlternate)
      print(computeLogZEss(aux$logUnnormalisedReweightedWeightsResampled) - aux$logEvidenceEstimateEssResampledAlternate)
      
      save(
        list  = ls(envir = environment(), all.names = TRUE), 
        file  = file.path(pathToOutputBase, 
          paste("smc_nParticlesUpper_", nParticlesUpper, 
          "_nParticlesLower_", nParticlesLower, 
          "_nSimulations_", nSimulations, 
          "_model_", MODELS[ii], 
          "_simulation_run_", mm, 
          sep='')),
        envir = environment()
      ) 
      rm(aux)
      
      
    }
  
  }
  

  
}


## ========================================================================= ##
## Plotting parameter estimates from final step of one of the SMC samplers
## ========================================================================= ##

# load(paste(pathToOutputBase, "smc_estimating_model_evidence_nParticlesUpper_4000nParticlesLowerr_2000_nSimulations_1_all_models_", sep=''))


nSteps <- length(aux$inverseTemperatures)
theta  <- array(unlist(aux$theta), c(MODEL_PAR$dimTheta, nParticlesUpper, nSteps))
# theta  <- THETA[,nSteps,] ## only using the particles from the last step here! TODO: we may need to resample these first or use the weights attached to them!

#modelType <- 5 #####################


WIDTH  <- 10
HEIGHT <- 6
COL_LL  <- c("royalblue4", "red4", "royalblue", "red2")
# LTY_NN  <- c(2,1)
LTY_NN  <- c(1)
LAG_MAX <- 500
PAR_LIM <- cbind(rep(-5, times=MODEL_PAR$dimTheta), rep(5, times=MODEL_PAR$dimTheta))
KDE_LIM <- rep(5, times=MODEL_PAR$dimTheta)
CEX_LEGEND <- 0.9


plotThetaSmc <- function(title, theta, idx)
{
  II <- length(idx)

  pdf(file=file.path(MISC_PAR$pathToFigures, paste(title, ".pdf", sep='')), width=WIDTH, height=HEIGHT)
      
  for (ii in 1:II) {
    plot(density(theta[ii,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]), type='l', col="white", xlab=MODEL_PAR$thetaNames[idx[ii]], ylab="Density", xlim=PAR_LIM[idx[ii],], ylim=c(0, KDE_LIM[idx[ii]]), main='')
    
    lines(density(c(theta[ii,,]), weights=c(aux$selfNormalisedWeightsEss)), type='l', col="red", lty=2, main='')
    
#     for (tt in 1:nSteps) {
#       theta  <- THETA[,,tt]
#       lines(density(theta[ii,], weights=aux$selfNormalisedWeights[,tt]), type='l', col="red", lty=1, main='')
#     }

    lines(density(theta[ii,,nSteps], weights=aux$selfNormalisedWeights[,nSteps]), type='l', col="red", lty=1, main='')

    grid <- seq(from=min(PAR_LIM[idx[ii],]), to=max(PAR_LIM[idx[ii],]), length=10000)
    lines(grid, dnorm(grid, mean=MODEL_PAR$meanHyper[idx[ii]], sd=MODEL_PAR$sdHyper[idx[ii]]), col="black", lty=2)
#     if (simulateData) {
#       abline(v=thetaTrue[ii,], col="blue")
#     }
    legend("topleft", legend=c("SMC sampler", "Prior"), col=c("red", "black"), bty='n', lty=c(1, 2), cex=CEX_LEGEND)
  }
  dev.off()
}

plotThetaSmc(paste("smc_theta_modelType_", modelType, sep=''), theta, 1:MODEL_PAR$dimTheta)



## ========================================================================= ##
## Plotting the log-model evidence
## ========================================================================= ##

load(paste(pathToOutputBase, "smc_estimating_model_evidence_nParticlesUpper_100nParticlesLowerr_1000_nSimulations_10_all_models_", sep=''))
modelIndices <- 0:7



# library(tikzDevice)
# options( 
#   tikzDocumentDeclaration = c(
#     "\\documentclass[12pt]{beamer}",
#     "\\usepackage{amssymb,amsmath,graphicx,mathtools,mathdots,stmaryrd}",
#     "\\usepackage{tikz}" 
#   )
# )

pdf(file=file.path(pathToFigures, "owls_evidence_boxplot.pdf"), width=5, height=7)

boxplot(t(outputLogEvidenceEstimate[,1:mm]), 
    ylab="Log-Evidence",
    xlab="Model",
    range=0,
    las=2, 
    names=modelIndices+1,
    main=""
  )
dev.off()

## ========================================================================= ##
## Plotting the log-model evidence (standard vs. importance tempering)
## ========================================================================= ##

load(file.path(pathToOutputBase, "smc_estimating_model_evidence_nParticlesUpper_5000nParticlesLowerr_1000_nSimulations_100_all_models_"))

logEvidenceTrue <- matrix(
c(-323.5212, -316.5992, -316.6001,
-315.7488, -315.7566, -315.7648,
-322.6757, -315.7818, -315.7888), 3, 3, byrow=FALSE)


pdf(file=file.path(pathToFigures, "owls_evidence_boxplot_2.pdf"), width=10, height=12)

op <- par(mfrow=c(3,1))

for (ii in 1:length(MODELS)) {
  boxplot(t(outputLogEvidenceEstimate[ii,1,,1:mm]), 
      ylab=paste("log-evidence estimate for Model ", MODELS[ii], sep=''),
      xlab="Importance tempering schemes",
      range=0,
      las=2, 
      ylim=c(-340,-310),
      names=1:5,
      main=""
    )
  for (jj in 1:3) {
    abline(h=logEvidenceTrue[jj,ii], lty=jj)
  }
  print(sqrt(diag(var(t(outputLogEvidenceEstimate[ii,1,,1:mm])))))
}



dev.off()
par(op)











u <- aux$logUnnormalisedReweightedWeights
W <- aux$selfNormalisedReweightedWeights
ess <- 1/colSums(W^2)
normalisedEss <- ess / sum(ess)


  nSteps <- dim(u)[2]
  logL <- rep(NA, nSteps)
  for (tt in 1:nSteps) {
    logL[tt] <- - log(var.welford(u[,tt]) + 1)
  }
  logL[is.nan(logL)] <- -Inf
  normalisedL <- exp(logL) / sum(exp(logL))
  
  
  logEss <- rep(NA, nSteps)
  for (tt in 1:nSteps) {
    maxU <- max(u[,tt])
    logEss[tt] <- 2*log(sum(exp(u[,tt] - maxU))) - log(sum(exp(2*(u[,tt] - maxU))))
  }
  logEss[is.nan(logEss)] <- 0
  normalisedEss <- exp(logEss) / sum(exp(logEss))


plot(1:nSteps, normalisedEss, type='l', col="black", ylim=c(0,1))
lines(1:nSteps, normalisedL, type='l', col="red")


# 
# 
# ## Numerically stable computation of the variance 
# ## associated with very small values represented 
# ## in log-space
# var.welford <- function(z){ # from https://www.r-bloggers.com/numerical-pitfalls-in-computing-variance/
#   n = length(z)
#   M = list()
#   S = list()
#   M[[1]] = z[[1]]
#   S[[1]] = 0
# 
#   for(k in 2:n){
#     M[[k]] = M[[k-1]] + ( z[[k]] - M[[k-1]] ) / k
#     S[[k]] = S[[k-1]] + ( z[[k]] - M[[k-1]] ) * ( z[[k]] - M[[k]] )
#   }
# #   return(S[[n]] / (n - 1)) ## TODO: check the normalisation here!
#   return(S[[n]] / n ) ## TODO: check the normalisation here!
# }
# ## Computes the log-model evidence using importance-tempering weights
# ## based on the (generalised) effective sample size associated with the
# ## unnormalised weights
# computeLogZAlternate <- function(u) {
# 
#   nSteps <- dim(u)[2]
#   logL <- rep(NA, nSteps)
#   for (tt in 1:nSteps) {
#     logL[tt] <- - log(var.welford(u[,tt]) + 1)
#   }
#   logL[is.nan(logL)] <- -Inf
#   normalisedL <- exp(logL) / sum(exp(logL))
#   
#   logC <- max(u)
#   
#   aux <- 0
#   for (tt in 1:nSteps) {
#     aux <- aux + normalisedL[tt] * sum(exp(u[,tt]-logC))
#   }
#   return(logC + log(aux));
# }
# 
# ## Computes the log-model evidence using importance-tempering weights
# ## based on the effective sample size associated with the
# ## self-normalised weights
# computeLogZEss <- function(u) {
# 
#   nSteps <- dim(u)[2]
#   
#   logEss <- rep(NA, nSteps)
#   for (tt in 1:nSteps) {
#     maxU <- max(u[,tt])
#     logEss[tt] <- 2*log(sum(exp(u[,tt] - maxU))) - log(sum(exp(2*(u[,tt] - maxU))))
#   }
#   logEss[is.nan(logEss)] <- 0
#   normalisedEss <- exp(logEss) / sum(exp(logEss))
#   
#   logC <- max(u)
#   
#   aux <- 0
#   for (tt in 1:nSteps) {
#     aux <- aux + normalisedEss[tt] * sum(exp(u[,tt]-logC))
#   }
#   return(logC + log(aux));
# }
# 
# 
# 
# 
# computeLogZAlternate(u)
# computeLogZEss(u)
# 
# 
