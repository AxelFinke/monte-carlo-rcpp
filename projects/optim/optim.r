library(MASS)

################################################################################
## Numerical approximation of the MLE
## (if the marginal likelihood is available in closed form)
################################################################################
# Numerically approximates the MLE. The numerical approximation is obtained by
# starting the numerical optimisation algorithm from (possibly) multiple
# initial values drawn from the prior and then returning the value associated
# with the highest marginal likelihood.
numericallyApproximateMle <- function(
  nRuns, 
  hyperparameters, 
  dimTheta, 
  support, 
  observations, 
  repar=(function(x) {return(x)}), 
  inverseRepar=(function(x) {return(x)}), 
  method="Nelder-Mead", 
  nCores=1)
{
  # (-1) * exact log-marginal likelihood:
  F <- function(reparTheta) {
    logLike <- evaluateLogMarginalLikelihoodCpp(hyperparameters, inverseRepar(reparTheta), observations, nCores)
    return(-logLike)
  }
  
  # Initial values:
  parInit <- matrix(NA, dimTheta, nRuns)
  for (ii in 1:nRuns) {
    parInit[,ii] <- sampleFromPriorCpp(dimTheta, hyperparameters, support, nCores)
  }
  
  # Numerical optimisation:
  argminF <- matrix(NA, dimTheta, nRuns)
  minF    <- rep(NA, nRuns)
  
  for (nn in 1:nRuns) {
    if (dimTheta == 1) {
      aux <- optim(repar(parInit[nn]), F, lower=support[1], upper=support[2], method="Brent")
    } else {

      aux <- optim(repar(parInit[,nn]), F, method=method)
    }
    argminF[,nn] <- inverseRepar(as.numeric(aux$par))
    minF[nn]     <- aux$value
  }
  kk <- which.min(minF)
  
  return(list(theta=argminF[,kk], value=minF[kk], allTheta=argminF, allValues=minF))
}

################################################################################
## Cooling schedule for simulated annealing and related algorithms
################################################################################
# Creates a piecewise-linear cooling schedule. For the last iterations
# (more specifically, for a specific proportion of the total number of 
# iterations) the inverse temperature is kept constant. This has been done to
# ensure that 
createCoolingSchedule <- function(
  betaMin, 
  betaMax, 
  nSteps, 
  proportionBetaFixed
) 
{
  beta <- seq(from=betaMin, to=betaMax, length=ceiling(nSteps*(1-proportionBetaFixed)))
  return(c(beta, rep(betaMax, times=nSteps - length(beta))))
}

################################################################################
## Determine the number of lower-level particles
################################################################################
# Calculates the number of lower-level particles used within each individual 
# lower-level SMC algorithm/importance-sampling scheme. This has been done to 
# allow for a fair comparison of all pseudo-marginal/pseudo-Gibbs-sampling 
# schemes, by ensuring that the total amount of work (see definition below)
# is roughly identical for all algorithms.
adjustN <- function(lower, nParticles, beta)
{
  if (lower %in%  c(0,1)) {
    nParticles <- ceiling(nParticles/ceiling(beta))
  } else if (lower == 2) {
    nParticles <- ceiling(nParticles/(2*ceiling(beta))) 
    # divide by "2" because we need to approximate the normalising constant in both the numerator and the denominator
  } else if (lower %in% c(3,4)) {
    nParticles <- ceiling(nParticles)
  } else if (lower == 5) {
    nParticles <- ceiling(nParticles/2)
    # divide by "2" because we need to approximate the normalising constant in both the numerator and the denominator
  } else if (lower == 7) {
    nParticles <- ceiling(nParticles/ceiling(beta))
    #print("Constant number of particles per lower-level SMC algorithm in the pseudo-Gibbs sampler") 
    #avgWork <- ceiling(sum(nParticles)/sum(ceiling(beta))) # average number of particles per lower-level SMC algorithm
    #nParticles <- rep(avgWork, times=length(beta))
  }
  return(nParticles)
}

################################################################################
## Calculate the total amount of work for the different optimisation algorithms
################################################################################
# Calculates the total amount of work involved in each optimisation algorithm.
# Here: work is defined as the number of times a (C)SMC algorithm is called, 
# multiplied by the number of particles used in each call.
calculateWork <- function(
 nParticlesLowerMcmc, 
 nParticlesUpper, 
 nParticlesLowerSmc, 
 betaMcmc, 
 betaSmc, 
 nStepsMcmc, 
 nStepsSmc
) 
{
  workMcmc <- c(
    sum(adjustN(0,nParticlesLowerMcmc,betaMcmc)*ceiling(betaMcmc)),
    sum(adjustN(1,nParticlesLowerMcmc,betaMcmc)*ceiling(betaMcmc)),
    sum(adjustN(2,nParticlesLowerMcmc,betaMcmc)*ceiling(betaMcmc)*2),
    sum(adjustN(3,nParticlesLowerMcmc,betaMcmc)),
    sum(adjustN(4,nParticlesLowerMcmc,betaMcmc)),
    sum(adjustN(5,nParticlesLowerMcmc,betaMcmc)*2),
    # Algorithm 6 is simulated annealing
    sum(adjustN(7,nParticlesLowerMcmc,betaMcmc)*ceiling(betaMcmc))
    # Algorithm 8 is the standard Gibbs sampler
  )

  workSmc <- c(
    sum(adjustN(0,nParticlesLowerSmc,betaSmc)*ceiling(betaSmc))*nParticlesUpper,
    sum(adjustN(1,nParticlesLowerSmc,betaSmc)*ceiling(betaSmc))*nParticlesUpper,
    sum(adjustN(2,nParticlesLowerSmc,betaSmc)*ceiling(betaSmc)*2)*nParticlesUpper,
    sum(adjustN(3,nParticlesLowerSmc,betaSmc))*nParticlesUpper,
    sum(adjustN(4,nParticlesLowerSmc,betaSmc))*nParticlesUpper,
    sum(adjustN(5,nParticlesLowerSmc,betaSmc)*2)*nParticlesUpper,
    # Algorithm 6 is simulated annealing
    sum(adjustN(7,nParticlesLowerSmc,betaSmc)*ceiling(betaSmc))*nParticlesUpper
    # Algorithm 8 is the standard Gibbs sampler
  )

  cost <- as.table(matrix(rbind(workMcmc, workSmc), nrow=2, ncol=length(workMcmc)))
  row.names(cost) <- c("TOP: MCMC", "TOP: SMC")
  colnames(cost) <- c("0", "1", "2", "3", "4", "5", "7")

  return(cost)
}

################################################################################
## Run the simulation study
################################################################################
# Runs a simulation study to compare the performance of the optimisation 
# algorithms.
runSimulationStudy <- function(
  pathToOutput,
  dimTheta, 
  hyperparameters,
  support,
  observations,
  nSmcStepsLower,
  kern = 0,
  useGradients = FALSE,
  rwmhSd = rep(1, times=dimTheta),
  crankNicolsonScaleParameter = 0.5,
  proportionCorrelated = 0.5,
  upper = c(0,1), 
  lower = 0:9, 
  prop = 0:1, 
  nSimulations = 100,
  nStepsMcmc = 2000,
  betaMinMcmc = 1, 
  betaMaxMcmc = 20,
  betaMinSmc = betaMinMcmc, 
  betaMaxSmc = betaMaxMcmc,
  alpha = 1,
  fixedLagSmoothingOrder = 0,
  nParticlesLowerMin = 5*nSmcStepsLower,
  nParticlesUpper = 50,
  nThetaUpdates = 50,
  onlyTemperObservationDensity = FALSE,
  areInverseTemperaturesIntegers = FALSE,
  proportionBetaFixedMcmc = 0.1,
  proportionBetaFixedSmc = proportionBetaFixedMcmc,
  backwardSamplingType = 1,
  proposalDownscaleProbability = 0.8,
  essResamplingThresholdUpper = 0.8,
  essResamplingThresholdLower = 0.8,
  useNonCentredParametrisation = FALSE, # should the (pseudo-)Gibbs samplers use an NCP?
  nonCentringProbability = 1.0, # probability of using the NCP if useNonCentredParametrisation = TRUE
  nCores = 1
) 
{
  nStepsSmc <- ceiling(nStepsMcmc/nParticlesUpper) # number of (top-level) SMC sampler steps
  
  # Piecewise linear cooling schedule:
  betaMcmc <- createCoolingSchedule(betaMinMcmc, betaMaxMcmc, nStepsMcmc, proportionBetaFixedMcmc)
  betaSmc  <- createCoolingSchedule(betaMinSmc, betaMaxSmc, nStepsSmc,  proportionBetaFixedSmc)

  # Number of lower-level particles (needs to be multiplied by ceiling(betaMcmc) 
  # to obtain the number of particles in the algorithm from Rubenthaler et al.):
  nParticlesLowerMcmc <- nParticlesLowerMin*ceiling(betaMcmc)^(1+alpha) # upper level: MCMC
  nParticlesLowerSmc  <- nParticlesLowerMin*ceiling(betaSmc)^(1+alpha) # upper level: SMC
  
  # Display the amount of work required for all the particle-based algorithms
  print(calculateWork(nParticlesLowerMcmc, nParticlesUpper, nParticlesLowerSmc, betaMcmc, betaSmc, nStepsMcmc, nStepsSmc))

  outputMcmc <- array(NA, c(dimTheta, nStepsMcmc, length(lower), length(prop), nSimulations))
  outputSmc  <- array(NA, c(dimTheta, nParticlesUpper, nStepsSmc, length(lower), length(prop), nSimulations))
  
  for (mm in 1:nSimulations) {
    for (uu in 1:length(upper)) {
      for (ll in 1:length(lower)) {
        for (pp in 1:length(prop)) {
        
          print(paste("mm: ", mm, " upper: ", upper[uu], " lower: ", lower[ll], " prop: ", prop[pp] ))
          print(paste("betaMinMcmc: ", betaMinMcmc, " betaMaxMcmc: ", betaMaxMcmc, " nStepsMcmc: ", 
                       nStepsMcmc, " nStepsSmc: ", nStepsSmc, " nParticlesUpper: ", nParticlesUpper))
        
          if (upper[uu] == 0) {
          
            aux <- optimMcmcCpp(
                lower[ll], 
                betaMcmc, 
                areInverseTemperaturesIntegers,
                dimTheta,
                hyperparameters,
                support,
                observations, 
                proposalDownscaleProbability, 
                kern,
                useGradients,
                rwmhSd,
                fixedLagSmoothingOrder,
                crankNicolsonScaleParameter,
                proportionCorrelated,
                prop[pp], 
                nThetaUpdates, 
                onlyTemperObservationDensity,
                nSmcStepsLower, 
                adjustN(lower[ll], nParticlesLowerMcmc, betaMcmc), 
                essResamplingThresholdLower, 
                backwardSamplingType, 
                useNonCentredParametrisation,
                nonCentringProbability,
                nCores
              )
              
            outputMcmc[,,ll,pp,mm] <- matrix(unlist(aux$theta), dimTheta, nStepsMcmc)
            
          } else if (upper[uu] == 1) {
          
            aux <- optimSmcCpp(
                lower[ll], 
                betaSmc, 
                areInverseTemperaturesIntegers, 
                dimTheta,
                hyperparameters,
                support,
                observations, 
                proposalDownscaleProbability, 
                kern,
                useGradients,
                rwmhSd,
                fixedLagSmoothingOrder,
                crankNicolsonScaleParameter,
                proportionCorrelated,
                prop[pp], 
                nThetaUpdates, 
                onlyTemperObservationDensity,
                nSmcStepsLower, 
                adjustN(lower[ll], nParticlesLowerSmc, betaSmc), 
                essResamplingThresholdLower, 
                backwardSamplingType, 
                nParticlesUpper, 
                essResamplingThresholdUpper, 
                useNonCentredParametrisation,
                nonCentringProbability,
                nCores
              )

            outputSmc[,,,ll,pp,mm] <- array(unlist(aux$theta), c(dimTheta, nParticlesUpper, nStepsSmc))
            
          }
        }
      }
    }
    
    save(
      list  = ls(envir = environment(), all.names = TRUE), 
      file  = paste(pathToOutput, "optim_nObservations_", nObservations,  
                    "_nSimulations_", nSimulations, "_betaMaxMcmc_", betaMaxMcmc, "_alpha_", alpha, "_nParticlesLowerMin_", nParticlesLowerMin, sep=''),
      envir = environment()
    ) 
                          
  }
  return(0)
}


################################################################################
## Plot results of the simulation study
################################################################################
# Plots the output of the simulation study.
plotSimulationStudyResults <- function(
  inputName,
  outputName,
  thetaMle,
  dimTheta,
  yLim,
  yLimBoxplot = yLim,
  yLabel,
  upperPlot = -1,
  lowerPlot = -1,
  propPlot = -1,
  colPlot = "magenta3",
  widthPlot = 16,
  heightPlot = 10,
  colTrue = "black",   
  mycol = as.numeric(col2rgb(colPlot))/256,
  marPlot = c(1.5, 1.5, 1.5, 0.5) + 0.1,
  padLeft = 2.5,
  padBottom = 1.5,
  alphaPlot = c(0.1, 0.15, 0.2, 0.25),
  quantPlot = matrix(c(0,1,0.05,0.95,0.1,0.9,0.25,0.75), 2, length(alphaPlot)),
  propTitle =  c("Prior proposal", "Alternate proposal"),
  upperTitle = c("MCMC algorithm", "SMC sampler"),
  lowerTitle = c("Pseudo-marginal SAME", "Pseudo-Marginal SAME (correlated)", "Pseudo-marginal SAME (noisy)", "Rubenthaler", "Rubenthaler (correlated)", "Rubenthaler (noisy)", "Simulated annealing", "Pseudo-Gibbs SAME", "Gibbs SAME")
) {

  load(inputName)
  if (upperPlot[1] == -1) {
    upperPlot <- upper
  }
  if (lowerPlot[1] == -1) {
    lowerPlot <- lower
  }
  if (propPlot[1] == -1) {
    propPlot <- prop
  }
  
  pdf(file=paste(outputName, ".pdf", sep=''), width=widthPlot, height=heightPlot)
  
  for (pp in 1:length(propPlot)) {
    for (uu in 1:length(upperPlot)) {
    
      op <- par(mfrow=c(dimTheta, length(lowerPlot)))
      
      for (kk in 1:dimTheta) {
        for (ll in 1:length(lowerPlot)) {
        
          marAux <- marPlot
          marAux[1] <- ifelse(kk == dimTheta, marAux[1] + padBottom, marAux[1])
          marAux[2] <- ifelse(ll == 1, marAux[2] + padLeft, marAux[2])
          par(mar=marAux)
        
          if (upperPlot[uu] == 0) {
            X <- outputMcmc[kk,, which(lower == lowerPlot[ll]), which(prop == propPlot[pp]),1:mm]
            nIterations <- nStepsMcmc
          } else if (upperPlot[uu] == 1) {
            nIterations <- nStepsSmc
            ##X <- apply(outputSmc[kk,,, which(lower == lowerPlot[ll]), which(prop == propPlot[pp]),1:mm], c(2,3), mean)
            
            X <- matrix(NA, nStepsSmc, nParticlesUpper*mm)
            for (n in 1:nParticlesUpper) {
              X[,((n-1)*mm+1):(n*mm)] <- outputSmc[kk,n,, which(lower == lowerPlot[ll]), which(prop == propPlot[pp]),1:mm]
            }
          }
          
          plot(1:nIterations, rep(0,times=nIterations), type='l', col="white", 
            xlim=c(1, nIterations), ylim=yLim[,kk], 
            xlab=ifelse(kk==dimTheta, "Iteration", ""), ylab=ifelse(ll==1, yLabel[kk], ""),
            main=ifelse(kk==1, lowerTitle[lowerPlot[ll]+1], ""), yaxs='i', xaxs='i', 
            xaxt="n",yaxt="n") 
          
          if (ll == 1) {
            axis(side=2, at=c(yLim[1,kk], yLim[2,kk]))
          }
          if (kk == dimTheta) {
            axis(side=1, at=c(1, ceiling(nIterations/2), nIterations))
          }

          Xquant <- matrix(NA, nIterations, 2)
          XMedian <- rep(NA, times=nIterations)
          
          if (mm > 1) {
            for (jj in 1:length(alphaPlot)) {
              for (gg in 1:nIterations) {
                Xquant[gg,] <- quantile(X[gg,], probs=quantPlot[,jj])
              }
              polygon(c(1:nIterations, nIterations:1), c(Xquant[,2], Xquant[nIterations:1,1]), border=NA, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot[jj]))
            }
            for (gg in 1:nIterations) {
              XMedian[gg] <- quantile(X[gg,], probs=0.5)
            }
          } else {
            XMedian = X
          }
          
          lines(1:nIterations, XMedian, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
          abline(h=thetaMle[kk], col=colTrue)
        }
      }
      par(op)
    }
  }
  dev.off()
  
  if (mm > 1) {
    pdf(file=paste(outputName, "_boxplots.pdf", sep=''), width=widthPlot, height=heightPlot)
    for (pp in 1:length(propPlot)) {
      for (uu in 1:length(upperPlot)) {
      
        op <- par(mfrow=c(dimTheta, 1))
        
        for (kk in 1:dimTheta) {
          
            marAux <- marPlot
            marAux[1] <- ifelse(kk == dimTheta, marAux[1] + padBottom, marAux[1])
            marAux[2] <- marAux[2] + padLeft
            
            par(mar=marAux)
          
            if (upperPlot[uu] == 0) {
              auxSteps <- ceiling(nStepsMcmc*(1-proportionBetaFixedMcmc)):nStepsMcmc
              X <- matrix(NA, nrow=length(lowerPlot), ncol=mm*length(auxSteps))
              for (ll in 1:length(lowerPlot)) {
                X[ll,] <- c(outputMcmc[kk,auxSteps,ll,which(prop == propPlot[pp]),1:mm])
              }
            } else if (upperPlot[uu] == 1) {
              auxSteps <- ceiling(nStepsSmc*(1-proportionBetaFixedSmc)):nStepsSmc
  #             if (length(auxSteps) > 1) {
  #               X <- apply(outputSmc[kk,,auxSteps,,which(prop == propPlot[pp]),1:mm], c(3,4), mean)
  #             } else {
  #               X <- apply(outputSmc[kk,,auxSteps,,which(prop == propPlot[pp]),1:mm], c(2,3), mean)  
  #             }
              X <- matrix(NA, length(lowerPlot), mm*length(auxSteps)*nParticlesUpper)
              for (ll in 1:length(lowerPlot)) {
                X[ll,] <- c(outputSmc[kk,,auxSteps,ll, which(prop == propPlot[pp]),1:mm])
              }
            }
            
            if (kk == dimTheta) {
              boxplot(t(X), ylim=yLimBoxplot[,kk], ylab=yLabel[kk], range=0, las=2, names=lowerTitle[lowerPlot+1], main="", border=rgb(mycol[1], mycol[2], mycol[3], 1.0))
            } else {
              boxplot(t(X), ylim=yLimBoxplot[,kk], ylab=yLabel[kk], range=0, las=2, main="", border=rgb(mycol[1], mycol[2], mycol[3], 1.0))   
            }
            abline(h=thetaMle[kk], col=colTrue)
            
        }
        par(op)
      }
    }
    dev.off()
  
    pdf(file=paste(outputName, "_kde.pdf", sep=''), width=widthPlot, height=heightPlot)
    
    for (pp in 1:length(propPlot)) {
      for (uu in 1:length(upperPlot)) {
      
        op <- par(mfrow=c(dimTheta, length(lowerPlot)))
        
        for (kk in 1:dimTheta) {
          for (ll in 1:length(lowerPlot)) {
          
            marAux <- marPlot
            marAux[1] <- ifelse(kk == dimTheta, marAux[1] + padBottom, marAux[1])
            marAux[2] <- ifelse(ll == 1, marAux[2] + padLeft, marAux[2])
            par(mar=marAux)
          
            if (upperPlot[uu] == 0) {
              X <- outputMcmc[kk,ceiling(nStepsMcmc*(1-proportionBetaFixedMcmc)):nStepsMcmc,which(lower == lowerPlot[ll]),which(prop == propPlot[pp]),1]
            } else if (upperPlot[uu] == 1) {
              auxSteps <- ceiling(nStepsSmc*(1-proportionBetaFixedSmc)):nStepsSmc
              X <- outputSmc[kk,,auxSteps, which(lower == lowerPlot[ll]), which(prop == propPlot[pp]),1]
            }
            
            plot(density(X), type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0),
              xlim=yLim[,kk], 
              ylab=ifelse(ll==1, "Density", ""), xlab=yLabel[kk],
              main=ifelse(kk==1, lowerTitle[lowerPlot[ll]+1], ""), yaxs='i', xaxs='i', 
              xaxt="n",yaxt="n")
            
            for (m in 2:mm) {
              if (upperPlot[uu] == 0) {
                X <- outputMcmc[kk,ceiling(nStepsMcmc*(1-proportionBetaFixedMcmc)):nStepsMcmc,which(lower == lowerPlot[ll]),which(prop == propPlot[pp]),m]
              } else if (upperPlot[uu] == 1) {
                auxSteps <- ceiling(nStepsSmc*(1-proportionBetaFixedSmc)):nStepsSmc
                X <- outputSmc[kk,,auxSteps, which(lower == lowerPlot[ll]), which(prop == propPlot[pp]),m]
              }
              lines(density(X), type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
            }
            axis(side=1, at=c(yLim[1,kk], yLim[2,kk]))
            
            abline(v=thetaMle[kk], col=colTrue)
          }
        }
        par(op)
      }
    }
    dev.off()
  }
  
  # Display the amount of work required for all the particle-based algorithms
  print(calculateWork(nParticlesLowerMcmc, nParticlesUpper, nParticlesLowerSmc, betaMcmc, betaSmc, nStepsMcmc, nStepsSmc))
  
  return(0)
}
