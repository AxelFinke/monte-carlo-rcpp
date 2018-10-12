
################################################################################
## Plot results of the simulation study
################################################################################
# Plots the output of the simulation study.
plotResults <- function(
  inputName,
  outputName,
  thetaTrue,
  dimTheta,
  yLim,
  yLimBoxplot = yLim,
  yLabel,
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
  lag.max.theta = 100,
  lag.max.states = 50,
  lowerTitle,
  upperTitle,
  burnin
) {

  load(inputName)
  
  for (uu in 1:length(UPPER)) {

    pdf(file=paste(outputName, "_", upperTitle[uu], "_traceplots.pdf", sep=''), width=widthPlot, height=heightPlot)
    
    op <- par(mfrow=c(dimTheta, length(LOWER)))
    
    for (kk in 1:dimTheta) {
      for (ll in 1:length(LOWER)) {
      
        marAux <- marPlot
        marAux[1] <- ifelse(kk == dimTheta, marAux[1] + padBottom, marAux[1])
        marAux[2] <- ifelse(ll == 1, marAux[2] + padLeft, marAux[2])
        par(mar=marAux)
      

        X <- outputTheta[kk,, ll, uu,1:mm]
       
        plot(1:nIterations, rep(0,times=nIterations), type='l', col="white", 
          xlim=c(1, nIterations), ylim=yLim[,kk], 
          xlab=ifelse(kk==dimTheta, "Iteration", ""), ylab=ifelse(ll==1, yLabel[kk], ""),
          main=ifelse(kk==1, lowerTitle[ll], ""), yaxs='i', xaxs='i', 
          xaxt="n",yaxt="n") 
        
        if (ll == 1) {
          axis(side=2, at=c(yLim[1,kk], yLim[2,kk]))
        }
        if (kk == dimTheta) {
          axis(side=1, at=c(1, ceiling(nIterations/2), nIterations))
        }

#         Xquant <- matrix(NA, nIterations, 2)
#         XMedian <- rep(NA, times=nIterations)
#         
#         if (mm > 1) {
#           for (jj in 1:length(alphaPlot)) {
#             for (gg in 1:nIterations) {
#               Xquant[gg,] <- quantile(X[gg,], probs=quantPlot[,jj])
#             }
#             polygon(c(1:nIterations, nIterations:1), c(Xquant[,2], Xquant[nIterations:1,1]), border=NA, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot[jj]))
#           }
#           for (gg in 1:nIterations) {
#             XMedian[gg] <- quantile(X[gg,], probs=0.5)
#           }
#         } else {
#           XMedian = X
#         }
#         lines(1:nIterations, XMedian, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))

        if (mm > 1) {
          for (m in 1:mm) {
            lines(1:nIterations, X[,m], type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
          }
        } else {
          lines(1:nIterations, X, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
        }
        abline(h=thetaTrue[kk], col=colTrue)
      }

    }
    par(op)
    dev.off()
  }

  
   for (uu in 1:length(UPPER)) {
       
      pdf(file=paste(outputName, "_", upperTitle[uu], "_acf_of_states.pdf", sep=''), width=widthPlot, height=heightPlot)
      op <- par(mfrow=c(length(TIMES), length(LOWER)))
       
      for (kk in 1:length(TIMES)) {
      
        for (ll in 1:length(LOWER)) {
        
          marAux <- marPlot
#           marAux[1] <- ifelse(kk == length(TIMES), marAux[1] + padBottom, marAux[1])
#           marAux[2] <- ifelse(ll == 1, marAux[2] + padLeft, marAux[2])
          par(mar=marAux)
  
          X <- outputStates[kk,(burnin+1):nIterations,ll,uu,1]

          ACF <- as.numeric(acf(X, lag.max=lag.max.states, plot=FALSE)$acf)
          
          plot(0:lag.max.states, ACF, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0),
            xlim=c(0,lag.max.states), ylim=c(-0.1,1.05),
            ylab=ifelse(ll==1, paste("Time ", TIMES[kk]+1, "; Component: ", COMPONENTS[kk]+1), ""), xlab="Lag",
            main=ifelse(kk==1, lowerTitle[ll], ""), yaxs='i', xaxs='i', xaxt="n",yaxt="n")
          
          if (mm > 1) {
            for (m in 2:mm) {
              X <- outputStates[kk,(burnin+1):nIterations,ll,uu,m]
              ACF <- as.numeric(acf(X, lag.max=lag.max.states, plot=FALSE)$acf)
              lines(0:lag.max.states, ACF, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
            }
          }
          axis(side=1, at=c(0, lag.max.states))
          if (ll == 1) {
            axis(side=2, at=c(0,1))
          }
          abline(h=0, col="black")
        }
      }
      par(op)
      dev.off()
    }
    
     for (uu in 1:length(UPPER)) {
       
      pdf(file=paste(outputName, "_", upperTitle[uu], "_kde_of_states.pdf", sep=''), width=widthPlot, height=heightPlot)
      op <- par(mfrow=c(length(TIMES), length(LOWER)))
       
      for (kk in 1:length(TIMES)) {
      
        for (ll in 1:length(LOWER)) {
        
          marAux <- marPlot
          marAux[1] <- ifelse(kk == length(TIMES), marAux[1] + padBottom, marAux[1])
          marAux[2] <- ifelse(ll == 1, marAux[2] + padLeft, marAux[2])
          par(mar=marAux)
  
          X <- outputStates[kk,(burnin+1):nIterations,ll,uu,1]
          
          plot(density(X), type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0),
            xlim=c(-3,3), ylim=c(0,2),
            ylab=ifelse(ll==1, paste("Time ", TIMES[kk]+1, "; Component: ", COMPONENTS[kk]+1), ""), xlab="Lag",
            main=ifelse(kk==1, lowerTitle[ll], ""), yaxs='i', xaxs='i')
          
          if (mm > 1) {
            for (m in 2:mm) {
              X <- outputStates[kk,(burnin+1):nIterations,ll,uu,m]
              lines(density(X), type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
            }
          }
#           axis(side=1, at=c(-3, 3))
#           if (ll == 1) {
#             axis(side=2, at=c(0,5))
#           }
        }
      }
      par(op)
      dev.off()
    }
    
  for (uu in 1:length(UPPER)) {
       
      pdf(file=paste(outputName, "_", upperTitle[uu], "_ess.pdf", sep=''), width=heightPlot, height=widthPlot)
      
      II <- length(LOWER)-1############# 
      if (UPPER[uu] == 0) {
        N <- nParticlesLinearMetropolis
      } else {
        N <- nParticlesLinearGibbs
      }
      
      op <- par(mfrow=c(II, 1))
      
        for (ll in 1:II) {
          marAux <- marPlot
          marAux[1] <- ifelse(ll == II, marAux[1] + padBottom + 1, marAux[1]) ################# note the +1
          marAux[2] <- marAux[2] + padLeft
          par(mar=marAux)
  
          X <- outputEss[,(burnin+1):nIterations,ll,uu,1] ##/ N ################# change this later when we store the normalised ESS
          
          plot(1:nObservations, rep(1, times=nObservations), type='l', col="white",
            xlim=c(1,nObservations), ylim=c(0,1),
            ylab="ESS", xlab=ifelse(ll==II, "Time [t]", ''),
            main=lowerTitle[ll])
 
          Xquant <- matrix(NA, nObservations, 2) 
          XMedian <- rep(NA, times=nObservations)
          
          for (jj in 1:length(alphaPlot)) {
            for (tt in 1:nObservations) {
              Xquant[tt,] <- quantile(X[tt,], probs=quantPlot[,jj])
            }
            polygon(c(1:nObservations, nObservations:1), c(Xquant[,2], Xquant[nObservations:1,1]), border=NA, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot[jj]))
          }
          for (tt in 1:nObservations) {
            XMedian[tt] <- quantile(X[tt,], probs=0.5)
          }
 
          lines(1:nObservations, XMedian, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
        }
      par(op)
      
      dev.off()
    }
    
#   if (mm > 1) {

    for (uu in 1:length(UPPER)) {
    
      pdf(file=paste(outputName, "_", upperTitle[uu], "_acf_of_parameters.pdf", sep=''), width=widthPlot, height=heightPlot)
        
      op <- par(mfrow=c(dimTheta, length(LOWER)))
        
      for (kk in 1:dimTheta) {
        for (ll in 1:length(LOWER)) {
        
          marAux <- marPlot
          marAux[1] <- ifelse(kk == dimTheta, marAux[1] + padBottom, marAux[1])
          marAux[2] <- ifelse(ll == 1, marAux[2] + padLeft, marAux[2])
          par(mar=marAux)
  
          X <- outputTheta[kk,(burnin+1):nIterations,ll,uu,1]

          ACF <- as.numeric(acf(X, lag.max=lag.max.theta, plot=FALSE)$acf)
          
          plot(0:lag.max.theta, ACF, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0),
            xlim=c(0,lag.max.theta), ylim=c(0,1.05),
            ylab=ifelse(ll==1, paste("Autocorrelation (", yLabel[kk], ")"), ""), xlab="Lag",
            main=ifelse(kk==1, lowerTitle[ll], ""), yaxs='i', xaxs='i', 
            xaxt="n",yaxt="n")
          if (mm > 1) {
            for (m in 2:mm) {
                X <- outputTheta[kk,(burnin+1):nIterations,ll,uu,m]
                ACF <- as.numeric(acf(X, lag.max=lag.max.theta, plot=FALSE)$acf)
                lines(0:lag.max.theta, ACF, type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
            }
          }
        }
      }
      par(op)
      dev.off()
    }


    for (uu in 1:length(UPPER)) {
    
      pdf(file=paste(outputName, "_", upperTitle[uu], "_kde.pdf", sep=''), width=widthPlot, height=heightPlot)
        
      op <- par(mfrow=c(dimTheta, length(LOWER)))
        
      for (kk in 1:dimTheta) {
        for (ll in 1:length(LOWER)) {
        
          marAux <- marPlot
          marAux[1] <- ifelse(kk == dimTheta, marAux[1] + padBottom, marAux[1])
          marAux[2] <- ifelse(ll == 1, marAux[2] + padLeft, marAux[2])
          par(mar=marAux)
  
          X <- outputTheta[kk,(burnin+1):nIterations,ll,uu,1]

          plot(density(X), type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0),
            xlim=yLim[,kk], 
            ylab=ifelse(ll==1, "Density", ""), xlab=yLabel[kk],
            main=ifelse(kk==1, lowerTitle[ll], ""), yaxs='i', xaxs='i', 
            xaxt="n",yaxt="n")
          
          if (mm > 1) {
            for (m in 2:mm) {
              X <- outputTheta[kk,(burnin+1):nIterations,ll,uu,m]
              lines(density(X), type='l', col=rgb(mycol[1], mycol[2], mycol[3], 1.0))
            }
          }
          axis(side=1, at=c(yLim[1,kk], yLim[2,kk]))
          
          abline(v=thetaTrue[kk], col=colTrue)
        }
      }
      par(op)
      dev.off()
    }
#   }
  
  return(0)
}

