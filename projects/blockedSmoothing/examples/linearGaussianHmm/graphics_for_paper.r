## Graphics for use in the TSP paper

## TODO: change back the plotting functions so that we create .tex and not .pdf files

imageType <- "tex" # either "pdf" or "tex" (the latter uses tikz)
paperType <- "tsp" # either "tsp", "arXiv" or "none"


library(tikzDevice)
options( 
  tikzDocumentDeclaration = c(
    "\\documentclass[12pt]{beamer}",
    "\\usepackage{amssymb,amsmath,graphicx,mathtools,mathdots,stmaryrd}",
    "\\usepackage{tikz}" 
  )
)

## for graphs
mv ~/Dropbox/research/output/cpp/mc/blockedSmoothing/linearGaussianHmm/figures/*q1*.tex ~/Dropbox/AF-SSS/blocked_smoothing/tsp_submission/


mv ~/Dropbox/research/output/cpp/mc/blockedSmoothing/linearGaussianHmm/for_revised_tsp_paper/figures/osga.tex ~/Dropbox/AF-SSS/blocked_smoothing/tsp_submission/
cd ~/Dropbox/AF-SSS/blocked_smoothing/tsp_submission/
pdflatex block_graphics.tex
pdftk block_graphics.pdf burst output finke%02d.pdf
pdflatex block.tex
pdflatex block.tex

pdflatex block_supp.tex
pdflatex block_supp.tex


## for manuscript
cd ~/Dropbox/AF-SSS/blocked_smoothing/tsp_submission/
pdftk block_supp.pdf cat 13-17 output supplementary_materials.pdf



#pdftk block.pdf cat 1-12 output block_main.pdf

#################### Concatenating samples:

## FROM LEGION 
PATH    <- pathToResults




FILES   <- list.files(path=PATH)

MM <- 0
for (ii in 1:length(FILES)) {
  load(file.path(PATH, FILES[ii]))
  MM <- MM + mm
}

# True sufficient statistics:
suffHTrueAux         <- array(NA, c(K+1, K+1, V1, MM))
suffQTrueAux         <- array(NA, c(K+1, V1, MM))
suffRTrueAux         <- array(NA, c(V1, MM))
suffSTrueAux         <- array(NA, c(V1, MM))
gradCompTrueAux      <- array(NA, c(K+3, V1, MM))

# Standard forward smoothing:
suffHFsAux           <- array(NA, c(K+1, K+1, I1, J1,V1, L1, MM))
suffQFsAux           <- array(NA, c(K+1, I1, J1,V1, L1, MM))
suffRFsAux           <- array(NA, c(I1, J1,V1, L1, MM))
suffSFsAux           <- array(NA, c(I1, J1,V1, L1, MM))
gradCompFsAux        <- array(NA, c(K+3, I1, J1,V1,L1, MM))

# Standard backward sampling:
suffHBsAux           <- array(NA, c(K+1, K+1, I1, J1,V1, L1, MM))
suffQBsAux           <- array(NA, c(K+1, I1, J1,V1, L1, MM))
suffRBsAux           <- array(NA, c(I1, J1,V1, L1, MM))
suffSBsAux           <- array(NA, c(I1, J1,V1, L1, MM))
gradCompBsAux        <- array(NA, c(K+3, I1, J1,V1,L1, MM))

# Blocked forward smoothing:
suffHBlockFsAux      <- array(NA, c(K+1, K+1, I1, J1, V1, L1, MM))
suffQBlockFsAux      <- array(NA, c(K+1, I1, J1, V1, L1, MM))
suffRBlockFsAux      <- array(NA, c(I1, J1, V1, L1, MM))
suffSBlockFsAux      <- array(NA, c(I1, J1, V1, L1, MM))
gradCompBlockFsAux   <- array(NA, c(K+3, I1, J1, V1,L1, MM))

# Blocked backward sampling:
suffHBlockBsAux      <- array(NA, c(K+1, K+1, I1, J1, V1, L1, MM))
suffQBlockBsAux      <- array(NA, c(K+1, I1, J1, V1, L1, MM))
suffRBlockBsAux      <- array(NA, c(I1, J1, V1, L1, MM))
suffSBlockBsAux      <- array(NA, c(I1, J1, V1, L1, MM))
gradCompBlockBsAux   <- array(NA, c(K+3, I1, J1, V1,L1, MM))


mm0 <- 0
for (ii in 1:length(FILES)) {

  load(file.path(PATH, FILES[ii]))
  mmAux <- (mm0+1):(mm0+mm)
  
  # True sufficient statistics:
  suffHTrueAux[,,,mmAux]   <- suffHTrue[,,,1:mm]
  suffQTrueAux[,,mmAux]    <- suffQTrue[,,1:mm]
  suffRTrueAux[,mmAux]     <- suffRTrue[,1:mm]
  suffSTrueAux[,mmAux]     <- suffSTrue[,1:mm]
  gradCompTrueAux[,,mmAux] <- gradCompTrue[,,1:mm]

  # Standard forward smoothing:
  suffHFsAux[,,,,,,mmAux]    <- suffHFs[,,,,,,1:mm]
  suffQFsAux[,,,,,mmAux]     <- suffQFs[,,,,,1:mm]
  suffRFsAux[,,,,mmAux]      <- suffRFs[,,,,1:mm]
  suffSFsAux[,,,,mmAux]      <- suffSFs[,,,,1:mm]
  gradCompFsAux[,,,,,mmAux]  <- gradCompFs[,,,,,1:mm]

  # Standard backward sampling:
  suffHBsAux[,,,,,,mmAux]    <- suffHBs[,,,,,,1:mm]
  suffQBsAux[,,,,,mmAux]     <- suffQBs[,,,,,1:mm]
  suffRBsAux[,,,,mmAux]      <- suffRBs[,,,,1:mm]
  suffSBsAux[,,,,mmAux]      <- suffSBs[,,,,1:mm]
  gradCompBsAux[,,,,,mmAux]  <- gradCompBs[,,,,,1:mm]

  # Blocked forward smoothing:
  suffHBlockFsAux[,,,,,,mmAux]    <- suffHBlockFs[,,,,,,1:mm]
  suffQBlockFsAux[,,,,,mmAux]     <- suffQBlockFs[,,,,,1:mm]
  suffRBlockFsAux[,,,,mmAux]      <- suffRBlockFs[,,,,1:mm]
  suffSBlockFsAux[,,,,mmAux]      <- suffSBlockFs[,,,,1:mm]
  gradCompBlockFsAux[,,,,,mmAux]  <- gradCompBlockFs[,,,,,1:mm]

  # Blocked backward sampling:
  suffHBlockBsAux[,,,,,,mmAux]    <- suffHBlockBs[,,,,,,1:mm]
  suffQBlockBsAux[,,,,,mmAux]     <- suffQBlockBs[,,,,,1:mm]
  suffRBlockBsAux[,,,,mmAux]      <- suffRBlockBs[,,,,1:mm]
  suffSBlockBsAux[,,,,mmAux]      <- suffSBlockBs[,,,,1:mm]
  gradCompBlockBsAux[,,,,,mmAux]  <- gradCompBlockBs[,,,,,1:mm]
  
  mm0 <- mm0 + mm
}

mm <- mm0

## Renaming the auxiliary quantities:

# True sufficient statistics:
suffHTrue    <- suffHTrueAux
suffQTrue    <- suffQTrueAux
suffRTrue    <- suffRTrueAux
suffSTrue    <- suffSTrueAux
gradCompTrue <- gradCompTrueAux

# Standard forward smoothing:
suffHFs      <- suffHFsAux
suffQFs      <- suffQFsAux
suffRFs      <- suffRFsAux
suffSFs      <- suffSFsAux
gradCompFs   <- gradCompFsAux

# Standard backward sampling:
suffHBs      <- suffHBsAux
suffQBs      <- suffQBsAux
suffRBs      <- suffRBsAux
suffSBs      <- suffSBsAux
gradCompBs   <- gradCompBsAux

# Blocked forward smoothing:
suffHBlockFs    <- suffHBlockFsAux
suffQBlockFs    <- suffQBlockFsAux
suffRBlockFs    <- suffRBlockFsAux
suffSBlockFs    <- suffSBlockFsAux
gradCompBlockFs <- gradCompBlockFsAux

# Blocked backward sampling:
suffHBlockBs    <- suffHBlockBsAux
suffQBlockBs    <- suffQBlockBsAux
suffRBlockBs    <- suffRBlockBsAux
suffSBlockBs    <- suffSBlockBsAux
gradCompBlockBs <- gradCompBlockBsAux

## Restoring paths
pathToOutput  <- "/home/axel/Dropbox/research/output/cpp/mc/blockedSmoothing/linearGaussianHmm/for_revised_tsp_paper"
pathToResults <- "/home/axel/Dropbox/research/output/cpp/mc/blockedSmoothing/linearGaussianHmm/for_revised_tsp_paper/results"
pathToFigures <- "/home/axel/Dropbox/research/output/cpp/mc/blockedSmoothing/linearGaussianHmm/for_revised_tsp_paper/figures"

####################




setwd(pathToFigures)

colBlock    <- "forestgreen" # colour for blocked particle smoothing
colStandard <- "magenta3" # colour for standard particle smoothing
ltyFs <- 1
ltyBs <- 2

ltyPlot <- 1
lwdPlot <- 2
# cexPlot <- 1
segLenPlot <- 1.5

quantPlot  <- c(0, 1) # quantile represented by the "error bars" / shaded area
alphaPlot  <- 0.17 # transparency parameter 1
alphaPlot2 <- 0.8 # transparency parameter 2

suffYLim <- c(-5,5) # 2nd axis limits for the plots of the sufficient statistics 
rmseYLim <- c(0,4) # 2nd axis limits for the root mean-square error plots

plotPointSize <- 10 ##########

plotMar <- c(0.7, 0.6, 0.9, 1) + 0.1 # c(bottom, left, top, right)
plotOma <- c(2.85,3,0.5,0.5) # c(bottom, left, top, right)

padLeft   <- 0
padBottom <- 0

xLabLineOuter <- 1.7
yLabLineOuter <- 1.5

xLabLineInner <- 2.4
yLabLineInner <- 2.4

if (paperType == "arXiv") {

  plotWidth      <- 5.5
  plotHeight     <- 2.25
  plotWidthRmse  <- plotWidth
  plotHeightRmse <- plotHeight
  plotWidthPar   <- 5.5
  plotHeightPar  <- 8
  
} else if (paperType == "tsp") {

  plotWidth      <- 3.58
  plotHeight     <- 2.35
  plotWidthRmse  <- plotWidth
  plotHeightRmse <- plotHeight
  plotWidthPar   <- 3.5
  plotHeightPar  <- 5
  
} else if (paperType == "none") {

  plotWidth      <- 12
  plotHeight     <- 8
  plotWidthRmse  <- plotWidth
  plotHeightRmse <- plotHeight
  plotWidthPar   <- 8
  plotHeightPar  <- 12
  
}



#onlyShow <- c(1,2)#############


plotSuffAverage <- function(suffname, label, X, XBlock, truth, suffylim, normalise=TRUE) {

  # Here, we plot (X - truth)/truth, where X is the estimate

  for (ll in 1:length(FILT)) {
  
    if (imageType == "tex") {
      tikz(paste(suffname, "_", ll, ".tex", sep=''),  width=plotWidth, height=plotHeight)
    } else if (imageType == "pdf") {
      pdf(paste(suffname, "_", ll, ".pdf", sep=''),  width=plotWidth, height=plotHeight)
    }
    op <- par(mfrow=c(J1, I1), oma=plotOma)
  
    for (jj in 1:J1) {
      for (ii in 1:I1) {
      
        aux <- setBlocks(bInn[ii], bOut[jj], V)
        blockInn <- aux$blockInn
        blockOut <- aux$blockOut  
        bI <- blockInn + 1
        bO <- blockOut + 1
        
        plotMarAux <- plotMar
        plotMarAux[1] <- ifelse(ii==I1, plotMarAux[1]+padBottom, plotMarAux[1])
        plotMarAux[2] <- ifelse(jj==1, plotMarAux[2]+padLeft, plotMarAux[2])
        par(mar=plotMarAux)
        
#         plot(SPACE, rep(0,V1), col="white", type="l", ylim=suffylim, 
#           main=paste("\\footnotesize{$\\mathop{\\mathrm{card}} K = ", bInn[ii], "; \\overline{K} = \\mathcal{N}_", bOut[jj], "(K)$}", sep=''), 
#           xlab="", ylab="", yaxs='i', xaxs='i', 
#           xaxt="n",yaxt="n", xlim=range(SPACE)
#         )

        plot(SPACE, rep(0,V1), col="white", type="l", ylim=suffylim, 
          main=paste("\\footnotesize{$\\mathop{\\mathrm{card}} K = ", bInn[ii], "; i = ", bOut[jj], "$}", sep=''), 
          xlab="", ylab="", yaxs='i', xaxs='i', 
          xaxt="n",yaxt="n", xlim=range(SPACE)
        )
        
        if (ii == 1) {
          axis(side=2, at=c(suffylim,0))
          if (jj == 2) {
            mtext(label, side=2, outer=TRUE, line=yLabLineOuter)
          }
        }
        if (jj == J1) {
            axis(side=1, at=c(10, 250, 500))
          if (ii == 2) {
            mtext("\\footnotesize{Model dimension $V$}", side=1, outer=TRUE, line=xLabLineOuter)
          }
        }
        abline(h=0, col="black", lwd=1, lty=1)
          
        
        ## Plot "error" bars, i.e. some suitable quantile of the empirical distribution
        for (ss in 0:1) {
        
          Xquant <- matrix(NA, V1, 2)
          truth1 <- rep(NA, V1)
          X1     <- matrix(NA, mm, V1)
        
          if (ss == 0) { # standard particle smoothing
            mycol <- as.numeric(col2rgb(colStandard))/256
            for (vv in 1:V1) {
              X1[,vv] <- X[ii,jj,vv,ll,1:mm] - truth[vv,1:mm]
            }
          } else if (ss == 1) { # blocked particle smoothing
            mycol <- as.numeric(col2rgb(colBlock))/256
            for (vv in 1:V1) {
              X1[,vv] <- XBlock[ii,jj,vv,ll,1:mm] - truth[vv,1:mm]
            }
          }

          for (vv in 1:V1) {
            if (normalise == "SPACE") {
              Xquant[vv,] <- quantile(X1[,vv]/SPACE[vv], probs=quantPlot)
            }
            else if (normalise == FALSE){
              Xquant[vv,] <- quantile(X1[,vv], probs=quantPlot)
            }
          }
          polygon(c(SPACE, SPACE[V1:1]), c(Xquant[,2], Xquant[V1:1,1]), border=NA, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot)) 
        }
        
        

        ## Plot the empirical mean of the approximations
        for (ss in 0:1) {
        
          Xmean <- rep(NA, V1)
        
          if (ss == 0) { # standard particle smoothing
            mycol <- as.numeric(col2rgb(colStandard))/256
            for (vv in 1:V1) {
              Xmean[vv] <- sum(X[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])/mm  
            }
          } else if (ss == 1) { # blocked particle smoothing
            mycol <- as.numeric(col2rgb(colBlock))/256
            for (vv in 1:V1) {
              Xmean[vv] <- sum(XBlock[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])/mm  
            }
          }

          if (normalise == "SPACE") {
            lines(SPACE[1:V1], Xmean/SPACE[1:V1], lty=ltyPlot, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot2), lwd=lwdPlot)
          }
          else if (normalise == FALSE){
            lines(SPACE[1:V1], Xmean, lty=ltyPlot, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot2), lwd=lwdPlot)
          }
        }
        
#         if (ii == 3 && jj == 1) {
#           legend(x=-20, y=max(suffYLim)*1.2,
#                  legend=c("\\footnotesize{standard FS}", "\\footnotesize{blocked FS}"),
#                  col=c(colStandard, colBlock), 
#                  lty=c(1,1), lwd=rep(lwdPlot, times=2),
#                  bty='n', pt.lwd=lwdPlot, seg.len=segLenPlot)
#         }
        
      }
    }   
    par(op)
    dev.off()
  }
  return(0)
}



plotSuffRmse <- function(
  suffname, label, XFs, XBs, XBlockFs, XBlockBs, truth, suffylim, WIDTH=plotWidthRmse, HEIGHT=plotHeightRmse, normalise=TRUE) {

  # Here, we plot (X - truth)/truth, where X is the estimate

  for (ll in 1:length(FILT)) {
  
    if (imageType == "tex") {
      tikz(paste(suffname, "_", ll, ".tex", sep=''),  width=WIDTH, height=HEIGHT)
    } else if (imageType == "pdf") {
      pdf(paste(suffname, "_", ll, ".pdf", sep=''),  width=WIDTH, height=HEIGHT)
    }
    
    op <- par(mfrow=c(J1,I1), oma=plotOma)
  
    for (jj in 1:J1) {
      for (ii in 1:I1) {
      
        aux <- setBlocks(bInn[ii], bOut[jj], V)
        blockInn <- aux$blockInn
        blockOut <- aux$blockOut  
        bI <- blockInn + 1
        bO <- blockOut + 1
        
        plotMarAux <- plotMar
        plotMarAux[1] <- ifelse(ii==I1, plotMarAux[1]+padBottom, plotMarAux[1])
        plotMarAux[2] <- ifelse(jj==1, plotMarAux[2]+padLeft, plotMarAux[2])
        par(mar=plotMarAux)
             
#         plot(SPACE, rep(0,V1), col="white", type="l", ylim=suffylim, main=paste("\\footnotesize{$\\mathop{\\mathrm{card}} K = ", bInn[ii], ";\\; \\overline{K} = \\mathcal{N}_", bOut[jj], "(K)$}", sep=''), 
        plot(SPACE, rep(0,V1), col="white", type="l", ylim=suffylim, 
        main=paste("\\footnotesize{$\\mathop{\\mathrm{card}} K = ", bInn[ii], "; i = ", bOut[jj], "$}", sep=''), 
        xlab="", ylab="", yaxs='i', xaxs='i', 
        xaxt="n",yaxt="n", xlim=range(SPACE))
             
        if (ii == 1) {
          axis(side=2, at=c(suffylim,0))
          if (jj == 2) {
            mtext(label, side=2, outer=TRUE, line=yLabLineOuter)
          }
        }
        
        if (jj == J1) {
          axis(side=1, at=c(10, 250, 500))
          if (ii == 2) {
            mtext("\\footnotesize{Model dimension $V$}", side=1, outer=TRUE, line=xLabLineOuter)
          }
        }

        
        Xmse <- rep(NA, V1)
                    
        for (ss in 0:1) { # 0: standard particle smoothing; 1: blocked particle smoothing
          for (rr in 0:1) { # 0: (blocked) forward smoothing; 1: (blocked) backward sampling
          
            # Plot the empirical MSE of the approximations
            if (ss == 0 && rr == 0) {
                               
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XFs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colStandard))/256
              mylty <- ltyFs
            
            } else if (ss == 0 && rr == 1) {
            
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XBs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colStandard))/256
              mylty <- ltyBs
              
            } else if (ss == 1 && rr == 0) {
            
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XBlockFs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colBlock))/256
              mylty <- ltyFs
              
            } else if (ss == 1 && rr == 1) {
            
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XBlockBs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colBlock))/256
              mylty <- ltyBs
            
            }
            
            if (normalise == "SPACE") {
              lines(SPACE[1:V1], Xmse/SPACE[1:V1], lty=mylty, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot2), lwd=lwdPlot)
            }
            else if (normalise == FALSE){
              lines(SPACE[1:V1], Xmse, lty=mylty, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot2), lwd=lwdPlot)
            }   
          }
        }
        
#         if (ii == 3 && jj == 1) {
#           legend(x=-20, y=max(suffYLim)*1.1,
#                  legend=c("\\footnotesize{standard FS}", "\\footnotesize{standard BS}", "\\footnotesize{blocked FS}", "\\footnotesize{blocked BS}"), 
#                  col=c(colStandard, colStandard, colBlock, colBlock), 
#                  lty=c(1,2,1,2), lwd=rep(lwdPlot, times=4),
#                  bty='n', pt.lwd=lwdPlot, seg.len=segLenPlot)
#         }
      }
    }
  par(op)
  dev.off()
  }
  return(0)
}



## For tsp paper:

plotSuffAverage("fs_avg_q1_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffQFs[1,,,,,], suffQBlockFs[1,,,,,], suffQTrue[1,,], suffylim=suffYLim,  normalise="SPACE")
plotSuffAverage("bs_avg_q1_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffQBs[1,,,,,], suffQBlockBs[1,,,,,], suffQTrue[1,,], suffylim=suffYLim,  normalise="SPACE")
plotSuffAverage("fs_avg_h11_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffHFs[1,1,,,,,], suffHBlockFs[1,1,,,,,], suffHTrue[1,1,,], suffylim=suffYLim,  normalise="SPACE")
plotSuffAverage("bs_avg_h11_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffHBs[1,1,,,,,], suffHBlockBs[1,1,,,,,], suffHTrue[1,1,,], suffylim=suffYLim,  normalise="SPACE")
plotSuffAverage("fs_avg_r_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffRFs, suffRBlockFs, suffRTrue, suffylim=suffYLim,  normalise="SPACE")
plotSuffAverage("bs_avg_r_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffRBs, suffRBlockBs, suffRTrue, suffylim=suffYLim,  normalise="SPACE")
plotSuffAverage("fs_avg_s_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffSFs, suffSBlockFs, suffSTrue, suffylim=suffYLim,  normalise="SPACE")
plotSuffAverage("bs_avg_s_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffSBs, suffSBlockBs, suffSTrue, suffylim=suffYLim,  normalise="SPACE")





plotSuffRmse("Rmse_avg_h11_normalised_space", "\\footnotesize{RMSE of ${\\mathbb{F}}_T^{N}/V$}", suffHFs[1,1,,,,,], suffHBs[1,1,,,,,], suffHBlockFs[1,1,,,,,], suffHBlockBs[1,1,,,,,], suffHTrue[1,1,,],  suffylim=rmseYLim, normalise="SPACE")

plotSuffRmse("Rmse_avg_h12_normalised_space", "\\footnotesize{RMSE of ${\\mathbb{F}}_T^{N}/V$}", suffHFs[1,2,,,,,], suffHBs[1,2,,,,,], suffHBlockFs[1,2,,,,,], suffHBlockBs[1,2,,,,,], suffHTrue[1,2,,],  suffylim=rmseYLim, normalise="SPACE")

plotSuffRmse("Rmse_avg_q1_normalised_space", "\\footnotesize{RMSE of ${\\mathbb{F}}_T^{N}/V$}", suffQFs[1,,,,,], suffQBs[1,,,,,], suffQBlockFs[1,,,,,], suffQBlockBs[1,,,,,], suffQTrue[1,,],  suffylim=rmseYLim, normalise="SPACE")

plotSuffRmse("Rmse_avg_q2_normalised_space", "\\footnotesize{RMSE of ${\\mathbb{F}}_T^{N}/V$}", suffQFs[2,,,,,], suffQBs[2,,,,,], suffQBlockFs[2,,,,,], suffQBlockBs[2,,,,,], suffQTrue[2,,],  suffylim=rmseYLim, normalise="SPACE")

plotSuffRmse("Rmse_avg_r_normalised_space", "\\footnotesize{RMSE of ${\\mathbb{F}}_T^{N}/V$}", suffRFs, suffRBs, suffRBlockFs, suffRBlockBs, suffRTrue,  suffylim=rmseYLim, normalise="SPACE")

plotSuffRmse("Rmse_avg_s_normalised_space", "\\footnotesize{RMSE of ${\\mathbb{F}}_T^{N}/V$}", suffSFs, suffSBs, suffSBlockFs, suffSBlockBs, suffSTrue,  suffylim=rmseYLim, normalise="SPACE")



## for slides:


plotSuffRmseSlides <- function(
  suffname, label, XFs, XBs, XBlockFs, XBlockBs, truth, suffylim, WIDTH=plotWidthRmse, HEIGHT=plotHeightRmse, normalise=TRUE) {

  # Here, we plot (X - truth)/truth, where X is the estimate

  for (ll in 1:length(FILT)) {
  
    if (imageType == "tex") {
      tikz(paste(suffname, "_", ll, ".tex", sep=''),  width=WIDTH, height=HEIGHT)
    } else if (imageType == "pdf") {
      pdf(paste(suffname, "_", ll, ".pdf", sep=''),  width=WIDTH, height=HEIGHT)
    }
    
    op <- par(mfrow=c(J1,I1), oma=plotOma)
  
    for (jj in 1:J1) {
      for (ii in 1:I1) {
      
        aux <- setBlocks(bInn[ii], bOut[jj], V)
        blockInn <- aux$blockInn
        blockOut <- aux$blockOut  
        bI <- blockInn + 1
        bO <- blockOut + 1
        
        plotMarAux <- plotMar
        plotMarAux[1] <- ifelse(ii==I1, plotMarAux[1]+padBottom, plotMarAux[1])
        plotMarAux[2] <- ifelse(jj==1, plotMarAux[2]+padLeft, plotMarAux[2])
        par(mar=plotMarAux)
             
#         plot(SPACE, rep(0,V1), col="white", type="l", ylim=suffylim, main=paste("\\footnotesize{$\\mathop{\\mathrm{card}} K = ", bInn[ii], ";\\; \\overline{K} = \\mathcal{N}_", bOut[jj], "(K)$}", sep=''), 
        plot(SPACE, rep(0,V1), col="white", type="l", ylim=suffylim, 
#         main=paste("\\footnotesize{block size: ", bInn[ii], "; overlap on each side: ", bOut[jj], "}", sep=''), 
        xlab="", ylab="", yaxs='i', xaxs='i', 
        xaxt="n",yaxt="n", xlim=range(SPACE))
        mtext(paste("\\scriptsize{block size: ", bInn[ii], "; overlap: ", bOut[jj], "}", sep=''), side=3, outer=FALSE, line=0)
             
        if (ii == 1) {
          axis(side=2, at=c(suffylim,0))
          if (jj == 2) {
            mtext(label, side=2, outer=TRUE, line=yLabLineOuter)
          }
        }
        
        if (jj == J1) {
          axis(side=1, at=c(10, 250, 500))
          if (ii == 2) {
            mtext("\\footnotesize{Model dimension $V$}", side=1, outer=TRUE, line=xLabLineOuter)
          }
        }

        
        Xmse <- rep(NA, V1)
                    
        for (ss in 0:1) { # 0: standard particle smoothing; 1: blocked particle smoothing
          for (rr in 0:1) { # 0: (blocked) forward smoothing; 1: (blocked) backward sampling
          
            # Plot the empirical MSE of the approximations
            if (ss == 0 && rr == 0) {
                               
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XFs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colStandard))/256
              mylty <- ltyFs
            
            } else if (ss == 0 && rr == 1) {
            
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XBs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colStandard))/256
              mylty <- ltyBs
              
            } else if (ss == 1 && rr == 0) {
            
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XBlockFs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colBlock))/256
              mylty <- ltyFs
              
            } else if (ss == 1 && rr == 1) {
            
              for (vv in 1:V1) {
                Xmse[vv] <- sqrt(sum((XBlockBs[ii,jj,vv,ll,1:mm]-truth[vv,1:mm])^2)/mm)  
              }
              mycol <- as.numeric(col2rgb(colBlock))/256
              mylty <- ltyBs
            
            }
            
            if (normalise == "SPACE") {
              lines(SPACE[1:V1], Xmse/SPACE[1:V1], lty=mylty, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot2), lwd=lwdPlot)
            }
            else if (normalise == FALSE){
              lines(SPACE[1:V1], Xmse, lty=mylty, col=rgb(mycol[1], mycol[2], mycol[3], alphaPlot2), lwd=lwdPlot)
            }   
          }
        }
        
        if (ii == 3 && jj == 1) {
          legend(x=-20, y=4*1.05,
                 legend=c("\\scriptsize{standard forward smoothing}", "\\scriptsize{standard backward sampling}", "\\scriptsize{blocked forward smoothing}", "\\scriptsize{blocked backward sampling}"), 
                 col=c(colStandard, colStandard, colBlock, colBlock), 
                 lty=c(1,2,1,2), lwd=rep(lwdPlot, times=4),
                 bty='n', pt.lwd=lwdPlot, seg.len=segLenPlot)
        }
      }
    }
  par(op)
  dev.off()
  }
  return(0)
}




plotWidthSlides  <- 5.1
plotHeightSlides <- 3.2

plotSuffAverage("fs_avg_q1_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffQFs[1,,,,,], suffQBlockFs[1,,,,,], suffQTrue[1,,], suffylim=suffYLim, normalise="SPACE")
plotSuffAverage("bs_avg_q1_normalised_space", "\\footnotesize{Error of ${\\mathbb{F}}_T^{N}/V$}", suffQBs[1,,,,,], suffQBlockBs[1,,,,,], suffQTrue[1,,], suffylim=suffYLim,  normalise="SPACE")
plotSuffRmseSlides("Rmse_avg_q1_normalised_space", "\\footnotesize{$\\mathop{\\mathrm{RMSE}}({\\mathbb{F}}^{N})/V$}", suffQFs[1,,,,,], suffQBs[1,,,,,], suffQBlockFs[1,,,,,], suffQBlockBs[1,,,,,], suffQTrue[1,,],  suffylim=rmseYLim, WIDTH=plotWidthSlides, HEIGHT=plotHeightSlides , normalise="SPACE")







mv ~/Dropbox/research/output/cpp/mc/blockedSmoothing/linearGaussianHmm/for_revised_tsp_paper/figures/*q1*.tex ~/Dropbox/research/seminars/2017-06-02_imperial/slides/
cd ~/Dropbox/research/seminars/2017-06-02_imperial/slides/
pdflatex slides.tex
pdflatex slides.tex
%




# suffname <- "testFile"
# label <- "testLabel"
# XFs <- suffQFs[1,,,,,]
# XBs <- suffQBs[1,,,,,]
# XBlockFs<- suffQBlockFs[1,,,,,]
# XBlockBs<- suffQBlockBs[1,,,,,]
# truth <- suffQTrue[1,,]
# suffylim <- rmseYLim


























######################


# Assumption:
# X has dimension: c(nrow, ncol, LL, xgrid, y), where $y$ has at least mm values
plotTikzMat <- function(
  name, nrow, ncol, x, y, height, width, mm=1,
  xlab, ylab, xmin=min(x), xmax=max(x), ymin, ymax, 
  xlabLineInner=2.4, ylabLineInner=2.4, xlabLineOuter=1.7, ylabLineOuter=1.5,
  main="", 
  oma= c(2.85,3,0.5,0.5), mar=c(0.7,0.7,0.9,0.7)+0.1, pad=c(0,0,0,0), # order: c(bottom, left, top, right)
  lty=1, lwd=1, col, 
  quantilesMin=c(0,0.05), quantilesMax=c(1,0.95), alpha=c(0.17, 0.3), #  quantilesMin=0, quantilesMax=1, alpha=0.17,
  plotHline=FALSE, hline=0,
  ltyHline=1, lwdHline=1, colHline="black",
  subtractValues=FALSE, subVal #an (nrow, ncol)-matrix whose elements are subtracted from y
) 
{
  LL <- dim(y)[3]
                        
  tikz(paste(name, ".tex", sep=''),  width=width, height=height)
  
  ###pdf(paste(name, ".pdf", sep=''),  width=width, height=height)
  
  op <- par(mfrow=c(nrow,ncol), oma=oma)
  

  
  for (ii in 1:nrow) {
    for (jj in 1:ncol) {
    
    
      marAux <- mar
      marAux[1] <- ifelse(ii==nrow, marAux[1]+pad[1], marAux[1])
      marAux[2] <- ifelse(jj==1,    marAux[2]+pad[2], marAux[2])
      marAux[3] <- ifelse(ii==1,    marAux[3]+pad[3], marAux[3])
      marAux[4] <- ifelse(jj==ncol, marAux[4]+pad[4], marAux[4])
      
      par(mar=marAux)
    
      xMin <- ifelse(length(xmin) == 1, xmin, xmin[jj])
      xMax <- ifelse(length(xmax) == 1, xmax, xmax[jj])
      yMin <- ifelse(length(ymin) == 1, ymin, xmin[ii])
      yMax <- ifelse(length(ymax) == 1, ymax, ymax[ii])
      if (subtractValues==TRUE) {
        Y <- y[ii,jj,,,]-subVal[ii,jj]
      } else {
        Y <- y[ii,jj,,,]
      }
      
      plot(x, rep(0,length(x)), 
           col="white", type="l", 
           xlim=c(xMin, xMax), 
           ylim=c(yMin, yMax), 
           main=ifelse(ii==1,main[jj],"") ,
           xlab="", ylab="", #ifelse(jj==1, ylab[ii], ""), 
           yaxs='i', xaxs='i', 
           xaxt="n",yaxt="n")
            
      if (jj == 1) {
        axis(side=2, at=c(-0.4,0,0.4)) ##, at=c(ylim,0))
        #if (ii == ceil(nrow/2)) {
         ### mtext(ylab[ii], side=2, outer=TRUE, line=ylabLineOuter)
             mtext(ylab[ii], side=2, outer=FALSE, line=ylabLineInner)
        #}
      }
      
      if (ii == nrow) {
        axis(side=1, at=c(1,500,1000)) ##, at=c(seq(from=1, to=G, by=250), G))
        #if (jj == ceiling(ncol/2)) {
         ### mtext(xlab[jj], side=1, outer=TRUE, line=xlabLineOuter)
              mtext(xlab[jj], side=1, outer=FALSE, line=xlabLineInner)
        #}
      }
        
      # Range of values
      if (mm > 1) {
        for (ll in 1:LL) {
        
          mycol <- as.numeric(col2rgb(col[ll]))/256
          
          for (pp in 1:length(quantilesMin)) {
            Xquant = matrix(NA, length(x), 2)
            for (gg in 1:length(x)) {
              Xquant[gg,] <- quantile(
                Y[ll,gg,],
                probs=c(quantilesMin[pp], quantilesMax[pp]))
            }  
            polygon(c(x, rev(x)), c(Xquant[,2], rev(Xquant[,1])), border=NA, col=rgb(mycol[1], mycol[2], mycol[3], alpha[pp]))
            
          }
        }
      }
      
      if (plotHline == TRUE) {
        abline(h=hline, lty=ltyHline, lwd=lwdHline, col=colHline)
      }
  
      # Mean value
      if (mm > 1) {
        for (ll in 1:LL) {
          lines(x, apply(Y[ll,,], 1, median), col=col[ll], lty=lty[ll], lwd=lwd[ll])
        }
      } else {
        for (ll in 1:LL) {
          lines(x, Y[ll,], col=col[ll], lty=lty[ll], lwd=lwd[ll])
        }
      }
    } 
  }
   
  par(op)
  dev.off()
                        
}


# load(file.path(pathToOutput, "osem_osga_new_T_10_V_100_N_500_M_200_MM_25_G_1000"))
# X <- array(NA, c(4, G, 2, 2, mm))
# X[,,,,] <- thetaMleEst[,,1,1,3:4,1:2,1:mm]
# X <- aperm(X, c(1,4,3,2,5))


load(file.path(pathToResults, "osem_osga_new_T_10_V_100_N_500_M_200_M1_50_G_1000_01"))
mm0 <- mm
for (i in 1:5) {
  load(file.path(pathToResults, paste("osem_osga_new_T_10_V_100_N_500_M_200_M1_50_G_1000_0", i, sep='')))
  print(mm)
  if (mm < mm0) {mm0 <- mm }
}
mm <- mm0
X <- array(NA, c(4, G, 2, 2, mm0*5))
for (i in 1:5) {
  load(file.path(pathToResults, paste("osem_osga_new_T_10_V_100_N_500_M_200_M1_50_G_1000_0", i, sep='')))
  X[,,1,,((i-1)*mm0+1):(mm0*5)] <- thetaMleBs[,,2,,1:mm0]
  X[,,2,,((i-1)*mm0+1):(mm0*5)] <- thetaMleBlockBs[,,2,,1:mm0]
}
X <- aperm(X, c(1,4,3,2,5))


plotTikzMat(
  "osga", nrow=4, ncol=2, x=1:G, y=X, height=plotHeightPar, width=plotWidthPar, mm=mm,
  xlab=rep("\\footnotesize{Iteration $p$}", times=2), ylab=c("\\footnotesize{Error of $\\hat{\\theta}_0[p]$}", "\\footnotesize{Error of $\\hat{\\theta}_1[p]$}", "\\footnotesize{Error of $\\hat{\\theta}_2[p]$}", "\\footnotesize{Error of $\\hat{\\theta}_3[p]$}"), ymin=-0.4, ymax=0.4, 
  main=c("\\footnotesize{\\mdseries{Stochastic gradient-ascent}}", "\\footnotesize{\\mdseries{Stochastic EM}}"), 
  col=c("magenta3", "forestgreen"), lty=rep(1,6), lwd=rep(1,6),
  subtractValues=TRUE, subVal=matrix(thetaML,4,2), plotHline=TRUE, hline=0, ltyHline=2, colHline="black"
)






for (i in 1:1000) { dev.off() }
