## Some additional R functions for use with the integrated model 
## for the little owls

###############################################################################
## Returns a number of (known) auxiliary model variables.
###############################################################################
getAuxiliaryModelParameters <- function(modelType, miscParameters) {

  ## ----------------------------------------------------------------------- ##
  ## Model Index
  ## ----------------------------------------------------------------------- ##
  
  # Model to be estimated
  #
  #  1: 1-component model: risk-premium parameters: 0; linear-leverage parameters: 0
  #  2: 1-component model: risk-premium parameters: 0; linear-leverage parameters: 1
  #  3: 1-component model: risk-premium parameters: 1; linear-leverage parameters: 0
  #  4: 1-component model: risk-premium parameters: 1; linear-leverage parameters: 1
  #
  #  5: 2-component model: risk-premium parameters: 0; linear-leverage parameters: 0
  #  6: 2-component model: risk-premium parameters: 0; linear-leverage parameters: 1
  #  7: 2-component model: risk-premium parameters: 0; linear-leverage parameters: 2
  #
  #  8: 2-component model: risk-premium parameters: 1; linear-leverage parameters: 0
  #  9: 2-component model: risk-premium parameters: 1; linear-leverage parameters: 1
  # 10: 2-component model: risk-premium parameters: 1; linear-leverage parameters: 2
  #
  # 11: 2-component model: risk-premium parameters: 2; linear-leverage parameters: 0
  # 12: 2-component model: risk-premium parameters: 2; linear-leverage parameters: 1
  # 13: 2-component model: risk-premium parameters: 2; linear-leverage parameters: 2
  
  N_COMPONENTS                 <- c(1,1,1,1, 2,2,2, 2,2,2, 2,2,2)
  N_RISK_PREMIUM_PARAMETERS    <- c(0,0,1,1, 0,0,0, 1,1,1, 2,2,2)
  N_LINEAR_LEVERAGE_PARAMETERS <- c(0,1,0,1, 0,1,2, 0,1,2, 0,1,2)
  N_WEIGHT_PARAMETERS          <- 2*(N_COMPONENTS==2) + 0*(N_COMPONENTS==1)
  
  ## Number of unknown parameters, i.e. the length of the vector theta:
  nComponents               <- N_COMPONENTS[modelType]
  nRiskPremiumParameters    <- N_RISK_PREMIUM_PARAMETERS[modelType]
  nLinearLeverageParameters <- N_LINEAR_LEVERAGE_PARAMETERS[modelType]
  nWeightParameters         <- N_WEIGHT_PARAMETERS[modelType]
  
  dimThetaMarginalised <- nComponents + nWeightParameters + 2 # the number of unknown model parameters excluding those which can be integrated out analytically
  dimTheta <- dimThetaMarginalised + 1 + nRiskPremiumParameters + nLeverageParameters # total numbero of unknown model parameters
  
  
  ## ----------------------------------------------------------------------- ##
  ## Specifying Hyperparameters and other known parameters
  ## ----------------------------------------------------------------------- ##

  # Hyperparameters:
  shapeHyperDelta <- 1
  scaleHyperDelta <- 1
  meanhyperAlpha <- 0
  varHyperAlpha <- 1
  shapeHyperXi <- 2
  scaleHyperXi <- 5
  shapeHyperInvZeta <- 2
  scaleHyperInvZeta <- 5
  meanHyperObsEq <- 0
  varHyperObsEq <- 100
  
  # Collecting all the known parameters in a vector:
  hyperParameters <- c(
    nComponents,
    nRiskPremiumParameters,
    nLinearLeverageParameters,
    dimThetaMarginalised,
    dimTheta,
    shapeHyperDelta,
    scaleHyperDelta,
    meanhyperAlpha,
    varHyperAlpha,
    shapeHyperXi,
    scaleHyperXi,
    shapeHyperInvZeta,
    scaleHyperInvZeta,
    meanHyperObsEq,
    varHyperObsEq
  )

  
  ## ----------------------------------------------------------------------- ##
  ## Names of the Parameters (only used for graphical output)
  ## ----------------------------------------------------------------------- ##
  
  thetaNames <- c()
  for (k in 1:nComponents) {
    thetaNames <- c(thetaNames, paste("log(Delta_kappa^", k, ")", sep=''))
  }
  if (nWeightParameters > 0) {
    for (k in 1:nWeightParameters) {
      thetaNames <- c(thetaNames, paste("alpha^", k, sep=''))
    }
  }
  thetaNames <- c(thetaNames, "xi", "log(1/zeta)", "mu")
  if (nRiskPremiumParameters == 1) {
      thetaNames <- c(thetaNames, "beta")
  } else if (nRiskPremiumParameters == 2) {
    for (k in 1:nRiskPremiumParameters) {
      thetaNames <- c(thetaNames, paste("beta^", k, sep=''))
    }
  }
  if (nLinearLeverageParameters == 1) {
      thetaNames <- c(thetaNames, "rho")
  } else if (nLinearLeverageParameters == 2) {
    for (k in 1:nLinearLeverageParameters) {
      thetaNames <- c(thetaNames, paste("rho^", k, sep=''))
    }
  }
  
  ## ----------------------------------------------------------------------- ##
  ## Support of the Unknown Parameters
  ## ----------------------------------------------------------------------- ##
  
  supportMin <- rep(-Inf, times=dimTheta) # minimum
  supportMax <- rep( Inf, times=dimTheta) # maximum
  support    <- matrix(c(supportMin, supportMax), dimTheta, 2, byrow=FALSE)

  return(list(modelType=modelType, nComponents=nComponents, nRiskPremiumParameters=nRiskPremiumParameters, nLinearLeverageParameters=nLinearLeverageParameters, dimTheta=dimTheta, dimThetaMarginalised=dimThetaMarginalised, hyperParameters=hyperParameters, support=support, thetaNames=thetaNames))
}