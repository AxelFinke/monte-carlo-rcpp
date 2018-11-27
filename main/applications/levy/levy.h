/// \file
/// \brief Generating data and performing inference in the Levy-driven SV model.
///
/// This file contains the functions for implementing the abstract model
/// class for the Levy-process driven stochastic volatility model.

#ifndef __LEVY_H
#define __LEVY_H

#include "main/templates/dynamic/stateSpace/stateSpace.h"
#include "main/algorithms/mwg/Mwg.h"
#include "main/algorithms/smc/Smc.h"
// #include "/ehmm/Ehmm.h" // TODO: implement this!
#include "main/rng/gaussian.h"

// [[Rcpp::depends("RcppArmadillo")]]


////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions which are needed repeatedly in the model
////////////////////////////////////////////////////////////////////////////////

/// Computes the partially observed vectors Z 
/// needed for evaluateing the (complete-data) likelihood.
void auxComputeZ(
  arma::mat& Z, 
  arma::colvec& varSigma, 
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta
)
{  
  unsigned int M = initialValues.size(); // number of component processes
  unsigned int P = observationTimesAux().size()-1; // number of observations
//   arma::colvec observationTimesAux(P+1)
//   observationTimesAux(0) = 0;
//   observationTimesAux(arma::span(1,P)) = observationTimes;

  Z.zeros(2*M+1,P); 
  Z.row(0) = arma::diff(observationTimesAux);
  arma::mat V(M,P, arma::mat::zeros);  // each row contains the values of the integrated processes $V^m$ at the observation times.

  std::vector<std::vector<unsigned int>> binContents; // binContents[i] contains the indices of the jumps that fall between observation $i-1$ and $i$.
  unsigned int idxMin, idxMax; // auxiliary indices of the first and last jump between two observations
  double sumJumpSizes; // sum of the jump sizes between two observations
  
  for (unsigned int m=0; m<M; k++)
  {
    // Calculating which jumps fall between specific observation times.
    binContents = computeBinContents(jumpTimes[m], observationTimesAux(arma::span(1,P)), true); // TODO: make this work with observationTimesAux rather than observationTimes!

    V(m,0) = initialValues[m] * std::exp(-kappa(m) * (observationTimesAux(1) - observationTimesAux(0)));
    if (!binContents[0].empty())
    { 
      idxMin = binContents[0][0];
      idxMax = binContents[0].size()-1;
      
      V(m,0)      += arma::accu(jumpSizes[m](arma::span(idxMin, idxMax)) % arma::exp(-kappa(m) * (observationTimesAux(1) - jumpTimes[m](arma::span(idxMin, idxMax)))));
      sumJumpSizes = arma::accu(jumpSizes[m](arma::span(idxMin, idxMax)));
    }
    else
    {
      sumJumpSizes = 0;
    }
    Z(1+m,0)   = (initialValues[m] - V(m,0) + sumJumpSizes)/kappa(m);
    Z(M+1+m,0) = sumJumpSizes - (observationTimesAux(1) - observationTimesAux(0)) * lambda(m) / zeta;
    
    for (unsigned int p=1; p<P; p++)
    {
      V(m,p) = V(m,p-1) * std::exp(-kappa(m) * (observationTimesAux(p+1) - observationTimesAux(p)));
      if (!binContents[p].empty())
      { 
        idxMin = binContents[p][0];
        idxMax = binContents[p].size()-1;
        
        V(m,p)      += arma::accu(jumpSizes[m](arma::span(idxMin, idxMax)) % arma::exp(-kappa(m) * (observationTimesAux(p+1) - jumpTimes[m](arma::span(idxMin, idxMax)))));
        sumJumpSizes = arma::accu(jumpSizes[m](arma::span(idxMin, idxMax)));
      }
      else
      {
        sumJumpSizes = 0;
      }
      Z(1+m,p)    = (V(m,p-1) - V(m,p) + sumJumpSizes)/kappa(m);
      Z(M+1+m,p) = sumJumpSizes - (observationTimesAux(p+1) - observationTimesAux(p)) * lambda(m) / zeta;
    }
  }
  
  // each row contains the values of the integrated processes $V^m$ at the observation times:
  varSigma = arma::trans(arma::sum(Z.rows(arma::span(M+1,2*M)), 0));
}
/// Computes the means and standard deviations 
/// needed for evaluating the observation density
/// (conditional on the all the latent variables and parameters).
void auxComputeObservationDensityParameters( // TODO: do we need a version of this which also stores finalValues along the way?
  arma::colvec& means, 
  arma::colvec& standardDeviations, 
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta, 
  const double mu, 
  const arma::colvec& beta, 
  const arma::colvec& rho
)
{
  unsigned int M = initialValues.size(); // number of component processes
  arma::mat Z, varSigma;
  auxComputeZ(Z, varSigma, initialValues, jumpTimes, jumpSizes, observationTimesAux, kappa, lambda, zeta);
  means = mu * Z.row(0) + beta.t() * Z.rows(arma::span(1,M)) + rho.t() * Z.rows(arma::span(M+1,2*M));
  standardDeviations = arma::sqrt(varSigma);
}
/// Computes the mean vector and precision matrix
/// needed for evaluating the observation density
/// (conditional on the all the latent variables and 
/// all parameters except those in the observation equation
/// as these are integrated out analytically). More precisely,
/// this function computes the parameters $\tilde{\mu}_p$ and $\tilde{\varSigma}_p$
/// conditional on $\tilde{\mu}_n$ and $\tilde{\varSigma}_n$, for $n < p$,
/// for some interval of observations $y_{n+1}, \dotsc, y_p$.
void auxComputeObservationDensityParametersMarginalised(
  const arma::colvec& initialMuTilde,
  const arma::mat& initialVarSigmaTildeInv,
  arma::colvec& finalMuTilde, 
  arma::mat& finalVarSigmaTildeInv,
  arma::colvec& varSigma,
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& observations,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta
)
{
// TODO check whether this function is still correct!
  unsigned int M = initialValues.size(); // number of component processes
  unsigned int P = observations.size(); // number of observations
  arma::mat Z;
//   arma::colvec varSigma;
  auxComputeZ(Z, varSigma, initialValues, jumpTimes, jumpSizes, observationTimesAux, kappa, lambda, zeta); 
  
  finalVarSigmaTildeInv = initialVarSigmaTildeInv;
  finalMuTilde          = initialVarSigmaTildeInv * initialMuTilde;
  for (unsigned int p=0; p<P; p++)
  {
    finalVarSigmaTildeInv = finalVarSigmaTildeInv + Z.col(p) * arma::trans(Z.col(p)) / varSigma(p);
    finalMuTilde          = finalMuTilde     + Z.col(p) * observations(p) / varSigma(p);
  }
  finalMuTilde = arma::inv(finalVarSigmaTildeInv) * finalMuTilde;
}
/// Evaluates the logarithm of the observation density in the case that 
/// the parameters of the observation equation are integrated out
/// (conditional on the latent variables and the remaining parameters).
/// This function assumes that finalMuTilde and finalVarSigmaTildeInv 
/// have already been computed.
double auxEvaluateLogObservationDensityMarginalised( 
  const arma::colvec& initialMuTilde,
  const arma::mat& initialVarSigmaTildeInv,
  const arma::colvec& finalMuTilde, 
  const arma::mat& finalVarSigmaTildeInv,
  const arma::colvec& varSigma, 
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& observations,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta
)
{
  return (
    - std::log(arma::det(finalVarSigmaTildeInv)) 
    + std::log(arma::det(initialVarSigmaTildeInv)) 
    - arma::accu(arma::log(varSigma)) 
    - arma::accu(observations % observations / varSigma) 
    + finalMuTilde.t() * finalVarSigmaTildeInv * finalMuTilde 
    - initialMuTilde.t() * initialVarSigmaTildeInv * initialMuTilde
  ) / 2.0;
}
/// Evaluates the logarithm of the observation density in the case that 
/// the parameters of the observation equation are integrated out
/// (conditional on the latent variables and the remaining parameters).
/// This function also computes finalMuTilde and finalVarSigmaTildeInv.
double auxEvaluateLogObservationDensityMarginalised(
  const arma::colvec& initialMuTilde,
  const arma::mat& initialVarSigmaTildeInv,
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& observations,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta
)
{
  arma::colvec finalMuTilde;
  arma::mat finalVarSigmaTildeInv;
  aram::colvec varSigma;
  auxEvaluateLogObservationDensityMarginalised( 
    initialMuTilde, initialVarSigmaTildeInv, finalMuTilde, finalVarSigmaTildeInv, varSigma, 
    initialValues, jumpTimes, jumpSizes, observationTimesAux, observations, kappa, lambda, zeta
  );
  return auxEvaluateLogObservationDensityMarginalised( 
    initialMuTilde, initialVarSigmaTildeInv, finalMuTilde, finalVarSigmaTildeInv, varSigma, 
    initialValues, jumpTimes, jumpSizes, observationTimesAux, observations, kappa, lambda, zeta
  );
}
/// Evaluates the log observation density for a specific interval
/// in the case that the parameters of the observation equation
/// are not integrated out
double evaluateLogObservationDensity(
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& observations,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta, 
  const double mu, 
  const arma::colvec& beta, 
  const arma::colvec& rho
)
{
  arma::colvec means(observations.size());
  arma::colvec standardDeviations(observations.size());
  auxComputeObservationDensityParameters(means, standardDeviations, initialValues, jumpTimes, jumpSizes, observationTimesAux, kappa, lambda, zeta, mu, beta, rho);
  double logDensity;
  for (unsigned int p=0; p<observations.size(); p++)
  {
    logDensity += R::dnorm(observations(p), means(p), standardDeviations(p), true);
  }
  return logDensity;
}
/// Computes samples the parameters of the observation equation from their
/// full conditional posterior distribution.
arma::colvec auxSampleMarginalisedParameters(
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& observations,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta,
  const arma::colvec& initialMuTilde,
  const arma::mat& initialVarSigmaTildeInv
)
{
//   unsigned int M = initialValues.size(); // number of component processes
//   unsigned int P = observations.size(); // number of observations
//   arma::mat Z;
//   arma::colvec varSigma;
//   auxComputeZ(Z, varSigma, initialValues, jumpTimes, jumpSizes, observationTimesAux, kappa, lambda, zeta);
//   
//   arma::mat finalVarSigmaTildeInv = initialVarSigmaTildeInv;
//   arma::mat finalMuTilde          = initialVarSigmaTildeInv * initialMuTilde;
//   for (unsigned int p=0; p<P; p++)
//   {
//     finalVarSigmaTildeInv = finalVarSigmaTildeInv + Z.col(p) * arma::trans(Z.col(p)) / varSigma(p);
//     finalMuTilde          = finalMuTilde          + Z.col(p) * observations(p) / varSigma(p);
//   }
//   arma::mat varSigmaTilde = arma::inv(finalVarSigmaTildeInv);
//   arma::colvec muTilde    = varSigmaTilde * finalMuTilde;
//   
//   arma::colvec finalMuTilde;
//   arma::mat finalVarSigmaTildeInv;
//   aram::colvec varSigma;
//   
  auxEvaluateLogObservationDensityMarginalised( 
    initialMuTilde, initialVarSigmaTildeInv, finalMuTilde, finalVarSigmaTildeInv, varSigma, 
    initialValues, jumpTimes, jumpSizes, observationTimesAux, observations, kappa, lambda, zeta
  );
  
  return muTilde + arma::chol(varSigmaTilde) * arma::randn<arma::colvec>(2*M+1);
}
/// Computes the logarithm of the conditional density of the PPPs,
/// conditional on the model parameters
double auxEvaluateLogCompoundPoissonPriorDensity(
  const std::vector<double>& initialValues, 
  const std::vector<arma::colvec>& jumpTimes, 
  const std::vector<arma::colvec>& jumpSizes,
  const arma::colvec& observationTimesAux,
  const arma::colvec& kappa, 
  const arma::colvec& lambda, 
  const double zeta,
  const arma::colvec& stationaryShape, 
  const double stationaryScale
)
{
  unsigned int M = initialValues.size(); // number of component processes
  double logDensity = 0.0;
  for (unsigned int m=0; m<M; m++)
  {
    logDensity += 
      R::dpois(jumpTimes[m].size(), lambda(m) * (arma::max(observationTimesAux) - arma::min(observationTimesAux)), true)
      + std::log(zeta) - zeta*arma::accu(zeta * jumpTimes[m])
      + dGamma(initialValues[m], stationaryShape(m), stationaryScale);
  }
  return logDensity;
}

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the state-space model
////////////////////////////////////////////////////////////////////////////////

/// Holds all the static model parameters.
class ModelParameters
{
public:
  
  /// Determines the model parameters from arma::colvec theta.
  void setUnknownParameters(const arma::colvec& theta)
  {
    delta_.set_size(nComponents_);
    kappa_.set_size(nComponents_);
    w_.set_size(nComponents_);
    lambda_.set_size(nComponents_);
          
    if (nComponents_ > 1)
    {
      delta_    = arma::exp(theta(arma::span(0,nComponents_-1)));
      kappa_    = arma::cumsum(delta_);    
      w_        = normaliseWeights(theta(arma::span(nComponents_, 2*nComponents_-1))); 
      xi_       = theta(2*nComponents_);
      zeta_     = 1.0 / theta(2*nComponents_+1);
    }
    else // i.e. if there is only one component
    {
      delta_(0) = std::exp(theta(0));
      kappa_(0) = delta_(0);
      w_(0)     = 1.0;
      xi_       = theta(1);
      zeta_     = 1.0 / theta(2);
    }
    lambda_ = w_ % kappa_ * xi_ * zeta_;
    stationaryShape_ = lambda_ / kappa_;
    stationaryScale_ = 1.0 / (xi_ * zeta_);
    
    mu_ = theta(dimTheta_-nRiskPremiumParameters_-nLinearLeverageParameters_-1);
    beta_.set_size(nComponents);
    rho_.set_size(nComponents);
   
    if (nRiskPremiumParameters_ == 0)
    {
      beta_.zeros();
    }
    else if (nRiskPremiumParameters_ == 1)
    {
      beta_.fill(theta(arma::span(dimTheta_-1-nLinearLeverageParameters_, dimTheta_-nLinearLeverageParameters_-1)));
    }
    else
    {
      beta_ = theta(arma::span(dimTheta_-nRiskPremiumParameters_-nLinearLeverageParameters_, dimTheta_-nLinearLeverageParameters_-1));
    }
    
    if (nLinearLeverageParameters_ == 0)
    {
      rho_.zeros();
    }
    else if (nLinearLeverageParameters_ == 1)
    {
      rho_.fill(theta(arma::span(dimTheta_-1, dimTheta_-1)));
    }
    else 
    {
      rho_ = theta(arma::span(dimTheta_-nLinearLeverageParameters_, dimTheta_-1)); 
    }
    
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyp)
  {
    nComponents_ = hyp(0); // the number of latent compound-Possion processes
    nRiskPremiumParameters_ = hyp(1); // number of distinct risk-premium parameters
    nLinearLeverageParameters_ = hyp(2); // number of distinct linear-leverage parameters
    dimThetaMarginalised_ = hyp(3); // length of the parameter vector after integrating out the parameters of the observation equation
    dimTheta_ = hyp(4); // length of the parameter vector
    
    // Hyperparameters:
    shapeHyperDelta_ = hyp(5);
    scaleHyperDelta_ = hyp(6);
    meanHyperAlpha_ = hyp(7);
    varHyperAlpha_ = hyp(8);
    sdHyperAlpha_ = std::sqrt(varHyperAlpha_);
    shapeHyperXi_ = hyp(9);
    scaleHyperXi_ = hyp(10);
    shapeHyperInvZeta_ = hyp(11);
    scaleHyperInvZeta_ = hyp(12);
    meanHyperObsEq_ = hyp(13);
    varHyperObsEq_ = hyp(14);
    sdHyperObsEq_ = std::sqrt(varHyperObsEq_);
    
    initialVarSigmaTildeInv_ = (1.0 / varHyperObsEq_) * arma::eye(2*nComponents_+1, 2*nComponents_+1);
    initialMuTilde_ = meanHyperObsEq_ *  arma::ones(2*nComponents_ + 1);
  

//     // Times at which observations are recorded:
//     observationTimes_ = hyp(arma::span(15, hyp.size()-1));   
//     lastObservationTime_ = arma::max(observationTimes_);
  }
  /// Returns the number of observations.
  unsigned int getNObservations() const {return nObservations_;}
  /// Returns the number of latent compound-Poisson processes.
  unsigned int getNComponents() const {return nComponents_;}
    /// Returns the number of distinct risk-premium parameters.
  unsigned int getNRiskPremiumParameters() const {return nRiskPremiumParameters_;}
    /// Returns the number of distinct linear-leverage parameters.
  unsigned int getNLinearLeverageParameters() const {return nLinearLeverageParameters_;}
  /// Returns the differences in the decay-rate parameters of all component processes.
  const arma::colvec& getDelta() const {return delta_;}
  /// Returns the difference in the decay-rate parameter of the $m$th component process.
  double getDelta(const unsigned int m) const {return delta_(m);}
  /// Returns the decay-rate parameters of all component processes.
  const arma::colvec& getKappa() const {return kappa_;}
  /// Returns the decay-rate parameter of the $m$th component processes.
  double getKappa(const unsigned int m) const {return kappa_(m);}
  /// Returns the component weights.
  const arma::colvec& getW() const {return w_;}
  /// Returns the stationary mean of the latent processes (needs to be weighted by w_);
  double getXi() const {return xi_;}
  /// Returns the stationary variance of the latent processes (needs to be weighted by w_);
  double getOmegaSquared() const {return xi_ / zeta_;}
  /// Returns the rate parameters of latent Poisson processes for the jump times.
  const arma::colvec& getLambda() const {return lambda_;}
  /// Returns the rate parameters of latent Poisson processes for the jump times given some parameter vector theta.
  arma::colvec getLambda(const arma::colvec& theta) 
  {
    setUnknownParameters(theta);
    return lambda_;
  } 
  /// Returns the drift parameter.
  double getMu() const {return mu_;}
  /// Returns the risk-premium parameters.
  const arma::colvec& getBeta() const {return beta_;}
  /// Returns the linear-leverage parameters.
  const arma::colvec& getRho() const {return rho_;}
    /// Returns one risk-premium parameter.
  double getBeta(const unsigned int i) const {return beta_(i);}
  /// Returns one linear-leverage parameter.
  double getRho(const unsigned int i) const {return rho_(i);}
  /// Returns the inverse of the exponential jump-size rate.
  double getInvZeta() const {return 1.0 / zeta_;}
  /// Returns the inverse of the exponential jump-size rate.
  double getZeta() const {return zeta_;}
  /// Returns the shape parameter of the gamma prior on the differences in the decay-rate parameters.
  double getShapeHyperDelta() const {return shapeHyperDelta_;}
  /// Returns the scale parameter of the gamma prior on the differences in the decay-rate parameters.
  double getScaleHyperDelta() const {return scaleHyperDelta_;}
  /// Returns the prior mean of the Gaussian priors on the reparametrised component weights.
  double getMeanHyperAlpha() const {return meanHyperAlpha_;}
  /// Returns the prior variance of the Gaussian priors on the reparametrised component weights.
  double getVarHyperAlpha() const {return varHyperAlpha_;}
  /// Returns the prior standard deviation of the Gaussian priors on the reparametrised component weights.
  double getSdHyperAlpha() const {return sdHyperAlpha_;}
  /// Returns the shape parameter of the gamma prior on the stationary mean of the latent processes.
  double getShapeHyperXi() const {return shapeHyperXi_;}
  /// Returns the scale parameter of the gamma prior on the stationary mean of the latent processes.
  double getScaleHyperXi() const {return scaleHyperXi_;}
  /// Returns the shape parameter of the gamma prior on the inverse of the exponential jump-size rate.
  double getShapeHyperInvZeta() const {return shapeHyperInvZeta_;}
  /// Returns the scale parameter of the gamma prior on the inverse of the exponential jump-size rate.
  double getScaleHyperInvZeta() const {return scaleHyperInvZeta_;}
  /// Returns the mean of the IID normal priors on each of the parameters in the observation equation.
  double getMeanHyperObsEq() const {return meanHyperObsEq_;}
  /// Returns the variance of the IID normal priors on each of the parameters in the observation equation.
  double getVarHyperObsEq() const {return varHyperObsEq_;}
  /// Returns the standard deviatiation of the IID normal priors on each of the parameters in the observation equation.
  double getSdHyperObsEq() const {return sdHyperObsEq_;}
  /// Returns the times at which observations are taken (assuming the process is started at time 0).
  const arma::colvec& getObservationTimes() const {return observationTimes_;}
  /// Returns the concatenated vector (0, observationTimes_).
  const arma::colvec& getObservationTimesAux() const {return observationTimesAux_;}
  /// Returns the time at which the last observations are taken.
  double getLastObservationTime() const {return lastObservationTime_;}
  /// Returns the shape parameter of Gamma stationary distribution of the $m$th latent component process.
  const arma::colvec& getStationaryShape() const {return stationaryShape_(m);}
  /// Returns the shape parameter of Gamma stationary distribution of the $m$th latent component process.
  double getStationaryShape(const unsigned int m) const {return stationaryShape_(m);}
 /// Returns the scale parameter of Gamma stationary distribution of the latent processes.
  double getStationaryScale() const {return stationaryScale_;}
  /// Returns the mean vector of the Gaussian prior on the parameters of the observation equation
  const arma::colvec& getInitialMuTilde() const {return initialMuTilde_;}
  /// Returns the inverse of the covariance matrix of the Gaussian prior on the parameters of the observation equation
  const arma::mat& getInitialVarSigmaTildeInv() const {return initialVarSigmaTildeInv_;}
  /// Specifies the times at which observations are taken (assuming the process is started at time 0).
  void setObservationTimes(const arma::colvec& observationTimes)
  {
    observationTimes_ = observationTimes;
    nObservations_ = observationTimes.size();
    observationTimesAux_.zeros(nObservations_+1);
    observationTimesAux_(0) = 0;
    observationTimesAux_(arma::span(1,nObservations)) = observationTimes_; 
    lastObservationTime_ = arma::max(observationTimes_);
  }
  
private:
  
  // Unknown model parameters:
  arma::colvec delta_; // differences in the decay-rate parameters of all component processes
  arma::colvec kappa_; // decay-rate parameters of all component processes
  arma::colvec w_; // normalised component weights
  arma::colvec lambda_; // rate parameters for the latent Poisson processes for the jump times
  double xi_; // mean-parameteter for the latent processes
  double zeta_; // rate of the jump-size distribution
  double mu_; // drift parameter in the observation equation
  arma::colvec beta_; // risk-premium parameters in the observation equation (can be integrated out analytically)
  arma::colvec rho_; // linear-leverage parameters in the observation equation (can be integrated out analytically)
  arma::colvec stationaryShape_; // shape parameters of the Gamma stationary distributions of each of the latent volatility process
  double stationaryScale_; // scale parameter of the Gamma stationary distributions of each of the latent volatility process (identical for all processes!)
  
  // Known hyperparameters for the prior distribution:
  double shapeHyperDelta_, scaleHyperDelta_; // parameters of the gamma priors on the differences in the decay-rate parameters.
  double meanHyperAlpha_, varHyperAlpha_, sdHyperAlpha_; // mean and variance (and standard deviation) of the Gaussian priors on the reparametrised component weights
  double shapeHyperXi_, scaleHyperXi_; // parameters of the gamma prior on the stationary mean of the latent processes.
  double shapeHyperInvZeta_, scaleHyperInvZeta_; // parameters of the gamma prior on the inverse of the exponential jump-size rate.
  double meanHyperObsEq_, varHyperObsEq_, sdHyperObsEq_; // mean and variance (and standard deviation) of the IID normal priors on each of the parameters in the observation equation (these parameters are integrated out in the algorithm).
  
  arma::colvec initialMuTilde_; // mean vector of the normal prior on the parameters of the observation equation
  arma::mat initialVarSigmaTildeInv_; // inverse of the covariance matrix of the normal prior on the parameters of the observation equation
  
  // Other known parameters:
  unsigned int nComponents_; // (known) number of latent processestimes at which observations are taken 
  unsigned int nRiskPremiumParameters_; // (known) number of distinct risk-premium parameters
  unsigned int nLinearLeverageParameters_; // (known) number of distinct linear-leverage parameters
  unsigned int nObservations_; // number of observations
  arma::colvec observationTimes_; // times at which observations are taken
  arma::colvec observationTimesAux_; // vector whose first element is zero and whose remaining elements are equal to observationTimes_
  double lastObservationTime_; // the time at which the last observations are taken

};
/// Holds all latent variables in the model under the 
/// centred parametrisation. //TODO: do we need to store the posterior mean/variance of the marginalised parameters in here?
class LatentPath
{
public:
  
  /// Returns the number of jumps in the $m$th component process.
  unsigned int getNJumps(const unsigned int m)
  {
    return jumpTimes_[m].size();
  }
  /// Initialises the components.
  void setup(const unsigned int nComponents)
  {
    jumpTimes_.resize(nComponents);
    jumpSizes_.resize(nComponents);
    initialValues_.resize(nComponents);
  }
  /// Samples initial values and jumps from the prior distribution.
  void sampleFromPrior(
    const double T, const arma::colvec& lambda, const double zeta, 
    const arma::colvec& stationaryShape, const double stationaryScale
  )
  {
    unsigned int nJumps;
    for (unsigned int m=0; m<M; m++)
    {
      initialValues_[m] = arma::as_scalar(arma::randg(1, distr_param(stationaryShape(m), stationaryScale))); 
      nJumps            = static_cast<unsigned int>(R::rpois(lambda(m) * T)); // number of jumps in the $m$th component process
      jumpTimes_[m]     = T * arma::sort(arma::randu<arma::colvec>(nJumps));
      jumpSizes_[m]     = - arma::log(arma::randu<arma::colvec>(nJumps)) / zeta;
    } 
  }
  
  std::vector<arma::colvec> jumpTimes_; // each vector component contains all jump times of a single component process (first entry must be > 0)
  std::vector<arma::colvec> jumpSizes_; // each vector component contains all jump sizes of a single component process
  std::vector<double> initialValues_;   // each vector component contains the initial values of a single component process
  
};
/// Holds all latent variables in the model under a different 
/// parametrisation. NOTE: This is currently not used!
class LatentPath LatentPathRepar;
/// Holds all latent variables in the model
class LatentVariable
{
public:
  
  /// Returns the number of jumps in the $m$th component process.
  unsigned int getNJumps(const unsigned int m)
  {
    return jumpTimes_[m].size();
  }
  // Initialises the components.
  void setup(const unsigned int nComponents)
  {
    jumpTimes_.resize(nComponents);
    jumpSizes_.resize(nComponents);
    initialValues_.resize(nComponents);
    finalValues_.resize(nComponents);
    mostRecentJumpTime_.resize(nComponents);
    mostRecentValue_.resize(nComponents);
  }
  /// Samples the initial values from the stationary distribution.
  void sampleInitialValues(const arma::colvec& stationaryShape, const double stationaryScale)
  {
    unsigned int M = initialValues_.size();
    for (unsigned int m=0; m<M; m++)
    {
      initialValues_[m] = rng_.randomGamma(stationaryShape(m), stationaryScale);
    }
  }
  /// Specifies the initial values.
  void setInitialValues(const std::vector<double>& initialValues) {initialValues_ = initialValues;}
  /// Samples number of jumps, jump times and jump sizes
  /// in some interval (t0, t1] and computes finalValues_
  /// conditional on initialValues.
  void sampleJumps(const double t0, const double t1, const arma::colvec lambda, const arma::colvec& kappa, const double zeta)
  {
    unsigned int M = jumpTimes_.size();
    for (unsigned int m=0; m<M; m++)
    {
      nJumps        = static_cast<unsigned int>(R::rpois(lambda(m) * (t1 - t0))); // number of jumps in the $m$th component process
      jumpTimes_[m] = t0 + (t1 - t0) * arma::sort(arma::randu<arma::colvec>(nJumps));
      jumpSizes_[m] = - arma::log(arma::randu<arma::colvec>(nJumps)) / zeta;
    }
  }
  /// Specifies the initial values.
  void computeFinalValues(const double t0, const double t1, const arma::colvec& kappa)
  {
    unsigned int M = jumpTimes_.size();
    for (unsigned int m=0; m<M; m++)
    {
      finalValues_[m] = initialValues_[m] *std::exp(-kappa(m) * (t1 - t0));
      if (getNJumps(m) > 0)
      { 
        finalValues_[m] += arma::accu(jumpSizes_[m] % arma::exp(-kappa(m) * (t1 - jumpTimes_[m])));
      }
    }
  }
  /// Returns the logarithm of the observation density for the current time interval
  /// conditional on the latent processes in that interval; the parameters of the 
  /// observation equation are integrated out analytically.
  double evaluateLogObservationDensityMarginalised(const arma::colvec& observationTimesAux, const arma::colvec& observations, const arma::colvec& kappa, const arma::colvec& lambda, const double zeta)
  {
    return auxEvaluateLogObservationDensityMarginalised
    (
      initialMuTilde_, initialVarSigmaTildeInv_,
      finalMuTilde_, finalVarSigmaTildeInv_, 
      initialValues_, jumpTimes_, jumpSizes_,
      observationTimesAux, observations,
      kappa, lambda, zeta
    )
  }
  double evaluateLogObservationDensity(const arma::colvec& observationTimesAux, const arma::colvec& observations, const arma::colvec& kappa)
  {
    // TODO
    
  }
  
  std::vector<arma::colvec> jumpTimes_; // jump times in a particular interval
  std::vector<arma::colvec> jumpSizes_; // corresponding jump sizes
  std::vector<double> initialValues_; // the value of the process at the beginning of the interval
  std::vector<double> finalValues_; // the value of the process at the end of the interval
  std::vector<double> mostRecentJumpTime_; // last jump time before the current interval
  std::vector<double> mostRecentValue_; // more precisely, the value of the latent process at the most recent jupp
  arma::colvec initialMuTilde_; // conditional posterior mean of the parameters in the observation equation at the beginning of the current interval
  arma::mat initialVarSigmaTildeInv_; //  conditional posterior covariance matrix of the parameters in the observation equation at the beginning of the current interval
  arma::colvec finalMuTilde_; // conditional posterior mean of the parameters in the observation equation at the end of the current interval
  arma::mat finalVarSigmaTildeInv_; //  conditional posterior covariance matrix of the parameters in the observation equation at the end of the current interval
  arma::colvec varSigma_; // the matrices $\varSigma_p$ associated with the observations in the current interval
};
/// Holds (some of the) Gaussian auxiliary variables generated as part of 
/// the SMC algorithm.
/// Can be used to hold the normal random variables used for implementing
/// correlated pseudo-marginal kernels. Otherwise, this may not need to 
/// be used.
typedef double Aux; // NOTE: correlated pseudo-marginal kernels are cannot easily be used in this model

/// Holds all observations.
typedef arma::colvec Observations;

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the class <<Model>>
///////////////////////////////////////////////////////////////////////////////

/// Evaluates the log-prior density of the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogPriorDensity()
{
  unsigned int K = modelParameters_.getNComponents();
  double logDensity = 0;
  
  // Gamma priors on the differences in the decay-rate parameters:
  for (unsigned int k=0; k<K; k++)
  {
    logDensity += R::dgamma(modelParameters_.getDeltaKappa(k), modelParameters_.getShapeHyperDelta(), modelParameters_.getScaleHyperDelta(), true);
  }
  
  // Dirichlet prior on the component weights:
  logDensity += std::lgamma(modelParameters_.getHyperW() * K) - K * std::lgamma(modelParameters_.getHyperW() * K) + (modelParameters_.getHyperW()-1) * arma::accu(arma::log(modelParameters_.getW()));
  
  // Gamma prior on the stationary mean of the latent processes:
  logDensity += R::dgamma(modelParameters_.getXi(), modelParameters_.getShapeHyperXi(), modelParameters_.getScaleHyperXi(), true);
  
  // Gamma prior on the inverse of the exponential jump-size rate:
  logDensity += R::dgamma(modelParameters_.getInvZeta(), modelParameters_.getShapeHyperInvZeta(), modelParameters_.getScaleHyperInvZeta(), true);
  
  // Log-prior density of the parameters associated with the observation equation (if these are not integrated out analytically).
  logDensity += R::dnorm(modelParameters_.getMu(), modelParameters_.getMeanHyperObsEq(), modelParameters_.getSdHyperObsEq(), true);
  if (!marginaliseParameters_)
  {
    for (unsigned int i=0; i<modelParameters_.getNRiskPremiumParameters(); i++)
    {
      logDensity += R::dnorm(modelParameters_.getBeta(i), modelParameters_.getMeanHyperObsEq(), modelParameters_.getSdHyperObsEq(), true);
    }
    for (unsigned int i=0; i<modelParameters_.getNLinearLeverageParameters(); i++)
    {
      logDensity += R::dnorm(modelParameters_.getRho(i), modelParameters_.getMeanHyperObsEq(), modelParameters_.getSdHyperObsEq(), true);
    }
  }
  
  return logDensity;
}
/// Samples the set of parameters from the prior.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromPrior(arma::colvec& theta)
{
  theta.set_size(dimTheta_);
  unsigned int K = modelParameters_.getNComponents();
  
  // Gamma priors on the differences in the decay-rate parameters:
  for (unsigned int k=0; k<K; k++)
  {
    theta(k) = rng_.randomGamma(modelParameters_.getShapeHyperDelta(), modelParameters_.getScaleHyperDelta());
  }
  
  // Dirichlet prior on the component weights:
  for (unsigned int k=K; k<2*K; k++)
  {
    theta(k) = rng_.randomGamma(modelParameters_.getHyperW(), 1.0);
  }
  theta(arma::span(K, 2*K-1)) = theta(arma::span(K, 2*K-1)) / arma::accu(theta(arma::span(K, 2*K-1)));
  
  // Gamma prior on the stationary mean of the latent processes:
  theta(2*K) = rng_.randomGamma(modelParameters_.getShapeHyperXi(), modelParameters_.getScaleHyperXi());
  
  // Gamma prior on the inverse of the exponential jump-size rate:
  theta(2*K+1) = rng_.randomGamma(modelParameters_.getShapeHyperInvZeta(), modelParameters_.getScaleHyperInvZeta());
  
  // Log-prior density of the parameters associated with the observation equation (if these are not integrated out analytically).
  if (!marginaliseParameters_)
  {
    for (unsigned int i=modelParameters_.getDimThetaMarginalised(); i<dimTheta_; i++)
    {
      theta(i) = modelParameters_.getMeanHyperObsEq() +  arma::randn() * modelParameters_.getSdHyperObsEq();
    }
  }
  
}
/// Increases the gradient by the gradient of the log-prior density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogPriorDensity(arma::colvec& gradient)
{
  // Empty: we do not use gradient information in this example.
}
/// Increases the gradient by the gradient of the log-initial density of
/// the latent states.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogInitialDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient)
{
  // Empty: we do not use gradient information in this example.
}
/// Increases the gradient by the gradient of the log-transition density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld, arma::colvec& gradient)
{
  // Empty: we do not use gradient information in this example.
}
/// Increases the gradient by the gradient of the log-observation density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogObservationDensity(const unsigned int t,  const LatentVariable& latentVariable, arma::colvec& gradient)
{
  // Empty: we do not use gradient information in this example.
}
/// Evaluates the marginal likelihood of the parameters (with the latent 
/// variables integrated out).
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihood()
{
  // Empty: the score is intractable in this model.
  return 0;
}
/// Evaluates the score.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateScore(arma::colvec& score)
{
  // Empty: the score is intractable in this model.
}
/// Generates latent variables from their full conditional distribution.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::runGibbs(LatentPath& latentPath)
{
  // Empty: the full conditional distribution of the latent variables is intractable in this model.
}
/// Simulates count data from state-space model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::simulateData(const arma::colvec& extraParameters)
{
  
  nObservations_ = modelParameters_.getNObservations(); // TODO: implement this
//   unsigned int M = modelParameters_.getNComponents(); 
//   unsigned int nJumps;
  
  latentPath_.sampleFromPrior(modelParameters_.getLastObservationTime(), modelParameters_.getLambda(m), modelParameters_.getZeta(), modelParameters_.getStationaryShape(), modelParameters_.getStationaryScale());
  
   
//   for (unsigned int m=0; m<M; m++)
//   {
//     latentPath_.initialValues_[m] = arma::as_scalar(arma::randg(1, distr_param(modelParameters_.getStationaryShape(m), modelParameters_.getStationaryScale()))); 
//     nJumps = static_cast<unsigned int>(R::rpois(modelParameters_.getLambda(m)) * modelParameters_.getLastObservationTime())); // number of jumps in the $m$th component process
//     latentPath_.jumpTimes_[m] = modelParameters_.getLastObservationTime() * arma::sort(arma::randu<arma::colvec>(nJumps));
//     latentPath_.jumpSizes_[m] = - modelParameters_.getInvZeta() * arma::log(arma::randu<arma::colvec>(nJumps));
//   } 
  
  // Computing the means and standard deviations needed for evaluating the likelihood 
  // (conditional on the latent variables and parameters).
  arma::colvec means(nObservations_);
  arma::colvec standardDeviations(nObservations_);

  auxComputeObservationDensityParameters(
    means, standardDeviations, 
    latentPath_.initialValues_, 
    latentPath_.jumpTimes_, latentPath_.jumpSizes_,
    modelParameters_.getObservationTimesAux(),
    modelParameters_.getKappa(),
    modelParameters_.getLambda(),
    modelParameters_.getZeta(),
    modelParameters_.getMu(),
    modelParameters_.getBeta(),
    modelParameters_.getRho()
  )
  
  observations_ = means + standardDeviations % arma::randn(nObservations_);
  
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihood(
  const LatentPath& latentPath
)
{

  double logDensity = 0.0;
  
  // Likelihood term:
  if (marginaliseParameters_)
  {
//     arma::colvec finalMuTilde;
//     arma::mat finalVarSigmaTildeInv;
//     logDensity += 
//     auxEvaluateLogObservationDensityMarginalised(
//       modelParameters_.getInitialMuTilde(),
//       modelParameters_.getInitialVarSigmaTildeInv(), 
//       finalMuTilde,
//       finalVarSigmaTildeInv,
//       latentPath.initialValues_, 
//       latentPath.jumpTimes_, 
//       latentPath.jumpSizes_,
//       modelParameters_.getObservationTimesAux(),
//       observations_,
//       modelParameters_.getKappa(),
//       modelParameters_.getLambda(),
//       modelParameters_.getZeta()
//     )  
    
    auxEvaluateLogObservationDensityMarginalised
    (
      modelParameters_.getInitialMuTilde(),
      modelParameters_.getInitialVarSigmaTildeInv(), 
      latentPath.initialValues_, 
      latentPath.jumpTimes_, 
      latentPath.jumpSizes_,
      modelParameters_.getObservationTimesAux(),
      observations_,
      modelParameters_.getKappa(),
      modelParameters_.getLambda(),
      modelParameters_.getZeta()
    )
  }
  else
  {  
//     arma::colvec means(observations_.size());
//     arma::colvec standardDeviations(observations_.size());
//     auxComputeObservationDensityParameters(
//       means, 
//       standardDeviations, 
//       latentPath.initialValues_, 
//       latentPath.jumpTimes_, 
//       atentPath.jumpSizes_,
//       modelParameters_.getObservationTimesAux(),
//       modelParameters_.getKappa(),
//       modelParameters_.getLambda(),
//       modelParameters_.getZeta(),
//       modelParameters_.getMu(),
//       modelParameters_.getBeta(),
//       modelParameters_.getRho()
//     )
//     for (unsigned int p=0; p<observations_.size(); p++)
//     {
//       logDensity += R::dnorm(observations_(p), means(p), standardDeviations(p), true);
//     }
    logDensity = evaluateLogObservationDensity(
      latentPath.initialValues_, 
      latentPath.jumpTimes_, 
      atentPath.jumpSizes_,
      modelParameters_.getObservationTimesAux(),
      observations_,
      modelParameters_.getKappa(),
      modelParameters_.getLambda(),
      modelParameters_.getZeta(),
      modelParameters_.getMu(),
      modelParameters_.getBeta(),
      modelParameters_.getRho()
    )
    
  } 
  return logDensity
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables using a (partially) non-centred parametrisation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihoodRepar(const LatentPathRepar& latentPathRepar)
{
  // NOTE: this function is unused!
  return 0.0y
}
/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromInitialDistribution()
{
  // TODO
  LatentVariable x;
  return x;
}
/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromTransitionEquation(const unsigned int t, const LatentVariable& latentVariableOld)
{
  // TODO
  LatentVariable x;
  return x;
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogInitialDensity(const LatentVariable& latentVariable)
{
  // TODO
  return 0.0;
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld)
{
  // TODO
  return 0.0;
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable)
{
  // TODO
  return 0.0;
}
/// Samples a single observation according to the observation equation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromObservationEquation(const unsigned int t, Observations& observations, const LatentVariable& latentVariable)
{
  // NOTE: unused here because we implement simulateData() directly.
}
/// Evaluates the log of the likelihood associated with some subset of the 
/// (static) model parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodFirst(LatentPath& latentPath)
{
  // Unused in this example
  return 0.0;
}



///////////////////////////////////////////////////////////////////////////////
/// Member functions of the class <<Mwg>>
///////////////////////////////////////////////////////////////////////////////

/// Type of reversible-jump move performed for one of the 
/// uniformly-at-random selected component processes
enum MwgMoveType
{  
  MWG_MOVE_INITIAL_VALUE = 0, // proposes to change the initial jump size (at time 0) 
  MWG_MOVE_MODIFY_JUMP_TIME, // selects a jump uniformly at random and proposes to modify its jump time by sampling it from a uniform distribution between the previous and next jump time
  MWG_MOVE_MODIFY_JUMP_SIZE, // selects a jump uniformly at random and proposes to modify its jump size by sampling it from its prior distribution
  MWG_MOVE_BIRTH, // proposes to add a jump uniformly at random in the relevant time interval with jump time drawn from the prior
  MWG_MOVE_DEATH // selects a jump uniformly at random and proposes to delete it
};
/// Holds some additional auxiliary parameters for the SMC algorithm.
class MwgParameters
{
public:
  
};
/// Runs the Metropolis-within-Gibbs algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class MwgParameters>
void Mwg<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, MwgParameters>::runSamplerBase(const arma::colvec& theta, LatentPath& latentPath)
{
  model_.setUnknownParameters(theta);
  unsigned int M               = model_.getModelParameters().getNComponents();
  arma::colvec stationaryShape = model_.getModelParameters().getStationaryShape();
  double stationaryScale       = model_.getModelParameters().getStationaryScale();
  double zeta                  = model_.getModelParameters().getZeta();
  double T                     = model_.getModelParameters().getLastObservationTime();
  
  unsigned int m;
  MwgMoveType selectedMove;
  LatentPath latentPathProp;
  double logProposalDensityNUm, logProposalDensityDen, logAlpha, logCompleteLikelihood, logCompleteLikelihoodProp;
  unsigned int idx; // auxiliary index variables
  unsigned int nJumps; 
  
  logCompleteLikelihood = 
    auxEvaluateLogCompoundPoissonPriorDensity
    (
      latentPath.initialValues_, 
      latentPath.jumpTimes_, 
      latentPath.jumpSizes_,
      model_.getModelParameters().getObservationTimesAux(),
      model_.getModelParameters().getKappa(),
      model_.getModelParameters().getLambda(),
      model_.getModelParameters().getZeta(),
      model_.getModelParameters().getStationaryShape(),
      model_.getModelParameters().getStationaryScale()
    )
    + model_.evaluateLogCompleteLikelihood(latentPath);
  
  for (unsigned int n=0; n<nUpdates_; n++)
  {
    latentPathProp = latentPath;
    m              = static_cast<unsigned int>(rng_.randomUniformInt(0, M-1));
    selectedMove   = static_cast<MwgMoveType>(randomlySelectMove());
    nJumps         = latentPath.getNJumps(m);
    
    /// If there are no jumps, enforce a birth move (unless a change of the initial value was attempted).
    if (nJumps == 0 && selectedMove != MWG_MOVE_INITIAL_VALUE)
    {
      selectedMove = MWG_MOVE_BIRTH;
    }
  
    if (selectedMove == MWG_MOVE_INITIAL_VALUE)
    {
      latentPathProp.initialValues_[m] = rng_.randomGamma(stationaryShape(m), stationaryScale);
      logProposalDensityNum = dGamma(latentPath.initialValues_[m],     stationaryShape(m), stationaryScale);
      logProposalDensityDen = dGamma(latentPathProp.initialValues_[m], stationaryShape(m), stationaryScale);
    } 
    else if (selectedMove == MWG_MOVE_MODIFY_JUMP_TIME)
    {
      // Select jump of the $m$th process uniformly at random:
      idx = static_cast<unsigned int>(rng_.randomUniformInt(0, nJumps-1));
      
      double jumpTimeNext, jumpTimePrev;
      
      if (nJumps == 1)
      {
        jumpTimeNext = T;
        jumpTimePrev = 0;
      }
      else if (idx == nJumps-1)
      {
        jumpTimeNext = T;
        jumpTimePrev = latentPath.jumpTimes_[m](idx-1);
      }
      else if (idx == 0)
      {
        jumpTimeNext = latentPath.jumpTimes_[m](idx+1);
        jumpTimePrev = 0;
      }
      else
      {
        jumpTimeNext = latentPath.jumpTimes_[m](idx+1);
        jumpTimePrev = latentPath.jumpTimes_[m](idx-1);
      }
      latentPathProp.jumpTimes_[m](idx) = jumpTimePrev + (jumpTimeNext - jumpTimePrev) * arma::randu();
      logProposalDensityNum = 0.0; 
      logProposalDensityDen = 0.0;
    }
    else if (selectedMove == MWG_MOVE_MODIFY_JUMP_SIZE)
    {
      // Select jump of the $m$th process uniformly at random:
      idx = static_cast<unsigned int>(rng_.randomUniformInt(0, nJumps-1));
      latentPathProp.jumpSizes_[m](idx) = -std::log(arma::randu()) / zeta;
      logProposalDensityNum = - zeta * latentPath.jumpSizes_[m](idx);
      logProposalDensityDen = - zeta * latentPathProp.jumpSizes_[m](idx); 
    } 
    else if (selectedMove == MWG_MOVE_BIRTH)
    {
      arma::colvec jumpTimeBorn(1); 
      arma::colvec jumpSizeBorn(1);
      
      jumpTimeBorn(0) = = T * arma::randu();
      jumpSizeBorn(0) = -std::log(arma::randu()) / zeta;
      
      // TODO: check if these indices are handled correctly!
      arma::uvec idxVec = arma::find(latentPath.jumptimes_[m] < jumpTimeBorn, 1, "last");
      if (!idxVec.is_empty())
      { 
        latentPathProp.jumpTimes_[m].insert_rows(jumpTimeBorn, idxVec(0)+1);
        latentPathProp.jumpSizes_[m].insert_rows(jumpSizeBorn, idxVec(0)+1);
      }
      else
      {
        latentPathProp.jumpTimes_[m].insert_rows(jumpTimeBorn, 0);
        latentPathProp.jumpSizes_[m].insert_rows(jumpSizeBorn, 0);
      }
      
      logProposalDensityNum = - std::log(latentPathProp.getNJumps(m));
      logProposalDensityDen = std::log(zeta) - zeta * jumpSizeBorn - std::log(T); 
    }
    else if (selectedMove == MWG_MOVE_DEATH)
    {
      // Select jump of the $m$th process uniformly at random:
      idx = static_cast<unsigned int>(rng_.randomUniformInt(0, nJumps-1));
      
      latentPathProp.jumpTimes_[m].shed_row(idx);
      latentPathProp.jumpSizes_[m].shed_row(idx);
      
      logProposalDensityNum = std::log(zeta) - zeta * latentPath.jumpSizes_[m](idx) - std::log(T); 
      logProposalDensityDen = - std::log(latentPath.getNJumps(m));
    }
    

    logAlpha = logProposalDensityNum - logProposalDensityDen;
    
    if (std::isfinite(logAlpha))
    {
      logCompleteLikelihoodProp = 
        auxEvaluateLogCompoundPoissonPriorDensity
        (
          latentPathProp.initialValues_, 
          latentPathProp.jumpTimes_, 
          latentPathProp.jumpSizes_,
          model_.getModelParameters().getObservationTimesAux(),
          model_.getModelParameters().getKappa(),
          model_.getModelParameters().getLambda(),
          model_.getModelParameters().getZeta(),
          model_.getModelParameters().getStationaryShape(),
          model_.getModelParameters().getStationaryScale()
        )
        + model_.evaluateLogCompleteLikelihood(latentPathProp);
        
      
      logAlpha += logCompleteLikelihoodProp - logCompleteLikelihood;
    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }  
    
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    { 
      std::cout << "######################### acceptance #########################" << std::endl;
      latentPath = latentPathProp;
      logCompleteLikelihood = logCompleteLikelihoodProp;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the class <<Smc>>
///////////////////////////////////////////////////////////////////////////////

/// Holds all latent variables associated with a single 
/// conditionally IID observation/single time step.
typedef LatentVariable Particle;
/// Holds some additional auxiliary parameters for the SMC algorithm.
class SmcParameters
{
public: // NOTE: the following functions need to be executed before the start of the particle Gibbs sampler!
 
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& algorithmParameters)
  {
    // Empty;
  }
  
  /// Returns upper limit on the time-interval whose observations are
  /// introduced in the $t$th SMC step.
  double getStepTimes(const unsigned int t) const {return stepTimes_(t);}
  /// Returns the vector of observations which are
  /// introduced in the $t$th SMC step.
  const arma::colvec& getBinnedObservations(const unsigned int t) const {return binnedObservations_[t];}
  /// Returns the vector of the times of the observations which are
  /// introduced in the $t$th SMC step.
  const arma::colvec& getBinnedObservationTimes(const unsigned int t) const {return binnedObservationTimes_[t];}
  /// Returns upper limits on the time-intervals whose observations are
  /// introduced in the SMC steps.
  const arma::colvec& getStepTimes() const {return stepTimes_;}
  /// Returns the threshold for switching to the centred parametrisation 
  /// for the backward/ancestor sampling weights.
  double getParametrisationThreshold() const {return parametrisationThreshold_;}
  /// Determines the threshold for switching to the centred parametrisation 
  /// for the backward/ancestor sampling weights.
  void setParametrisationThreshold(const double parametrisationThreshold) {parametrisationThreshold_ = parametrisationThreshold;}
  /// Specifies the step times.
  void setStepTimes(const arma::colvec& stepTimes) {stepTimes_ = stepTimes;}
  /// Computes the time differences between successive SMC steps.
  void computeDiffStepTimes()
  {
    diffStepTimes_.set_size(stepTimes_.size()-1);
    if (diffStepTimes_.size() > 1) 
    {
      for (unsigned int i=0; i<diffStepTimes_.size(); i++)
      {
        diffStepTimes_(i) = stepTimes_(i+1) - stepTimes_(i);
      }
    }
  } 
  /// Computes the time differences between successive SMC steps.
  void computeDiffStepTimes(const arma::colvec& stepTimes)
  {
    stepTimes_ = stepTimes;
    computeDiffStepTimes();
  }
  /// Determines the observations used for each SMC step.
  void computeBinnedObservations(const arma::colvec& observations, const arma::colvec& observationTimes)
  {
    // TODO: change this function so that it uses the function from helperFunctions.h!
    
    
    
    // increase the upper bounds on the bin ranges by the following amount 
    // because arma::histc assumes ranges of the form $[a, b)$:
    double rangeIncrease = arma::min(arma::diff(observationTimes)) / 2.0; 
    
    arma::colvec binRanges(stepTimes.size()); // the ranges of the bins
    binRanges    = stepTimes_ + rangeIncrease;
    binRanges(0) = 0.0; // because we do not want to increase the lower bound on the first bin
    
    // TODO: check how the bin ranges are computed exactly, we want the bins to INCLUDE their upper bounds:
    std::uvec binIndices =  arma::histc(observationTimes, binRanges); 
    
    binnedObservations_.resize(stepTimes_.size()-1);
    for (unsigned int i=0; i<binnedObservations_.size(); i++)
    {
      /// TODO: need to deal with empty bins here!
      binnedObservations_[i] = observations.elem(arma::find(binIndices == i));
    }
  }
  /// Determines the observations used for each SMC step.
  void computeBinnedObservations(const arma::colvec& observations, const arma::colvec& observationTimes, const arma::colvec& stepTimes)
  {
    stepTimes_ = stepTimes;
    computeBinnedObservations(observations, observationTimes);
  }
  
  
private:
  
  arma::colvec diffStepTimes_; // time differences between successive SMC steps
  arma::colvec stepTimes_; // upper boundaries of the intervals corresponding to each SMC step NOTE: the first entry is a zero!
  std::vector<arma::colvec> binnedObservations_; // observations in each SMC update interval
  std::vector<arma::colvec> binnedObservationTimes_; // observations in each SMC update interval
  double parametrisationThreshold_; // threshold for switching to the centred parametrisation for the backward/ancestor sampling weights
  
};



/// Samples particles at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleParticles
(
  const unsigned int t, 
  std::vector<Particle>& particlesNew,
  const std::vector<Particle>& particlesOld
)
{
  unsigned int M = model_.getModelParameters().getNComponents();
  double t0 = smcParameters_.getStepTimes(t-1);
  double t1 = smcParameters_.getStepTimes(t);

  for (unsigned int n=0; n<getNParticles(); n++)
  {
    particlesNew[n].setInitialValues(particlesOld[n].finalValues_);
    particlesNew[n].sampleJumps(t0, t1, model_.getModelParameters().getLambda(), model_.getModelParameters().getKappa(), model_.getModelParameters().getZeta());
    particlesNew[n].computeFinalValues(t0, t1, model_.getModelParameters().getKappa());
  }
  if (isConditional_)
  {
    particlesNew[particleIndicesIn_(t)] = particlePath_[t];
    particlesNew[particleIndicesIn_(t)].setInitialValues(particlesOld[particleIndicesIn_(t)].finalValues_);
    particlesNew[n].computeFinalValues(t0, t1, model_.getModelParameters().getKappa());
  }
}
/// Computes a particle weight at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogParticleWeights
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew,
  const std::vector<Particle>& particlesOld,
  arma::colvec& logWeights
)
{

  double t0 = smcParameters_.getStepTimes(t-1);
  double t1 = smcParameters_.getStepTimes(t);

  for (unsigned int n=0; n<getNParticles(); n++)
  {
    logWeights(n) += particlesNew[n].evaluateLogLikelihood(t0, t1, smcParameters_.getBinnedObservations(t), smcParameters_.getBinnedObservationTimes(t), model_.getModelParameters().getKappa());
  }
}
/// Reparametrises particles at Step 0 to obtain the values of Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromParticles
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew, 
  const std::vector<Particle>& particlesOld, 
  std::vector<Aux>& aux1
)
{
  std::cout << "Warning: correlated pseudo-marginal kernels are currently not implemented!" << std::endl;
}
/// Reparametrises Gaussians at Step 0 to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineParticlesFromGaussians
(
  const unsigned int t,
  std::vector<Particle>& particlesNew, 
  const std::vector<Particle>& particlesOld,
  const std::vector<Aux>& aux1
)
{
  std::cout << "Warning: correlated pseudo-marginal kernels are currently not implemented!" << std::endl;
}
/// Samples particles at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleInitialParticles
(
  std::vector<Particle>& particlesNew
)
{
  double t1 = smcParameters_.getStepTimes(0);
  double t0 = 0;
  for (unsigned int n=0; n<getNParticles(); n++)
  {
    particlesNew[n].sampleInitialValues(model_.getModelParameters().getStationaryShape(), model_.getModelParameters().getStationaryScale());
    particlesNew[n].sampleJumps(t0, t1, model_.getModelParameters().getLambda(), model_.getModelParameters().getKappa(), model_.getModelParameters().getZeta());
    particlesNew[n].computeFinalValues(t0, t1, model_.getModelParameters().getKappa());
  }
  if (isConditional_) 
  {
    particlesNew[particleIndicesIn_(0)] = particlePath_[0];
    // NOTE: the initial values for the initial particles should already be set appropriately
    particlesNew[n].computeFinalValues(t0, t1, model_.getModelParameters().getKappa());
  }
}
/// Computes the incremental particle weights at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogInitialParticleWeights
(
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{ 
  for (unsigned int n=0; n<getNParticles(); n++)
  {
    logWeights(n) += particlesNew[n].evaluateLogLikelihood(t0, t1, ); // TODO
  }
}
/// Reparametrises the particles at Step t to obtain the value of 
/// Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromInitialParticles
(
  const std::vector<Particle>& particlesNew, 
  std::vector<Aux>& aux1
)
{ 
  std::cout << "Warning: correlated pseudo-marginal kernels are currently not implemented!" << std::endl;
}
/// Reparametrises Gaussians at Step 0 to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineInitialParticlesFromGaussians
(
  std::vector<Particle>& particlesNew,  
  const std::vector<Aux>& aux1
)
{
  std::cout << "Warning: correlated pseudo-marginal kernels are currently not implemented!" << std::endl;
}
/// Computes (part of the) unnormalised "future" target density needed for 
/// backward or ancestor sampling.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::logDensityUnnormalisedTarget
(
  const unsigned int t,
  const Particle& particle
)
{
 // TODO
}

/// Converts a particle path into the set of all latent variables in the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath)
{
  
  std::cout << "Start: convertParticlePathToLatentPath()" std::endl;
  
  unsigned int M = model_.getModelParameters().getNComponents();
  unsigned int nJumpsAux, nJumps;
  
  for (unsigned int m=0; m<M; m++)
  {
    nJumpsAux = 0; // total number of jumps up to the current time
    for (unsigned int t=0; t<getNSteps(); t++)
    {
      nJumpsAux += particlePath[t].getNJumps(m);
    }
    latentPath.jumpTimes_[m].set_size(nJumpsAux);
    latentPath.jumpSizes_[m].set_size(nJumpsAux);
    nJumpsAux = 0;
    latentPath.initialValues_[m] = particlePath[0].initialValues_[m];
    
    for (unsigned int t=0; t<getNSteps(); t++)
    { 
      nJumps = particlePath[t].getNJumps(m); // number of jumps in the current interval
      if (nJumps > 0)
      {
        latentPath.jumpTimes_[m](arma::span(nJumpsAux, nJumpsAux+nJumps-1)) += particlePath[t].jumpTimes_[m];
        latentPath.jumpSizes_[m](arma::span(nJumpsAux, nJumpsAux+nJumps-1)) += particlePath[t].jumpSizes_[m];
        nJumpsAux += nJumps;
      }
    }
  }
  
  std::cout << "End: convertParticlePathToLatentPath()" std::endl;
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  std::cout << "Start: convertLatentPathToParticlePath()" std::endl;
  
  particlePath.resize(getNSteps()); 
  unsigned int M = model_.getModelParameters().getNComponents();
  std::vector<std::vector<unsigned int>> idx;
 
  for (unsigned int m=0; m<M; m++)
  {
    particlePath[0].initialValues_[m] = latentPath.initialValues_[m];
    if (latentPath.getNJumps(m) > 0)
    {
      idx = computeBinContents(latentPath.jumpTimes_[m], smcParameters_.getStepTimes(), true)
      
      for (unsigned int t=0; t<getNSteps(); t++)
      {
        if (idx[t].empty())
        {
          particlePath[t].jumpTimes_[m].clear();
          particlePath[t].jumpSizes_[m].clear();
        }
        else
        {
          particlePath[t].jumpTimes_[m] = latentPath.jumpTimes_[m](arma::span(idx[t](0), idx[t](idx[t].size()-1)));
          particlePath[t].jumpSizes_[m] = latentPath.jumpSizes_[m](arma::span(idx[t](0), idx[t](idx[t].size()-1)));
        }
      }
    }
  }
  std::cout << "End: convertLatentPathToParticlePath()" std::endl;
  
  // TODO: we still need to make sure that the members "initialValues_" and "finalValues_" are computed within the SMC algorithm
}



///////////////////////////////////////////////////////////////////////////////
/// Member functions of the class <<Ehmm>>
///////////////////////////////////////////////////////////////////////////////





///////////////////////////////////////////////////////////////////////////////
/// Member functions of the class <<GibbsSampler>>
///////////////////////////////////////////////////////////////////////////////

/// Proposes new set of reparametrised latent variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters, class EhmmParameters> 
double GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, MwgParameters, SmcParameters, EhmmParameters>::evaluateLogCompleteLikelihood(const arma::colvec& theta, const LatentPath& latentPath, const bool includeLatentPriorDensity)
{
  double logDensity = 0.0;
  model_.setUnknownParameters(theta);
  if (computeLatentPriorDensity)
  {
    /// Computes the logarithm of the conditional density of the PPPs,
    /// conditional on the model parameters
    logDensity = 
    auxEvaluateLogCompoundPoissonPriorDensity(
      latentPath.initialValues_, 
      latentPath.jumpTimes_, 
      latentPath.jumpSizes_,
      model_.getModelParameters().getObservationTimesAux(),
      model_.getModelParameters().getKappa(),
      model_.getModelParameters().getLambda(),
      model_.getModelParameters().getZeta(),
      model_.getModelParameters().getStationaryShape(),
      model_.getModelParameters().getStationaryScale()
    ) 
  }
  logDensity += model_.evaluateLogCompleteLikelihood(latentPath);
  return logDensity;
}
/// Samples parameters which had been analytically integrated out during the 
/// parameter-update steps from their full conditional posterior distribution.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters, class EhmmParameters> 
void GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, MwgParameters, SmcParameters, EhmmParameters>::sampleMarginalisedParameters(arma::colvec& theta, const LatentPath& latentPath)
{
  model_.setUnknownParameters(theta);
  theta(arma::span(model_.getDimTheta()-model_.getModelParameters().getNRiskPremiumParameters()-model_.getModelParameters().getNLinearLeverageParameters()-1, model_.getDimTheta())) = 
  auxSampleMarginalisedParameters(
    latentPath.initialValues_, 
    latentPath.jumpTimes_, 
    latentPath.jumpSizes_,
    model_.getModelParameters().getObservationTimesAux(),
    model_.getObservations(),
    model_.getModelParameters().getKappa(),
    model_.getModelParameters().getLambda(),
    model_.getModelParameters().getZeta(),
    model_.getModelParameters().getInitialMuTilde(),
    model_.getModelParameters().getInitialVarSigmaTildeInv()
  );
}
/// Initialises the latent-variable path object.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters, class EhmmParameters> 
void GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, MwgParameters, SmcParameters, EhmmParameters>::initialiseLatentPath(const arma::colvec& theta, LatentPath& latentPath)
{
  unsigned int nComponents = model_.getModelParameters().getNComponents();
  latentPath.setup(nComponents);
}
/// Initialises the vector in which the output is to be stored.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters, class EhmmParameters> 
void GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, MwgParameters, SmcParameters, EhmmParameters>::initialiseOutput(std::vector<arma::colvec>& output)
{
  unsigned int nComponents = model_.getModelParameters().getNComponents();
  unsigned int dimTheta    = model_.getModelParameters().getDimTheta()
  output.resize(nIterations_);
}
/// Stores the output of the Gibbs sampler
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters, class EhmmParameters> 
void GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, MwgParameters, SmcParameters, EhmmParameters>::storeOutput(const unsigned int g, std::vector<arma::colvec>& output, const arma::colvec& theta, const LatentPath& latentPath)
{
  unsigned int nComponents = model_.getModelParameters().getNComponents();
  unsigned int dimTheta    = model_.getModelParameters().getDimTheta()
  output[g].set_size(dimTheta + nComponents);    
  output[g](arma::span(0, dimTheta-1)) = theta;  
  for (unsigned int k=0; k<nComponents; k++)
  {
    output[g](dimTheta + k) = latentPath.getNJumps(k);
  }
}
/// Proposes new set of reparametrised latent variables
/// via the NCP.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class MwgParameters, class SmcParameters, class EhmmParameters> 
void GibbsSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, MwgParameters, SmcParameters, EhmmParameters>::proposeLatentPath(LatentPathProp& latentPathProp, const LatentPath& latentPath, const arma::colvec& thetaProp, const arma::colvec& theta)
{
  model_.setUnknownParameters(theta);
  arma::colvec lambda     = model_.getModelParameters().getLambda();
  double zeta             = model_.getModelParameters().getZeta();
  
  model_.setUnknownParameters(thetaProp);
  arma::colvec lambdaProp = model_.getModelParameters().getLambda();
  double zetaProp         = model_.getModelParameters().getZeta();
  
  unsigned int M = model_.getModelParameters().getNComponents();
  double T       = model_.getModelParameters().getLastObservationTime();
  
  unsigned int nAdditionalJumps; // number of additional (two-dimensional) points sampled if lambdaProp(m) > lambda(m)
  arma::colvec jumpTimesAux, jumpSizesAux;
  arma::uvec idx; // indices for sorting/
  double lambdaDiff; // lambdaProp(m) - lambda(m)
  
  for (unsigned int m=0; m<M; m++)
  {
    lambdaDiff = lambdaProp(m) - lambda(m);
    
    if (lambdaDiff > 0) // i.e. then we need to add new points
    {
      nJumpsAux = R::pois(T * lambdaDiff); // number of additional points
      
      // WARNING: this may be slow! 
      jumpTimesAux = arma::join_rows(latentPath.jumpTimes_[m], arma::randu<arma::colvec>(nAdditionalJumps)); // existing and newly generated jump times
      jumpSizesAux = - arma::log(arma::join_rows(
                       lambda(m) * arma::exp(- zeta * latentPath.jumpSizes_[m]), // existing jump sizes (transformed to a PPP on [0, lambda(m))
                       lambda(m) + lambdaDiff * arma::randu<arma::colvec>(nAdditionalJumps) // newly generated jump sizes (transformed to a PPP on (lambda(m), lambdaProp(m))  
                     ) / lambdaProp(m) ) / zetaProp;
      
      idx = arma::sort_index(jumpTimesAux);
      latentPathProp.jumpTimes_[m] = jumpTimesAux.elem(idx);
      latentPathProp.jumpSizes_[m] = jumpSizesAux.elem(idx);
    }
    else // i.e. if lambda(m) >= lambdaProp(m); we need to remove points
    {
      // Indices of the jumps that are large enough to be "kept"
      idx = arma::find(latentPath.jumpSizes[m] >= (std::log(lambda(m)) - std::log(lambdaProp(m))) / zeta);
      if (!idx.is_empty())
      {
        latentPathProp.jumpTimes_[m] = jumpTimesAux.elem(idx);
        latentPathProp.jumpSizes_[m] = (zeta * jumpSizesAux.elem(idx) + std::log(lambdaProp(m)) - std::log(lambda(m))) / zetaProp;
      }
    }
    
    latentPathProp.initialValues_[m] = latentPath.initialValues_[m];
//     latentPathProp.initialValues_[m] = latentPath.initialValues_[m] * zeta / zetaProp; // TODO: should we use an NCP for the initial values, too?
  }
}

#endif
