/// \file
/// \brief Generating data and performing inference in the multivariate copula stochastic volatility model. 
///
/// This file contains the functions for implementing the abstract model
/// class for the multivariate factor stochastic volatility model 
/// with copula dependence. 

#ifndef __FACTORCOPULASV_H
#define __FACTORCOPULASV_H

#include "main/templates/dynamic/stateSpace/stateSpace.h"
#include "main/algorithms/mcmc/Mcmc.h"
#include "main/rng/gaussian.h"

// [[Rcpp::depends("RcppArmadillo")]]

/// Type of model to use.
enum ModelType 
{ 
  MODEL_EXACT_VOLATILITY_MEASUREMENTS = 0, // probabilities $p_{g,t}$ and $\gamma_t$ are time-varying.
  MODEL_NOISY_VOLATILITY_MEASUREMENTS,     // same as above but with $\delta_1 = 0$
};

/// Type of copula used to model the dependence between
/// the noise variables and latent factors.
enum CopulaType 
{ 
  COPULA_GAUSSIAN = 0, // use a Gaussian copula
  COPULA_90,           // rotated by 90 degrees
  COPULA_270           // rotated by 270 degrees
};

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the state-space model
////////////////////////////////////////////////////////////////////////////////

/// Holds all the static model parameters.
class ModelParameters
{
public:
  
  /// Determines the model parameters from arma::colvec theta. 
  /// Note that $\theta = (\vartheta, h_C(\lambda^{HZ})$ for modelType = 0 and  $\theta = (\vartheta, h_C(\lambda^{HZ}, \log \omega)$ for modelType = 1, where
  /// $\vartheta = (\alpha_{0:K}, \beta_{0:K}, \log \kappa_{0:K}, \log(2 \kappa \mu - \sigma^2 )_{1:K}, \log \sigma_{1:K}, h_C(\lambda^H)_{1:K}, h_C(\lambda^Z)_{1:K})$
  void setUnknownParameters(const arma::colvec& theta)
  {
//     std::cout << "start: setUnknownParameters()" << std::endl;
    // TODO
    theta_  = theta;
    unsigned int K = nExchangeRates_;
//     alpha_.set_size(K);
//     beta_.set_size(K);
//     kappa_.set_size(K);
//     mu_.set_size(K);
//     sigma_.set_size(K);
//     lambdaH_.set_size(K);
//     lambdaZ_.set_size(K);
    
    alpha_ = theta(arma::span(0*K,  K-1));
    beta_  = theta(arma::span(1*K,2*K-1)); 
//     kappa_ = arma::exp(theta(arma::span(2*K,3*K-1)));
//     sigma_ = arma::exp(theta(arma::span(4*K,5*K-1)));
//     mu_ = (arma::exp(theta(arma::span(3*K,4*K-1))) + arma::pow(sigma_, 2.0)) / (2.0 * kappa_);
    
    mu_.set_size(K);
    kappa_.set_size(K);
    sigma_.set_size(K);
    lambdaH_.set_size(K);
    lambdaZ_.set_size(K);
    for (unsigned int k=0; k<K; k++)
    {
      kappa_(k)    = std::exp(theta(2*K+k));
      sigma_(k)    = std::exp(theta(4*K+k));
      mu_(k)       = (std::exp(theta(3*K+k)) + sigma_(k)*sigma_(k)) / (2.0 * kappa_(k));
      lambdaH_(k)  = transformThetaToLambda(theta(5*K+k), copulaTypeH_); // NOTE: not sure if the vectorised version of this function works correctly!
      lambdaZ_(k)  = transformThetaToLambda(theta(6*K+k), copulaTypeZ_);// NOTE: not sure if the vectorised version of this function works correctly!
    }
    lambdaHZ_ = transformThetaToLambda(theta(7*K), copulaTypeHZ_); 
    
//     //////////////
//     arma::colvec backtransformed(K);
//     for (unsigned int k=0; k<K; k++)
//     {
//       backtransformed(k) = std::log(2.0 * kappa_(k) * mu_(k) - sigma_(k) * sigma_(k));
//     }
//     /////////////
    
//                 std::cout << "start: transformThetaToLambda()" << std::endl;
    // Transforming elements of theta to the corresponding scalar copula parameters:
//     lambdaH_  = transformThetaToLambda(theta(arma::span(5*K,6*K-1)), copulaTypeH_);
//     lambdaZ_  = transformThetaToLambda(theta(arma::span(6*K,7*K-1)), copulaTypeZ_);
//     lambdaHZ_ = transformThetaToLambda(theta(7*K), copulaTypeHZ_); 
//             std::cout << "end: transformThetaToLambda()" << std::endl;
    
//     std::cout << "alpha: " << alpha_.t() << std::endl;
//     std::cout << "beta: " << beta_.t() << std::endl;
//     std::cout << "log kappa: " << arma::trans(arma::log(kappa_)) << std::endl;
    //     std::cout << "log sigma: " << arma::trans(arma::log(sigma_)) << std::endl;
    
    
    // NOTE: the following parameters often contain NaNs! 
//     std::cout << "backtransformed: " << backtransformed.t()  << std::endl;
//     std::cout << "theta original : " << arma::trans(theta(arma::span(3*K,4*K-1))) << std::endl;
//     std::cout << "log lambdaH: " << arma::trans(arma::log(lambdaH_)) << " " ;
//     std::cout << "log lambdaZ: " << arma::trans(arma::log(lambdaZ_)) << std::endl;
        
    
    if (modelType_ == MODEL_NOISY_VOLATILITY_MEASUREMENTS)
    {
      omega_ = std::exp(theta(7*K+1));
    }
    
//         std::cout << "end: setUnknownParameters()" << std::endl;
      
  }
  
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyp)
  {
    
//         std::cout << "start: setKnownParameters()" << std::endl;
    modelType_      = static_cast<ModelType>(hyp(0));
    copulaTypeH_    = static_cast<CopulaType>(hyp(1));
    copulaTypeZ_    = static_cast<CopulaType>(hyp(2));
    copulaTypeHZ_   = static_cast<CopulaType>(hyp(3)); 
    nExchangeRates_ = static_cast<unsigned int>(hyp(4));
    dimTheta_       = static_cast<unsigned int>(hyp(5));
    delta_          = hyp(6);
    rootDelta_      = std::sqrt(delta_);
    meanInitialLogVolatility_ = hyp(arma::span(7,6+nExchangeRates_));
    sdInitialLogVolatility_   = hyp(arma::span(7+nExchangeRates_,6+2*nExchangeRates_));
    meanHyper_      = hyp(arma::span(7+2*nExchangeRates_,6+2*nExchangeRates_+dimTheta_));
    sdHyper_        = hyp(arma::span(7+2*nExchangeRates_+dimTheta_, 6+2*nExchangeRates_+2*dimTheta_));
    
//     std::cout << "copulaTypeH: " << static_cast<unsigned int>(copulaTypeH_) << std::endl;
//     std::cout << "copulaTypeZ: " << static_cast<unsigned int>(copulaTypeZ_) << std::endl;
//     std::cout << "copulaTypeHZ: " << static_cast<unsigned int>(copulaTypeHZ_) << std::endl;
    
    
//     std::cout << "meanHyper: " << meanHyper_.t() << std::endl;
//     std::cout << "sdHyper: " << sdHyper_.t() << std::endl;
    
    if (modelType_ == MODEL_EXACT_VOLATILITY_MEASUREMENTS)
    {
      dimLatentVariable_ = 2;
    }
    else 
    {
      dimLatentVariable_ = 2 + nExchangeRates_;
    }
//             std::cout << "end: setKnownParameters()" << std::endl;
  }
  
  /// Samples a value of the conditional distribution of one of the 
  /// components of a bivariate distribution with Gaussian marginals
  /// and dependence structure modelled via some copula.
  double sampleFromConditionalCopula(
    const double x, // value of the other component
    const double lambda, // scalar copula parameter
    const CopulaType& copulaType // type of copula
  ) const
  {

    
    // Evaluation of the conditional copula CDF
    if (copulaType == COPULA_GAUSSIAN)
    {
//       std::cout << "sampling from Gaussian copula!" <<std::endl;
//       std::cout << "lambda: " << lambda << "; x: " << x << "; lambda * x: " << lambda * x << "; std::sqrt(1.0-lambda*lambda): " << std::sqrt(1.0-lambda*lambda) << std::endl;
      double u = arma::randn();
      double y = (u - lambda * x)/std::sqrt(1.0-lambda*lambda);
      return y;
//       return arma::randn();
    }
    else if (copulaType == COPULA_90)
    {
//        std::cout << "sampling from 90 copula!" <<std::endl;
      double u = R::pnorm(x, 0.0, 1.0, true, false); // value of the other component transformed to $[0,1]$
      double v = arma::randu();
      return R::qnorm(std::pow((std::pow(v, -lambda)-1)/(std::pow(1-u, -lambda))+1.0, -(1.0+lambda)/lambda), 0.0, 1.0, true, false);
    }
    else if (copulaType == COPULA_270)
    {
//        std::cout << "sampling from 270 copula!" <<std::endl;
      double u = R::pnorm(x, 0.0, 1.0, true, false); // value of the other component transformed to $[0,1]$
      double v = arma::randu();
      return R::qnorm(1.0 - std::pow(std::pow(u, lambda) / std::pow(1-v, lambda)+1.0, -(1.0+lambda)/lambda), 0.0, 1.0, true, false);
    }
    else
    {
      return 0.0;
    }
  }
  /// Evaluates the logarithm of the derivative of the conditional
  /// copula CDF.
  double evaluateLogDerivativeCopulaCdf(
    const double qnormV,
    const double qnormU, // value of the other component
    const double lambda, // scalar copula parameter
    const CopulaType& copulaType // type of copula
  ) const
  {
    // Evaluation of the condition copula CDF
    if (copulaType == COPULA_GAUSSIAN)
    {
      double aux = 1 - lambda * lambda;
      return R::dnorm((qnormV - lambda*qnormU)/std::sqrt(aux), 0.0, 1.0, true) 
        - R::dnorm(qnormV, 0.0, 1.0, true) 
        - std::log(aux)/2.0;
    }
    else if (copulaType == COPULA_90)
    {
      // TODO: this is inefficient!
      double u = R::pnorm(qnormU, 0.0, 1.0, true, false);
      double v = R::pnorm(qnormV, 0.0, 1.0, true, false);
      return -(1.0+2.0*lambda)/lambda * std::log((std::pow(v, -lambda)-1)/(std::pow(1-u, -lambda))+1) - (1+lambda) *std::log(v) + std::log(1+lambda) + lambda * std::log(1-u);
    }
    else if (copulaType == COPULA_270)
    {
      // TODO: this is inefficient!
      double u = R::pnorm(qnormU, 0.0, 1.0, true, false);
      double v = R::pnorm(qnormV, 0.0, 1.0, true, false);
      return -(1+lambda)/lambda * std::log(std::pow(u, lambda) / std::pow(1-v, lambda)+1) + std::log(1+lambda) + lambda * std::log(u) - (1+lambda) * std::log(1-v);
    }
    else 
    {
      return 0.0;
    }
  }
  /// Evaluates the logarithm of the derivative of the 
  /// inverse conditional copula CDF.
  double evaluateLogDerivativeCopulaInverseCdf(
    const double w,
    const double u, // value of the other component (transformed to [0,1])
    const double lambda, // scalar copula parameter
    const CopulaType& copulaType // type of copula
  ) const
  {
    // Evaluation of the condition copula CDF
    
    double out = 0.0;
    if (copulaType == COPULA_GAUSSIAN)
    {
      
      double aux = std::sqrt(1.0 - lambda*lambda);
      double qnormW = R::qnorm(w, 0.0, 1.0, true, false);
      double qnormU = R::qnorm(u, 0.0, 1.0, true, false);
      
      out =
        R::dnorm(aux*qnormW - lambda*qnormU, 0.0, 1.0, true) 
        - R::dnorm(qnormW, 0.0, 1.0, true) 
        + std::log(aux);
    }
    else if (copulaType == COPULA_90)
    {
      // TODO: this is inefficient! 
      out =
      - (2.0*lambda+1.0)/(lambda+1) * std::log(w) 
      - std::log(lambda+1) 
      - lambda * std::log(1.0-u) 
      - (lambda+1)/lambda * std::log((std::pow(w, -lambda/(lambda+1))-1.0)/std::pow(1.0-u, lambda)+1.0);
    }
    else if (copulaType == COPULA_270)
    {
      // TODO: this is inefficient!
      out =
      - lambda * std::log(u)
      - std::log(lambda+1) 
      - (2*lambda+1)/(lambda+1) * std::log(1.0 - w)
      - (lambda+1)/lambda * std::log((std::pow(1.0-w,-lambda/(lambda+1)) - 1.0)* std::pow(w, -lambda));
    }
//     else 
//     {
//       return 0.0;
//     }

    ////////////////////////
//     if (!std::isfinite(out)) 
//     {
//       if (copulaType == COPULA_GAUSSIAN)
//       {
//         double aux = std::sqrt(1.0 - lambda*lambda);
//         double qnormW = R::qnorm(w, 0.0, 1.0, true, false);
//         double qnormU = R::qnorm(u, 0.0, 1.0, true, false);
//         
// //         std::cout << " u: " << u << "; w: " << w << " ";
//       }
//       else if (copulaType == COPULA_90)
//       {
//         // TODO: this is inefficient! 
//         out =
//         - (2.0*lambda+1.0)/(lambda+1) * std::log(w) 
//         - std::log(lambda+1) 
//         - lambda * std::log(1.0-u) 
//         - (lambda+1)/lambda * std::log((std::pow(w, -lambda/(lambda+1))-1.0)/std::pow(1.0-u, lambda)+1.0);
//       }
//       else if (copulaType == COPULA_270)
//       {
//         // TODO: this is inefficient!
//         out
//         - lambda * std::log(u)
//         - std::log(lambda+1) 
//         - (2*lambda+1)/(lambda+1) * std::log(1.0 - w)
//         - (lambda+1)/lambda * std::log((std::pow(1.0-w,-lambda/(lambda+1)) - 1.0)* std::pow(w, -lambda));
//       }
//     }
    ////////////////////////

    return out;
  }
  /// Generates a log-exchange rate at time $t$ given the log-exchange rate at time $t-1$,
  /// the log-volatility at time $t$, 
  /// the latent factors and the model parameters.
  double sampleLogExchangeRate(const unsigned int k, const double sOld, const double xNew, const double h) const
  {
    double eta = sampleFromConditionalCopula(h, getLambdaH(k), getCopulaTypeH());
    double sNew = sOld 
                  + (getAlpha(k) + getBeta(k) * std::exp(xNew)) * getDelta() 
                  + std::exp(xNew/2.0) * getRootDelta() * eta;
    return sNew;
  }
  /// Generates a log-volatility at time $t$ given the log-volatility
  /// at time $t-1$, the latent factors and the model parameters.
  double sampleLogVolatility(
    const unsigned int k, 
    const double xOld, 
    const double z
  ) const
  {
    double zeta = sampleFromConditionalCopula(z, getLambdaZ(k), getCopulaTypeZ());
    
//         std::cout << "lambdaZ: " << getLambdaZ(k) << std::endl;
    
//          std::cout << "copulaTypeZ: " << getCopulaTypeZ() << std::endl;
    
    //////////////////////
//     std::cout << "Warning: sampleLogVolatility() changed!" << std::endl;
//     zeta = arma::randn();
    ///////////////////////
    double xNew = xOld 
                  + getDelta() * (getKappa(k) * (getMu(k) - std::exp(xOld)) - getSigma(k) * getSigma(k)/2.0) * std::exp(-xOld) 
                  + getRootDelta() * getSigma(k) * std::exp(-xOld/2.0) * zeta;
                  
    if (k == 0)
    {
      std::cout << xOld << "/" << getDelta() * (getKappa(k) * (getMu(k) - std::exp(xOld)) - getSigma(k) * getSigma(k)/2.0) * std::exp(-xOld) << "/" << getRootDelta() * getSigma(k) * std::exp(-xOld/2.0) << std::endl;
      std::cout << "--- xOld/xNew/z/zeta: " << xOld << "/" << xNew << "/" << z << "/" << zeta << std::endl;
    }
    return xNew;
  }
  /// Generates the factors at some time step.
  arma::colvec sampleFactors() const
  {
    arma::colvec x(2);
    x(0) = arma::randn(); // factor H
    x(1) = sampleFromConditionalCopula(x(0), getLambdaHZ(), getCopulaTypeHZ());
    
//     std::cout << "lambdaHZ: " << getLambdaHZ() << std::endl;
    
//      std::cout << "copulaTypeHZ: " << getCopulaTypeHZ() << std::endl;

    return x;
  }
  /// Computes the implied value of the noise variable eta.
  double computeEta(
    const unsigned int k,
    const double sNew, 
    const double sOld, 
    const double xNew
  ) const
  {
    return (sNew - sOld - (getAlpha(k) + getBeta(k) * std::exp(xNew)) * getDelta()) / (std::exp(xNew/2.0) * getRootDelta());
  }
  /// Computes the implied value of the noise variable zeta.
  double computeZeta(
    const unsigned int k,
    const double xNew, const double xOld
  ) const
  {
    return (xNew - xOld - ((getKappa(k) * (getMu(k) - std::exp(xOld))) - getSigma(k) * getSigma(k) / 2.0) * getDelta() / std::exp(xOld)) / (getSigma(k) * std::exp(-xOld/2.0) * getRootDelta());
  }
  /// Evaluates the log-observation density in the case of exact volatility measurements.
  /// Note that everything in here except for evaluateLogDerivativeCopulaCdf() is identical
  /// for each particle. Thus, computational savings are possible by avoiding 
  /// these duplicate calculations. 
  double computeLogObservationDensityExact(
    const unsigned int k, 
    const double h, 
    const double z,
    const double sNew, const double sOld, 
    const double xNew, const double xOld
  ) const
  { 
    double eta  = computeEta(k,  sNew, sOld, xNew);
    double zeta = computeZeta(k, xNew, xOld);
    double pnormEta  = R::pnorm(eta,  0.0, 1.0, true, false);
    double pnormH    = R::pnorm(h,    0.0, 1.0, true, false);
    double pnormZeta = R::pnorm(zeta, 0.0, 1.0, true, false);
    double pnormZ    = R::pnorm(z,    0.0, 1.0, true, false);
    
//     return -std::log(getSigma(k)) + std::log(getDelta()) + xNew / 2.0 + xOld / 2.0 + R::dnorm(eta, 0.0, 1.0, true) + R::dnorm(zeta, 0.0, 1.0, true) 
    
//         std::cout << "sigma delta x: " <<  - std::log(getSigma(k)) 
//     - std::log(getDelta()) 
//     - xNew/2.0 + xOld/2.0 << "; dnorm: " << R::dnorm(eta,  0.0, 1.0, true) 
//     + R::dnorm(zeta, 0.0, 1.0, true) << " ";
    
    
    
//     std::cout << "C Eta: " << evaluateLogDerivativeCopulaInverseCdf(pnormEta,  pnormH, getLambdaH(k), getCopulaTypeH()) << "; C Zeta: " << evaluateLogDerivativeCopulaInverseCdf(pnormZeta, pnormZ, getLambdaZ(k), getCopulaTypeZ()) << " ";
    
    double out = 
    - std::log(getSigma(k)) 
    - std::log(getDelta()) 
    - xNew/2.0 + xOld/2.0 
    + R::dnorm(eta,  0.0, 1.0, true) 
    + R::dnorm(zeta, 0.0, 1.0, true);
    
    if (std::isfinite(out))
    {
      double out2 = out + evaluateLogDerivativeCopulaInverseCdf(pnormEta,  pnormH, getLambdaH(k), getCopulaTypeH()) + evaluateLogDerivativeCopulaInverseCdf(pnormZeta, pnormZ, getLambdaZ(k), getCopulaTypeZ());
      
      if (std::isfinite(out2))
      {
        return out2;
      }
      else
      {
//         std::cout << "WARNING: NaNs caused by evaluateLogDerivativeCopulaInverseCdf()!" << std::endl;
        return - std::numeric_limits<double>::infinity();
      }
    }
    else 
    {
      return - std::numeric_limits<double>::infinity();
    }
//     + evaluateLogDerivativeCopulaCdf(eta, h, getLambdaH(k), getCopulaTypeH())
//     + evaluateLogDerivativeCopulaCdf(zeta, z, getLambdaZ(k), getCopulaTypeZ());
  }
  /// Evaluates the log-observation density in the case of exact volatility measurements.
  /// Calculations common to all particles. WARNING: this derivation of the weight may not be correct
  double computeLogObservationDensityExactCommon(
    const unsigned int k,
    const double eta, const double zeta,
    const double xNew, const double xOld
  ) const
  {
    return -std::log(getSigma(k)) - std::log(getDelta()) - xNew/2.0 + xOld / 2.0 + R::dnorm(eta, 0.0, 1.0, true) + R::dnorm(zeta, 0.0, 1.0, true);
  }
  /// Evaluates the log-observation density in the case of exact volatility measurements.
  /// Calculations which need to be performed for each particle individually. WARNING: this derivation of the weight may not be correct
  double computeLogObservationDensityExactIndividual(
    const unsigned int k,
    const double pnormEta, const double pnormZeta,
    const double h, const double z
  ) const
  {
    double pnormH = R::pnorm(h, 0.0, 1.0, true, false);
    double pnormZ = R::pnorm(z, 0.0, 1.0, true, false);
    return 
      evaluateLogDerivativeCopulaInverseCdf(pnormEta, pnormH, getLambdaH(k), getCopulaTypeH())
      + evaluateLogDerivativeCopulaInverseCdf(pnormZeta, pnormZ, getLambdaZ(k), getCopulaTypeZ());
//      evaluateLogDerivativeCopulaCdf(eta, h, getLambdaH(k), getCopulaTypeH())
//      + evaluateLogDerivativeCopulaCdf(zeta, z, getLambdaZ(k), getCopulaTypeZ());
  } 
  
  /// Evaluates the log-observation density in the case of noisy volatility measurements. WARNING: this derivation of the weight may not be correct
  double computeLogObservationDensityNoisy(
    const unsigned int k, 
    const double h, const double z, const double xNew,
    const double sNew, const double sOld, 
    const double yxNew
  ) const
  {
    double eta  = computeEta(k, sNew, sOld, xNew);
    double pnormEta = R::pnorm(eta, 0.0, 1.0, true, false);
    double pnormH   = R::pnorm(h,   0.0, 1.0, true, false);
    
    //////////////////////
//     std::cout << "; eta/xNew: " << eta << "/" << xNew << "";
//     std::cout << "P1: " << R::pnorm(yxNew, xNew, getOmega(), true, true) << "; P2: " << R::dnorm(eta, 0.0, 1.0, true) << "; P3: " << evaluateLogDerivativeCopulaCdf(eta, h, getLambdaH(k), getCopulaTypeH())<< " ";
    /////////////////////
    
//     return R::pnorm(yxNew, xNew, getOmega(), true, true) - R::dnorm(eta, 0.0, 1.0, true) - evaluateLogDerivativeCopulaCdf(eta, h, getLambdaH(k), getCopulaTypeH());
    
    // TODO: re-do the calculation of the observation density!
    return 0.0
    - std::log(getRootDelta()) 
    - xNew/2.0 
    + R::dnorm(yxNew, xNew, getOmega(), true) 
    + R::dnorm(eta, 0.0, 1.0, true) 
    + evaluateLogDerivativeCopulaInverseCdf(pnormEta, pnormH, getLambdaH(k), getCopulaTypeH());
  }
  /// Samples the initial (at time $-1$) value of the $k$th log-exchange rate.
  double sampleInitialLogExchangeRate(const unsigned int k) const
  { 
    return 0.0;
  }
  /// Samples the initial (at time $-1$) value of the $k$th log-volatility.
  double sampleInitialLogVolatility(const unsigned int k) const
  {
//     std::cout << "meanInitialLogVolatility: " << getMeanInitialLogVolatility(k) << std::endl;
    
    return getMeanInitialLogVolatility(k) + getSdInitialLogVolatility(k) * arma::randn();
  }
  
  /// Returns the parameter vector $\alpha_{1:K}$.
  arma::colvec getAlpha() const {return alpha_;}
  /// Returns the element $\alpha_k$.
  double getAlpha(const unsigned int k) const {return alpha_(k);}
  /// Returns the parameter vector $\beta_{1:K}$.
  arma::colvec getBeta() const {return alpha_;}
  /// Returns the element $\beta_k$.
  double getBeta(const unsigned int k) const {return beta_(k);}
  /// Returns the parameter vector $\kappa_{1:K}$.
  arma::colvec getKappa() const {return kappa_;}
  /// Returns the element $\kappa_k$.
  double getKappa(const unsigned int k) const {return kappa_(k);}
  /// Returns the parameter vector $\mu_{1:K}$.
  arma::colvec getMu() const {return mu_;}
  /// Returns the element $\mu_k$.
  double getMu(const unsigned int k) const {return mu_(k);}
  /// Returns the parameter vector $\sigma_{1:K}$.
  arma::colvec getSigma() const {return sigma_;}
  /// Returns the element $\beta_k$.
  double getSigma(const unsigned int k) const {return sigma_(k);}
  /// Returns the parameter vector $\lambda^H_{1:K}$.
  arma::colvec getLambdaH() const {return lambdaH_;}
  /// Returns the element $\lambda^H_k$.
  double getLambdaH(const unsigned int k) const {return lambdaH_(k);}
  /// Returns the parameter vector $\lambda^Z_{1:K}$.
  arma::colvec getLambdaZ() const {return lambdaZ_;}
  /// Returns the element $\lambda^Z_k$.
  double getLambdaZ(const unsigned int k) const {return lambdaZ_(k);}
  /// Returns $\lambda^{HZ}$.
  double getLambdaHZ() const {return lambdaHZ_;}
  /// Returns $\omega$.
  double getOmega() const {return omega_;}

  /// Returns the number of count-data observations $T$.
  unsigned int getT() const {return T_;}
   /// Returns the model index.
  ModelType getModelType() const {return modelType_;}
  /// Returns the type of copula used for modelling dependence between factors and noise variables.
  CopulaType getCopulaTypeH() const {return copulaTypeH_;}
  CopulaType getCopulaTypeZ() const {return copulaTypeZ_;}
  /// Returns the type of copula used for modelling dependence between the factors.
  CopulaType getCopulaTypeHZ() const {return copulaTypeHZ_;}
  /// Returns the number modelled exchange rates.
  unsigned int getNExchangeRates() const {return nExchangeRates_;}
  /// Returns the means of the normal distributions for the initial log-volatilities.
  arma::colvec getMeanInitialLogVolatility() const {return meanInitialLogVolatility_;}
  /// Returns the mean of the normal distribution for the $k$th initial log-volatility.
  double getMeanInitialLogVolatility(const unsigned int k) const {return meanInitialLogVolatility_(k);}
  /// Returns the standard deviations of the normal distributions for the initial log-volatilities.
  arma::colvec getSdInitialLogVolatility() const {return sdInitialLogVolatility_;}
  /// Returns the standard deviation of the normal distribution for the $k$th initial log-volatility.
  double getSdInitialLogVolatility(const unsigned int k) const {return sdInitialLogVolatility_(k);}
  /// Returns the stepsize parameter $\delta$.
  double getDelta() const {return delta_;}
  /// Returns the square-root of the stepsize parameter $\delta$.
  double getRootDelta() const {return rootDelta_;}
  
  /// Returns the Gaussian prior mean of the $i$th parameter
  double getMeanHyper(const unsigned int i) const {return meanHyper_(i);}
  /// Returns the Gaussian prior standard deviation of the $i$th parameter
  double getSdHyper(const unsigned int i) const {return sdHyper_(i);}
  /// Returns the Gaussian prior mean of all parameters.
  arma::colvec getMeanHyper() const {return meanHyper_;}
  /// Returns the Gaussian prior standard deviation of all parameters.
  arma::colvec getSdHyper() const {return sdHyper_;}
  /// Returns the $i$th parameter.
  double getTheta(const unsigned int i) const {return theta_(i);}
  /// Returns the full parameter vector.
  arma::colvec getTheta() const {return theta_;}
  /// Returns the dimension of the latent state.
  unsigned int getDimLatentVariable() const {return dimLatentVariable_;}
  
private:
  
  /// Computes the copula parameter from the corresponding (transformed)
  /// element of theta.
  double transformThetaToLambda(const double theta, const CopulaType& copulaType)
  {
    double lambda = 0.0;
    if (copulaType == COPULA_GAUSSIAN)
    {
//             std::cout << "using Gaussian copula for HZ" << std::endl;
      lambda = 2.0 * inverseLogit(theta) - 1;
    } 
    else if (copulaType == COPULA_90)
    {
//                  std::cout << "using 90 copula for HZ" << std::endl;
      lambda = std::exp(theta);
    }
    else if (copulaType == COPULA_270)
    {
//                    std::cout << "using 270 copula for HZ" << std::endl;
      lambda = std::exp(theta);
    }
    return lambda;
  }
  /// Computes the copula parameter from the corresponding (transformed)
  /// element of theta. Overload for transforming a whole vector.
  arma::colvec transformThetaToLambda(const arma::colvec& theta, const CopulaType& copulaType)
  {
    arma::colvec lambda;
    if (copulaType == COPULA_GAUSSIAN)
    {
      std::cout << "using Gaussian copula for H or Z" << std::endl;
      lambda = 2.0 * inverseLogit(theta) - 1;
    } 
    else if (copulaType == COPULA_90)
    {
            std::cout << "using 90 copula for H or Z" << std::endl;
      lambda = arma::exp(theta);
    }
    else if (copulaType == COPULA_270)
    {
            std::cout << "using 270 copula for H or Z" << std::endl;
      lambda = arma::exp(theta);
    }
    return lambda;
  }
 
  // Unknown model parameters
  arma::colvec theta_; // the full parameter vector
  arma::colvec alpha_, beta_, kappa_, mu_, sigma_, lambdaH_, lambdaZ_;
  double lambdaHZ_, omega_;

  // Known hyperparameters for the prior distribution:
  ModelType modelType_; // index for model specification
  CopulaType copulaTypeH_, copulaTypeZ_, copulaTypeHZ_; // index for the choice of copula functions for modelling noise-variable--factor and factor--factor dependence
  unsigned int nExchangeRates_; // number of exchange rates
  double delta_, rootDelta_; // the time between observations/discretisation parameter
  arma::colvec meanInitialLogVolatility_; // means of the normal distribution for the initial true log-volatilities
  arma::colvec sdInitialLogVolatility_; // standard deviations of the normal distribution for the initial true log-volatilities
  unsigned int dimTheta_; // length of the parameter vector theta
  unsigned int T_; // number of observation periods
  arma::colvec meanHyper_, sdHyper_; // means and standard deviations of the Gaussian priors on all parameters
  unsigned int dimLatentVariable_; // dimension of the state vector
  
};
/// Holds some additional auxiliary parameters for the SMC algorithm.
class SmcParameters
{
  
public:
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& algorithmParameters)
  {
    // Empty;
  }
  
private:
  
};
/// Holds some additional auxiliary parameters for the SMC algorithm.
class McmcParameters
{
    
public:
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& algorithmParameters)
  {
    // Empty;
  }
  
private:
  
};

/// Holds all latent variables in the model
typedef arma::colvec LatentVariable;

/// Holds all latent variables in the model
typedef arma::mat LatentPath;

/// Holds all latent variables in the model under 
/// the non-centred parametrisation.
typedef LatentPath LatentPathRepar;

/// Holds all latent variables associated with a single 
/// conditionally IID observation/single time step.
typedef LatentVariable Particle;

/// Holds (some of the) Gaussian auxiliary variables generated as part of 
/// the SMC algorithm.
/// Can be used to hold the normal random variables used for implementing
/// correlated pseudo-marginal kernels. Otherwise, this may not need to 
/// be used.

typedef arma::colvec Aux; // TODO: can we use correlated pseudo-marginal approaches in this model?

/// Holds all observations.
class Observations
{
public:
 
  arma::colvec initialLogExchangeRates_; // length-$K$ vector of the initial log-exchange rates, i.e. $S_{-1,k}$.
  arma::colvec initialLogVolatilities_; // length-$K$ vector of the initial log-exchange rates, i.e. $S_{-1,k}$.
  arma::mat logExchangeRates_; // a $(K \times T)$ matrix of the log-exchange rates, where $K$ denotes the number of exchange rates
  arma::mat logVolatilities_;  // a $(K \times T)$ matrix of the observed (with or without error) log-volatilities of the log-exchange rates.

};

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the class <<Model>>
///////////////////////////////////////////////////////////////////////////////

/// Evaluates the log-prior density of the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogPriorDensity()
{
  double logDensity = 0;
  for (unsigned int i=0; i < modelParameters_.getTheta().size(); i++)
  {
    logDensity += R::dnorm(modelParameters_.getTheta(i), modelParameters_.getMeanHyper(i), modelParameters_.getSdHyper(i), true);
  }
  return logDensity;
}
/// Samples the set of parameters from the prior.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromPrior(arma::colvec& theta)
{
  // NOTE: here we simply use the same (non-truncated) Gaussian prior on all parameters!
  theta = modelParameters_.getMeanHyper() + modelParameters_.getSdHyper() % arma::randn<arma::colvec>(theta.size());
}
/// Increases the gradient by the gradient of the log-prior density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogPriorDensity(arma::colvec& gradient)
{
  // TODO (though at the moment, we're not using any gradient information)
}
/// Increases the gradient by the gradient of the log-initial density of
/// the latent states.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogInitialDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient)
{
  // TODO (though at the moment, we're not using any gradient information)
}
/// Increases the gradient by the gradient of the log-transition density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld, arma::colvec& gradient)
{
  // TODO (though at the moment, we're not using any gradient information)
}
/// Increases the gradient by the gradient of the log-observation density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogObservationDensity(const unsigned int t,  const LatentVariable& latentVariable, arma::colvec& gradient)
{
  // TODO (though at the moment, we're not using any gradient information)
}
/// Evaluates the marginal likelihood of the parameters (with the latent 
/// variables integrated out).
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihood()
{
  // Empty: the marginal likelihood is intractable in this model.
  return 0.0;
}
/// Evaluates the second part of the marginal likelihood of the parameters (with the latent 
/// variables integrated out).
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodSecond(LatentPath& latentPath)
{
  // Empty: the marginal likelihood is intractable in this model.
  return 0.0;
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
/// Simulates data from the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::simulateData(const arma::colvec& extraParameters)
{
//   std::cout << "running simulateData()" << std::endl;
  unsigned int K = modelParameters_.getNExchangeRates();
  observations_.logExchangeRates_.set_size(K, nObservations_);
  observations_.logVolatilities_.set_size(K, nObservations_);
  observations_.initialLogExchangeRates_.set_size(K);
  observations_.initialLogVolatilities_.set_size(K);
  latentPath_.set_size(modelParameters_.getDimLatentVariable(), nObservations_);

//   std::cout << "sampleFromInitialDistribution()" << std::endl;
  latentPath_.col(0) = sampleFromInitialDistribution();
//   std::cout << "sampleFromObservationEquation(0)" << std::endl;
  sampleFromObservationEquation(0, observations_, latentPath_.col(0));
  for (unsigned int t=1; t<nObservations_; ++t)
  {
//     std::cout << "sampleFromInitialDistribution(t)" << std::endl;
    latentPath_.col(t) = sampleFromTransitionEquation(t, latentPath_.col(t-1));
//     std::cout << "sampleFromObservationEquation(t)" << std::endl;
    sampleFromObservationEquation(t, observations_, latentPath_.col(t));
  }
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihood(const LatentPath& x)
{
  std::cout << "WARNING: evalutation of the log-complete likelihood is not implemented yet! But this is only needed for Gibbs-sampling type algorithms" << std::endl;
  
  return 0;
}
/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromInitialDistribution()
{
  arma::colvec xi(modelParameters_.getDimLatentVariable());
  
//   std::cout << "sampleFactors" << std::endl;
  xi(arma::span(0,1)) = modelParameters_.sampleFactors();
    
  if (modelParameters_.getModelType() == MODEL_NOISY_VOLATILITY_MEASUREMENTS)
  {
//     arma::colvec x0(modelParameters_.getNExchangeRates());
    for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
    {
      // The sampled log-volatility at time $-1$:
//       x0(k) = modelParameters_.sampleInitialLogVolatility(k);
      
      // NOTE: for simplicity, we assume that the initial volatility is measured without error
      // The sampled log-volatility at time $0$:
      xi(2+k) = modelParameters_.sampleLogVolatility(k, observations_.initialLogVolatilities_(k), xi(1));
    }
  }
  return xi;
}
/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromTransitionEquation(const unsigned int t, const LatentVariable& latentVariableOld)
{
  arma::colvec xi(modelParameters_.getDimLatentVariable());
  xi(arma::span(0,1)) = modelParameters_.sampleFactors();
  if (modelParameters_.getModelType() == MODEL_NOISY_VOLATILITY_MEASUREMENTS)
  {
    for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
    {
      // The sampled log-volatility at time $t$:
      xi(2+k) = modelParameters_.sampleLogVolatility(k, latentVariableOld(2+k), xi(1));
    }
  }
  return xi;
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogInitialDensity(const LatentVariable& latentVariable)
{
  return 0.0; // TODO: we do not need to implement this because we are proposing from the model
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld)
{
  return 0.0; // TODO: we do not need to implement this because we are proposing from the model
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable)
{
  double logDensity = 0.0;
  if (modelParameters_.getModelType() == MODEL_EXACT_VOLATILITY_MEASUREMENTS)
  {
    if (t == 0)
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {
        logDensity += R::dnorm(observations_.initialLogVolatilities_(k), modelParameters_.getMeanInitialLogVolatility(k), modelParameters_.getSdInitialLogVolatility(k), true);
        
//         std::cout << "initial density: " << std::endl;
//         std::cout << R::dnorm(observations_.initialLogVolatilities_(k), modelParameters_.getMeanInitialLogVolatility(k), modelParameters_.getSdInitialLogVolatility(k), true) << std::endl;
        
//         std::cout << "logInitialDensity: " <<R::dnorm(observations_.initialLogVolatilities_(k), modelParameters_.getMeanInitialLogVolatility(k), modelParameters_.getSdInitialLogVolatility(k), true) << " ---- ";
        
//         std::cout << "logDensity: " << modelParameters_.computeLogObservationDensityExact(k, latentVariable(0), latentVariable(1), observations_.logExchangeRates_(k,t), observations_.initialLogExchangeRates_(k), observations_.logVolatilities_(k,t), observations_.initialLogVolatilities_(k)) << " ---- ";
        
        logDensity += modelParameters_.computeLogObservationDensityExact(k, latentVariable(0), latentVariable(1), observations_.logExchangeRates_(k,t), observations_.initialLogExchangeRates_(k), observations_.logVolatilities_(k,t), observations_.initialLogVolatilities_(k));
        
      
//         double h = latentVariable(0);
//         double z = latentVariable(1);
//         
//         double sNew = observations_.logExchangeRates_(k,t);
//         double sOld = observations_.initialLogExchangeRates_(k);
//         double xNew = observations_.logVolatilities_(k,t);
//         double xOld = observations_.initialLogVolatilities_(k);
//         
//         double eta  = modelParameters_.computeEta(k,  sNew, sOld, xNew);
//         double zeta = modelParameters_.computeZeta(k, xNew, xOld);
//         double pnormEta  = R::pnorm(eta,  0.0, 1.0, true, false);
//         double pnormH    = R::pnorm(h,    0.0, 1.0, true, false);
//         double pnormZeta = R::pnorm(zeta, 0.0, 1.0, true, false);
//         double pnormZ    = R::pnorm(z,    0.0, 1.0, true, false);
//         
//         std::cout << "logDensity for k=" << k << ": " << logDensity <<"; ICDF_H: " <<  modelParameters_.evaluateLogDerivativeCopulaInverseCdf(pnormEta, pnormH, modelParameters_.getLambdaH(k), modelParameters_.getCopulaTypeH()) << "; ICDF_Z: " << modelParameters_.evaluateLogDerivativeCopulaInverseCdf(pnormZeta, pnormZ, modelParameters_.getLambdaZ(k), modelParameters_.getCopulaTypeZ()) << "; ---- ";
        

      }
    }
    else
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {       
        logDensity += modelParameters_.computeLogObservationDensityExact(k, latentVariable(0), latentVariable(1), observations_.logExchangeRates_(k,t), observations_.logExchangeRates_(k,t-1), observations_.logVolatilities_(k,t), observations_.logVolatilities_(k,t-1));
      }
    }
  }
  else if (modelParameters_.getModelType() == MODEL_NOISY_VOLATILITY_MEASUREMENTS)
  {
    if (t == 0)
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {
        

//         std::cout << "; xNew/xOld/h/z/logDensity/: " << latentVariable(0), latentVariable(1)
        
        // NOTE: for simplicity, we assume that the initial volatility (at time $-1$) is measured without error
        logDensity += R::dnorm(observations_.initialLogVolatilities_(k), modelParameters_.getMeanInitialLogVolatility(k), modelParameters_.getSdInitialLogVolatility(k), true);
        logDensity += modelParameters_.computeLogObservationDensityNoisy(k, latentVariable(0), latentVariable(1), latentVariable(2+k), observations_.logExchangeRates_(k,t), observations_.initialLogExchangeRates_(k), observations_.logVolatilities_(k,t));
        
//           std::cout << "logDensity for k=" << k << ": " << logDensity << "; ---- ";
      }
    }
    else
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {
        logDensity += modelParameters_.computeLogObservationDensityNoisy(k, latentVariable(0), latentVariable(1), latentVariable(2+k), observations_.logExchangeRates_(k,t), observations_.logExchangeRates_(k,t-1), observations_.logVolatilities_(k,t));
      }
    }
  }
  
//   std::cout << "logDensity: " << logDensity 
//   <<"; Xi "<< latentVariable(2+0) <<" "<< latentVariable(2+1) <<" "<< latentVariable(2+2) 
//   << "S: " << observations_.logExchangeRates_(0,t) << " "<< observations_.logExchangeRates_(1,t) << " "<< observations_.logExchangeRates_(2,t) 
//   <<" "<< observations_.initialLogExchangeRates_.t()
//   <<" X: "<< observations_.logVolatilities_(0,t) <<" "<< observations_.logVolatilities_(1,t)   <<" "<< observations_.logVolatilities_(2,t) 
//   << " ----------- ";
  return logDensity;
}
/// Samples a single observation according to the observation equation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromObservationEquation(const unsigned int t, Observations& observations, const LatentVariable& latentVariable)
{
  
  if (t == 0) {
    for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
    {
      observations.initialLogExchangeRates_(k) = modelParameters_.sampleInitialLogExchangeRate(k);
      observations.initialLogVolatilities_(k)  = modelParameters_.sampleInitialLogVolatility(k);
    }
  }
      
  if (modelParameters_.getModelType() == MODEL_EXACT_VOLATILITY_MEASUREMENTS)
  {
    if (t == 0)
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {
        observations.logVolatilities_(k,t)  
        = modelParameters_.sampleLogVolatility(k, observations.initialLogVolatilities_(k), latentVariable(1));
        
        observations.logExchangeRates_(k,t) 
        = modelParameters_.sampleLogExchangeRate(k, observations.initialLogExchangeRates_(k), observations.logVolatilities_(k,t), latentVariable(0));
      }
    }
    else
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {
        observations.logVolatilities_(k,t)  = modelParameters_.sampleLogVolatility(k, observations.logVolatilities_(k,t-1), latentVariable(1));
        observations.logExchangeRates_(k,t) = modelParameters_.sampleLogExchangeRate(k, observations.logExchangeRates_(k,t-1), observations.logVolatilities_(k,t), latentVariable(0));
      }
    }
  }
  else if (modelParameters_.getModelType() == MODEL_NOISY_VOLATILITY_MEASUREMENTS)
  {
    if (t == 0)
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {
        observations.logVolatilities_(k,t)  = latentVariable(2+k) + modelParameters_.getOmega() * arma::randn();
        observations.logExchangeRates_(k,t) = modelParameters_.sampleLogExchangeRate(k, observations.initialLogExchangeRates_(k), latentVariable(2+k), latentVariable(0));
      }
    }
    else
    {
      for (unsigned int k=0; k<modelParameters_.getNExchangeRates(); k++)
      {
        observations.logVolatilities_(k,t)  = latentVariable(2+k) + modelParameters_.getOmega() * arma::randn();
        observations.logExchangeRates_(k,t) = modelParameters_.sampleLogExchangeRate(k, observations.logExchangeRates_(k,t-1), latentVariable(2+k), latentVariable(0));
      }
    }
  }
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables using a (partially) non-centred parametrisation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihoodRepar(const LatentPathRepar& latentPathRepar)
{
  return evaluateLogCompleteLikelihood(latentPathRepar);
}
/// Evaluates the log of the likelihood associated with some subset of the 
/// (static) model parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodFirst(LatentPath& latentPath)
{
//   std::cout << "Warning: we are not using delayed-acceptance approaches in this model!" << std::endl;
  return 0.0;
}

///////////////////////////////////////////////////////////////////////////////
/// Member functions of Smc class
///////////////////////////////////////////////////////////////////////////////

/// Samples particles at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleParticles
(
  const unsigned int t, 
  std::vector<Particle>& particlesNew,
  const std::vector<Particle>& particlesOld
)
{
  for (unsigned int n=0; n<getNParticles(); n++)
  {
    particlesNew[n] = model_.sampleFromTransitionEquation(t, particlesOld[n]);
  }
  if (isConditional_) {particlesNew[particleIndicesIn_(t)] = particlePath_[t];}
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
  //////////////////////////
//   arma::colvec logWeightsAux = logWeights;
    //////////////////////////
  
  /// Avoiding some duplicate calculations in order to speed up the algorithm
  /// in the case that we measure the volatilities exactly.
  if (model_.getModelParameters().getModelType() == MODEL_EXACT_VOLATILITY_MEASUREMENTS)
  {
//     double eta, zeta, pnormEta, pnormZeta;
//     double sNew, sOld, xNew, xOld;
//     
//     for (unsigned int k=0; k<model_.getModelParameters().getNExchangeRates(); k++)
//     {
//       // NOTE: these are no longer correct
//       sNew = model_.getObservations().logExchangeRates_(k,t);
//       sOld = model_.getObservations().logExchangeRates_(k,t-1);
//       xNew = model_.getObservations().logVolatilities_(k,t);
//       xOld = model_.getObservations().logVolatilities_(k,t-1);
//       eta  = model_.getModelParameters().computeEta(k,  sNew, sOld, xNew);
//       zeta = model_.getModelParameters().computeZeta(k, xNew, xOld);
//       
//       pnormEta  = R::pnorm(eta, 0.0, 1.0, true, false);
//       pnormZeta = R::pnorm(zeta, 0.0, 1.0, true, false);
//       
//       logWeights = logWeights + model_.getModelParameters().computeLogObservationDensityExactCommon(k, eta, zeta, xNew, xOld);
//       
//       for (unsigned int n=0; n<getNParticles(); n++)
//       {
//         logWeights(n) += model_.getModelParameters().computeLogObservationDensityExactIndividual(k, pnormEta, pnormZeta, particlesNew[n](0), particlesNew[n](1));
//       }
//     }
//     
      //////////////////////////
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]);
//       logWeightsAux(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]);
//       std::cout << logWeightsAux(n) - logWeights(n) << " ";
    }
//     std::cout << "Absolute difference in log weights: " << arma::sum(arma::abs(logWeightsAux - logWeights)) << std::endl;
      //////////////////////////
    

  }
  else
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]);
    }
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
  for (unsigned int n=0; n<getNParticles(); n++)
  {
    particlesNew[n] = model_.sampleFromInitialDistribution();
  }
  if (isConditional_) {particlesNew[particleIndicesIn_(0)] = particlePath_[0];}
}
/// Computes the incremental particle weights at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogInitialParticleWeights
(
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{
  /// Avoiding some duplicate calculations in order to speed up the algorithm
  /// in the case that we measure the volatilities exactly.
  if (model_.getModelParameters().getModelType() == MODEL_EXACT_VOLATILITY_MEASUREMENTS)
  {
//     double eta, zeta, pnormEta, pnormZeta;
//     double sNew, sOld, xNew, xOld;
//     
//     for (unsigned int k=0; k<model_.getModelParameters().getNExchangeRates(); k++)
//     {
// 
//       sNew = model_.getObservations().logExchangeRates_(k,0);
//       sOld = model_.getObservations().initialLogExchangeRates_(k);
//       xNew = model_.getObservations().logVolatilities_(k,0);
//       xOld = model_.getObservations().initialLogVolatilities_(k);
//       eta  = model_.getModelParameters().computeEta(k,  sNew, sOld, xNew);
//       zeta = model_.getModelParameters().computeZeta(k, xNew, xOld);
//       pnormEta  = R::pnorm(eta, 0.0, 1.0, true, false);
//       pnormZeta = R::pnorm(zeta, 0.0, 1.0, true, false);
//       
//       logWeights = logWeights + model_.getModelParameters().computeLogObservationDensityExactCommon(k, eta, zeta, xNew, xOld);
//       
//       for (unsigned int n=0; n<getNParticles(); n++)
//       {
//         logWeights(n) += model_.getModelParameters().computeLogObservationDensityExactIndividual(k, pnormEta, pnormZeta, particlesNew[n](0), particlesNew[n](1));
//       }
//     }
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += model_.evaluateLogObservationDensity(0, particlesNew[n]);
//       logWeightsAux(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]);
//       std::cout << logWeightsAux(n) - logWeights(n) << " ";
    }
  }
  else
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += model_.evaluateLogObservationDensity(0, particlesNew[n]);
    }
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
  std::cout << "Warning: evaluation of the transition density is currently not implemented!" << std::endl;
  return 0.0;
}

/// Converts a particle path into the set of all latent variables in the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath)
{
  convertStdVecToArmaMat(particlePath, latentPath);
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  convertArmaMatToStdVec(latentPath, particlePath);
}

#endif
