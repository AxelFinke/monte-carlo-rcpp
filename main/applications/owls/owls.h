/// \file
/// \brief Generating data and performing inference in "Little Owls" model. 
///
/// This file contains the functions for implementing the abstract model
/// class for the "Little Owls" model.

#ifndef __OWLS_H
#define __OWLS_H

#include "main/templates/dynamic/stateSpace/stateSpace.h"
#include "main/algorithms/mcmc/Mcmc.h"
#include "main/rng/gaussian.h"

// [[Rcpp::depends("RcppArmadillo")]]


enum ModelType 
{ 
  MODEL_VARYING_CAPTURE_VARYING_PRODUCTIVITY = 0,                          // probabilities $p_{g,t}$ and $\gamma_t$ are time-varying.
  MODEL_VARYING_CAPTURE_VARYING_PRODUCTIVITY_NO_DELTA_1,                   // same as above but with $\delta_1 = 0$
  
  MODEL_CONSTANT_CAPTURE_VARYING_PRODUCTIVITY,                             // probabilities $p_{g,t}$ are constant in $t$; $\gamma_t$ are time-varying.
  MODEL_CONSTANT_CAPTURE_VARYING_PRODUCTIVITY_NO_DELTA_1,                  // same as above but with $\delta_1 = 0$
  
  MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY,                             // probabilities $p_{g,t}$ are time-varying in $t$; $\gamma_t$ are constant.
  MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_DELTA_1,                  // same as above but with $\delta_1 = 0$
  MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3,                  // same as above but with $\alpha_3 = 0$
  MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3_DELTA_1,          // same as above but with $\delta_1 = \alpha_3 = 0$
  MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1,           // same as above but with $\alpha_1 = \beta_1 = 0$
  MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1_DELTA_1,   // same as above but with $\alpha_1 = \beta_1 = \delta_1 = 0$
  
  MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY,                            // probabilities $p_{g,t}$ are constant in $t$; $\gamma_t$ are constant in $t$.
  MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_DELTA_1,                 // same as above but with $\delta_1 = 0$
  MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3,                 // same as above but with $\alpha_3 = 0$
  MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3_DELTA_1,         // same as above but with $\delta_1 = \alpha_3 = 0$
  MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1,          // same as above but with $\alpha_1 = \beta_1 = 0$
  MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1_DELTA_1,  // same as above but with $\alpha_1 = \beta_1 = \delta_1 = 0$
  
};

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the state-space model
////////////////////////////////////////////////////////////////////////////////

/// Holds all the static model parameters.
class ModelParameters
{
public:
  
  /// Determines the model parameters from arma::colvec theta.
  /// Note that $\theta = (\alpha_{0:3}, \beta_0, \delta_{0:1}, \beta_1,{2:T}, \gamma_{1:T})$
  void setUnknownParameters(const arma::colvec& theta)
  {
    
    theta_  = theta;
    gamma_.set_size(T_);
    beta1_.set_size(T_-1);
    
    if (modelType_ == MODEL_VARYING_CAPTURE_VARYING_PRODUCTIVITY)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = theta(6);
      gamma_  = theta(arma::span(7,T_+6));
      beta1_  = theta(arma::span(T_+7,2*T_+5));
    }
    else if (modelType_ == MODEL_VARYING_CAPTURE_VARYING_PRODUCTIVITY_NO_DELTA_1)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = 0.0;
      gamma_  = theta(arma::span(6,T_+5));
      beta1_  = theta(arma::span(T_+6,2*T_+4));
    }
    else if (modelType_ == MODEL_CONSTANT_CAPTURE_VARYING_PRODUCTIVITY)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = theta(6);
      gamma_  = theta(arma::span(7,T_+6));
      beta1_.fill(theta(T_+7));
    }
    else if (modelType_ == MODEL_CONSTANT_CAPTURE_VARYING_PRODUCTIVITY_NO_DELTA_1)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = 0.0;
      gamma_  = theta(arma::span(6,T_+7));
      beta1_.fill(theta(T_+6));
    }
    else if (modelType_ == MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = theta(6);
      gamma_.fill(theta(7));
      beta1_ = theta(arma::span(8, 6+T_));
    }
    else if (modelType_ == MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_DELTA_1)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = 0.0;
      gamma_.fill(theta(6));
      beta1_ = theta(arma::span(7, 5+T_));
    }
    else if (modelType_ == MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = 0.0;
      beta0_  = theta(3);
      delta0_ = theta(4);
      delta1_ = theta(5);
      gamma_.fill(theta(6));
      beta1_ = theta(arma::span(7, 5+T_));
    }
    else if (modelType_ == MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3_DELTA_1)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = 0.0;
      beta0_  = theta(3);
      delta0_ = theta(4);
      delta1_ = 0.0;
      gamma_.fill(theta(5));
      beta1_ = theta(arma::span(6, 4+T_));
    }
    else if (modelType_ == MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1)
    {
      alpha0_ = theta(0);
      alpha1_ = 0.0;
      alpha2_ = theta(1);
      alpha3_ = theta(2);
      beta0_  = 0.0;
      delta0_ = theta(3);
      delta1_ = theta(4);
      gamma_.fill(theta(5));
      beta1_ = theta(arma::span(6, 4+T_));
    }
    else if (modelType_ == MODEL_VARYING_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1_DELTA_1)
    {
      alpha0_ = theta(0);
      alpha1_ = 0.0;
      alpha2_ = theta(1);
      alpha3_ = theta(2);
      beta0_  = 0.0;
      delta0_ = theta(3);
      delta1_ = 0.0;
      gamma_.fill(theta(4));
      beta1_ = theta(arma::span(5, 3+T_));
    }
    else if (modelType_ == MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY)
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = theta(6);
      gamma_.fill(theta(7));
      beta1_.fill(theta(8));
    }
    else if (modelType_ == MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_DELTA_1) 
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = theta(3);
      beta0_  = theta(4);
      delta0_ = theta(5);
      delta1_ = 0.0;
      gamma_.fill(theta(6));
      beta1_.fill(theta(7));
    }
    else if (modelType_ == MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3) 
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = 0.0;
      beta0_  = theta(3);
      delta0_ = theta(4);
      delta1_ = theta(5);
      gamma_.fill(theta(6));
      beta1_.fill(theta(7));
    }
    else if (modelType_ == MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_3_DELTA_1) 
    {
      alpha0_ = theta(0);
      alpha1_ = theta(1);
      alpha2_ = theta(2);
      alpha3_ = 0.0;
      beta0_  = theta(3);
      delta0_ = theta(4);
      delta1_ = 0.0;
      gamma_.fill(theta(5));
      beta1_.fill(theta(6));
    }
    else if (modelType_ == MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1) 
    {
      alpha0_ = theta(0);
      alpha1_ = 0.0;
      alpha2_ = theta(1);
      alpha3_ = theta(2);
      beta0_  = 0.0;
      delta0_ = theta(3);
      delta1_ = theta(4);
      gamma_.fill(theta(5));
      beta1_.fill(theta(6));
    }
     else if (modelType_ == MODEL_CONSTANT_CAPTURE_CONSTANT_PRODUCTIVITY_NO_ALPHA_1_BETA_1_DELTA_1) 
    {
      alpha0_ = theta(0);
      alpha1_ = 0.0;
      alpha2_ = theta(1);
      alpha3_ = theta(2);
      beta0_  = 0.0;
      delta0_ = theta(3);
      delta1_ = 0.0;
      gamma_.fill(theta(4));
      beta1_.fill(theta(5));
    }

    
    
    rho_ = arma::exp(gamma_);
    eta_ = arma::exp(delta0_ + delta1_ * arma::conv_to<arma::colvec>::from(voleCovar_));
    
    phiFemaleFirst_ = inverseLogit(alpha0_ + alpha3_ * timeNormCovar_);
    phiMaleFirst_   = inverseLogit(alpha0_ + alpha1_ + alpha3_ * timeNormCovar_);
    phiFemaleAdult_ = inverseLogit(alpha0_ + alpha2_ + alpha3_ * timeNormCovar_);
    phiMaleAdult_   = inverseLogit(alpha0_ + alpha1_ + alpha2_+ alpha3_ * timeNormCovar_);
    
    pFemale_ = inverseLogit(beta1_);
    pMale_   = inverseLogit(beta0_ + beta1_);

    computeQ(qFemaleFirst_, pFemale_, phiFemaleFirst_, phiFemaleAdult_); 
    computeQ(qFemaleAdult_, pFemale_, phiFemaleAdult_, phiFemaleAdult_); 
    computeQ(qMaleFirst_, pMale_, phiMaleFirst_, phiMaleAdult_); 
    computeQ(qMaleAdult_, pMale_, phiMaleAdult_, phiMaleAdult_); 
    
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyp)
  {
    dimLatentVariable_ = 2;
    modelType_ = static_cast<ModelType>(hyp(0));
    T_ = static_cast<unsigned int>(hyp(1)); // number of observations
    dimTheta_ = hyp(2);
    
    meanHyper_     = hyp(arma::span(3,dimTheta_+2));
    sdHyper_       = hyp(arma::span(dimTheta_+3,2*dimTheta_+2));
    minHyperInit_  = static_cast<unsigned int>(hyp(2*dimTheta_+3));
    maxHyperInit_  = static_cast<unsigned int>(hyp(2*dimTheta_+4));
    timeNormCovar_ = hyp(arma::span(2*dimTheta_+5,2*dimTheta_+3 + T_)); // "normalised" years
    voleCovar_     = arma::conv_to<arma::uvec>::from(hyp(arma::span(2*dimTheta_+4 +T_,2*dimTheta_+2 + 2*T_))); // binary vole abundance data
 
//     thetaIndicesSecond_.clear();
    
  }
  
  /// Returns the indices of the model parameters that only depend on the count data.
//   arma::uvec getThetaIndicesSecond() const {return thetaIndicesSecond_;} 
  
  /// Returns the parameter vector $\phi_{1,m}$.
  arma::colvec getPhiMaleFirst() const {return phiMaleFirst_;}
  /// Returns the parameter vector $\phi_{1,f}$.
  arma::colvec getPhiFemaleFirst() const {return phiFemaleFirst_;}
  /// Returns the parameter vector $\phi_{A,m}$.
  arma::colvec getPhiMaleAdult() const {return phiMaleAdult_;}
  /// Returns the parameter vector $\phi_{A,f}$.
  arma::colvec getPhiFemaleAdult() const {return phiFemaleAdult_;}
  
  /// Returns the vector $\phi_{1,f,i}$.
  double getPhiFemaleFirst(const unsigned int i) const {return phiFemaleFirst_(i);}
  /// Returns the parameter $\phi_{A,f,i}$.
  double getPhiFemaleAdult(const unsigned int i) const {return phiFemaleAdult_(i);}
  /// Returns the parameter vector $\phi_{1,m,i}$.
  double getPhiMaleFirst(const unsigned int i) const {return phiMaleFirst_(i);}
  /// Returns the parameter vector $\phi_{A,m,i}$.
  double getPhiMaleAdult(const unsigned int i) const {return phiMaleAdult_(i);}
  
  /// Returns the relevant elements of the parameter vector $\q_{1,f,i}$.
  arma::colvec getQFemaleFirst(const unsigned int t) const 
  {
    return arma::trans(arma::conv_to<arma::rowvec>::from(qFemaleFirst_(arma::span(t,t), arma::span(t,qFemaleFirst_.n_cols-1))));
  }
  /// Returns the relevant elements of the parameter vector $\q_{A,f,i}$.
  arma::colvec getQFemaleAdult(const unsigned int t) const 
  {
    return arma::trans(arma::conv_to<arma::rowvec>::from(qFemaleAdult_(arma::span(t,t), arma::span(t,qFemaleAdult_.n_cols-1))));  
  }
  /// Returns the relevant elements of the parameter vector $\q_{1,m,i}$.
  arma::colvec getQMaleFirst(const unsigned int t) const 
  {
    return arma::trans(arma::conv_to<arma::rowvec>::from(qMaleFirst_(arma::span(t,t), arma::span(t,qMaleFirst_.n_cols-1))));
  }
  /// Returns the relevant elements of the parameter vector $\q_{A,m,i}$.
  arma::colvec getQMaleAdult(const unsigned int t) const 
  {
    return arma::trans(arma::conv_to<arma::rowvec>::from(qMaleAdult_(arma::span(t,t), arma::span(t,qMaleAdult_.n_cols-1))));
  }
  
  /// Returns the parameter vector $\rho$.
  arma::colvec getRho() const {return rho_;}
  /// Returns $\rho_t$.
  double getRho(const unsigned int t) const {return rho_(t);}
  /// Returns the parameter vector $\eta$.
  arma::colvec getEta() const {return eta_;}
  /// Returns $\eta_t$.
  double getEta(const unsigned int t) const {return eta_(t);}
  /// Returns the parameter vector $\gamma_{1:T}$.
  arma::colvec getGamma() const {return gamma_;}
  /// Returns $\gamma_t$.
  double getGamma(const unsigned int t) const {return gamma_(t);}
  /// Returns $\beta_{2,t}$.
  double getBeta1(const unsigned int t) const {return beta1_(t);}
  /// Returns the parameter vector $\beta_{1,2:T}$.
  arma::colvec getBeta1() const {return beta1_;}
  
  /// Returns $\alpha_0$.
  double getAlpha0() const {return alpha0_;}
  /// Returns $\alpha_1$.
  double getAlpha1() const {return alpha1_;}
  /// Returns $\alpha_2$.
  double getAlpha2() const {return alpha2_;}
  /// Returns $\alpha_3$.
  double getAlpha3() const {return alpha3_;}
  /// Returns $\beta_0$.
  double getBeta0() const {return beta0_;}
  /// Returns $\delta_0$.
  double getDelta0() const {return delta0_;}
  /// Returns $\delta_1$.
  double getDelta1() const {return delta1_;}
  
  /// Returns the parameter vector $p_m$.
  arma::colvec getPMale() const {return pMale_;}
  /// Returns the parameter vector $p_f$.
  arma::colvec getPFemale() const {return pFemale_;}
  
  /// Returns the number of count-data observations $T$.
  unsigned int getT() const {return T_;}
   /// Returns the model index.
  ModelType getModelType() const {return modelType_;}
  
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
  
  /// Returns the minimum of the support of the uniform prior on the initial state.
  unsigned int getMinHyperInit() const {return minHyperInit_;}
  /// Returns the maximum of the support of the uniform prior on the initial state.
  unsigned int getMaxHyperInit() const {return maxHyperInit_;}
  
  /// Returns the dimension of the latent state.
  unsigned int getDimLatentVariable() const {return dimLatentVariable_;}
  
private:
  
  void computeQ(arma::mat& q, const arma::colvec& p, const arma::colvec& phiFirst, const arma::colvec& phiAdult)
  {
    q.zeros(T_-1,T_);
    for (unsigned int i=0; i<T_-1; i++)
    {
      // NOTE: the rows only differ by the phiFirst element. 
      // Hence, it may be more efficient to traverse the other direction first
      
      // NOTE: q(i,j) is the probability of observing an individual at time $j+1$(!) 
      // given that it was last observed at time $i$.
      q(i,i) = std::exp(std::log(phiFirst(i)) + std::log(p(i))); 

      // NOTE: the $t$th element of p corresponds to $p_{t+1}$ because no parameter $p_1$ is used/defined.
      for (unsigned int j=i+1; j<T_-1; j++)
      {
        // NOTE: 
        // The $j$th column corresponds to the number of observed individuals at time $j+1$.
        // p(j) is the probability of observing an individual at time $j+1$.
        // phiAdult(j) is the probability of an adult individual surviving to time $j+1$.
        q(i,j) = std::exp(
          std::log(phiFirst(i)) 
          + std::log(p(j)) 
          + arma::accu(arma::log(phiAdult(arma::span(i+1,j))) + arma::log(1.0 - p(arma::span(i,j-1))))
        ); 
      }
      
      q(i,T_-1) = 1 - arma::accu(q.row(i));
    }
  }
    
  // Unknown model parameters
  arma::colvec theta_; // the full parameter vector
  double alpha0_, alpha1_, alpha2_, alpha3_, beta0_, delta0_, delta1_;
  arma::colvec beta1_; // (T-1)-dimensional vector 
  arma::colvec gamma_; // T-dimensional vector
  
  // Unknown model parameters (these are deterministic functions of the covariates and of alpha, beta, gamma and delta
  arma::colvec phiMaleFirst_; // length-(T-1) vectors
  arma::colvec phiFemaleFirst_; // length-(T-1) vector
  arma::colvec phiMaleAdult_; // length-(T-1) vectors
  arma::colvec phiFemaleAdult_; // length-(T-1) vector
  
  arma::colvec rho_; // productivity rates (vector of length T)
  arma::colvec eta_; // immigration rates (vector of length T-1)
  arma::colvec pMale_; // ovserving a live male individual (vector of length T-1)
  arma::colvec pFemale_; // ovserving a live female individual (vector of length T-1)
  arma::mat qMaleFirst_; // (T-1, T)-matrix of multinomial cell probabilities for release and recapture of first-year males
  arma::mat qFemaleFirst_; // (T-1, T)-matrix of multinomial cell probabilities for release and recapture of first-year females
  arma::mat qMaleAdult_; // (T-1, T)-matrix of multinomial cell probabilities for release and recapture of adult males
  arma::mat qFemaleAdult_; // (T-1, T)-matrix of multinomial cell probabilities for release and recapture of adult females
  
  // Known hyperparameters for the prior distribution:
  ModelType modelType_; // index for model specification
  unsigned int dimTheta_; // length of the parameter vector theta
  unsigned int T_; // number of observation periods
  arma::colvec meanHyper_, sdHyper_; // means and standard deviations of the Gaussian priors on all parameters
  unsigned int minHyperInit_, maxHyperInit_; // minimum and maximum of the support of the uniform prior on the initial state.
  unsigned int dimLatentVariable_; // dimension of the state vector
  
//   unsigned int thetaIndicesSecond_; // indices of the model parameters that only depend on the count data.
  
  // Other known covariates:
  arma::colvec timeNormCovar_; // length-(T-1) vector of "normalised" years
  arma::uvec voleCovar_; // length-(T-1) vector of vole abundance data (note that the data file contains T entries but the last one isn't used for inference)

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
typedef arma::uvec LatentVariable;

/// Holds all latent variables in the model
typedef arma::umat LatentPath;

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
  
  /// Returns the relevant entries in row $t$ of the 
  /// capture-recapture data matrix for first-year female individuals
  arma::uvec getCapRecapFemaleFirst(const unsigned int t) const
  {
    return arma::trans(arma::conv_to<arma::urowvec>::from(capRecapFemaleFirst_(arma::span(t,t), arma::span(t,capRecapFemaleFirst_.n_cols-1))));
  }
  /// Returns the relevant entries in row $t$ of the 
  /// capture-recapture data matrix for adult female individuals
  arma::uvec getCapRecapFemaleAdult(const unsigned int t) const
  {
    return arma::trans(arma::conv_to<arma::urowvec>::from(capRecapFemaleAdult_(arma::span(t,t), arma::span(t,capRecapFemaleAdult_.n_cols-1))));
  }
  /// Returns the relevant entries in row $t$ of the 
  /// capture-recapture data matrix for first-year male individuals
  arma::uvec getCapRecapMaleFirst(const unsigned int t) const
  {
    return arma::trans(arma::conv_to<arma::urowvec>::from(capRecapMaleFirst_(arma::span(t,t), arma::span(t,capRecapMaleFirst_.n_cols-1))));
  }
  /// Returns the relevant entries in row $t$ of the 
  /// capture-recapture data matrix for adult male individuals
  arma::uvec getCapRecapMaleAdult(const unsigned int t) const
  {
    return arma::trans(arma::conv_to<arma::urowvec>::from(capRecapMaleAdult_(arma::span(t,t), arma::span(t,capRecapMaleAdult_.n_cols-1))));
  }
  
  /// Returns the number of juvenile females released in year $t$.
  unsigned int getReleasedFemaleFirst(const unsigned int t) const { return releasedFemaleFirst_(t); }
  /// Returns the number of adult females released in year $t$.
  unsigned int getReleasedFemaleAdult(const unsigned int t) const { return releasedFemaleAdult_(t); }
  /// Returns the number of juvenile females released in year $t$.
  unsigned int getReleasedMaleFirst(const unsigned int t) const { return releasedMaleFirst_(t); }
  /// Returns the number of adult females released in year $t$.
  unsigned int getReleasedMaleAdult(const unsigned int t) const { return releasedMaleAdult_(t); }
  
  arma::uvec releasedFemaleFirst_; // length-$(T-1)$ vector total number of female first-years released in each year (again, not data on the number of individuals released at time $T$ is included here since there is not recapture data on these).
  arma::uvec releasedMaleFirst_; // length-$(T-1)$ vector of total number of male first-years released in each year.
  arma::uvec releasedFemaleAdult_; // length-$(T-1)$ vector of total number of female adults released in each year.
  arma::uvec releasedMaleAdult_; // length-$(T-1)$ vector of total number of male adults released in each year.
  
  arma::umat capRecapFemaleFirst_; // (T-1, T)-matrices of female capture-recapture data (for $j<T$, element $(i,j)$ (counting from $1$ and not counting from $0$ as C++ does) represents the number of individuals last seen at time $i$ that are recaptured at time $j+1$ (!). The $T$th column represents the number of individuals never recaptured. In other words, we do not include an initial column of zeros because animals released at time $1$ cannot be recaptured at time $1$. Furthermore, the matrix only has $T-1$ rows because there is no recapture data on the animals released at time $T$.
  arma::umat capRecapMaleFirst_; // same as above but for male first-years
  arma::umat capRecapFemaleAdult_; // same as above but for female adults
  arma::umat capRecapMaleAdult_; // same as above but for male adults
  
  arma::uvec count_; // length-T vector of count data
  arma::umat fecundity_; // (T,2) matrix of fecundity data (the first column represents the number of chicks that survive to leave the nest and the second column is the number of chicks that are produced).

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
  // Empty: the marginal likelihood in this model.
  return 0.0;
}
/// Evaluates the second part of the marginal likelihood of the parameters (with the latent 
/// variables integrated out).
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodSecond(LatentPath& latentPath)
{
  // Empty: the marginal likelihood in this model.
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
/// Simulates count data from state-space model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::simulateData(const arma::colvec& extraParameters)
{
  observations_.count_.set_size(nObservations_); // TODO: need to make sure that nObservations_ is set correctly
  latentPath_.set_size(modelParameters_.getDimLatentVariable(), nObservations_);

  latentPath_.col(0) = sampleFromInitialDistribution();
  sampleFromObservationEquation(0, observations_, latentPath_.col(0));
  for (unsigned int t=1; t<nObservations_; ++t)
  {
    latentPath_.col(t) = sampleFromTransitionEquation(t, latentPath_.col(t-1));
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
  return arma::randi<arma::uvec>(modelParameters_.getDimLatentVariable(),
                                 arma::distr_param(static_cast<int>(modelParameters_.getMinHyperInit()), static_cast<int>(modelParameters_.getMaxHyperInit())));
}
/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromTransitionEquation(const unsigned int t, const LatentVariable& latentVariableOld)
{
  arma::uvec x(2);
  unsigned int totalPopSize = arma::accu(latentVariableOld);
  
//   x(0) = static_cast<unsigned int>(R::rpois(totalPopSize * modelParameters_.getRho(t-1) * modelParameters_.getPhiFemaleFirst(t-1) / 2.0));
//   x(1) = static_cast<unsigned int>(R::rpois(totalPopSize * modelParameters_.getEta(t-1)) + R::rbinom(totalPopSize, modelParameters_.getPhiFemaleAdult(t-1)));
  
  x(0) = R::rpois(totalPopSize * modelParameters_.getRho(t-1) * modelParameters_.getPhiFemaleFirst(t-1) / 2.0);
  x(1) = R::rpois(totalPopSize * modelParameters_.getEta(t-1)) + R::rbinom(totalPopSize, modelParameters_.getPhiFemaleAdult(t-1));
  
//   std::cout << x(0) << " " << x(1) << std::endl; /////////////////////
  return x;
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogInitialDensity(const LatentVariable& latentVariable)
{
  return 0.0; //static_cast<unsigned int>(R::rpois(arma::sum(latentVariable)));
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld)
{
  std::cout << "WARNING: evalutation of the log-transition density is not implemented yet! This requires the evalutation of a Poisson-Binomial convolution which may be computationally costly especially for large population sizes" << std::endl;
  return 0;
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable)
{
  return R::dpois(observations_.count_(t), arma::sum(latentVariable), true);
}
/// Samples a single observation according to the observation equation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromObservationEquation(const unsigned int t, Observations& observations, const LatentVariable& latentVariable)
{
  // NOTE: unused here
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
/// TODO: this needs to be added to the model class
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodFirst(LatentPath& latentPath)
{
  double logLike = 0.0;

  // Likelihood based on fecundity data
  for (unsigned int t=0; t<nObservations_; t++)
  {
    logLike += R::dpois(observations_.fecundity_(t,1), observations_.fecundity_(t,0) * modelParameters_.getRho(t), true);
  }
  // Likelihood based on capture--recapture data
  for (unsigned int t=0; t<nObservations_-1; t++)
  {
    // TODO: we need to only use the columns that are to the right of the main diagonal!
    logLike += logMultinomialDensity(observations_.getCapRecapFemaleFirst(t), observations_.getReleasedFemaleFirst(t), modelParameters_.getQFemaleFirst(t));
    logLike += logMultinomialDensity(observations_.getCapRecapFemaleAdult(t), observations_.getReleasedFemaleAdult(t), modelParameters_.getQFemaleAdult(t)); 
    logLike += logMultinomialDensity(observations_.getCapRecapMaleFirst(t), observations_.getReleasedMaleFirst(t), modelParameters_.getQMaleFirst(t)); 
    logLike += logMultinomialDensity(observations_.getCapRecapMaleAdult(t), observations_.getReleasedMaleAdult(t), modelParameters_.getQMaleAdult(t)); 
  }
  return logLike;
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
  for (unsigned int n=0; n<getNParticles(); n++)
  {
    logWeights(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]);
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
  for (unsigned int n=0; n<getNParticles(); n++)
  {
    logWeights(n) += model_.evaluateLogObservationDensity(0, particlesNew[n]);
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
