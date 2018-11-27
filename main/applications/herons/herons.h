/// \file
/// \brief Generating data and performing inference in "Herons" model. 
///
/// This file contains the functions for implementing the abstract model
/// class for the "Herons" model.

#ifndef __HERONS_H
#define __HERONS_H

#include <stdlib.h> // provides system("pause")
#include "main/templates/dynamic/stateSpace/stateSpace.h"
#include "main/algorithms/mcmc/Mcmc.h"
#include "main/rng/gaussian.h"


// [[Rcpp::depends("RcppArmadillo")]]

/// Type of model for the productivity rate.
enum ModelType 
{ 
  MODEL_CONSTANT_PRODUCTIVITY = 0,         // productivity rate constant over time
  MODEL_REGRESSED_ON_FDAYS,                // productivity rates logistically regressed on fDays,
  MODEL_DIRECT_DENSITY_DEPENDENCE,         // direct density dependence (log-productivity rates specified as a linear function of abundance),
  MODEL_THRESHOLD_DEPENDENCE_OBSERVATIONS, // threshold dependence (of the productivity rate) on the observed heron counts (thresholds unknown),
  MODEL_THRESHOLD_DEPENDENCE_TRUE_COUNTS,  // threshold dependence (of the productivity rate) on the true heron counts (thresholds unknown).
  MODEL_MARKOV_SWITCHING                   // a Markov-regime switching state-space model
};

/// Type of prior distribution on the initial latent state.
enum InitialDistributionType
{ 
  INITIAL_DISTRIBUTION_POISSON = 0,        // product of Poisson distributions whose means are estimated by the algorithm (potentially approximated by product of Gaussians),
  INITIAL_DISTRIBUTION_NEGATIVE_BINOMIAL,  // product of negative binomials distributions with known parameters (potentially approximated by product of Gaussians),
  INITIAL_DISTRIBUTION_DISCRETE_UNIFORM    // product of (discrete) uniform distributions with known support.
};

/// Type of observation equation associated with the state-space model.
enum ObservationEquationType
{ 
  OBSERVATION_EQUATION_POISSON = 0,        // Poisson measurement error (potentially approximated by a Gaussian),
  OBSERVATION_EQUATION_NEGATIVE_BINOMIAL,  // negative binomial measurement error whose overdispersion parameter is estimated by the algorithm (potentially approximated by a Gaussian),
};


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
    
    ///////////////////////////////////////////////////////////////////////////
    // Storing the parameters
    ///////////////////////////////////////////////////////////////////////////
    theta_  = theta;

    // Regression coefficients for the specification of the recovery probabilities:
    alpha0_ = theta(0);
    beta0_  = theta(1);
    lambda_ = inverseLogit(alpha0_ + beta0_ * timeNormCovar_); 
    
    // Regression coefficients for the specification of the survival probabilities:
    alpha_.set_size(nAgeGroups_);
    beta_.set_size(nAgeGroups_);
    alpha_  = theta(arma::span(2,1+nAgeGroups_));
    beta_   = theta(arma::span(2+nAgeGroups_,1+2*nAgeGroups_));
    
    // Parameters for the measurement equation
    if (observationEquationType_ == OBSERVATION_EQUATION_NEGATIVE_BINOMIAL)
    {
      omega_ = theta(2+2*nAgeGroups_);
      kappa_ = inverseLogit(omega_);
    }
    else
    {
      kappa_ = 1.0;
    }
    
    // Parameters for the distribution of the state at time $0$:
    if (initialDistributionType_ == INITIAL_DISTRIBUTION_POISSON)
    {
      delta0_ = theta(nModelIndependentParameters_-2);
      delta1_ = theta(nModelIndependentParameters_-1);
      chi0_   = std::exp(delta0_);
      chi1_   = std::exp(delta1_);
      m0_.fill(chi0_);
      m0_(nAgeGroups_-1) = chi1_;
      C0_ = arma::diagmat(m0_);
    }
    
    // Parameters for the step function governing the productivity rate (in some models)
    rho_.set_size(nObservationsCount_-1);
    
    if (modelType_ == MODEL_REGRESSED_ON_FDAYS)
    {
      // Coefficients for logistically regressing the productivity rate on fDays:
      gamma0_ = theta(nModelIndependentParameters_);
      gamma1_ = theta(nModelIndependentParameters_+1);
      rho_    = arma::exp(gamma0_ + gamma1_ * fDaysCovar_); // vector of length nObservationsCount_; 
    }
    else if (modelType_ == MODEL_CONSTANT_PRODUCTIVITY)
    {
      psi_ = theta(nModelIndependentParameters_); // the log-productivity rate
      rho_.fill(std::exp(psi_)); // the productivity rate
    }
    else if (modelType_ == MODEL_DIRECT_DENSITY_DEPENDENCE)
    {
      // Coefficients for regressing the log-productivity rate on the observations:
      epsilon0_ = theta(nModelIndependentParameters_);
      epsilon1_ = theta(nModelIndependentParameters_+1);
      
      // NOTE: for this model, rho_ is specified at the the first step of the SMC filter 
      // or in the Kalman filter implemented in evaluateLogMarginalLikelihood().
      setRho(arma::exp(epsilon0_ + epsilon1_ * countDataNormalised_));   
    }
    else if (modelType_ == MODEL_THRESHOLD_DEPENDENCE_OBSERVATIONS)
    {
      // NOTE: for threshold dependence on the true states,
      // $\rho_t$ must be specified individually for each particle; for threshold dependence
      // on the observations, $\rho_t$ is specified in sampleInitialParticles() or in the Kalman filter
      // implemented in evaluateLogMarginalLikelihood().
      
      zeta_.set_size(nLevels_);
      nu_.set_size(nLevels_);
      eta_.set_size(nLevels_);
//       tau_.set_size(nLevels_); // tau_ eventually has length nLevels_ - 1!
      
      zeta_  = theta(arma::span(nModelIndependentParameters_, nModelIndependentParameters_+1+nLevels_-2));
      eta_   = theta(arma::span(nModelIndependentParameters_+1+nLevels_-1, nModelIndependentParameters_+1+2*(nLevels_-1)));

      tau_ = arma::min(countData_) + arma::cumsum(arma::exp(eta_)) / arma::accu(arma::exp(eta_)) * static_cast<double>(arma::max(countData_) - arma::min(countData_));
      tau_ = tau_(arma::span(0, tau_.size()-1)); // because the last entry should be equal to the largest count, anyway!

      if (nLevels_ > 1)
      {
        for (unsigned int k=nLevels_-1; k>0; k--)
        {
          nu_(k) = arma::accu(arma::exp(zeta_(arma::span(k,nLevels_-1))));
        }
      }
      nu_(0) = arma::accu(arma::exp(zeta_));
      
      for (unsigned int t=0; t<nObservationsCount_-1; t++)
      {
        computeRhoFromStepFun(t, countData_(t));
      }
    

    }
    else if (modelType_ == MODEL_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
    {
      // NOTE: for threshold dependence on the true states,
      // $\rho_t$ must be specified individually for each particle; for threshold dependence
      // on the observations, $\rho_t$ is specified in sampleInitialParticles() or in the Kalman filter
      // implemented in evaluateLogMarginalLikelihood().
      
      // WARNING: threshold dependence on the true counts does no longer work because we are now rescaling the 
      // thresholds in such as way that they do no fall outside the range of observations. And this clearly 
      // cannot be done a-priori.
      
      zeta_.set_size(nLevels_);
      nu_.set_size(nLevels_);
      eta_.set_size(nLevels_-1);
      tau_.set_size(nLevels_-1);
      
      zeta_  = theta(arma::span(nModelIndependentParameters_, nModelIndependentParameters_+1+nLevels_-2));
      eta_   = theta(arma::span(nModelIndependentParameters_+1+nLevels_-1, nModelIndependentParameters_+1+2*(nLevels_-1)-1));

      tau_ = arma::cumsum(arma::exp(eta_));
     

      if (nLevels_ > 1)
      {
        for (unsigned int k=nLevels_-1; k>0; k--)
        {
          nu_(k) = arma::accu(arma::exp(zeta_(arma::span(k,nLevels_-1))));
        }
      }
      nu_(0) = arma::accu(arma::exp(zeta_));

    }
    else if (modelType_ == MODEL_MARKOV_SWITCHING)
    {
      
      zeta_.set_size(nLevels_);
      nu_.set_size(nLevels_);
            
      zeta_  = theta(arma::span(nModelIndependentParameters_, nModelIndependentParameters_+1+nLevels_-2));
      nu_    = arma::cumsum(arma::exp(zeta_)); // NOTE: the levels are now increasing rather than decreasing!
      
      varpi_.set_size(nLevels_, nLevels_);
      P_.set_size(nLevels_, nLevels_);
      for (unsigned int k=0; k<nLevels_; k++)
      {
        varpi_.row(k) = arma::trans(theta_(arma::span(nModelIndependentParameters_+1+nLevels_-1+k*nLevels_, nModelIndependentParameters_+1+nLevels_-2+nLevels_+k*nLevels_)));
        P_.row(k) = arma::exp(varpi_.row(k)) / arma::accu(arma::exp(varpi_.row(k)));
      }
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Compute the multinomial probabilites for the ring-recovery data:
    ///////////////////////////////////////////////////////////////////////////

    // Computing the matrix of survival probabilities.
    phi_.set_size(nAgeGroups_, nObservationsCount_-1);
    for (unsigned int a=0; a<nAgeGroups_; a++)
    {
      phi_.row(a) = arma::trans(inverseLogit(alpha_(a) + beta_(a) * fDaysCovar_(arma::span(1,fDaysCovar_.size()-1))));
    }
    
    // Computing the matrix of multinomial ring-recovery probabilities.
    Q_.zeros(nObservationsRing_, nObservationsRing_+1);
    
    // Auxiliary time indices 
    unsigned int t1 = t1Ring_ - t1Count_ + 1;

    double logQ;
    for (unsigned int i=0; i<Q_.n_rows; i++)
    { 
      for (unsigned int j=i; j<Q_.n_cols-1; j++)
      {
        logQ = 0.0;
        for (unsigned int k=i; k<j; k++)
        {
          logQ += std::log(phi_(std::min(k-i, nAgeGroups_-1), t1+k-1));
        }
        logQ += std::log(1.0 - phi_(std::min(j-i, nAgeGroups_-1),t1+j-1)) + std::log(lambda_(t1+j-1));
        Q_(i,j) = std::exp(logQ);
      }
      Q_(i,Q_.n_cols-1) = 1 - arma::accu(Q_.row(i));
    }
    
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyp)
  {
    modelType_ = static_cast<ModelType>(hyp(0));
    
    t1Count_ = static_cast<unsigned int>(hyp(1)); // first year for which count-data observations are available
    t2Count_ = static_cast<unsigned int>(hyp(2)); // last year for which count-data observations are available
    t1Ring_  = static_cast<unsigned int>(hyp(3)); // first year for which ring-recovery observations are available
    t2Ring_  = static_cast<unsigned int>(hyp(4)); // last year for which ring-recovery observations are available
    
    nObservationsCount_ = t2Count_ - t1Count_ + 1; // number of count-data observations.
    nObservationsRing_  = t2Ring_  - t1Ring_  + 1; // number of capture-recapture observation periods.
    
    dimTheta_      = static_cast<unsigned int>(hyp(5)); // number of model parameters to be estimated
    meanHyper_     = hyp(arma::span(6,dimTheta_+5)); // mean vector for the multivariate Gaussian prior
    sdHyper_       = hyp(arma::span(dimTheta_+6,2*dimTheta_+5)); // vector of standard deviations for the multivariate Gaussian prior
    nAgeGroups_    = static_cast<unsigned int>(hyp(2*dimTheta_+6)); // number of age groups
    nLevels_       = static_cast<unsigned int>(hyp(2*dimTheta_+7)); // number of levels for the threshold/regime-switching models
    
    initialDistributionType_ = static_cast<InitialDistributionType>(hyp(2*dimTheta_+8)); // type of prior on the initial latent state
    observationEquationType_ = static_cast<ObservationEquationType>(hyp(2*dimTheta_+9)); // type of measurement equation
    
    // Parameters on the prior of the initial latent state under various distributional assumptions.
    minHyperInit0_ = static_cast<unsigned int>(hyp(2*dimTheta_+10)); // minimum of the support of the discrete uniform prior on the first nAgeGroups-1 age groups
    minHyperInit1_ = static_cast<unsigned int>(hyp(2*dimTheta_+11)); // minimum of the support of the discrete uniform prior on the last age group
    maxHyperInit0_ = static_cast<unsigned int>(hyp(2*dimTheta_+12)); // maximum of the support of the discrete uniform prior on the first nAgeGroups-1 age groups
    maxHyperInit1_ = static_cast<unsigned int>(hyp(2*dimTheta_+13)); // maximum of the support of the discrete uniform prior on the last age group
    negativeBinomialSizeHyperInit0_ = hyp(2*dimTheta_+14); // size parameter of the negative-binomial prior on the first nAgeGroups-1 age groups
    negativeBinomialSizeHyperInit1_ = hyp(2*dimTheta_+15); // size parameter of the negative-binomial prior on the last age group
    negativeBinomialProbHyperInit0_ = hyp(2*dimTheta_+16); // probability parameter of the negative-binomial prior on the first nAgeGroups-1 age groups
    negativeBinomialProbHyperInit1_ = hyp(2*dimTheta_+17); // probability parameter of the negative-binomial prior on the last age group
    
    timeNormCovar_ = hyp(arma::span(2*dimTheta_+18, 2*dimTheta_+16 + nObservationsCount_)); // "normalised" years (i.e. nObservationsCount - 1 elements)
    fDaysCovar_    = hyp(arma::span(2*dimTheta_+17 + nObservationsCount_, 2*dimTheta_+16 + 2*nObservationsCount_)); // (i.e. nObservationsCount elements)
    countData_     = arma::conv_to<arma::uvec>::from(hyp(arma::span(2*dimTheta_+17 + 2*nObservationsCount_, 2*dimTheta_+16 + 3*nObservationsCount_)));
    
    countDataNormalised_ = hyp(arma::span(2*dimTheta_+17 + 2*nObservationsCount_, 2*dimTheta_+16 + 3*nObservationsCount_));
    double countDataMean = arma::accu(countDataNormalised_) / countDataNormalised_.size();
    double countDataSd = std::sqrt(arma::accu(arma::pow(countDataNormalised_, 2.0)) - std::pow(countDataMean, 2.0));
    countDataNormalised_ = (countDataNormalised_ - countDataMean)/countDataSd;
    
    nModelIndependentParameters_ = 2+2*nAgeGroups_; // number of model parameters which are identical for all model indices

    if (getInitialDistributionType() == INITIAL_DISTRIBUTION_POISSON) // i.e. if the prior on the initial state is Poisson
    {
      nModelIndependentParameters_ = nModelIndependentParameters_ + 2;
    } 
    if (getObservationEquationType() == OBSERVATION_EQUATION_NEGATIVE_BINOMIAL) // i.e. if the observation equation uses a negative-binomial distribution
    {
      nModelIndependentParameters_ = nModelIndependentParameters_ + 1;
    }
    
    m0_.set_size(nAgeGroups_);
    C0_.zeros(nAgeGroups_, nAgeGroups_); 
    
    if (initialDistributionType_ == INITIAL_DISTRIBUTION_NEGATIVE_BINOMIAL) // i.e. if the initial distribution is a negative binomial
    { 
      m0_.fill(negativeBinomialSizeHyperInit0_ * (1.0 - negativeBinomialProbHyperInit0_) / negativeBinomialProbHyperInit0_);
      
      m0_(nAgeGroups_-1) = negativeBinomialSizeHyperInit1_ * (1.0 - negativeBinomialProbHyperInit1_) / negativeBinomialProbHyperInit1_;
      
      for (unsigned int a=0; a<nAgeGroups_-1; a++)
      {
        C0_(a,a) = m0_(a) / negativeBinomialProbHyperInit0_;
      }
      C0_(nAgeGroups_-1, nAgeGroups_-1) = m0_(nAgeGroups_-1) / negativeBinomialProbHyperInit1_;
    }
    else if (initialDistributionType_ == INITIAL_DISTRIBUTION_DISCRETE_UNIFORM) // i.e. if the initial distribution is a (discrete) uniform
    { 
      m0_.fill(static_cast<double>(maxHyperInit0_ + minHyperInit0_) / 2.0);
      m0_(nAgeGroups_-1) = static_cast<double>(maxHyperInit1_ + minHyperInit1_) / 2.0;
      for (unsigned int a=0; a<nAgeGroups_-1; a++)
      {
        C0_(a,a) = std::pow(static_cast<double>(maxHyperInit0_- minHyperInit0_), 2.0) / 12.0;
      }
      C0_(nAgeGroups_-1, nAgeGroups_-1) = std::pow(static_cast<double>(maxHyperInit1_-minHyperInit1_), 2.0) / 12.0;
    }
  }
  
  /// Computes the productivity rate for one particular year in the case
  /// that rho follows a step function (based on either the true counts
  /// or on the observations).
  void computeRhoFromStepFun(const unsigned int t, const unsigned int population)
  {
    if (nLevels_ == 1)
    {
      rho_(t) = nu_(0);
    }
    else
    {
      if (tau_(nLevels_-2) <= population)
      {
        rho_(t) = nu_(nLevels_-1);
      }
      else
      {
        unsigned int idx = arma::as_scalar(arma::find(tau_ > population, 1, "first"));
        rho_(t) = nu_(idx);
      }
    }
  }
  
  /// Iterates a single step of the Kalman filter for some productivity rate rho.
  double iterateKalmanFilter(
    const unsigned int t, 
    const double observation, 
    const double rho,
    arma::colvec& mU, 
    arma::mat& CU
  )
  {
//     std::cout << "START: iteration " << t << " of the KF" << std::endl;
    arma::mat A(nAgeGroups_, nAgeGroups_, arma::fill::zeros);
    arma::rowvec C(nAgeGroups_, arma::fill::ones);
    C(0) = 0;
    
    // Predictive mean and covariance matrix
    arma::colvec mP(nAgeGroups_);
    arma::mat    CP(nAgeGroups_, nAgeGroups_);
    
    // Mean and covariance matrix of the incremental likelihood
    double mY;
    double CY;
    
    // Auxiliary quantities
    arma::colvec B2Diag(nAgeGroups_);
    B2Diag.zeros();
    arma::mat kg;
    
    // Prediction step
    if (t > 0) 
    {
      for (unsigned int a=1; a<nAgeGroups_; a++)
      {
        A(0,a) = rho * getPhi(0,t-1);
      }
      for (unsigned int a=1; a<nAgeGroups_; a++)
      {
        A(a,a-1) = getPhi(a,t-1);
      }
      A(nAgeGroups_-1,nAgeGroups_-1) = getPhi(nAgeGroups_-1,t-1);

      B2Diag(0) = rho * getPhi(0,t-1) * arma::accu(mU(arma::span(1,nAgeGroups_-1)));
      for (unsigned int a=1; a<nAgeGroups_-1; a++)
      {
        B2Diag(a) = getPhi(a,t-1) * (1.0 - getPhi(a,t-1)) * mU(a-1);
      }
      B2Diag(nAgeGroups_-1) = getPhi(nAgeGroups_-1,t-1) * (1.0 - getPhi(nAgeGroups_-1,t-1)) * arma::accu(mU(arma::span(nAgeGroups_-2,nAgeGroups_-1)));
      
      mP = A * mU;
      CP = A * CU * A.t() + arma::diagmat(B2Diag);    
    } 
    else 
    {
      mP = getM0();
      CP = getC0(); 
      
      
    }
    // Likelihood step
    mY = arma::conv_to<double>::from(C * mP);
    CY = arma::conv_to<double>::from(C * CP * C.t() + arma::accu(mP(arma::span(1,nAgeGroups_-1))) / getKappa());
 
    // Update step
    kg = (CP * C.t()) / CY;
    mU = mP + kg * (observation - mY);
    CU = CP - kg * C * CP;

    // Adding the incremental log-marginal likelihood
    return arma::as_scalar(gaussian::evaluateDensityUnivariate(observation, mY, CY, false, true));
  }
  /// Runs a full Kalman filter and Smoother
  void runKalman(
    const arma::uvec& y,
    arma::mat& mS,
    arma::cube& CS,
    arma::colvec& mY,
    arma::colvec& CY
  )
  {
    unsigned int T = nObservationsCount_;
    unsigned int A = nAgeGroups_;
    
    // The transition matrices (these depend on rho!)
    arma::cube transMat(A, A, T, arma::fill::zeros);
    
    // The matrix for the observation equation.
    arma::rowvec C(A, arma::fill::ones);
    C(0) = 0;
    
    // Predicted means and covariance matrices
    arma::mat  mP(A, T);
    arma::cube CP(A, A, T);
    
    // Updated means and covariance matrices
    arma::mat  mU(A, T);
    arma::cube CU(A, A, T);
    
    // Smoothed means and covariance matrices
    mS.set_size(A, T);
    CS.set_size(A, A, T);

    // Auxiliary quantities
    arma::colvec B2Diag(A);
    B2Diag.zeros();
    arma::mat kg;
    
    // Prediction step
    for (unsigned int t=0; t<T; t++)
    {
      if (t > 0) 
      {
        for (unsigned int a=1; a<A; a++)
        {
          transMat.slice(t)(0,a) = getRho(t-1) * getPhi(0,t-1);
        }
        for (unsigned int a=1; a<A; a++)
        {
          transMat.slice(t)(a,a-1) = getPhi(a,t-1);
        }
        transMat.slice(t)(A-1,A-1) = getPhi(A-1,t-1);

        B2Diag(0) = getRho(t-1) * getPhi(0,t-1) * arma::accu(mU(arma::span(1,A-1), arma::span(t-1,t-1)));
        for (unsigned int a=1; a<A-1; a++)
        {
          B2Diag(a) = getPhi(a,t-1) * (1.0 - getPhi(a,t-1)) * mU.col(t-1)(a-1);
        }
        B2Diag(A-1) = getPhi(A-1,t-1) * (1.0 - getPhi(A-1,t-1)) * arma::accu(mU(arma::span(A-2,A-1), arma::span(t-1,t-1)));
        
        mP.col(t)   = transMat.slice(t) * mU.col(t-1);
        CP.slice(t) = transMat.slice(t) * CU.slice(t-1) * transMat.slice(t).t() + arma::diagmat(B2Diag);    
      } 
      else 
      {
        mP.col(t)   = getM0();
        CP.slice(t) = getC0();        
      }
      // Likelihood step
      mY(t) = arma::conv_to<double>::from(C * mP.col(t));
      CY(t) = arma::conv_to<double>::from(C * CP.slice(t) * C.t() + arma::accu(mP(arma::span(1,A-1), arma::span(t,t))) / getKappa());
  
      // Update step
      kg = (CP.slice(t) * C.t()) / CY(t);
      mU.col(t) = mP.col(t) + kg * (y(t) - mY(t));
      CU.slice(t) = CP.slice(t) - kg * C * CP.slice(t);
    }
    
    /// Kalman smoother
    arma::mat Ck(A, A);
    mS.col(T-1)   = mU.col(T-1);
    CS.slice(T-1) = CU.slice(T-1);
    
    for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
    {
      Ck          = CU.slice(t) * transMat.slice(t+1).t() * arma::inv(CP.slice(t+1));
      mS.col(t)   = mU.col(t)   + Ck * (mS.col(t+1)   - mP.col(t+1));
      CS.slice(t) = CU.slice(t) + Ck * (CS.slice(t+1) - CP.slice(t+1)) * Ck.t();
    }
  }
  
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
  /// Returns the number of age groups.
  unsigned int getNAgeGroups() const {return nAgeGroups_;}
  /// Returns the number of levels in the step function for the productivity.
  unsigned int getNLevels() const {return nLevels_;}
  /// Returns the dimension of the latent state.
//   unsigned int getDimLatentVariable() const {return dimLatentVariable_;}
  /// Returns the number of count-data observation periods.
  unsigned int getNObservationsCount() const {return nObservationsCount_;}
  /// Returns the number of ring-recovery data observation periods.
  unsigned int getNObservationsRing() const {return nObservationsRing_;}
  
  /// Returns the $i$th parameter.
  double getTheta(const unsigned int i) const {return theta_(i);}
  /// Returns the full parameter vector.
  arma::colvec getTheta() const {return theta_;}
  /// Returns the parameter $\kappa$ used in the negative-binomial distribution of the measurement errors.
  double getKappa() const {return kappa_;}
  /// Returns the Poisson mean of the distribution of the number of $a$th years ($a < A$) at time 0.
  double getChi0() const {return chi0_;}
  /// Returns the Poisson mean of the distribution of the number of adults at time 0.
  double getChi1() const {return chi1_;}
  /// Returns the survival probabilities.
  double getPhi(const unsigned int a, const unsigned int t) const {return phi_(a,t);}
  /// Returns the productivity rate.
  double getRho(const unsigned int t) const {return rho_(t);}
  /// Returns the productivity rate.
  arma::colvec getRho() const {return rho_;}
  /// Returns the survival probabilities.
  arma::mat getPhi() const {return phi_;}
  /// Returns the recovery probabilities.
  arma::colvec getLambda() const {return lambda_;}
  /// Returns the $i$th row of the multinomial ring-recovery probabilities
  arma::rowvec getQ(const unsigned int i) const {return Q_.row(i);}
  /// Returns the mean of the approximate Gaussian prior on the initial state.
  arma::colvec getM0() const {return m0_;}
  /// Returns the covariance matrix of the approximate Gaussian prior on the initial state.
  arma::mat getC0() const {return C0_;}
  
  /// Returns the first regression parameter in the direct-density dependence case.
  double getEpsilon0() const {return epsilon0_;}
  /// Returns the second regression parameter in the direct-density dependence case.
  double getEpsilon1() const {return epsilon1_;}
  
  /// Returns the $i$th row of the transition matrix for the 
  /// regime indicator variable
  arma::rowvec getP(const unsigned int i) const {return P_.row(i);}
  /// Returns the element $(i,j)$ of the transition matrix for the 
  /// regime indicator variable
  double getP(const unsigned int i, const unsigned int j) const {return P_(i,j);}
  
  /// Returns the maximum of the support of the discrete uniform distribution for the initial population counts for age groups $1,\dotsc, A-1$
  int getMaxHyperInit0() const {return maxHyperInit0_;}
  /// Returns the maximum of the support of the discrete uniform distribution for the initial population counts for age group $A$
  int getMaxHyperInit1() const {return maxHyperInit1_;}
  /// Returns the minimum of the support of the discrete uniform distribution for the initial population counts for age groups $1,\dotsc, A-1$
  int getMinHyperInit0() const {return minHyperInit0_;}
  /// Returns the minimum of the support of the discrete uniform distribution for the initial population counts for age group $A$
  int getMinHyperInit1() const {return minHyperInit1_;}
  
  /// Returns the size parameter of the negative-binomial prior on the first nAgeGroups-1 age groups.
  double getNegativeBinomialSizeHyperInit0() const {return negativeBinomialSizeHyperInit0_;}
  /// Returns the size parameter of the negative-binomial prior on the last age group.
  double getNegativeBinomialSizeHyperInit1() const {return negativeBinomialSizeHyperInit1_;}
  /// Returns the probability parameter of the negative-binomial prior on the first nAgeGroups-1 age groups.
  double getNegativeBinomialProbHyperInit0() const {return negativeBinomialProbHyperInit0_;}
  /// Returns the probability parameter of the negative-binomial prior on the last age group.
  double getNegativeBinomialProbHyperInit1() const {return negativeBinomialProbHyperInit1_;}
  
  /// Returns the type of prior on the initial latent state used by the model.
  InitialDistributionType getInitialDistributionType() const {return initialDistributionType_;}
  /// Returns the type of measurement equation used by the model.
  ObservationEquationType getObservationEquationType() const {return observationEquationType_;}
  
  /// Specifies the productivity rate.
  void setRho(const arma::colvec& rho) {rho_ = rho;}
  /// Specifies the productivity rate.
  void setRho(const unsigned int t, const double rho) {rho_(t) = rho;}
  /// Returns the $i$th level of the productivity rates.
  double getNu(const unsigned int i) const {return nu_(i);}
  
  
private:
  
  // Unknown model parameters
  arma::colvec theta_; // the full parameter vector
  double omega_, kappa_; // parameters for the measurement equation
  double epsilon0_, epsilon1_, gamma0_, gamma1_; // regression parameters for the productivity rates
  double delta0_, delta1_, chi0_, chi1_; // parameters for the distribution of the state at time 0
  double psi_; // constant log-productivity rate
  arma::colvec zeta_, eta_, nu_, tau_; // parameters for the step functions
  arma::colvec rho_; // productivity rates
  arma::colvec alpha_, beta_; // regression parameters for the survival probabilities
  double alpha0_, beta0_; // regression parameters for the recovery probabilities
  arma::colvec lambda_; // recovery probabilities
  arma::mat phi_; // a (nAgeGroups, nObservationsRing)-matrix of survival probabilities; NOTE: the first column contains the survival probabilties $\phi_{a,0}$.
  arma::mat Q_; // multinomial probabilities for the ring-recovery data
  arma::mat P_, varpi_; // (normalised/unnormalised) transition matrix for the latent regime-switches
  arma::colvec m0_; // mean of the approximate Gaussian prior on the initial state.
  arma::mat C0_; // covariance matrix of the approximate Gaussian prior on the initial state.

  // Known hyperparameters for the prior distribution:
  ModelType modelType_; // index for model specification
  unsigned int dimTheta_; // length of the parameter vector theta
  arma::colvec meanHyper_, sdHyper_; // means and standard deviations of the Gaussian priors on all parameters
  int minHyperInit0_, minHyperInit1_, maxHyperInit0_, maxHyperInit1_; // minimum/maximum of the support of the discrete uniform distribution for the initial population counts (for age groups $1,\dotsc, A-1$ and $A$, repectively)
  double negativeBinomialSizeHyperInit0_, negativeBinomialSizeHyperInit1_; // size parameters of the negative binomial distribution for the initial population counts (for age groups $1,\dotsc, A-1$ and $A$, repectively)
  double negativeBinomialProbHyperInit0_, negativeBinomialProbHyperInit1_; // probability parameters of the negative binomial distribution for the initial population counts (for age groups $1,\dotsc, A-1$ and $A$, repectively)
  
  unsigned int nModelIndependentParameters_; // number of parameters which do not depend on the particular model type.
  
  InitialDistributionType initialDistributionType_; // type of prior on the initial latent state.
  ObservationEquationType observationEquationType_; // type of conditional distribution of the observations conditional on the latent states.

  unsigned int nAgeGroups_; // number of age groups
  unsigned int nLevels_;    // number of levels in the step function for productivity

  unsigned int nObservationsRing_;  // number of years for which ring-recovery observations are available
  unsigned int nObservationsCount_; // number of years for which count-data observations are available
  unsigned int t1Count_, t2Count_;  // first, last year for which count-data observations are available
  unsigned int t1Ring_, t2Ring_;    // first, last year for which ring-recovery observations are available
  
  // Other known covariates:
  arma::colvec timeNormCovar_; // length-(nObservationsCount_-1) vector of "normalised" years (1928--1997)
  arma::colvec fDaysCovar_;    // vector of length nObservationsCount_-1; from 1928(!) [we don't use fDays_{1927} and have deleted it from the data file!]to 1997
  arma::uvec countData_; // the count-data (with its mean subtracted and divided by its standard deviation)
  arma::colvec countDataNormalised_; // the count-data (with its mean subtracted and divided by its standard deviation)
  
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

  
  // All of the following parameters/functions are only used if we use the lookahead
  // proposal which uses a Kalman filter/smoother to construct the proposal kernel at time $t$.
  
  /// Determines some auxiliary parameters at the start of the algorithm.
  void setLookaheadParameters(const unsigned int K, const unsigned int nAgeGroups) 
  {
    // TODO: K needs to be included in the hyperparamters!
    K_ = K;
    C_.ones(nAgeGroups);
    C_(0) = 0;
  }
  /// Generates all particles at the $t$th iteration of the SMC algorithm.
  void proposeAllParticles(
    std::vector<arma::uvec>& particlesNew,
    const std::vector<arma::uvec>& particlesOld,
    const unsigned int t, // current time step
    const unsigned int T, // number of count-data observations
    const unsigned int nParticles, // number of particles 
    const unsigned int nAgeGroups, 
    const double kappa,
    const double chi0,
    const double chi1,
    const arma::mat& phi,
    const arma::colvec& rho,
    const arma::uvec& y
  )
  {
//     std::cout << "start setKalmanParameters()" << std::endl;
    setKalmanParameters(t, T, nAgeGroups, kappa, chi0, chi1, phi, rho, y);
//         std::cout << "end setKalmanParameters()" << std::endl;
        
    logProposalDensity_.zeros(nParticles);
        
//         std::cout << "start proposing particles" << std::endl;
    for (unsigned int n=0; n<nParticles; n++)
    {
      proposeParticle(particlesNew[n], particlesOld[n], t, n, nAgeGroups, y);
    }
  }
  /// Returns the particle weights.
  double getLogProposalDensity(const unsigned int n) 
  {
    return logProposalDensity_(n);
  }

  
private:
  
  /// Specifies the transition matrix for the state-equation
  /// in the approximate linear Gaussian state-space model.
  void setA(
    const unsigned int k, 
    const unsigned int t,
    const unsigned int nAgeGroups,
    const arma::mat& phi,
    const arma::colvec& rho
  )
  {
    if (t > 0 || k > 0)
    {
      A_[k] = arma::diagmat(phi(arma::span(1,nAgeGroups-1), arma::span(t+k-1, t+k-1)), -1);
      A_[k](nAgeGroups-1,nAgeGroups-1) = phi(nAgeGroups-1, t+k-1);
      A_[k].row(0).fill(rho(t+k-1) * phi(0, t+k-1));
      A_[k](0,0) = 0;
    }
  }
  /// Specifies the covariance matrix for the state-equation
  /// in the approximate linear Gaussian state-space model.
  void setB(
    const unsigned int k, 
    const unsigned int t,
    const unsigned int nAgeGroups,
    const arma::mat& phi,
    const arma::colvec& rho,
    const arma::uvec y
  )
  {
    if (t > 0 || k > 0)
    {
      arma::colvec v(nAgeGroups);
      v(0) = rho(t+k-1) * phi(0, t+k-1) * y(t+k-1);
      for (unsigned int a=1; a<nAgeGroups-1; a++)
      {
        v(a) = (phi(a, t+k-1) * (1 - phi(a, t+k-1))* y(t+k-1)) / nAgeGroups; 
      }
      v(nAgeGroups-1) = phi(nAgeGroups-1, t+k-1) * (1 - phi(nAgeGroups-1, t+k-1))* y(t+k-1);
      
      BBT_[k] = arma::diagmat(v); 
      B_[k]   = arma::diagmat(arma::sqrt(BBT_[k].diag()));
    }
  }
  /// Runs a Kalman filter/smoother starting at time $t$.
  void runKalmanSmoother(
    const unsigned int t, 
    const arma::uvec&  particleOld, 
    const unsigned int nAgeGroups, 
    const arma::uvec& y
  )
  {
  
    ///////////////////////////////////////////////////////////////////////////
    // Forward filtering
    ///////////////////////////////////////////////////////////////////////////
    
    std::vector<arma::colvec> mP, mU; // prediction and updated means
    std::vector<arma::mat> CP, CU; // prediction and updated covariance matrices
  
    mP.resize(L_-t+1);
    mU.resize(L_-t+1);
    CP.resize(L_-t+1);
    CU.resize(L_-t+1);
    
    // Prediction step at time $t$:
    if (t > 0)
    {
      mP[0] = A_[0] * particleOld;
      CP[0] = BBT_[0];
    }
    else if (t==0) 
    {
      mP[0] = chi_;
      CP[0] = arma::diagmat(chi_);
    }
    
    // Update step at time $t$:
    
//     std::cout << "Update step at time t" << std::endl;
    
    double z = arma::as_scalar(y(t) - C_ * mP[0]);
    double S = arma::as_scalar(C_ * CP[0] * C_.t() * DDT_(0));
    arma::colvec G = CP[0] * C_.t() / S;
    
    mU[0] = mP[0] + G * z;
    CU[0] = (arma::eye(nAgeGroups, nAgeGroups) - G * C_) * CP[0];
    
    
//           if (t <=45)
//       {
//     std::cout << "mP[0] " << mP[0].t() << "; mU[0] " << mU[0].t()  << std::endl;
//                std::cout << "CP[0] " << CP[0] << "; CU[0] " << CU[0]  << std::endl;
//       }
    
    // Prediction/update steps further into the future:
    for (unsigned int k=1; k<A_.size(); k++)
    {
      // Prediction step:
      
//        std::cout << "Prediction step at time " << t+k << std::endl;
      mP[k] = A_[k] * mU[k-1];
      CP[k] = A_[k] * CU[k-1] * A_[k].t() + BBT_[k];
      
      // Update Step:
//              std::cout << "Update step at time " << t+k << std::endl;
      z = arma::as_scalar(y(t+k) - C_ * mP[k]);
      S = arma::as_scalar(C_ * CP[k] * C_.t() * DDT_(k));
      G = CP[k] * C_.t() / S;
      mU[k] = mP[k] + G * z;
      CU[k] = (arma::eye(nAgeGroups, nAgeGroups) - G * C_) * CP[k];
      
//       if (t <=4)
//       {
//           std::cout << "mP[k] " << mP[k].t() << "; mU[k] " << mU[k].t()  << " ";
// //            std::cout << "CP[k] " << CP[k] << "; CU[k] " << CU[k]  << std::endl;
//       }
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Backward smoothing
    ///////////////////////////////////////////////////////////////////////////
    
    std::vector<arma::colvec> mS; // smoothed means
    std::vector<arma::mat> CS; // smoothed covariance matrices
    mS.resize(L_-t+1);
    CS.resize(L_-t+1);
    arma::mat J;
    mS[mS.size()-1] = mU[mS.size()-1];
    CS[CS.size()-1] = CU[CS.size()-1];
    
    
//                    if (t <=4)
//       {
//           std::cout << "mS[CS.size()-1] " << mS[CS.size()-1].t() << "";
// //            std::cout << "CS[CS.size()-1] " << CS[CS.size()-1] << std::endl;
//       }
//                  std::cout << "start Backward smoothing recursion" << std::endl;
                 
    for (unsigned int k=CS.size()-2; k != static_cast<unsigned>(-1); k--)
    {
      J = (arma::solve(CP[k+1].t(), A_[k+1] * CU[k].t())).t();
      
      mS[k] = mU[k] + J * (mS[k+1] - mP[k+1]);
      CS[k] = CU[k] + J * (CS[k+1] - CP[k+1]) * J.t();
      
//                if (t <=4)
//       {
//           std::cout << "mS[k] " << mS[k].t() << " ";
// //            std::cout << "CS[k] " << CS[k] << std::endl;
//       }
      
    }
  
  
//            std::cout << "end Backward smoothing recursion" << std::endl;
           
    mu_    = mS[0];
             
    sigma_ = CS[0];
    
//       if (t <=10)
//       {
// //           std::cout << "mu_ " << mu_.t() << "";
// //            std::cout << "sigma_ " << sigma_ << std::endl;
//       }
  }
  
  /// Determines some auxiliary parameters 
  /// used by the Kalman filter/smoother based particle 
  /// proposal kernel at the $t$th SMC step.
  void setKalmanParameters(
    const unsigned int t, 
    const unsigned int T, 
    const unsigned int nAgeGroups, 
    const double kappa,
    const double chi0,
    const double chi1,
    const arma::mat& phi,
    const arma::colvec& rho,
    const arma::uvec& y
  )
  {

    
    L_ = std::min(t+K_, T-1);
    
//          std::cout << "K_: " << K_ << std::endl;
//      std::cout << "T: " << T << std::endl;
//     std::cout << "L_: " << L_ << std::endl;
//     std::cout << "kappa: " << kappa << std::endl;
//         std::cout << "chi0: " << chi0 << std::endl;
//             std::cout << "chi1: " << chi1 << std::endl;
    
    A_.resize(L_-t+1);
    B_.resize(L_-t+1);
    BBT_.resize(L_-t+1);
//     D_.set_size(L_-t+1);
//     DDT_.set_size(L_-t+1);
    
    DDT_ = kappa * arma::conv_to<arma::colvec>::from(y(arma::span(t,L_)));
    
//             std::cout << "DDT_: " << DDT_.t() << std::endl;
    
    D_   = arma::sqrt(DDT_);
    

//             std::cout << "D_: " << D_.t() << std::endl;
    
    for (unsigned int k=0; k<A_.size(); k++)
    {
//       std::cout << "setting up A" << std::endl;
      setA(k, t, nAgeGroups, phi, rho);
      
//       std::cout << "A_: " << A_[k] << std::endl;
//             std::cout << "setting up B" << std::endl;
      setB(k, t, nAgeGroups, phi, rho, y);
      
//       std::cout << "B_: " << B_[k] << std::endl;
    }
    
    if (t == 0)
    {
      chi_.set_size(nAgeGroups);
      chi_.fill(chi0);
      chi_(nAgeGroups-1) = chi1;
    }
    
  }

  /// Proposes a particle and calculates the associated log-incremental weight.
  void proposeParticle
  (
    arma::uvec& particleNew, 
    const arma::uvec& particleOld, 
    const unsigned int t, 
    const unsigned int n, 
    const unsigned int nAgeGroups, 
    const arma::uvec& y
  )
  {
    
    runKalmanSmoother(t, particleOld, nAgeGroups, y);
    
    ///////////////////// START: Test using a Poisson proposal //////////////////////
    /*
    particleNew.set_size(nAgeGroups);
    for (unsigned int a=0; a<nAgeGroups; a++)
    {     
      particleNew(a) = R::rpois(mu_(a));
      logProposalDensity_(n) = R::dpois(particleNew(a), mu_(a), true);
    }
    */
    ///////////////////// END: Test using a Poisson proposal //////////////////////
    
   ///////////////////// Start: using the incorrect density //////////////
   /*
    arma::colvec x = mu_ + arma::chol(sigma_) * arma::randn<arma::colvec>(nAgeGroups);
    particleNew.set_size(nAgeGroups);

    for (unsigned int a=0; a<nAgeGroups; a++)
    {     
//          std::cout << "sample single particle for age group " << a << std::endl;
      if (x(a) <= 0)
      {
        particleNew(a) = 0;
      }
      else
      {
        particleNew(a) = ceil(x(a));
      }
      
      if (particleNew(a) > 0)
      {
        logProposalDensity_(n) += R::dnorm(particleNew(a), mu_(a), std::sqrt(sigma_(a,a)), true);
      }
      else if (particleNew(a) == 0)
      {
        logProposalDensity_(n) += std::log(R::pnorm(0, mu_(a), std::sqrt(sigma_(a,a)), true, false));
      }
    }
    */
    ///////////////////// End: using incorrect density //////////////
    
    
    ///////////////////// Start: using the original discrete Gaussian //////////////
    /*
    
//              std::cout << "FINISHED Kalman filter/smoother" << std::endl;
    
//                    std::cout << "sample x" << std::endl;
                   
    arma::colvec x = mu_ + arma::sqrt(sigma_.diag()) % arma::randn<arma::colvec>(nAgeGroups);
    particleNew.set_size(nAgeGroups);
    
//     std::cout << "x: " << x.t() << std::endl;

    for (unsigned int a=0; a<nAgeGroups; a++)
    {     
//          std::cout << "sample single particle for age group " << a << std::endl;
      if (x(a) <= 0)
      {
        particleNew(a) = 0;
      }
      else
      {
        particleNew(a) = ceil(x(a));
      }
      
      if (particleNew(a) > 0)
      {
        logProposalDensity_(n) += std::log(R::pnorm(particleNew(a), mu_(a), std::sqrt(sigma_(a,a)), true, false) - R::pnorm(particleNew(a)-1, mu_(a), std::sqrt(sigma_(a,a)), true, false));
      }
      else if (particleNew(a) == 0)
      {
        logProposalDensity_(n) += std::log(R::pnorm(0, mu_(a), std::sqrt(sigma_(a,a)), true, false));
      }
    }
    */
    ///////////////////// End: using the original discret Gaussian //////////////
    
    ///////////////////// Start: using univariate truncated discrete Gaussians //////////////
    
    
//              std::cout << "FINISHED Kalman filter/smoother" << std::endl;
    
//                    std::cout << "sample x" << std::endl;
                   
   
//     std::cout << "x: " << x.t() << std::endl;
    
    double ub, x, sqrtSigma;
    particleNew.set_size(nAgeGroups);

    for (unsigned int a=0; a<nAgeGroups; a++)
    {     
//          std::cout << "sample single particle for age group " << a << std::endl;
      
      // WARNING: not sure if this is useful:
      if (mu_(a) < 0)
      {
        mu_(a) = 0;
        
      }
      
      sqrtSigma = std::sqrt(sigma_(a,a)); // NOTE: multiplication by factor 2!
      
      
      if (t == 0 || a == 0)
      {
        ub = std::numeric_limits<double>::infinity();
      }
      else if (a > 0 && a < nAgeGroups - 1)
      {
        ub = particleOld(a-1);
      }
      else // i.e. if a == nAgeGroups-1
      {
        ub = static_cast<double>(particleOld(nAgeGroups-1)) + static_cast<double>(particleOld(nAgeGroups-2));
      }
      x = gaussian::rtnorm(-1.0, ub, mu_(a), sqrtSigma, true);
      
      if (x <= 0)
      {
        particleNew(a) = 0;
      }
      else
      {
        particleNew(a) = ceil(x);
      }
      

      logProposalDensity_(n) += 
        std::log(
          R::pnorm(static_cast<double>(particleNew(a)),   mu_(a), sqrtSigma, true, false) - 
          R::pnorm(static_cast<double>(particleNew(a))-1.0, mu_(a), sqrtSigma, true, false)
        )
        - 
        std::log(
          R::pnorm(ub, mu_(a), sqrtSigma, true, false) - 
          R::pnorm(-1.0, mu_(a), sqrtSigma, true, false)
        );
        
//       if (t <= 5) 
//       {
//       std::cout << "proposal weight components: particle(a), mu(a), sqrt(sigma(a,a)), ub: " << particleNew(a) << " " << mu_(a) << " " << sqrtSigma << " " << ub << " " <<
//         R::pnorm(particleNew(a),   mu_(a), sqrtSigma, true, false) 
//         << " " <<
//         R::pnorm(particleNew(a)-1, mu_(a), sqrtSigma, true, false)
//         << " " << 
//         R::pnorm(ub, mu_(a), sqrtSigma, true, false)
//         << " " <<
//         R::pnorm(-1, mu_(a), sqrtSigma, true, false)
//         << std::endl;
//       }
      
 
    }
    
    ///////////////////// End: using univariate truncated discrete Gaussians //////////////
//     if (n == 0) 
//     {
//       std::cout << "mu: " << mu_.t() << " " << " sigma_: " << arma::trans(sigma_.diag()) << " ";
//     }
    
  }
  
  unsigned int K_; // maximum number of lookahead steps
  unsigned int L_; // $L = min\{t+K_, T\}$ (needs to be set at the start of each SMC-filter step
  
  arma::colvec mu_; // smoothed mean-vector for the time-$t$ proposal.
  arma::mat sigma_; // smoothed covariance matrix for the time-$t$ proposal.
  
  // Parameters for the approximate linear-Gaussian state-space model 
  // as defined in the manuscript:
  arma::rowvec C_; 
  std::vector<arma::mat> A_, B_, BBT_;
  arma::colvec D_, DDT_;
  arma::colvec chi_;
  
  arma::colvec logProposalDensity_; // the log-proposal densities for all particles
  
};
/// Holds some additional auxiliary parameters for the SMC algorithm.
class McmcParameters
{
  
};

/// Holds all latent variables in the model (and some other variables of interest)
typedef arma::uvec LatentVariable;

/// Holds all latent variables in the model
// typedef arma::umat LatentPath;
class LatentPath
{
  
public:
  
//   /// Returns the vector of productivity rates.
//   arma::colvec getProductivityRates() const {return productivityRates_;}
//   /// Returns the matrix of true counts.
//   arma::umat getTrueCounts() const {return trueCounts_;}
//   /// Returns the vector of regimes for the regime-switching model.
//   arma::umat getRegimes() const {return regimes_;}
//   /// Returns the vector of means of the marginal smoothing distributions 
//   /// for the true counts for each time point.
//   std::vector<arma::colvec> getSmoothedMeans() const {return smoothedMeans_;}
//   /// Returns the vector of covariance matrices of the marginal smoothing distributions 
//   /// for the true counts for each time point.
//   std::vector<arma::mat> getSmoothedCovarianceMatrices() const {return smoothedCovarianceMatrices_;}
//   
//   /// Specifies the vector of productivity rates.
//   void setProductivityRates(const arma::colvec& productivityRates) {productivityRates_ = productivityRates;}
//   /// Specifies the matrix of true counts.
//   void setTrueCounts() (const arma::umat& trueCounts) {trueCounts_ = trueCounts;}
//   /// Specifies the vector of regimes for the regime-switching model.
//   void setRegimes() (const arma::uvec& regimes) {regimes_ = regimes;}
//   /// Specifies the vector of means of the marginal smoothing distributions 
//   /// for the true counts for each time point.
//   void setSmoothedMeans() (const std::vector<arma::colvec>& smoothedMeans) {smoothedMeans_ = smoothedMeans;}
//   /// Specifies the vector of covariance matrices of the marginal smoothing distributions 
//   /// for the true counts for each time point.
//   void setSmoothedCovarianceMatrices() (const std::vector<arma::mat>& smoothedCovarianceMatrices) {smoothedCovarianceMatrices_ = smoothedCovarianceMatrices;}
  

  
  arma::colvec productivityRates_; // the vector of productivity rates for each time point.
  arma::umat trueCounts_; // latent population counts (only used if these are not integrated out analytically under a linear-Gaussian approximation of the hidden Markov model) 
//   arma::uvec regimes_; // discrete regime indicators for each time point (only used by the Markov-regime switching model for the productivity rate).
  arma::mat smoothedMeans_; // means (in cols) of the marginal smoothing distributions for the true counts for each time point.
  arma::cube smoothedCovarianceMatrices_; // covariance matrices (in slices) of the marginal smoothing distributions for the true counts for each time point.
  
private: 
    
};

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
  
  /// Returns the relevant entries in the $i$th row of the 
  /// ring-recovery data matrix
  arma::urowvec getRingRecovery(const unsigned int i) const
  {
    return ringRecovery_.row(i);
  }
  
  /// Returns the number of juveniles ringed in the $i$th year 
  /// for which ring-recovery data is available
  unsigned int getNRinged(const unsigned int i) const { return nRinged_(i); }
  
  arma::uvec nRinged_; // number of individuals ringed in the years in which ring-recovery data is available
  arma::umat ringRecovery_; // ring-recovery data
  arma::uvec count_; // length-nObservationsCount_ vector of count data
//   unsigned int t1Count_, t2Count_, t1Ring_, t2Ring_; // first/last years for which count/ring-recovery data is available

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

/// Evaluates the log of the marginal likelihood of the full data set.
/// Not available in this model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihood()
{
  return 0.0;
}
/// Evaluates the marginal likelihood of the parameters using a
/// Kalman filter (i.e. with the latent 
/// variables integrated out). Note that this implies a linear-Gaussian approximation
/// and even then is not possible for all models. It also requires the initial distribution
/// not to be a uniform.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihoodSecond(LatentPath& latentPath)
{
//   std::cout << "started running KF" << std::endl;
  
  // setting some additional model parameters (these could not be computed earlier because 
  // they depend on the observations). 
//   if (modelParameters_.getModelType() == MODEL_THRESHOLD_DEPENDENCE_OBSERVATIONS)
//   {
//     for (unsigned int t=0; t<modelParameters_.getNObservationsCount()-1; t++)
//     {
//       // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_.
//       modelParameters_.computeRhoFromStepFun(t, getObservations().count_(t));
//     }
//   }
//   else if (modelParameters_.getModelType() == MODEL_DIRECT_DENSITY_DEPENDENCE)
//   {
//     // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_
//     arma::colvec countAux = arma::conv_to<arma::colvec>::from(getObservations().count_(arma::span(0,getObservations().count_.size()-2)));
// 
//     double meanCountAux = arma::accu(countAux) / countAux.size();
//     double sdCountAux = std::sqrt(arma::accu(arma::pow(countAux, 2.0)) - std::pow(meanCountAux, 2.0));
//   
//     modelParameters_.setRho(
//       arma::exp(
//         modelParameters_.getEpsilon0() + modelParameters_.getEpsilon1() * (countAux-meanCountAux)/sdCountAux
//       )
//     );
//   }
  
  bool isNumericallyStable = true;
  unsigned int T = modelParameters_.getNObservationsCount();
  
  // Log-marginal likelihood
  double logLikelihood = 0;
  
  if (modelParameters_.getModelType() == MODEL_MARKOV_SWITCHING)  // run a discrete particle filter to approximate the loglikelihood of the Markov-regime switching model with linear-Gaussian approximation
  {
    
    unsigned int M = 200;
    unsigned int K = modelParameters_.getNLevels();
    unsigned int N = K;
    double logC = arma::datum::inf;

//     arma::uvec particlesNew(M*K);
//     arma::uvec particlesOld(M*K);
    std::vector<arma::colvec> mUNew(M*K);
    std::vector<arma::colvec> mUOld(M*K);
    std::vector<arma::mat> CUNew(M*K);
    std::vector<arma::mat> CUOld(M*K);
    
    arma::umat parentIndicesFull(M*K, T);
    arma::umat particlesFull(M*K, T);
   
    arma::uvec parentIndices(M*K);
    
    arma::colvec logUnnormalisedWeights(M*K, arma::fill::zeros);
    arma::colvec selfNormalisedWeights(M*K);
    arma::colvec logSelfNormalisedWeights(M*K);

    for (unsigned int n=0; n<N; n++)
    {
      mUNew[n].set_size(modelParameters_.getNAgeGroups());
      CUNew[n].set_size(modelParameters_.getNAgeGroups(), modelParameters_.getNAgeGroups());
      
      // Determine particles.
      // ------------------------------------------------------------------- //
//       particlesNew(n) = n;
      particlesFull(n,0) = n;
      
      // Update weights.
      // ------------------------------------------------------------------- //
      logUnnormalisedWeights(n) 
        += modelParameters_.iterateKalmanFilter(0, static_cast<double>(getObservations().count_(0)), 1.0, mUNew[n], CUNew[n]) - std::log(K);
    }
    
//     particlesFull(arma::span(0,N-1), arma::span(0,0)) = particlesNew(arma::span(0,N-1));
    
    logLikelihood += std::log(arma::accu(arma::exp(logUnnormalisedWeights(arma::span(0,N-1)))));
    
    for (unsigned int t=1; t<T; t++)
    {
//       std::cout << "DPF, step " << t << " of " << T <<  std::endl;

      selfNormalisedWeights    = normaliseWeights(logUnnormalisedWeights(arma::span(0,N-1)));
      logSelfNormalisedWeights = normaliseExp(logUnnormalisedWeights(arma::span(0,N-1)));
      
      
      if (!arma::is_finite(selfNormalisedWeights))
      {
        std::cout << "WARNING: W contains NaNs!" << std::endl;
        isNumericallyStable = false;
        logLikelihood = - arma::datum::inf;
        break;
      }
      
      
      // Resampling scheme from Fearnhead (1998).
      // ------------------------------------------------------------------- //
      if (N <= M) // if the discrete particle filter does not resample (i.e. in the first few steps)
      {
        parentIndices(arma::span(0,N*K-1)) = arma::repmat(arma::linspace<arma::uvec>(0, N-1, N), 1, K);
        N = N * K;   
        for (unsigned int n=0; n<N; n++)
        {
          logUnnormalisedWeights(n) = logSelfNormalisedWeights(parentIndices(n));
        }
      }
      else // i.e. if $N > M$; recall that $N$ is the number of particles produced in the previous step.
      {
        // algorithm for calculating the constant needed for the resampling 
        // step in the DPF
        
        if (N < M * K) {N = M * K;}

        arma::colvec logWSorted = arma::sort(logSelfNormalisedWeights, "descend");
        arma::colvec WSorted    = arma::sort(selfNormalisedWeights,    "descend");

        unsigned int kNew = 0;
        unsigned int kOld = 0;
        bool isFirstIteration = true;
 
        while (isFirstIteration || kNew != kOld)
        {     
//           if (kOld == 0) // i.e. if this is the initial iteration
          if (isFirstIteration) // i.e. if this is the initial iteration
          {
//             kOld = 0;
//             kNew = 0;
            isFirstIteration = false;
          }
//           else
//           {
//             kOld = kNew;
//           }

          kOld = kNew;
          logC = std::log(M - kOld) - std::log(arma::accu(WSorted(arma::span(kOld,WSorted.size()-1))));
          for (unsigned int j=kOld; j<WSorted.size(); j++)
          {
            if (logWSorted(j) > -logC) 
            {
              kNew++;
            }
          }
//           std::cout << "logC: " << logC << "; kNew - kOld:" << kNew-kOld << " ";
        }
//          std::cout << "logC: " << logC << "; kNew - kOld:" << kNew-kOld << " ";

        // Testing whether Algorithm 5.2 from Fearnhead (1998) 
        // has managed to calculate logC without 
        // numerical issues caused by too many extremely small weights.
        double testSum = 0;
        for (unsigned int j=0; j<logSelfNormalisedWeights.size(); j++)
        {
          if (logSelfNormalisedWeights(j) >= -logC) 
          {
            testSum++;
          }
          else
          {
            testSum += std::exp(logC) * std::exp(logSelfNormalisedWeights(j));
          }
        }
        
        
        //////////////
//         testSum = 0; ///// WARNING: just for testing!
        //////////////
//         std::cout <<  "Testing value found for logC:" << testSum - M << std::endl;
   
       
        if (std::abs(testSum - M) < 0.00001) // i.e. if Algorithm 5.2 from Fearnhead (1998) works without numerical issues caused by too many extremely small weights.
        {
//           std::cout << " ------------- optimal resamling successful ------------- " << std::endl;
            
          arma::uvec indKep = arma::find(logSelfNormalisedWeights >  -logC); // indices of particles that are kept without any resampling
          arma::uvec indRes = arma::find(logSelfNormalisedWeights <= -logC); // indices of particles which are resampled via systematic resampling
          
          unsigned int L = indKep.size();   // number of "not resampled" particles
          arma::uvec parentIndicesAux(M-L); // parent indices obtained by systematic resampling 
          resample::systematicBase(arma::randu(), parentIndicesAux, normaliseWeights(logUnnormalisedWeights(indRes)), M-L);
          
          if (L > 0)
          {
            parentIndices(arma::span(0, L-1)) = indKep;
          }
          parentIndices(arma::span(L, M-1)) = parentIndicesAux;
          parentIndices = arma::repmat(parentIndices(arma::span(0,M-1)), 1, K);
          
          logUnnormalisedWeights.zeros();
          for (unsigned int n=0; n<N; n++)
          {
            if (logSelfNormalisedWeights(parentIndices(n)) > -logC)
            {
              logUnnormalisedWeights(n) += logSelfNormalisedWeights(parentIndices(n));
            }
            else
            {
              logUnnormalisedWeights(n) -= logC;
            }
          }
        }
        else // if rare numerical issues cause the optimal resampling scheme to fail.
        {
          std::cout <<  "Testing value found for logC at step " << t << ": " << testSum - M << std::endl;
//           std::cout <<  "selfNormalisedWeights: " << selfNormalisedWeights.t() << std::endl;
          logC = arma::datum::inf;
          arma::uvec parentIndicesAux(M);
          resample::systematicBase(arma::randu(), parentIndicesAux, normaliseWeights(logUnnormalisedWeights), M);
          parentIndices = arma::repmat(parentIndices(arma::span(0,M-1)), 1, K);
//           logUnnormalisedWeights.fill(-std::log(M));
          
          logUnnormalisedWeights.zeros();
          for (unsigned int n=0; n<N; n++)
          {
            logUnnormalisedWeights(n) += logSelfNormalisedWeights(parentIndices(n));
          }
          
//           getchar();
        } 
      }
      
      if (t>0)
      {
        parentIndicesFull(arma::span(0,N-1), arma::span(t-1,t-1)) = parentIndices(arma::span(0,N-1));
      }
      
      // Determining the parent particles based on the parent indices: 
      for (unsigned int n=0; n<N; n++)
      {
//         particlesOld(n) = particlesNew(parentIndices(n)); 
        mUOld[n]        = mUNew[parentIndices(n)];
        CUOld[n]        = CUNew[parentIndices(n)];
      }
      mUNew = mUOld;
      CUNew = CUOld;
      
      // Determine particles.
      // ------------------------------------------------------------------- //
      for (unsigned int k=0; k<K; k++) 
      {
//         particlesNew(arma::span(k*(N/K), (k+1)*(N/K)-1)).fill(k);
        particlesFull(arma::span(k*(N/K), (k+1)*(N/K)-1), arma::span(t,t)).fill(k);
      
      }
//       particlesFull(arma::span(0,N-1), arma::span(t,t)) = particlesNew(arma::span(0,N-1));
//       
      // Update weights.
      // ------------------------------------------------------------------- //
      for (unsigned int n=0; n<N; n++)
      {                
        logUnnormalisedWeights(n) 
          += modelParameters_.iterateKalmanFilter(t, static_cast<double>(getObservations().count_(t)), modelParameters_.getNu(particlesFull(n,t)), mUNew[n], CUNew[n]) 
          + std::log(modelParameters_.getP(particlesFull(parentIndices(n),t-1), particlesFull(n,t)));
//                     += modelParameters_.iterateKalmanFilter(t, static_cast<double>(getObservations().count_(t)), modelParameters_.getNu(particlesNew(n)), mUNew[n], CUNew[n]) 
//           + std::log(modelParameters_.getP(particlesOld(n), particlesNew(n)));
      }
      logLikelihood += std::log(arma::accu(arma::exp(logUnnormalisedWeights(arma::span(0,N-1)))));
    }
    
   
    if (isNumericallyStable) 
    {
      // ------------------------------------------------------------------------
      // Determining some additional output
      // This does not affect the evolution of the SMC sampler or PMMH algorithm.
      latentPath.trueCounts_.set_size(1,T);
      unsigned int idx = sampleInt(normaliseWeights(logUnnormalisedWeights(arma::span(0,N-1))));

      latentPath.trueCounts_(0,T-1) = particlesFull(idx,T-1);

      for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
      { 
        idx = parentIndicesFull(idx, t);
        latentPath.trueCounts_(0,t) = particlesFull(idx,t);
      }
    
      for (unsigned int t=1; t<T; t++)
      {
        modelParameters_.setRho(t-1, modelParameters_.getNu(latentPath.trueCounts_(0,t)));
      }
      // ------------------------------------------------------------------------
    }
    else
    {
      // ------------------------------------------------------------------------
      // Determining some additional output
      // This does not affect the evolution of the SMC sampler or PMMH algorithm.
      latentPath.trueCounts_.set_size(1,T);
      for (unsigned int t=0; t<T; t++)
      { 
        latentPath.trueCounts_(0,t) = 0;
      }
      for (unsigned int t=1; t<T; t++)
      {
        modelParameters_.setRho(t-1, 0);
      }
      // ------------------------------------------------------------------------
    }

  }
  /*
  // Running a standard "Rao--Blackwellised" PF
  double logLikelihoodPf = 0.0;
  if (modelParameters_.getModelType() == MODEL_MARKOV_SWITCHING)
  {
    unsigned int N = 1000000;
    unsigned int K = modelParameters_.getNLevels();

    arma::uvec particlesNew(N);
    arma::uvec particlesOld(N);
    std::vector<arma::colvec> mUNew(N);
    std::vector<arma::colvec> mUOld(N);
    std::vector<arma::mat> CUNew(N);
    std::vector<arma::mat> CUOld(N);
   
    arma::uvec parentIndices(N);
    
    arma::colvec logUnnormalisedWeights(N);
    logUnnormalisedWeights.fill(-std::log(N));
    arma::colvec selfNormalisedWeights(N);
    arma::colvec logSelfNormalisedWeights(N);
    
//     std::cout << "Standard PF, step " << 0 << " of " << T <<  std::endl;
    
    for (unsigned int n=0; n<N; n++)
    {
      mUNew[n].set_size(modelParameters_.getNAgeGroups());
      CUNew[n].set_size(modelParameters_.getNAgeGroups(), modelParameters_.getNAgeGroups());
      
      // Determine particles.
      // ------------------------------------------------------------------- //
      particlesNew(n) = arma::conv_to<unsigned int>::from(arma::randi<arma::uvec>(1, arma::distr_param(0, K-1)));
      
      // Update weights.
      // ------------------------------------------------------------------- //
      logUnnormalisedWeights(n) 
        += modelParameters_.iterateKalmanFilter(0, static_cast<double>(getObservations().count_(0)), 1.0, mUNew[n], CUNew[n]);
    }
    
    for (unsigned int t=1; t<T; t++)
    {

      selfNormalisedWeights = normaliseWeights(logUnnormalisedWeights);

      if (!arma::is_finite(selfNormalisedWeights))
      {
        std::cout << "WARNING: W contains NaNs!" << std::endl;
      }

      if (1.0 / arma::dot(selfNormalisedWeights, selfNormalisedWeights) < N * 0.9)
      {
        // update estimate of the normalising constant:
        logLikelihoodPf += std::log(arma::sum(arma::exp(logUnnormalisedWeights))); 
        resample::systematicBase(arma::randu(), parentIndices, selfNormalisedWeights, N);
        logUnnormalisedWeights.fill(-std::log(N));
      }
      else
      {
        parentIndices = arma::linspace<arma::uvec>(0, N-1, N);
      }
  
      // Determining the parent particles based on the parent indices: 
      for (unsigned int n=0; n<N; n++)
      {
        particlesOld(n) = particlesNew(parentIndices(n)); 
        mUOld[n]        = mUNew[parentIndices(n)];
        CUOld[n]        = CUNew[parentIndices(n)];
      }
      mUNew = mUOld;
      CUNew = CUOld;
      
      // Determine particles and update weights.
      // ------------------------------------------------------------------- //
      for (unsigned int n=0; n<N; n++) 
      {
        particlesNew(n) = sampleInt(modelParameters_.getP(particlesOld(n)));
        logUnnormalisedWeights(n) 
          += modelParameters_.iterateKalmanFilter(t, static_cast<double>(getObservations().count_(t)), modelParameters_.getNu(particlesNew(n)), mUNew[n], CUNew[n]);
      }
    }
    logLikelihoodPf += std::log(arma::sum(arma::exp(logUnnormalisedWeights))); 
    
    std::cout << "logLikelihood: " << logLikelihood << " ";
    std::cout << "; logLikelihoodPf: " << logLikelihoodPf << " ";
    std::cout << "; difference: " << logLikelihood - logLikelihoodPf << std::endl;
  }
  */

  
  // run a Kalman filter and smoother conditional on the productivity rates 
  // Incremental likelihood means and variances:
  arma::colvec mY(T);
  arma::colvec CY(T);
  if (isNumericallyStable) 
  {
    modelParameters_.runKalman(getObservations().count_, latentPath.smoothedMeans_, latentPath.smoothedCovarianceMatrices_, mY, CY);
    latentPath.productivityRates_ = modelParameters_.getRho();
  }

  // Exactly calculates the marginal count-data likelihood for the 
  // linear-Gaussian approximation to the hidden Markov model.
  if (modelParameters_.getModelType() != MODEL_MARKOV_SWITCHING)
  {
    for (unsigned int t=0; t<T; t++)
    {
      logLikelihood += R::dnorm(static_cast<double>(getObservations().count_(t)), mY(t), std::sqrt(CY(t)), true);
    }
  }
 
  if (!std::isfinite(logLikelihood))
  {
    return - arma::datum::inf;
    std::cout << "WARNING: loglikelihood obtained from KF: " << logLikelihood << ", has been replaced by -Inf!" << std::endl;
  }
  else
  {
    return logLikelihood;
  }
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

  observations_.ringRecovery_.zeros(modelParameters_.getNObservationsRing(), modelParameters_.getNObservationsRing()+1);
  observations_.nRinged_.set_size(observations_.ringRecovery_.n_rows);
  
  unsigned int k;
  
  for (unsigned int i=0; i<observations_.ringRecovery_.n_rows; i++)
  {
    observations_.nRinged_(i) = 800;
    for (unsigned int j=0; j<observations_.nRinged_(i); j++)
    { 
      k = sampleInt(modelParameters_.getQ(i));
      observations_.ringRecovery_(i,k) += 1;
    }
  }
  
//   observations_.nRinged_ = arma::sum(observations_.ringRecovery_, 1);
  
  
  observations_.count_.set_size(modelParameters_.getNObservationsCount()); // TODO: need to make sure that nObservations_ is set correctly
  
  if (modelParameters_.getModelType() == MODEL_MARKOV_SWITCHING)
  {
    latentPath_.trueCounts_.set_size(modelParameters_.getNAgeGroups()+1, modelParameters_.getNObservationsCount());
//     latentPath_.regimes_.set_size(modelParameters_.getNObservationsCount());
  }
  else 
  {
    latentPath_.trueCounts_.set_size(modelParameters_.getNAgeGroups(), modelParameters_.getNObservationsCount());
  }

  latentPath_.trueCounts_.col(0) = sampleFromInitialDistribution();
  sampleFromObservationEquation(0, observations_, latentPath_.trueCounts_.col(0));
  for (unsigned int t=1; t<modelParameters_.getNObservationsCount(); ++t)
  {
    latentPath_.trueCounts_.col(t) = sampleFromTransitionEquation(t, latentPath_.trueCounts_.col(t-1));
    sampleFromObservationEquation(t, observations_, latentPath_.trueCounts_.col(t));
  }
  
//   latentPath_.regimes_ = latentPath_.trueCounts_.row(modelParameters_.getNAgeGroups());
  
}
/// Evaluates the log of the likelihood of the parameters given the latent 
/// variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogCompleteLikelihood(const LatentPath& x)
{
//   std::cout << "WARNING: evalutation of the log-complete likelihood is not implemented yet! But this is only needed for Gibbs-sampling type algorithms" << std::endl;
  
//   if (modelParameters_.getModelType() == MODEL_THRESHOLD_DEPENDENCE_OBSERVATIONS)
//   {
//     for (unsigned int t=0; t<modelParameters_.getNObservationsCount()-1; t++)
//     {
//       // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_.
//       modelParameters_.computeRhoFromStepFun(t, getObservations().count_(t));
// //       std::cout << "rho(" << t << "): " << model_.getModelParameters().getRho(t) << " " ;
//     }
// 
//   }
//   else if (modelParameters_.getModelType() == MODEL_DIRECT_DENSITY_DEPENDENCE)
//   {
//       // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_
//     arma::colvec countAux = arma::conv_to<arma::colvec>::from(getObservations().count_(arma::span(0,getObservations().count_.size()-2)));
//     
//     ///////////////////
// //     std::cout << "NOTE: we're now normalising the observed counts in the direct-density dependence model!" << std::endl;
//     double meanCountAux = static_cast<double>(arma::accu(countAux)) / countAux.size();
//     double sdCountAux   = std::sqrt(arma::accu(arma::pow(countAux, 2.0)) - std::pow(meanCountAux, 2.0));
//   
//     modelParameters_.setRho(
//       arma::exp(
//         modelParameters_.getEpsilon0() + modelParameters_.getEpsilon1() * (countAux-meanCountAux)/sdCountAux
//       )
//     );
//   }
//   
  
  double logLike = evaluateLogInitialDensity(x.col(0)) + 
                   evaluateLogObservationDensity(0, x.col(0));
                   
  for (unsigned int t=1; t<modelParameters_.getNObservationsCount(); ++t)
  {
    logLike += evaluateLogTransitionDensity(t, x.col(t), x.col(t-1)) + 
               evaluateLogObservationDensity(t, x.col(t));
  }
  return logLike + evaluateLogMarginalLikelihoodFirst(x);
}
/// Samples a single latent variable at Time t=0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromInitialDistribution()
{

  unsigned int A = modelParameters_.getNAgeGroups();
  arma::uvec x;
  arma::uvec xx(1);
  
  if (modelParameters_.getModelType() == MODEL_MARKOV_SWITCHING)
  {
    x.set_size(A+1);
    xx = arma::randi<arma::uvec>(1, arma::distr_param(0, modelParameters_.getNLevels()-1));
    x(A) = xx(0);
  }
  else 
  {
    x.set_size(A);
  }

  if (modelParameters_.getInitialDistributionType() == INITIAL_DISTRIBUTION_NEGATIVE_BINOMIAL)
  {
    // Sampling the state at time 0 (counting "time $0$" as in the paper!)
    for (unsigned int a=0; a<A-1; a++)
    {
      x(a) = R::rnbinom(modelParameters_.getNegativeBinomialSizeHyperInit0(), modelParameters_.getNegativeBinomialProbHyperInit0());
    }
    x(A-1) = R::rnbinom(modelParameters_.getNegativeBinomialSizeHyperInit1(), modelParameters_.getNegativeBinomialProbHyperInit1());
  }
  else if (modelParameters_.getInitialDistributionType() == INITIAL_DISTRIBUTION_DISCRETE_UNIFORM)
  {
    x(arma::span(0,A-2)) = arma::randi<arma::uvec>(A-1,
                                 arma::distr_param(modelParameters_.getMinHyperInit0(), modelParameters_.getMaxHyperInit0()));
    xx = arma::randi<arma::uvec>(1,
                                 arma::distr_param(modelParameters_.getMinHyperInit1(), modelParameters_.getMaxHyperInit1()));
    x(A-1) = xx(0);
  }
  else if (modelParameters_.getInitialDistributionType() == INITIAL_DISTRIBUTION_POISSON)
  {
    // Sampling the state at time 0 (counting "time $0$" as in the paper!)
    for (unsigned int a=0; a<A-1; a++)
    {
      x(a) = R::rpois(modelParameters_.getChi0());
    }
    x(A-1) = R::rpois(modelParameters_.getChi1());
  }
  
  return x;
}
/// Samples a single latent variable at Time t>0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromTransitionEquation(const unsigned int t, const LatentVariable& latentVariableOld)
{
  unsigned int A = modelParameters_.getNAgeGroups();
  arma::uvec x;
 
  // Total population at the previous time step (except first-years and adults).
  unsigned int totalPopSize = arma::accu(latentVariableOld(arma::span(1,A-1)));
  
  // TODO: set rho_t here for the models which have this as being state-dependent; 
  if (modelParameters_.getModelType() == MODEL_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
  {
    modelParameters_.computeRhoFromStepFun(t-1, totalPopSize);
  }
  
  
  if (modelParameters_.getModelType() == MODEL_MARKOV_SWITCHING)
  {
    x.set_size(A+1);
    x(A) = sampleInt(modelParameters_.getP(latentVariableOld(A)));
    modelParameters_.setRho(t-1, modelParameters_.getNu(x(A)));
  }
  else 
  {
    x.set_size(A);
  }
    
  x(0) = R::rpois(modelParameters_.getRho(t-1) * modelParameters_.getPhi(0,t-1) * totalPopSize);
  
  for (unsigned int a=1; a<A-1; a++)
  {
    x(a) = R::rbinom(latentVariableOld(a-1), modelParameters_.getPhi(a,t-1));
  }
  x(A-1) = 
  R::rbinom(
    arma::accu(latentVariableOld(arma::span(A-2,A-1))), modelParameters_.getPhi(A-1,t-1)
  );
  
  return x;
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogInitialDensity(const LatentVariable& latentVariable)
{
  unsigned int A = modelParameters_.getNAgeGroups();
  double logDensity = 0.0;
  
  if (modelParameters_.getInitialDistributionType() == INITIAL_DISTRIBUTION_NEGATIVE_BINOMIAL)
  {
    for (unsigned int a=0; a<A-1; a++)
    {
      logDensity += R::dnbinom(latentVariable(a), modelParameters_.getNegativeBinomialSizeHyperInit0(), modelParameters_.getNegativeBinomialProbHyperInit0(), true);
    }
    logDensity += R::dnbinom(latentVariable(A-1), modelParameters_.getNegativeBinomialSizeHyperInit1(), modelParameters_.getNegativeBinomialProbHyperInit1(), true);
  }
  else if (modelParameters_.getInitialDistributionType() == INITIAL_DISTRIBUTION_DISCRETE_UNIFORM)
  {
    logDensity -= (A-1)*std::log(modelParameters_.getMaxHyperInit0() - modelParameters_.getMinHyperInit0());
    logDensity -= std::log(modelParameters_.getMaxHyperInit1() - modelParameters_.getMinHyperInit1());
  }
  else if (modelParameters_.getInitialDistributionType() == INITIAL_DISTRIBUTION_POISSON)
  {
    for (unsigned int a=0; a<A-1; a++)
    {
      logDensity += R::dpois(latentVariable(a), modelParameters_.getChi0(), true);
    }
    logDensity += R::dpois(latentVariable(A-1), modelParameters_.getChi1(), true);
  }

  return logDensity; 
}
/// Evaluates the log-conditional prior density of the Time-t latent variable.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld)
{
  double logDensity = 0.0;
  unsigned int A = modelParameters_.getNAgeGroups();
  
  // Total population at the previous time step (except first-years and adults).
  unsigned int totalPopSize = arma::accu(latentVariableOld(arma::span(1,A-1)));
  
  // TODO: if we use threshold dependence on the true counts, we need to calculate the thresholds here!

  // TODO: if we use threshold  dependence on the true counts, we need to make sure 
  // (a) that we still pass rho_ based on the observed counts to the Kalman filter
  // (b) rho(t-1) used by the Kalman filter is computed based on particleOld!
 
  // TODO: set rho_t here for the models which have this as being state-dependent; 
  if (modelParameters_.getModelType() == MODEL_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
  {
    modelParameters_.computeRhoFromStepFun(t-1, totalPopSize);
  }
  
//   std::cout << "WARNING: for the regime-switching model, the transition density does not yet take the transition for the discrete regime indicators into account!"<< std::endl;
  
  logDensity += R::dpois(latentVariableNew(0), modelParameters_.getRho(t-1) * modelParameters_.getPhi(0, t-1) * totalPopSize, true);
  
  for (unsigned int a=1; a<A-1; a++)
  {
    logDensity += R::dbinom(latentVariableNew(a), latentVariableOld(a-1), modelParameters_.getPhi(a,t-1), true);
  }
  logDensity += R::dbinom(latentVariableNew(A-1), latentVariableOld(A-1) + latentVariableOld(A-2), modelParameters_.getPhi(A-1,t-1), true);
  
  return logDensity;
}
/// Evaluates the log-observation density of the observations at Time t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable)
{

//   unsigned int A = modelParameters_.getNAgeGroups();
  unsigned int adultPopulation = arma::accu(latentVariable(arma::span(1,modelParameters_.getNAgeGroups()-1)));
//    
//   double sz = modelParameters_.getKappa() / (1.0 - modelParameters_.getKappa()) * adultPopulation;

  // NOTE: this assumes the original negative-binomial measurement equation:
  
  if (modelParameters_.getObservationEquationType() == OBSERVATION_EQUATION_NEGATIVE_BINOMIAL)
  {
    double sz = modelParameters_.getKappa() / (1.0 - modelParameters_.getKappa()) * adultPopulation;
    return R::dnbinom(observations_.count_(t), sz, modelParameters_.getKappa(), true);
  } 
  else if (modelParameters_.getObservationEquationType() == OBSERVATION_EQUATION_POISSON)
  {
    // NOTE: this assumes a Poisson measurement equation:
    return R::dpois(observations_.count_(t), adultPopulation, true);
  }
  else
  {
    return -arma::datum::inf;
  }


}
/// Samples a single observation according to the observation equation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromObservationEquation(const unsigned int t, Observations& observations, const LatentVariable& latentVariable)
{

  
  // TODO: we need to setup the productivity rates here (for some of the models)
  unsigned int A = modelParameters_.getNAgeGroups();
  unsigned int adultPopulation = arma::accu(latentVariable(arma::span(1,A-1)));
  double sz = modelParameters_.getKappa() / (1.0 - modelParameters_.getKappa()) * adultPopulation;
  
  if (modelParameters_.getObservationEquationType() == OBSERVATION_EQUATION_NEGATIVE_BINOMIAL)
  {
    // NOTE: this samples from the original negative-binomial distribution:
    observations_.count_(t) = R::rnbinom(sz, modelParameters_.getKappa());
  }
  else 
  {
    // NOTE: this samples from a Poisson distribution:
    observations_.count_(t) = R::rpois(adultPopulation);
  }

//   if (modelParameters_.getModelType() == MODEL_THRESHOLD_DEPENDENCE_OBSERVATIONS)
//   {
//     if (t < modelParameters_.getNObservationsCount()-1)
//     {
//       modelParameters_.computeRhoFromStepFun(t, observations_.count_(t));
//     }
//   }
//   else if (modelParameters_.getModelType() == MODEL_DIRECT_DENSITY_DEPENDENCE)
//   {
//     // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_
//     std::cout << "WARNING: we cannot use direct-density dependence with normalising the count data!" << std::endl; 
// //     arma::colvec countAux = arma::conv_to<arma::colvec>::from(observations_.count_(arma::span(0,observations_.count_.size()-2)));
// //     double meanCountAux = arma::accu(countAux) / countAux.size();
// //     double sdCountAux = std::sqrt(arma::accu(arma::pow(countAux, 2.0)) - std::pow(meanCountAux, 2.0));
// //   
// //     modelParameters_.setRho(t, 
// //       std::exp(
// //         modelParameters_.getEpsilon0() + modelParameters_.getEpsilon1() * (countAux-meanCountAux)/sdCountAux
// //       )
// //     );
//   }
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
  double logLike = 0.0;
  // Likelihood based on ring-recovery data
  for (unsigned int i=0; i<modelParameters_.getNObservationsRing(); i++)
  {
    logLike += logMultinomialDensity(observations_.getRingRecovery(i), observations_.getNRinged(i), modelParameters_.getQ(i));
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
  
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  { 
    // Sampling particles from the model transitions:
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      particlesNew[n] = model_.sampleFromTransitionEquation(t, particlesOld[n]);
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_ALTERNATE_0)
  {
    
    unsigned int A = model_.getModelParameters().getNAgeGroups();
    arma::uvec xx(1);
    arma::uvec x;

    if (model_.getModelParameters().getModelType() == MODEL_MARKOV_SWITCHING)
    {
      x.set_size(A+2);
    }
    else
    {
      x.set_size(A+1);
    }
    
    // Sampling herons at time $t$ in years $2$ to $A$ from the model transitions
    // but sampling first-years at time $t-1$.
    for (unsigned int n=0; n<getNParticles(); n++)
    {   

      x(A) = arma::accu(particlesOld[n](arma::span(1,A-1)));
            
      if (model_.getModelParameters().getModelType() == MODEL_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
      { 
        //model_.getRefModelParameters().computeRhoFromStepFun(t-1, x(A));
        if (t > 1)
        {
          model_.getRefModelParameters().computeRhoFromStepFun(t-2, particlesOld[n](A));
        }
      }
      else if (model_.getModelParameters().getModelType() == MODEL_MARKOV_SWITCHING)
      {
        // Sampling the level indicator for the latent Markov chain governing the 
        // regime switches
        if (t == 1)
        {
          xx = arma::randi<arma::uvec>(1, arma::distr_param(0, model_.getModelParameters().getNLevels()-1));
          x(A+1) = xx(0);
        }
        else if (t > 1)
        {
          x(A+1) = sampleInt(model_.getModelParameters().getP(particlesOld[n](A+1)));
          //model_.getRefModelParameters().setRho(t-2, model_.getModelParameters().getNu(x(A+1)));
          model_.getRefModelParameters().setRho(t-2, model_.getModelParameters().getNu(particlesOld[n](A+1))); // TODO: is particlesOld correct here???
        }
        
      }
      
      if (t == 1)
      {
        if (model_.getModelParameters().getInitialDistributionType() == INITIAL_DISTRIBUTION_NEGATIVE_BINOMIAL)
        {
          for (unsigned int a=0; a<A-1; a++)
          {
            x(a) = R::rnbinom(model_.getModelParameters().getNegativeBinomialSizeHyperInit0(), model_.getModelParameters().getNegativeBinomialProbHyperInit0());
          }
          x(A-1) = R::rnbinom(model_.getModelParameters().getNegativeBinomialSizeHyperInit1(), model_.getModelParameters().getNegativeBinomialProbHyperInit1()); 
        }
        else if (model_.getModelParameters().getInitialDistributionType() == INITIAL_DISTRIBUTION_POISSON)
        {
          for (unsigned int a=0; a<A-1; a++)
          {
            x(a) = R::rpois(model_.getModelParameters().getChi0());
          }
          x(A-1) = R::rpois(model_.getModelParameters().getChi1());
        }
        else if (model_.getModelParameters().getInitialDistributionType() == INITIAL_DISTRIBUTION_DISCRETE_UNIFORM)
        {
          x(arma::span(0,A-2)) = arma::randi<arma::uvec>(A-1,
                                  arma::distr_param(model_.getModelParameters().getMinHyperInit0(), model_.getModelParameters().getMaxHyperInit0()));
          xx = arma::randi<arma::uvec>(1,
                                  arma::distr_param(model_.getModelParameters().getMinHyperInit1(), model_.getModelParameters().getMaxHyperInit1()));
          x(A-1) = xx(0);
          
        }
      }
      else if (t > 1)
      {
        x(0) = R::rpois(model_.getModelParameters().getRho(t-2) * model_.getModelParameters().getPhi(0,t-2) * particlesOld[n](A));
        if (A > 2)
        {
          x(1)   = R::rbinom(x(0), model_.getModelParameters().getPhi(1,t-1));
          x(A-1) = R::rbinom(arma::accu(particlesOld[n](arma::span(A-2,A-1))), model_.getModelParameters().getPhi(A-1,t-1));
          
          for (unsigned int a=2; a<A-1; a++)
          {
            x(a) = R::rbinom(particlesOld[n](a-1), model_.getModelParameters().getPhi(a,t-1));
          }
          
        }
        else if (A == 2)
        {
          x(1) = R::rbinom(particlesOld[n](1) + x(0), model_.getModelParameters().getPhi(A-1,t-1));
        }
      }
      particlesNew[n] = x;
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_LOOKAHEAD)
  {
    smcParameters_.proposeAllParticles(particlesNew, particlesOld, t, model_.getModelParameters().getNObservationsCount(), nParticles_, model_.getModelParameters().getNAgeGroups(), model_.getModelParameters().getKappa(), model_.getModelParameters().getChi0(), model_.getModelParameters().getChi1(), model_.getModelParameters().getPhi(), model_.getModelParameters().getRho(), model_.getObservations().count_);
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
  
  
    /////////////////
//   double yy = model_.getObservations().count_(t);
//   std::cout << "diff between 1st particle and obs at time " << t << ": " << arma::accu(particlesNew[0](arma::span(1,model_.getModelParameters().getNAgeGroups()-1))) - yy << std::endl;
//   std::cout << "1st incremental logWeight: " << model_.evaluateLogObservationDensity(t, particlesNew[0]) << "; rho(t-1): " << model_.getModelParameters().getRho(t-1) << "; phi(t-1): " << model_.getModelParameters().getPhi(0,t-1) << ", " << model_.getModelParameters().getPhi(1, t-1) <<  std::endl;
  /////////////////////
  
  
  
//   std::cout << "particlesNew[0]:" << particlesNew[0].t() << " ";
  if (smcProposalType_ == SMC_PROPOSAL_LOOKAHEAD)
  {
    //////////////////////////////////////////////////////////////////////////////////
    /*
    if (t <= 3)
    {
    arma::colvec vTest1(nParticles_);
    arma::colvec vTest2(nParticles_);
    arma::colvec vTest3(nParticles_);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      vTest1(n) = model_.evaluateLogObservationDensity(t, particlesNew[n]);
    }
    
    std::cout << "log-observation densities: " << vTest1.t() << std::endl;
    
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      vTest2(n) = model_.evaluateLogTransitionDensity(t, particlesNew[n], particlesOld[n]);
    }
    
    std::cout << "log-transitions densities: " << vTest2.t() << std::endl;
    
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      vTest3(n) = smcParameters_.getLogProposalDensity(n);
    }
    
    std::cout << "log-proposal densities: " << vTest3.t() << std::endl;
    }
    */
    //////////////////////////////////////////////////////////////////////////////////
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]) + model_.evaluateLogTransitionDensity(t, particlesNew[n], particlesOld[n]) - smcParameters_.getLogProposalDensity(n);
      
//       std::cout << " obs: " << model_.evaluateLogObservationDensity(t, particlesNew[n]) << "; trans: " << model_.evaluateLogTransitionDensity(t, particlesNew[n], particlesOld[n]) << "; prop: " <<  smcParameters_.getLogProposalDensity(n) << std::endl;
    }
//     for (unsigned int n=0; n<getNParticles(); n++)
//     {
//       logWeights(n) += model_.evaluateLogObservationDensity(t, particlesNew[n]) + model_.evaluateLogTransitionDensity(t, particlesNew[n], particlesOld[n]) - smcParameters_.getLogProposalDensity(n);
//     }
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
  
//       std::cout << "started sampleInitialParticles()" << std::endl;
      
//   if (model_.getModelParameters().getModelType() == MODEL_THRESHOLD_DEPENDENCE_OBSERVATIONS)
//   {
//     for (unsigned int t=0; t<model_.getModelParameters().getNObservationsCount()-1; t++)
//     {
//       // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_.
//       model_.getRefModelParameters().computeRhoFromStepFun(t, model_.getObservations().count_(t));
//     }
// 
//   }
//   else if (model_.getModelParameters().getModelType() == MODEL_DIRECT_DENSITY_DEPENDENCE)
//   {
//     // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_
//     arma::colvec countAux = arma::conv_to<arma::colvec>::from(model_.getObservations().count_(arma::span(0,model_.getObservations().count_.size()-2)));
// 
//     double meanCountAux = arma::accu(countAux) / countAux.size();
//     double sdCountAux = std::sqrt(arma::accu(arma::pow(countAux, 2.0)) - std::pow(meanCountAux, 2.0));
//   
//     model_.getRefModelParameters().setRho(
//       arma::exp(
//         model_.getModelParameters().getEpsilon0() + model_.getModelParameters().getEpsilon1() * (countAux-meanCountAux)/sdCountAux
//       )
//     );
//   }
  
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  { 
    // Sampling particles from the model transitions:
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      particlesNew[n] = model_.sampleFromInitialDistribution();
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_ALTERNATE_0)
  {
    unsigned int A = model_.getModelParameters().getNAgeGroups();
//     unsigned int K = model_.getModelParameters().getNLevels();
    
    arma::uvec x;
    arma::uvec xx(1);
    
    if (model_.getModelParameters().getModelType() == MODEL_MARKOV_SWITCHING)
    {
      x.set_size(A+2);
    }
    else 
    {
      x.set_size(A+1);
    }

//       std::cout << "nAgeGroups: " << A << std::endl;

    if (model_.getModelParameters().getInitialDistributionType() == INITIAL_DISTRIBUTION_NEGATIVE_BINOMIAL)
    {
      for (unsigned int n=0; n<getNParticles(); n++)
      {
        // NOTE: x(0) is unused!
        for (unsigned int a=1; a<A-1; a++)
        {
          x(a) = R::rnbinom(model_.getModelParameters().getNegativeBinomialSizeHyperInit0(), model_.getModelParameters().getNegativeBinomialProbHyperInit0());
        }
        x(A-1) = R::rnbinom(model_.getModelParameters().getNegativeBinomialSizeHyperInit1(), model_.getModelParameters().getNegativeBinomialProbHyperInit1());
        x(A) = arma::accu(x(arma::span(1,A-1))); // Note: this is unused
        
        particlesNew[n] = x;
      }
    }
    else if (model_.getModelParameters().getInitialDistributionType() == INITIAL_DISTRIBUTION_POISSON)
    {
      for (unsigned int n=0; n<getNParticles(); n++)
      {
        // NOTE: x(0) is unused!
        for (unsigned int a=1; a<A-1; a++)
        {
          x(a) = R::rpois(model_.getModelParameters().getChi0());
        }
        x(A-1) = R::rpois(model_.getModelParameters().getChi1());
        x(A) = arma::accu(x(arma::span(1,A-1))); // Note: this is unused
        
        particlesNew[n] = x;
      }
    }
    else if (model_.getModelParameters().getInitialDistributionType() == INITIAL_DISTRIBUTION_DISCRETE_UNIFORM)
    {
      for (unsigned int n=0; n<getNParticles(); n++)
      {
        x(arma::span(0,A-2)) = arma::randi<arma::uvec>(A-1,
                                  arma::distr_param(model_.getModelParameters().getMinHyperInit0(), model_.getModelParameters().getMaxHyperInit0()));
        xx = arma::randi<arma::uvec>(1,
                                  arma::distr_param(model_.getModelParameters().getMinHyperInit1(), model_.getModelParameters().getMaxHyperInit1()));
        
        x(A-1) = xx(0);
        
        x(A) = arma::accu(x(arma::span(1,A-1))); // Note: this is unused
        
        particlesNew[n] = x;
      }
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_LOOKAHEAD)
  {
//     std::cout << "setting up lookahead parameters at the first step" << std::endl;
    smcParameters_.setLookaheadParameters(getNLookaheadSteps(), model_.getModelParameters().getNAgeGroups());
    
    std::vector<arma::uvec> particlesOld;
    
//         std::cout << "proposing all particles at first step" << std::endl;
        
    smcParameters_.proposeAllParticles(
      particlesNew, particlesOld, 0, model_.getModelParameters().getNObservationsCount(), nParticles_, model_.getModelParameters().getNAgeGroups(), model_.getModelParameters().getKappa(), model_.getModelParameters().getChi0(), model_.getModelParameters().getChi1(), model_.getModelParameters().getPhi(), model_.getModelParameters().getRho(), model_.getObservations().count_
    );
  }
  
  if (isConditional_) {particlesNew[particleIndicesIn_(0)] = particlePath_[0];}
  
//  std::cout << "1st particle at initial step: " << particlesNew[0].t() << std::endl;
 
//        std::cout << "finished sampleInitialParticles()" << std::endl;
}
/// Computes the incremental particle weights at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogInitialParticleWeights
(
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{
//    std::cout << "started computeLogInitialParticleWeights()" << std::endl;
  
  if (smcProposalType_ == SMC_PROPOSAL_LOOKAHEAD)
  {
    
    //////////////////////////////////////////////////////////////////////////////////
    /*
    arma::colvec vTest1(nParticles_);
    arma::colvec vTest2(nParticles_);
    arma::colvec vTest3(nParticles_);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      vTest1(n) = model_.evaluateLogObservationDensity(0, particlesNew[n]);
    }
    
    std::cout << "log-observation densities: " << vTest1.t() << std::endl;
    
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      vTest2(n) = model_.evaluateLogInitialDensity(particlesNew[n]);
    }
    
    std::cout << "log-initial densities: " << vTest2.t() << std::endl;
    
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      vTest3(n) = smcParameters_.getLogProposalDensity(n);
    }
    
    std::cout << "log-proposal densities: " << vTest3.t() << std::endl;
    */
    //////////////////////////////////////////////////////////////////////////////////
    
    
    
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += model_.evaluateLogObservationDensity(0, particlesNew[n]) + model_.evaluateLogInitialDensity(particlesNew[n]) - smcParameters_.getLogProposalDensity(n);
    }
  }
  else 
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
  //     std::cout << "particlesNew[n]:" << particlesNew[n].t() << "; particlesNew[n].size(): " << particlesNew[n].size();
      
      logWeights(n) += model_.evaluateLogObservationDensity(0, particlesNew[n]);
    }
  }
//    std::cout << "finished computeLogInitialParticleWeights()" << std::endl;
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
  return model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particle);
}

/// Converts a particle path into the set of all latent variables in the model.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertParticlePathToLatentPath(const std::vector<Particle>& particlePath, LatentPath& latentPath)
{
//   std::cout << "START: convertParticlePathToLatentPath()" << std::endl;
  
  if (model_.getModelParameters().getModelType() == MODEL_MARKOV_SWITCHING)
  {
    convertStdVecToArmaMat(particlePath, latentPath.trueCounts_); 
    latentPath.productivityRates_.set_size(model_.getModelParameters().getNObservationsCount()-1);
    for (unsigned int t=0; t<model_.getModelParameters().getNObservationsCount()-1; t++)
    {
      latentPath.productivityRates_(t) = model_.getModelParameters().getNu(latentPath.trueCounts_(model_.getModelParameters().getNAgeGroups(), t+1));
    }
  }
  else 
  {
    convertStdVecToArmaMat(particlePath, latentPath.trueCounts_);
    latentPath.productivityRates_ = model_.getModelParameters().getRho();
  }
//    std::cout << "END: convertParticlePathToLatentPath()" << std::endl;
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  // Not needed (unless a (particle) Gibbs sampler is required).
//   convertArmaMatToStdVec(latentPath_.trueCounts_, particlePath);
}

#endif
