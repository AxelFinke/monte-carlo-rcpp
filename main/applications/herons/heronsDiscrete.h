/// \file
/// \brief Generating data and performing inference in "Herons" model. 
///
/// This file contains the functions for implementing the abstract model
/// class for the "Herons" model if the latent states are discrete.

#ifndef __HERONSDISCRETE_H
#define __HERONSDISCRETE_H

#include "main/templates/dynamic/stateSpace/stateSpace.h"
#include "main/algorithms/mcmc/Mcmc.h"
#include "main/rng/gaussian.h"

// [[Rcpp::depends("RcppArmadillo")]]


enum ModelIndex 
{ 
  MODEL_CONSTANT_PRODUCTIVITY = 0,                 // productivity rate constant over time
  MODEL_REGRESSED_ON_FDAYS,                        // productivity rates logistically regressed on fDays,
  MODEL_DIRECT_DENSITY_DEPENDENCE,                 // direct density dependence (log-productivity rates specified as a linear function of abundance),
  MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_OBSERVATIONS, // threshold dependence (of the productivity rate) on the observed heron counts (thresholds unknown),
  MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_TRUE_COUNTS,  // threshold dependence (of the productivity rate) on the true heron counts (thresholds unknown).
  MODEL_MARKOV_SWITCHING,                          // a Markov-switching state-space model,
  MODEL_KNOWN_THRESHOLD_DEPENDENCE_OBSERVATIONS,   // threshold dependence (of the productivity rate) on the observed heron counts (thresholds known), TODO
  MODEL_KNOWN_THRESHOLD_DEPENDENCE_TRUE_COUNTS     // threshold dependence (of the productivity rate) on the true heron counts (thresholds known). TODO
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
    
//     std::cout << "started setting up unknown parameters" << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////
    // Storing the parameters
    ///////////////////////////////////////////////////////////////////////////
    theta_  = theta;

    unsigned int nModelIndependentParameters = 2+2*nAgeGroups_; // number of model parameters which are identical for all model indices

    if (getUsePoissonInitialDistribution()) // i.e. if the prior on the initial state is Poisson
    {
      nModelIndependentParameters = nModelIndependentParameters + 2;
    } 
    if (getUseNegativeBinomialObservationEquation()) // i.e. if the observation equation uses a negative-binomial distribution
    {
      nModelIndependentParameters = nModelIndependentParameters + 1;
    }

    // Regression coefficients for the specification of the recovery probabilities:
//         std::cout << "Regression coefficients for the recovery probabilities" << std::endl;
    alpha0_ = theta(0);
    beta0_  = theta(1);
    lambda_ = inverseLogit(alpha0_ + beta0_ * timeNormCovar_); // vector of length nObservationsCount_-1; from 1928(!) [we don't use fDays_{1927} and have deleted it from the data file!]to 1997
    // NOTE: only a subset of the lambdas are actually used!
    
    // Regression coefficients for the specification of the survival probabilities:
//         std::cout << "Regression coefficients for the survival probabilities" << std::endl;
    alpha_.set_size(nAgeGroups_);
    beta_.set_size(nAgeGroups_);
    
    alpha_  = theta(arma::span(2,1+nAgeGroups_));
    beta_   = theta(arma::span(2+nAgeGroups_,1+2*nAgeGroups_));
    
    // Parameters for the measurement equation
    if (getUseNegativeBinomialObservationEquation())
    {
      omega_ = theta(2+2*nAgeGroups_);
      kappa_ = inverseLogit(omega_);
    }
    
    // Parameters for the distribution of the state at time $0$:
    if (getUsePoissonInitialDistribution()) {
      delta0_ = theta(nModelIndependentParameters-2);
      delta1_ = theta(nModelIndependentParameters-1);
      chi0_   = std::exp(delta0_);
      chi1_   = std::exp(delta1_);   
    }
    
    // Parameters for the step function governing the productivity rate (in some models)
    rho_.set_size(nObservationsCount_-1);
    
    if (modelIndex_ == MODEL_REGRESSED_ON_FDAYS)
    {
      // Coefficients for logistically regressing the productivity rate on fDays:
      gamma0_ = theta(nModelIndependentParameters);
      gamma1_ = theta(nModelIndependentParameters+1);
      
      rho_    = arma::exp(gamma0_ + gamma1_ * fDaysCovar_); // vector of length nObservationsCount_; from 1928(!) [we don't use fDays_{1927} and have deleted it from the data file!] to 1997
      
    }
    else if (modelIndex_ == MODEL_CONSTANT_PRODUCTIVITY)
    {
      psi_ = theta(nModelIndependentParameters); // the log-productivity rate
      rho_.fill(std::exp(psi_)); // the productivity rate
    }
    else if (modelIndex_ == MODEL_DIRECT_DENSITY_DEPENDENCE)
    {
      // Coefficients for regressing the log-productivity rate on the observations:
      epsilon0_ = theta(nModelIndependentParameters);
      epsilon1_ = theta(nModelIndependentParameters+1);
      
      // NOTE: for this model, rho_ is specified at the the first step of the SMC filter
    }
    else if (modelIndex_ == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_OBSERVATIONS || modelIndex_ == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
    {
      
      // NOTE: for threshold dependence on the true states,
      // $\rho_t$ must be specified individually for each particle; for threshold dependence
      // on the observations, $\rho_t$ is specified in sampleInitialParticles()
      

      zeta_.set_size(nLevels_);
      nu_.set_size(nLevels_);
      
      eta_.set_size(nLevels_-1);
      tau_.set_size(nLevels_-1);

      
      zeta_  = theta(arma::span(nModelIndependentParameters, nModelIndependentParameters+1+nLevels_-2));
      eta_   = theta(arma::span(nModelIndependentParameters+1+nLevels_-1, nModelIndependentParameters+1+2*(nLevels_-1)-1));

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
    else if (modelIndex_ == MODEL_MARKOV_SWITCHING)
    {
      
      zeta_.set_size(nLevels_);
      nu_.set_size(nLevels_);
            
      zeta_  = theta(arma::span(nModelIndependentParameters, nModelIndependentParameters+1+nLevels_-2));
      nu_    = arma::cumsum(arma::exp(zeta_)); // NOTE: the levels are now increasing rather than decreasing!
      
      varpi_.set_size(nLevels_, nLevels_);
      P_.set_size(nLevels_, nLevels_);
      for (unsigned int k=0; k<nLevels_; k++)
      {
        varpi_.row(k) = arma::trans(theta_(arma::span(nModelIndependentParameters+1+nLevels_-1+k*nLevels_, nModelIndependentParameters+1+nLevels_-2+nLevels_+k*nLevels_)));
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
      phi_.row(a) = arma::trans(inverseLogit(alpha_(a) + beta_(a) * fDaysCovar_));
    }

//     phi_.fill(0.6); 
//     std::cout << "phi_: " << phi_ << std::endl;
//     std::cout << "WARNING: we're fixing phi!" << std::endl;
    
    // Computing the matrix of multinomial ring-recovery probabilities.
    Q_.zeros(nObservationsRing_, nObservationsRing_+1);
    
    // Auxiliary time indices 
    unsigned int t1 = t1Ring_ - t1Count_ + 1;
//     unsigned int t2 = t2Ring_ - t1Count_ + 1;
    
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
    
//     std::cout << "phi: " << phi_ << std::endl;
//      std::cout << "kappa_: " << kappa_ << std::endl;
//           std::cout << "chi0_: " << chi0_ << std::endl;
//      std::cout << "chi1_: " << chi1_ << std::endl;
//      std::cout << "epsilon0_: " << epsilon0_ << std::endl;
//      std::cout << "epsilon1_: " << epsilon1_ << std::endl;
//     std::cout << "Q: " << Q_ << std::endl;
    
//     std::cout << "Finished set up unknown parameters" << std::endl;
    
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyp)
  {
    std::cout << "WARNING: we have currently replaced the observation density by a Gaussian for test purposes!" << std::endl;
    
//     dimLatentVariable_ = 2;
    
    modelIndex_ = static_cast<ModelIndex>(hyp(0));
    
//     std::cout << "current model index: " << modelIndex_ << std::endl;
    
    t1Count_ = static_cast<unsigned int>(hyp(1)); // first year for which count-data observations are available
    t2Count_ = static_cast<unsigned int>(hyp(2)); // last year for which count-data observations are available
    t1Ring_  = static_cast<unsigned int>(hyp(3)); // first year for which ring-recovery observations are available
    t2Ring_  = static_cast<unsigned int>(hyp(4)); // last year for which ring-recovery observations are available
    
    nObservationsCount_ = t2Count_ - t1Count_ + 1;
    nObservationsRing_  = t2Ring_  - t1Ring_  + 1;
    
    dimTheta_      = static_cast<unsigned int>(hyp(5));
    meanHyper_     = hyp(arma::span(6,dimTheta_+5));
    sdHyper_       = hyp(arma::span(dimTheta_+6,2*dimTheta_+5));
    nAgeGroups_    = static_cast<unsigned int>(hyp(2*dimTheta_+6));
    nLevels_       = static_cast<unsigned int>(hyp(2*dimTheta_+7));
    
    usePoissonInitialDistribution_ = static_cast<bool>(hyp(2*dimTheta_+8));
    useNegativeBinomialObservationEquation_ = static_cast<bool>(hyp(2*dimTheta_+9));
    minHyperInit0_ = static_cast<unsigned int>(hyp(2*dimTheta_+10));
    minHyperInit1_ = static_cast<unsigned int>(hyp(2*dimTheta_+11));
    maxHyperInit0_ = static_cast<unsigned int>(hyp(2*dimTheta_+12));
    maxHyperInit1_ = static_cast<unsigned int>(hyp(2*dimTheta_+13));
    
    timeNormCovar_ = hyp(arma::span(2*dimTheta_+14,2*dimTheta_+12 + nObservationsCount_)); // "normalised" years
    fDaysCovar_    = hyp(arma::span(2*dimTheta_+13 +nObservationsCount_,2*dimTheta_+11 + 2*nObservationsCount_));
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
  
   /// Returns the model index.
  ModelIndex getModelIndex() const {return modelIndex_;}
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
  /// Returns the productivity rate (if calculated based on the true counts);
  double getRhoTrueCount(const unsigned int t) const {return rhoTrueCount_(t);}
  /// Returns the productivity rate.
  arma::colvec getRho() const {return rho_;}
  /// Returns the survival probabilities.
  arma::mat getPhi() const {return phi_;}
  /// Returns the recovery probabilities.
  arma::colvec getLambda() const {return lambda_;}
  /// Returns the $i$th row of the multinomial ring-recovery probabilities
  arma::rowvec getQ(const unsigned int i) const {return Q_.row(i);}
  
  /// Returns the first regression parameter in the direct-density dependence case.
  double getEpsilon0() const {return epsilon0_;}
  /// Returns the second regression parameter in the direct-density dependence case.
  double getEpsilon1() const {return epsilon1_;}
  
  /// Returns the $i$th row of the transition matrix for the 
  /// regime indicator variable
  arma::rowvec getP(const unsigned int i) const {return P_.row(i);}
  
  /// Returns the maximum of the support of the discrete uniform distribution for the initial population counts for age groups $1,\dotsc, A-1$
  int getMaxHyperInit0() const {return maxHyperInit0_;}
  /// Returns the maximum of the support of the discrete uniform distribution for the initial population counts for age group $A$
  int getMaxHyperInit1() const {return maxHyperInit1_;}
  /// Returns the minimum of the support of the discrete uniform distribution for the initial population counts for age groups $1,\dotsc, A-1$
  int getMinHyperInit0() const {return minHyperInit0_;}
  /// Returns the minimum of the support of the discrete uniform distribution for the initial population counts for age group $A$
  int getMinHyperInit1() const {return minHyperInit1_;}
  /// Returns whether a Poisson distribution is used to model the initial population counts (instead of a discrete-uniform distribution).
  bool getUsePoissonInitialDistribution() const {return usePoissonInitialDistribution_;}
  /// Returns whether the conditional distribution of the observations conditional on the latent states follows a negative-binomial distribution (instead of a Poisson distribution).
  bool getUseNegativeBinomialObservationEquation() const {return useNegativeBinomialObservationEquation_;}
  
  /// Specifies the productivity rate.
  void setRho(const arma::colvec& rho) {rho_ = rho;}
  /// Specifies the productivity rate.
  void setRho(const unsigned int t, const double rho) {rho_(t) = rho;}
  
  /// Returns the $i$th level of the productivity rates.
  double getNu(const unsigned int i) const { return nu_(i); }
  
  
  
private:
  
  // Unknown model parameters
  arma::colvec theta_; // the full parameter vector
  double omega_, kappa_; // parameters for the measurement equation
  double epsilon0_, epsilon1_, gamma0_, gamma1_; // regression parameters for the productivity rates
  double delta0_, delta1_, chi0_, chi1_; // parameters for the distribution of the state at time 0
  double psi_; // constant log-productivity rate
  arma::colvec zeta_, eta_, nu_, tau_; // parameters for the step functions
  arma::colvec rho_, rhoTrueCount_; // productivity rates
  arma::colvec alpha_, beta_; // regression parameters for the survival probabilities
  double alpha0_, beta0_; // regression parameters for the recovery probabilities
  arma::colvec lambda_; // recovery probabilities
  arma::mat phi_; // a (nAgeGroups, nObservationsRing)-matrix of survival probabilities; NOTE: the first column contains the survival probabilties $\phi_{a,0}$.
  arma::mat Q_; // multinomial probabilities for the ring-recovery data
  
  arma::mat P_, varpi_; // (normalised/unnormalised) transition matrix for the latent regime-switches
  
  
  // Known hyperparameters for the prior distribution:
  ModelIndex modelIndex_; // index for model specification
  unsigned int dimTheta_; // length of the parameter vector theta
  arma::colvec meanHyper_, sdHyper_; // means and standard deviations of the Gaussian priors on all parameters
  
  int minHyperInit0_, minHyperInit1_, maxHyperInit0_, maxHyperInit1_; // minimum/maximum of the support of the discrete uniform distribution for the initial population counts (for age groups $1,\dotsc, A-1$ and $A$, repectively)
  bool usePoissonInitialDistribution_; // should we use a Poisson initial distribution (rather than discrete-uniform) distribution for the first latent state?
  bool useNegativeBinomialObservationEquation_; // should the conditional distribution of the observations conditional on the latent states follow a negative-binomial distribution (instead of a Poisson distribution)?
  
  unsigned int nAgeGroups_; // number of age groups
  unsigned int nLevels_; // number of levels in the step function for productivity
//   unsigned int dimLatentVariable_; // dimension of the state vector
  
  unsigned int nObservationsRing_; // number of years for which ring-recovery observations are available
  unsigned int nObservationsCount_; // number of years for which count-data observations are available
  unsigned int t1Count_, t2Count_; // first, last year for which count-data observations are available
  unsigned int t1Ring_, t2Ring_; // first, last year for which ring-recovery observations are available
  
  // Other known covariates:
  arma::colvec timeNormCovar_; // length-(nObservationsCount_-1) vector of "normalised" years (1928--1997)
  arma::colvec fDaysCovar_; // vector of length nObservationsCount_-1; from 1928(!) [we don't use fDays_{1927} and have deleted it from the data file!]to 1997

};
/// Holds some additional auxiliary parameters for the SMC algorithm.
class SmcParameters
{
  
public:
  
  // All of the following parameters/functions are only used if we use the lookahead
  // proposal which uses a Kalman filter/smoother to construct the proposal kernel at time $t$.
  
  /// Determines some auxiliary parameters at the start of the algorithm.
  void setParameters(const unsigned int K, const unsigned int nAgeGroups) 
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
    
      if (t <=10)
      {
          std::cout << "mu_ " << mu_.t() << "";
//            std::cout << "sigma_ " << sigma_ << std::endl;
      }
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
        
      if (t <= 5) 
      {
      std::cout << "proposal weight components: particle(a), mu(a), sqrt(sigma(a,a)), ub: " << particleNew(a) << " " << mu_(a) << " " << sqrtSigma << " " << ub << " " <<
        R::pnorm(particleNew(a),   mu_(a), sqrtSigma, true, false) 
        << " " <<
        R::pnorm(particleNew(a)-1, mu_(a), sqrtSigma, true, false)
        << " " << 
        R::pnorm(ub, mu_(a), sqrtSigma, true, false)
        << " " <<
        R::pnorm(-1, mu_(a), sqrtSigma, true, false)
        << std::endl;
      }
      
 
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
  
//   std::cout << "meanHyper: " << arma::trans(modelParameters_.getMeanHyper()) << std::endl;
//   std::cout << "theta.size(): " << theta.size() << std::endl;
  
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
  
  // TODO: need to make sure that dimLatentVariable_ is set correctly!
  if (modelParameters_.getModelIndex() == MODEL_MARKOV_SWITCHING)
  {
    latentPath_.set_size(modelParameters_.getNAgeGroups()+1, modelParameters_.getNObservationsCount());
  }
  else 
  {
    latentPath_.set_size(modelParameters_.getNAgeGroups(), modelParameters_.getNObservationsCount());
  }

  latentPath_.col(0) = sampleFromInitialDistribution();
  sampleFromObservationEquation(0, observations_, latentPath_.col(0));
  for (unsigned int t=1; t<modelParameters_.getNObservationsCount(); ++t)
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
//   std::cout << "WARNING: evalutation of the log-complete likelihood is not implemented yet! But this is only needed for Gibbs-sampling type algorithms" << std::endl;
  
  if (modelParameters_.getModelIndex() == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_OBSERVATIONS)
  {
    for (unsigned int t=0; t<modelParameters_.getNObservationsCount()-1; t++)
    {
      // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_.
      modelParameters_.computeRhoFromStepFun(t, getObservations().count_(t));
//       std::cout << "rho(" << t << "): " << model_.getModelParameters().getRho(t) << " " ;
    }

  }
  else if (modelParameters_.getModelIndex() == MODEL_DIRECT_DENSITY_DEPENDENCE)
  {
      // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_
    arma::colvec countAux = arma::conv_to<arma::colvec>::from(getObservations().count_(arma::span(0,getObservations().count_.size()-2)));
    
    ///////////////////
//     std::cout << "NOTE: we're now normalising the observed counts in the direct-density dependence model!" << std::endl;
    double meanCountAux = static_cast<double>(arma::accu(countAux)) / countAux.size();
    double sdCountAux   = std::sqrt(arma::accu(arma::pow(countAux, 2.0)) - std::pow(meanCountAux, 2.0));
  
    modelParameters_.setRho(
      arma::exp(
        modelParameters_.getEpsilon0() + modelParameters_.getEpsilon1() * (countAux-meanCountAux)/sdCountAux
      )
    );
  }
  
  
  double logLike = evaluateLogInitialDensity(x.col(0)) + 
                   evaluateLogObservationDensity(0, x.col(0));
                   
  for (unsigned int t=1; t<modelParameters_.getNObservationsCount(); ++t)
  {
    logLike += evaluateLogTransitionDensity(t, x.col(t), x.col(t-1)) + 
               evaluateLogObservationDensity(t, x.col(t));
  }
  return logLike + evaluateLogPartialLikelihood();
}
/// Samples a single latent variable at Time t=0 from its conditional prior
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
LatentVariable Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromInitialDistribution()
{

  unsigned int A = modelParameters_.getNAgeGroups();
  arma::uvec x;
  arma::uvec xx(1);
  
  if (modelParameters_.getModelIndex() == MODEL_MARKOV_SWITCHING)
  {
    x.set_size(A+1);
    xx = arma::randi<arma::uvec>(1, arma::distr_param(0, modelParameters_.getNLevels()-1));
    x(A) = xx(0);
  }
  else 
  {
    x.set_size(A);
  }

  if (modelParameters_.getUsePoissonInitialDistribution())
  {
    // Sampling the state at time 0 (counting "time $0$" as in the paper!)
    for (unsigned int a=0; a<A-1; a++)
    {
      x(a) = R::rpois(modelParameters_.getChi0());
    }
    x(A-1) = R::rpois(modelParameters_.getChi1());
  }
  else
  {
    x(arma::span(0,A-2)) = arma::randi<arma::uvec>(A-1,
                                 arma::distr_param(modelParameters_.getMinHyperInit0(), modelParameters_.getMaxHyperInit0()));
    xx = arma::randi<arma::uvec>(1,
                                 arma::distr_param(modelParameters_.getMinHyperInit1(), modelParameters_.getMaxHyperInit1()));
    x(A-1) = xx(0);
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
  if (modelParameters_.getModelIndex() == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
  {
    modelParameters_.computeRhoFromStepFun(t-1, totalPopSize);
  }
  
  
  if (modelParameters_.getModelIndex() == MODEL_MARKOV_SWITCHING)
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
  
  if (modelParameters_.getUsePoissonInitialDistribution())
  {
    // Sampling the state at time 0 (counting "time $0$" as in the paper!)
    for (unsigned int a=0; a<A-1; a++)
    {
      logDensity += R::dpois(latentVariable(a), modelParameters_.getChi0(), true);
    }
    logDensity += R::dpois(latentVariable(A-1), modelParameters_.getChi1(), true);
  }
  else
  {
    logDensity -= (A-1)*std::log(modelParameters_.getMaxHyperInit0() - modelParameters_.getMinHyperInit0());
    logDensity -= std::log(modelParameters_.getMaxHyperInit1() - modelParameters_.getMinHyperInit1());
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
  if (modelParameters_.getModelIndex() == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
  {
    modelParameters_.computeRhoFromStepFun(t-1, totalPopSize);
  }
  
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
  
  if (modelParameters_.getUseNegativeBinomialObservationEquation())
  {
    double sz = modelParameters_.getKappa() / (1.0 - modelParameters_.getKappa()) * adultPopulation;
    return R::dnbinom(observations_.count_(t), sz, modelParameters_.getKappa(), true);
  } 
  else
  {
    // NOTE: this assumes a Poisson measurement equation:
    //////// return R::dpois(observations_.count_(t), adultPopulation, true);
    return R::dnorm(static_cast<double>(observations_.count_(t)), static_cast<double>(adultPopulation), 500,  true);
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
  
  if (modelParameters_.getUseNegativeBinomialObservationEquation())
  {
    // NOTE: this samples from the original negative-binomial distribution:
    observations_.count_(t) = R::rnbinom(sz, modelParameters_.getKappa());
  }
  else 
  {
    // NOTE: this samples from a Poisson distribution:
    observations_.count_(t) = R::rpois(adultPopulation);
  }

  if (modelParameters_.getModelIndex() == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_OBSERVATIONS)
  {
    if (t < modelParameters_.getNObservationsCount()-1)
    {
      modelParameters_.computeRhoFromStepFun(t, observations_.count_(t));
    }
  }
  else if (modelParameters_.getModelIndex() == MODEL_DIRECT_DENSITY_DEPENDENCE)
  {
    // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_
    std::cout << "WARNING: we cannot use direct-density dependence with normalising the count data!" << std::endl; 
//     arma::colvec countAux = arma::conv_to<arma::colvec>::from(observations_.count_(arma::span(0,observations_.count_.size()-2)));
//     double meanCountAux = arma::accu(countAux) / countAux.size();
//     double sdCountAux = std::sqrt(arma::accu(arma::pow(countAux, 2.0)) - std::pow(meanCountAux, 2.0));
//   
//     modelParameters_.setRho(t, 
//       std::exp(
//         modelParameters_.getEpsilon0() + modelParameters_.getEpsilon1() * (countAux-meanCountAux)/sdCountAux
//       )
//     );
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
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogPartialLikelihood()
{
  double logLike = 0.0;
  
//       std::cout << "TRing " << modelParameters_.getNObservationsRing() << std::endl;

  // Likelihood based on ring-recovery data
  for (unsigned int i=0; i<modelParameters_.getNObservationsRing(); i++)
  {
//     std::cout << "i: " << i << std::endl;
//     std::cout << modelParameters_.getQ(i) << std::endl;
//     std::cout << observations_.getNRinged(i) << std::endl;
//     std::cout << observations_.getRingRecovery(i) << std::endl;
//     
//     std::cout << "partial loglikelihood, term " << i << " " << logMultinomialDensity(observations_.getRingRecovery(i), observations_.getNRinged(i), modelParameters_.getQ(i)) << std::endl;
    
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
//   std::cout << "============= started sampleParticles() at time " << t << " ================== " << std::endl;
  
//   std::cout << "phi(,t-1): " << model_.getModelParameters().getPhi(0,t-1) << " " << model_.getModelParameters().getPhi(1,t-1) << " " << model_.getModelParameters().getPhi(2,t-1) << " " << std::endl;
  
  if (prop_ == SMC_PROPOSAL_PRIOR)
  { 
    // Sampling particles from the model transitions:
    for (unsigned int n=0; n<getNParticles(); n++)
    {
//               std::cout << particlesOld[n].t() << std::endl;
      particlesNew[n] = model_.sampleFromTransitionEquation(t, particlesOld[n]);
//       std::cout << particlesNew[n].t() << std::endl;
    }
  }
  else if (prop_ == SMC_PROPOSAL_ALTERNATE_0)
  {
    
    unsigned int A = model_.getModelParameters().getNAgeGroups();
    arma::uvec xx(1);
    arma::uvec x;
//     unsigned int K = model_.getModelParameters().getNLevels();
    
    if (model_.getModelParameters().getModelIndex() == MODEL_MARKOV_SWITCHING)
    {
      x.set_size(A+2);
    }
    else
    {
      x.set_size(A+1);
    }
    
    // Total population at the previous time step (except first-years and adults).
//     unsigned int totalPopSize = 0;
 
    // Sampling herons at time $t$ in years $2$ to $A$ from the model transitions
    // but sampling first-years at time $t-1$.
    for (unsigned int n=0; n<getNParticles(); n++)
    {   

      x(A) = arma::accu(particlesOld[n](arma::span(1,A-1)));
            
      if (model_.getModelParameters().getModelIndex() == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_TRUE_COUNTS)
      { 
        //model_.getRefModelParameters().computeRhoFromStepFun(t-1, x(A));
        if (t > 1)
        {
          model_.getRefModelParameters().computeRhoFromStepFun(t-2, particlesOld[n](A));
        }
      }
      else if (model_.getModelParameters().getModelIndex() == MODEL_MARKOV_SWITCHING)
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
          model_.getRefModelParameters().setRho(t-2, model_.getModelParameters().getNu(particlesOld[n](A+1)));
        }
        
      }
      
      if (t == 1)
      {
        if (model_.getModelParameters().getUsePoissonInitialDistribution())
        {
          for (unsigned int a=0; a<A-1; a++)
          {
            x(a) = R::rpois(model_.getModelParameters().getChi0());
          }
          x(A-1) = R::rpois(model_.getModelParameters().getChi1());
        }
        else
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
  else if (prop_ == SMC_PROPOSAL_LOOKAHEAD)
  {
    smcParameters_.proposeAllParticles(particlesNew, particlesOld, t, model_.getModelParameters().getNObservationsCount(), nParticles_, model_.getModelParameters().getNAgeGroups(), model_.getModelParameters().getKappa(), model_.getModelParameters().getChi0(), model_.getModelParameters().getChi1(), model_.getModelParameters().getPhi(), model_.getModelParameters().getRho(), model_.getObservations().count_);
  }
//    std::cout << " particleNew[0] at $t$th step: " << particlesNew[0].t() << " " << "; particleOld[0] at $t$th step: " << particlesOld[0].t() << std::endl;

//   std::cout << "============= finished sampleParticles() at time " << t << " ================== " << std::endl;
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
//   std::cout << "particlesNew[0]:" << particlesNew[0].t() << " ";
  if (prop_ == SMC_PROPOSAL_LOOKAHEAD)
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
      
      std::cout << " obs: " << model_.evaluateLogObservationDensity(t, particlesNew[n]) << "; trans: " << model_.evaluateLogTransitionDensity(t, particlesNew[n], particlesOld[n]) << "; prop: " <<  smcParameters_.getLogProposalDensity(n) << std::endl;
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
      
  if (model_.getModelParameters().getModelIndex() == MODEL_UNKNOWN_THRESHOLD_DEPENDENCE_OBSERVATIONS)
  {
    for (unsigned int t=0; t<model_.getModelParameters().getNObservationsCount()-1; t++)
    {
      // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_.
      model_.getRefModelParameters().computeRhoFromStepFun(t, model_.getObservations().count_(t));
//       std::cout << "rho(" << t << "): " << model_.getModelParameters().getRho(t) << " " ;
    }

  }
  else if (model_.getModelParameters().getModelIndex() == MODEL_DIRECT_DENSITY_DEPENDENCE)
  {
    // NOTE: getRefModelParameters returns a non-constant reference to modelParameters_
    arma::colvec countAux = arma::conv_to<arma::colvec>::from(model_.getObservations().count_(arma::span(0,model_.getObservations().count_.size()-2)));
//       model_.getRefModelParameters().setRho(
//         arma::exp(
//           model_.getModelParameters().getEpsilon0() + model_.getModelParameters().getEpsilon1() * countAux
//         )
//       );
    
    ///////////////////
//     std::cout << "NOTE: we're now normalising the observed counts in the direct-density dependence model!" << std::endl;
    double meanCountAux = arma::accu(countAux) / countAux.size();
    double sdCountAux = std::sqrt(arma::accu(arma::pow(countAux, 2.0)) - std::pow(meanCountAux, 2.0));
  
    model_.getRefModelParameters().setRho(
      arma::exp(
        model_.getModelParameters().getEpsilon0() + model_.getModelParameters().getEpsilon1() * (countAux-meanCountAux)/sdCountAux
      )
    );
    
//          std::cout << "rho: " << arma::trans(model_.getModelParameters().getRho()) << std::endl;
  }
  
  if (prop_ == SMC_PROPOSAL_PRIOR)
  { 
    // Sampling particles from the model transitions:
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      particlesNew[n] = model_.sampleFromInitialDistribution();
    }
  }
  else if (prop_ == SMC_PROPOSAL_ALTERNATE_0)
  {
    unsigned int A = model_.getModelParameters().getNAgeGroups();
//     unsigned int K = model_.getModelParameters().getNLevels();
    
    arma::uvec x;
    arma::uvec xx(1);
    
    if (model_.getModelParameters().getModelIndex() == MODEL_MARKOV_SWITCHING)
    {
      x.set_size(A+2);
    }
    else 
    {
      x.set_size(A+1);
    }

//       std::cout << "nAgeGroups: " << A << std::endl;
    if (model_.getModelParameters().getUsePoissonInitialDistribution())
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
    else
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
  else if (prop_ == SMC_PROPOSAL_LOOKAHEAD)
  {
//     std::cout << "setting up lookahead parameters at the first step" << std::endl;
    smcParameters_.setParameters(getNLookaheadSteps(), model_.getModelParameters().getNAgeGroups());
    
    std::vector<arma::uvec> particlesOld;
    
//         std::cout << "proposing all particles at first step" << std::endl;
        
    smcParameters_.proposeAllParticles(
      particlesNew, particlesOld, 0, model_.getModelParameters().getNObservationsCount(), nParticles_, model_.getModelParameters().getNAgeGroups(), model_.getModelParameters().getKappa(), model_.getModelParameters().getChi0(), model_.getModelParameters().getChi1(), model_.getModelParameters().getPhi(), model_.getModelParameters().getRho(), model_.getObservations().count_
    );
  }
  
  
  
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
  
  if (prop_ == SMC_PROPOSAL_LOOKAHEAD)
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
  convertStdVecToArmaMat(particlePath, latentPath);
}
/// Converts the set of all latent variables in the model into a particle path.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::convertLatentPathToParticlePath(const LatentPath& latentPath, std::vector<Particle>& particlePath)
{
  convertArmaMatToStdVec(latentPath, particlePath);
}

#endif
