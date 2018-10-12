/// \file
/// \brief Generating data and calculating the likelihood in a state-space model.
///
/// This file contains the functions for sampling data and running a Kalman
/// filter in a multivariate linear Gaussian state-space model.

#ifndef __LINEARGAUSSIANHMM_H
#define __LINEARGAUSSIANHMM_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
// #include <gperftools/profiler.h>
#include <iostream>
#include <vector>
#include "base/rng/Rng.h"
#include "base/rng/gaussian.h"
#include "base/helperFunctions.h"
#include "base/smc/resample.h"

////////////////////////////////////////////////////////////////////////////////
// Obtain log-marginal likelihood via the Kalman filter
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
class Kalman
{
  
public:
  // Constructor
  Kalman( 
    const arma::mat& A_,  // must be a (dx, dx)-matrix
    const arma::mat& B_,  // must be a (dx, dx)-matrix
    const arma::mat& C_,  // must be a (dy, dx)-matrix
    const arma::mat& D_,  // must be a (dy, dy)-matrix
    const arma::vec& m0_, // must be a (dx, 1)-vector
    const arma::mat& C0_, // must be a (dx, dx)-matrix
    const arma::mat& y_   // must be a (dy, T)-matrix
  ) :
    dx(A_.n_rows), dy(y_.n_rows), T(y_.n_cols),
    A(A_), B(B_), C(C_), D(D_), C0(C0_), y(y_),
    m0(m0_)
  { 
    // empty
  }
  
  // Destructor
  //~Kalman();
  
  double get_logLike()
  {
    return logLike;
  }

  // Member functions
  void runKalman()
  {
  
    // Predictive mean and covariance matrix
    arma::vec mP(dx);
    arma::mat CP(dx, dx);
    
    // Updated mean and covariance matrix
    arma::vec mU(dx);
    arma::mat CU(dx, dx);
    
    // Mean and covariance matrix of the incremental likelihood
    arma::vec mY(dy);
    arma::mat CY(dy, dy);
    
    // Log-marginal likelihood
    logLike = 0;
    
    // Auxiliary quantities
    arma::mat Q = B*B.t();
    arma::mat R = D*D.t();
    arma::mat kg;
    
    for (unsigned int t=0; t<T; t++)
    {
      
      // Prediction step
      if (t > 0) 
      {
        mP = A * mU;
        CP = A * CU * A.t() + Q; 
      } 
      else 
      {
	mP = m0;
        CP = C0; 
      }
      
      // Likelihood step
      mY = C * mP;
      CY = C * CP * C.t() + R;
 
      // Update step
      kg = (arma::solve(CY.t(), C * CP.t())).t();
      mU = mP + kg * (y.col(t) - mY);
      CU = CP - kg * C * CP;

      // Adding the incremental log-marginal likelihood
      logLike += arma::as_scalar(gaussian::evaluateDensityMultivariate(y.col(t), mY, CY, false, true));
      
    }
    
  }
   
private:
  const unsigned int dx, dy, T;
  const arma::mat A, B, C, D, C0, y;
  const arma::vec m0;
  double logLike;
};

////////////////////////////////////////////////////////////////////////////////
// Some types for use in SMC algorithms
////////////////////////////////////////////////////////////////////////////////
/// Proposal types for the particles.
enum ProposalType 
{ 
  SMC_PROPOSAL_PRIOR = 0, 
  SMC_PROPOSAL_OPTIMAL
};
/// Backward sampling types.
enum CsmcType 
{  
  CSMC_BACKWARD_KERNEL_NONE = 0,
  CSMC_BACKWARD_KERNEL_STANDARD,
  CSMC_BACKWARD_KERNEL_ANCESTOR
};
/// SMC filter types.
enum FilterType 
{ 
  SMC_FILTER_STANDARD = 0,
  SMC_FILTER_BLOCK,
  SMC_FILTER_EXACT
};
/// Backward sampling types.
enum BackwardKernelType 
{  
  SMC_BACKWARD_KERNEL_STANDARD = 0,
  SMC_BACKWARD_KERNEL_BLOCK
};
/// Type of parametrisation employed by parameter-estimation algorithms.
enum ParametrisationType 
{ 
  SMC_PARAMETRISATION_UNBOUNDED = 0, // parametrisation in terms of components of A, and log(B(0,0)), log(D(0,0))
  SMC_PARAMETRISATION_NATURAL        // parametrisation in terms of components of A, B, D 
};

////////////////////////////////////////////////////////////////////////////////
// Perform particle-based inference in linear Gaussian HMMs
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
class Model
{
public:
  // Constructors
  Model(
    Rng* rng_, unsigned int N_, double H_, 
    FilterType filt_, ProposalType prop_, ParametrisationType para_,
    arma::mat& A_, arma::mat& B_, arma::mat& C_, 
    arma::mat& D_, 
    arma::vec& m0_, arma::mat& C0_, const arma::mat& y_, 
    bool isSimple_, unsigned int cores_) : rng(rng_), y(y_)
  {
    N = N_; 
    H = H_; 
    A = A_; 
    B = B_; 
    C = C_; 
    D = D_; 
    m0 = m0_; 
    C0 = C0_; 
    V = A.n_rows; 
    dy = y.n_rows; 
    T = y.n_cols;
    K = static_cast<unsigned int>(arma::as_scalar(arma::find(A == 0, 1, "first")) -1 ); // number of non-zero elements in the first row of A
    filt = filt_;
    prop = prop_;
    para = para_;
    isSimple = isSimple_;
    theta.set_size(K+3);
    set_theta();
    setAuxiliaryParameters();
    storeHistory = false;
    storePath = false;
  }

  
  Model(
    Rng* rng_, unsigned int N_, double H_, 
    FilterType filt_, ProposalType prop_, ParametrisationType para_,
    arma::colvec& thetaInit_, arma::colvec& stepSizes_, arma::colvec& m0_, 
    arma::mat& C0_, const arma::mat& y_, 
    bool isSimple_, unsigned int cores_) : rng(rng_), y(y_)
  { 
    V = C0_.n_rows; 
    dy = y.n_rows; 
    isSimple = isSimple_;
    cores = cores_; 
    N = N_; 
    H = H_; 
    filt = filt_;
    prop = prop_;
    para = para_;
    m0 = m0_; 
    C0 = C0_;
    thetaInit = thetaInit_;
    theta = thetaInit;
    stepSizes = stepSizes_;
    K = theta.n_rows - 3; // number of non-zero elements in the first row of A
    setParameters();
    setAuxiliaryParameters();
    T = y.n_cols;
    storeHistory = false;
    storePath = false;
  }
  
  /// Destructor
  //~Model();

  /// Returns the number of unknown static parameters.
  unsigned int getDimTheta() 
  {
    return K + 3;
  }
  /// Obtains the estimate of the normalising constant
  double get_logLikeEst() 
  {
    return logLikeEst;
  }
  double get_logLikeTrue() 
  {
    return logLikeTrue;
  }
  /// Obtains the estimates of theta obtained from a gradient-ascent algorithm
  arma::mat get_thetaFull() 
  {
    return thetaFull;
  }
  /// Obtains the particle paths from backward sampling
  arma::cube get_xBack() 
  {
    return xBack;
  }
  /// Obtains exact smoothing means
  arma::mat get_mS() 
  {
    return mS;
  }
  /// Obtains exact smoothing covariance matrices
  arma::cube get_CS() 
  {
    return CS;
  }
  /// Obtains blocked particle approximations of marginal smoothing distributions
  /*
  arma::cube get_marginals() 
  {
    return marginals;
  }
  */
  /// Obtains componentwise approximation of the expectation of the sufficient statistic H
  arma::cube get_suffHCompEst() 
  {
    return suffHCompEst;
  }
  /// Obtains componentwise approximation of the expectation of the sufficient statistic H
  arma::mat get_suffHEst() 
  {
    return suffHEst;
  }
  /// Obtains componentwise approximation of the expectation of the sufficient statistic H
  arma::mat get_suffQCompEst() 
  {
    return suffQCompEst;
  }
  /// Obtains componentwise approximation of the expectation of the sufficient statistic H
  arma::vec get_suffQEst() 
  {
    return suffQEst;
  }
  /// Obtains componentwise approximation of the expectation of the sufficient statistic R
  arma::colvec get_suffRCompEst() 
  {
    return suffRCompEst;
  }
  /// Obtains componentwise approximation of the expectation of the sufficient statistic R
  double get_suffREst() 
  {
    return suffREst;
  }
  /// Obtains componentwise approximation of the expectation of the sufficient statistic S
  arma::colvec get_suffSCompEst() 
  {
    return suffSCompEst;
  }
  /// Obtains componentwise approximation of the expectation of the sufficient statistic S
  double get_suffSEst() 
  {
    return suffSEst;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic H
  arma::cube get_suffHCompTrue() 
  {
    return suffHCompTrue;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic H
  arma::mat get_suffHTrue() 
  {
    return suffHTrue;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic H
  arma::mat get_suffQCompTrue() 
  {
    return suffQCompTrue;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic H
  arma::vec get_suffQTrue() 
  {
    return suffQTrue;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic R
  arma::colvec get_suffRCompTrue() 
  {
    return suffRCompTrue;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic R
  double get_suffRTrue() 
  {
    return suffRTrue;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic S
  arma::colvec get_suffSCompTrue() 
  {
    return suffSCompTrue;
  }
  /// Obtains componentwise calculation of the expectation of the sufficient statistic S
  double get_suffSTrue() 
  {
    return suffSTrue;
  }
  /// Obtains componentwise approximation of the expectation of thesmoothing functionals needed
  /// to calculate the gradient
  arma::mat get_gradCompEst() 
  {
    return gradCompEst;
  }
  /// Obtains componentwise approximation of the expectation of thesmoothing functionals needed
  /// to calculate the gradient
  arma::vec get_gradEst() 
  {
    return gradEst;
  }
  /// Obtains componentwise calculation of the expectation of the smoothing functionals needed
  /// to calculate the gradient
  arma::mat get_gradCompTrue() 
  {
    return gradCompTrue;
  }
  /// Obtains componentwise calculation of the expectation of the smoothing functionals needed
  /// to calculate the gradient
  arma::vec get_gradTrue() 
  {
    return gradTrue;
  }
  /// Determines whether or not the entire particle system should be stored
  void setStoreHistory(const bool storeHistory_) 
  {
    storeHistory = storeHistory_;
  }
  /// Determines the type of (blocked) backward-sampling scheme to be used
  void set_back(const BackwardKernelType back_) 
  {
    back = back_;
    
    if (back == SMC_BACKWARD_KERNEL_STANDARD && filt != SMC_FILTER_BLOCK)
    {
      arma::mat blockAux(2,1);
      blockAux(0,0) = 0;
      blockAux(1,0) = V-1;
      setBlocks(blockAux, blockAux);
    }
  }
  /// Determines the number of backward particles
  void setM(unsigned int M_) 
  {
    M = M_;
  }
  /// Determines the number of backward particles
  void set_isSimple(const bool isSimple_) 
  {
    isSimple = isSimple_;
  }
  /// Determines whether marginal smoothing approximations should be stored
  /*
  void set_storeMarginals(bool storeMarginals_) 
  {
    storeMarginals = storeMarginals_;
  }
  */
  /// Determines whether marginal smoothing approximations should be stored
  void setStoreSuff(const bool storeSuff_) 
  {
    storeSuff = storeSuff_;
  }
  /// Determines the blocks for use in some blocking approximation.
  void setBlocks(const arma::mat& blockInn_, const arma::mat& blockOut_) 
  {
    blockInn = blockInn_;
    blockOut = blockOut_;
    nBlocks  = blockInn.n_cols;
  }
  /// Determines the hyperparameters for the prior distribution of 
  /// the static parameters.
  void setHyperParameters(arma::colvec& hyperParameters_) 
  {
    hyperParameters = hyperParameters_;
  }
  /// Determines the standard deviations used for the proposal kernel
  /// of a Gaussian random-walk MH update for theta.
  void setRwmhSd(arma::colvec& rwmhSd_) 
  {
    rwmhSd = rwmhSd_;
  }
  // Member functions
  /*
  void run_kalman_original()
  {
  
    // Predictive mean and covariance matrix
    arma::colvec mP(V);
    arma::mat CP(V, V);
    
    // Updated mean and covariance matrix
    arma::colvec mU(V);
    arma::mat CU(V, V);
    
    // Mean and covariance matrix of the incremental likelihood
    arma::colvec mY(dy);
    arma::mat CY(dy, dy);
    
    // Log-marginal likelihood
    logLikeTrue = 0;
    
    // Auxiliary quantities
    arma::mat kg;
    
    for (unsigned int t=0; t<T; t++) {
      
      // Prediction step
      if (t > 0) {
        mP = A * mU;
        CP = A * CU * A.t() + BBT; 
      } 
      else {
        mP = m0;
        CP = C0; 
      }
      
      // Likelihood step
      mY = C * mP;
      CY = C * CP * C.t() + DDT;
 
      // Update step
      kg = (arma::solve(CY.t(), C * CP.t())).t();
      mU = mP + kg * (y.col(t) - mY);
      CU = CP - kg * C * CP;

      // Adding the incremental log-marginal likelihood
      logLikeTrue += arma::as_scalar(gaussian::evaluateDensityMultivariate(y.col(t), mY, CY, false, true));
      
    }
  }
  */
  
  // Member functions
  void runKalman()
  {
  
    // Predictive mean and covariance matrix
    mP.set_size(V, T);
    CP.set_size(V, V, T);
    
    // Updated mean and covariance matrix
    mU.set_size(V, T);
    CU.set_size(V, V, T);
    
    // Smoothed mean and covariance matrix
    mS.set_size(V, T);
    CS.set_size(V, V, T);
    
    // Mean and covariance matrix of the incremental likelihood
    arma::colvec mY(dy);
    arma::mat CY(dy, dy);
    
    // Log-marginal likelihood
    logLikeTrue = 0;
    
    // Auxiliary quantities
    arma::mat kg;
    
    for (unsigned int t=0; t<T; t++) {
      
      // Prediction step
      if (t > 0) {
        mP.col(t) = A * mU.col(t-1);
        CP.slice(t) = A * CU.slice(t-1) * A.t() + BBT; 
      } 
      else {
        mP.col(0) = m0;
        CP.slice(0) = C0; 
      }
      
      // Likelihood step
      mY = C * mP.col(t);
      CY = C * CP.slice(t) * C.t() + DDT;
 
      // Update step
      kg = (arma::solve(CY.t(), C * arma::trans(CP.slice(t)))).t();
      mU.col(t)   = mP.col(t) + kg * (y.col(t) - mY);
      CU.slice(t) = CP.slice(t) - kg * C * CP.slice(t);

      // Adding the incremental log-marginal likelihood
      logLikeTrue += arma::as_scalar(gaussian::evaluateDensityMultivariate(y.col(t), mY, CY, false, true));
    }
    
  }
  
  // Member functions
  void runKalmanSmoother()
  {   
    ///////////////////////////////////////////////////////////////////////////
    // Backward smoothing
    ///////////////////////////////////////////////////////////////////////////

    arma::mat Ck(V, V);
    mS.col(T-1) = mU.col(T-1);
    CS.slice(T-1) = CU.slice(T-1);
    
    for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
    {
      Ck = CU.slice(t) * A.t() * arma::inv(CP.slice(t+1));
      mS.col(t)   = mU.col(t)   + Ck * (mS.col(t+1)   - mP.col(t+1));
      CS.slice(t) = CU.slice(t) + Ck * (CS.slice(t+1) - CP.slice(t+1)) * Ck.t();
    }
    
  }
  
  /// Runs the SMC algorithm
  void runSmcBase(arma::mat& particlePath_)
  {       
    
    //std::cout << "start runSmcBase" << std::endl;
    arma::mat particlesOld(V, N); // particles from current step
    arma::mat particlesNew(V, N); // particles from previous step
    arma::uvec parentIndices(N); // parent indices
    unsigned int parentIndex, particleIndex; // parent/particle index for a single particle
    arma::colvec logWeights(N); // unnormalised log-weights
    logWeights.fill(-log(N)); // start with uniform weights
    arma::colvec W(N); // unnormalised log-weights
    arma::colvec WAux; // unnormalised log-weights
    double ESS; // effective sample size
    logLikeEst = 0.0; // log of the estimated marginal likelihood   
    logComponentWeightsFull.set_size(V,N,T);
    
    if (isConditional) // i.e. if we run a conditional SMC algorithm 
    {
      //std::cout << "CONDITIONAL SMC" << std::endl;
      storePath = true;
      WAux.set_size(N);
    }
    
    if (storePath)
    {
      storeHistory = true;
    }
    
    if (storeHistory)
    {
      particlesFull.set_size(V, N, T);
      parentIndicesFull.set_size(N, T-1);
      logWeightsFull.set_size(N, T);
    }
    
    if (storePath) 
    {
      particleIndices.set_size(T);
    }

    switch (filt)
    {
      case SMC_FILTER_STANDARD: // run a standard SMC filter
        
//          std::cout << "----------- Standard PF -----------" << std::endl;

        //std::cout << "SMC, Step " << 1 << std::endl; 
        
        // Step 1 of the SMC algorithm:
        if (isConditional) // i.e. if we run a conditional SMC algorithm 
        {
          particleIndices(0) = arma::as_scalar(arma::randi(1, arma::distr_param(0,N-1)));
        }
        initialiseSmc(particlesNew, logWeights, particlePath_); 

        if (storeHistory)
        {
          particlesFull.slice(0) = particlesNew;
          logWeightsFull.col(0)  = logWeights;
        }

        for (unsigned int t=1; t<T; t++)
        {
          //std::cout << "SMC, Step " << (t+1) << std::endl; 
          
          //Step t, t > 1, of the SMC algorithm:
          W = normaliseWeights(logWeights); // Self-normalised weights
          ESS = 1.0 / arma::dot(W, W); // Effective sample size
          
          if (!arma::is_finite(W))
          {
            std::cout << "WARNING: W contains NaNs!" << std::endl;
          }
          
          if (isConditional) // i.e. if we run a conditional SMC algorithm 
          {
            // Determining the parent index of the current input particle:
            if (csmc == CSMC_BACKWARD_KERNEL_ANCESTOR) // via ancestor sampling
            {
              WAux = normaliseWeights(
                logWeights + 
                gaussian::evaluateDensityMultivariate(particlePath_.col(t), A*particlesFull.slice(t-1), B, true, true)
              );  
              if (!arma::is_finite(WAux))
              {
                std::cout << "WARNING: WAux contains NaNs!" << std::endl;
              }
              parentIndex = sampleInt(WAux); 
            }
            else // not via ancestor sampling
            {
              parentIndex = particleIndices(t-1);
            }
          }
        
          if (ESS < H*N) // Checking ESS resampling threshold
          {
            // Updating the estimate of the normalising constantÖ
            logLikeEst += log(arma::sum(exp(logWeights)));
            
            // Obtaining the parent indices via adaptive systematic resampling:           
            if (isConditional) // "conditional" resampling
            {
              resample::conditionalSystematic(parentIndices, particleIndex, W, N, parentIndex);
              particleIndices(t) = particleIndex;
            }
            else // "unconditional" resampling
            {
              resample::systematic(parentIndices, W, N);
            }
            
            // Resetting the weights:
            logWeights.fill(-log(N)); 
          } 
          else // I.e. no resampling:
          {
            if (isConditional) 
            {
              particleIndices(t) = particleIndices(t-1);
            }
            parentIndices = arma::linspace<arma::uvec>(0, N-1, N);
          }
          
          // Determining the parent particles based on the parent indices: 
          particlesOld = particlesNew.cols(parentIndices); 
          
           // Sampling and reweighting:
          iterateSmc(t, particlesNew, particlesOld, logWeights, particlePath_);
        
          if (storeHistory)
          {
            // Storing the entire particle system:
            particlesFull.slice(t)     = particlesNew;
            parentIndicesFull.col(t-1) = parentIndices;
            logWeightsFull.col(t)      = logWeights;
          }
        }
        
        // Updating the estimate of the normalising constant:
        logLikeEst += log(arma::sum(exp(logWeights)));
        
        if (storePath)
        {
          // Sampling a single particle path:
          particlePath_.set_size(V,T);
          
          // Final-time particle:
          W = normaliseWeights(logWeights);; // final-time self-normalised weights
          particleIndices(T-1) = sampleInt(W);
          particlePath_.col(T-1) = particlesFull.slice(T-1).col(particleIndices(T-1));
          
          // Recursion for the particles at previous time steps:
          if (isConditional == true && csmc == CSMC_BACKWARD_KERNEL_STANDARD)
          { // i.e. we emply the usual backward-sampling recursion
            for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
            { 
              WAux = normaliseWeights(
                logWeightsFull.col(t) + 
                gaussian::evaluateDensityMultivariate(particlePath_.col(t+1), A*particlesFull.slice(t), B, true, true)
              );
              
              if (!arma::is_finite(WAux))
              {
                std::cout << "WARNING: WAux contains NaNs!" << std::endl;
              }
          
              particleIndices(t) = sampleInt(WAux);
              particlePath_.col(t) = particlesFull.slice(t).col(particleIndices(t));
            }
          }
          else // i.e we just trace back the ancestral lineage
          {  
            for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
            { 
              particleIndices(t) = parentIndicesFull(particleIndices(t+1), t);
              particlePath_.col(t) = particlesFull.slice(t).col(particleIndices(t));
            }
          }
        }
        
        
        break;
      
      case SMC_FILTER_BLOCK: // run the blocked particle filter
        {
          
//           std::cout << "----------- Blocked PF -----------" << std::endl;
          
          // Step 1 
          double logZ = 0;
          //std::cout << "BPF, Step " << 0 << std::endl;  
          initialiseSmc(particlesNew, logWeights, particlePath_); // logWeights is not actually used, here
          logLikeEst = 0;
          if (storeHistory)
          {
            particlesFull.slice(0) = particlesNew;
            logWeightsFull.col(0) = logWeights;
          }

          // Step t, t > 1
          for (unsigned int t=1; t<T; t++)
          {  
            //std::cout << "BPF, Step " << t << std::endl;  
            for (unsigned int j=0; j<nBlocks; j++) 
            {
                         //   std::cout << "compute block parameters" << std::endl;
              computeBlockParameters(j);
                          //  std::cout << "compute local weights" << std::endl;
              computeLocalWeights(logWeights, t-1, lInn, uInn);
              
              //std::cout << "calculating normalising constant estimate" << std::endl;
              W = normaliseWeights(logWeights, logZ); // self-normalised weights for the jth block
              logLikeEst += logZ; // approximate normalising constant
              
              parentIndices = sampleInt(N, W);
              for (unsigned int n=0; n<N; n++) // TODO: maybe this loop can be avoided
              {
                particlesOld(arma::span(lInn,uInn), arma::span(n,n)) = particlesNew(arma::span(lInn,uInn), arma::span(parentIndices(n),parentIndices(n))); // Determining the parent particles for the current block  
              }
            }
            iterateSmc(t, particlesNew, particlesOld, logWeights, particlePath_); // logWeights is not actually used, here

            // Potentially storing the entire particle system
            if (storeHistory)
            {
              particlesFull.slice(t) = particlesNew;
              logWeightsFull.col(t) = logWeights;
            }
          }
          for (unsigned int j=0; j<nBlocks; j++) 
          {
            W = normaliseWeights(logWeights, logZ); // self-normalised weights for the jth block
            logLikeEst += logZ - (T-1)*log(N); // approximate normalising constant
          }
          
        }
        break;
        
      case SMC_FILTER_EXACT: // obtains samples from the exact (marginal) filtering distributions
        
//          std::cout << "----------- Exact filter -----------" << std::endl;
        
        runKalman();
        if (storeHistory)
        {
          logWeightsFull.fill(-log(N));
          logComponentWeightsFull.zeros();
        }
        for (unsigned int t=0; t<T; t++)
        {  
          if (storeHistory)
          {
            particlesFull.slice(t) = gaussian::sampleMultivariate(N, mU.col(t), CU.slice(t), false);
          }
        }
        break;
    } 
  }
  /// A wrapper for runSmcBase().
  void runSmc()
  {
    isConditional = false;
    runSmcBase(particlePath);
  }
  /// A wrapper for runSmcBase().
  void runSmc(arma::mat& particlePath_)
  {
    isConditional = false;
    runSmcBase(particlePath_);
  }
  /// A wrapper for runSmcBase().
  void runCsmc(arma::mat& particlePath_)
  {
    isConditional = true;
    runSmcBase(particlePath_);
  }
  /// Performs backward sampling.
  void runParticleSmoother()
  {
    
  /////////////////////////////
  /////////////////////////////
//   ProfilerStart("/home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/blockedSmoothing/linearGaussianHmm/profile_runParticleSmoother.log");
  /////////////////////////////
  /////////////////////////////

    
    arma::colvec backwardKernels(N);
    arma::colvec logWeights(N); // log-unnormalised backward weights (randomDiscrete() needs the weights-argument to be an STL vector)
    unsigned int b;    // auxiliary particle index
    arma::mat meanAux; // auxiliary mean vectors used for standard and blocked backward sampling
    
    if (storeSuff)
    {
      suffHCompEst.zeros(K+1,K+1,V);
      suffQCompEst.zeros(K+1,V);
      suffRCompEst.zeros(V);
      suffR1CompEst.zeros(V);
      suffSCompEst.zeros(V);
      gradCompEst.zeros(K+3,V);
    }
    
    if (M < N && back == SMC_BACKWARD_KERNEL_STANDARD) // backward sampling use standard backward kernels
    {
//       std::cout << "running standard BS" << std::endl;
      
//       xBack.set_size(V, M, T); // sample paths obtained via backward sampling
//       backwardKernels = normaliseWeights(logWeightsFull.col(T-1));
      
      
      if (filt == SMC_FILTER_BLOCK)
      {
        subsampleParticlesFromBpfApproximation();
        runStandardBackwardSampler(subsampledParticlesFromBpfApproximationFull, logWeightsForSubsampledBpfApproximationFull);
      }
      else
      {
        runStandardBackwardSampler(particlesFull, logWeightsFull);
      }
    
      
    
//         //std::cout << "Backward sampling, Step T" << std::endl;
//         for (unsigned int m=0; m<M; m++)
//         {  
//           //b = rng->randomDiscrete(W);
//           // std::cout << backwardKernels.t() << std::endl;
//           b = sampleInt(backwardKernels);
//           xBack.slice(T-1)(arma::span::all, arma::span(m,m)) = 
//           particlesFull.slice(T-1)(arma::span::all, arma::span(b,b));
//         }
//         
//         for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
//         {  
//           //std::cout << "Backward sampling, Step " << t << std::endl;
//           meanAux = A*particlesFull.slice(t); // only needs to be calculated once per (backward) time step
//           for (unsigned int m=0; m<M; m++)
//           {  
//             logWeights = logWeightsFull.col(t) + gaussian::evaluateDensityMultivariate(
//                     xBack.slice(t+1)(arma::span::all, arma::span(m,m)), 
//                     meanAux, B, true, true, cores);
//                     
//             backwardKernels = normaliseWeights(logWeights);
//             b = sampleInt(backwardKernels); 
//             xBack.slice(t)(arma::span::all, arma::span(m,m)) = 
//             particlesFull.slice(t)(arma::span::all, arma::span(b,b));
//           }
//         }
//       
//       
//       if (storeSuff)
//       {
//         suff_est();
//       }
        
    }
    else if (M < N && back == SMC_BACKWARD_KERNEL_BLOCK) // backward sampling using blocked backward kernels
    {
//       std::cout << "running blocked BS" << std::endl;
      
      for (unsigned int j=0; j<nBlocks; j++)
      {
        //std::cout << "Blocked backward sampling, Step T-1, Block, " << j << std::endl;
        computeBlockParameters(j);
        if (sizeOutNei != xBackBlock.n_rows)
        {
          xBackBlock.set_size(sizeOutNei, M, T);
        }
        /// Blocked backward sampling for Step T:
        computeLocalWeights(logWeights, T-1, lOutNei, uOutNei); // calculates the local weights
        backwardKernels = normaliseWeights(logWeights);
        
        //std::cout << "loop for m" << std::endl;
        for (unsigned int m = 0; m < M; m++)
        {  
          b = sampleInt(backwardKernels);
          xBackBlock.slice(T-1)(arma::span::all, arma::span(m,m)) = 
          particlesFull.slice(T-1)(arma::span(lOutNei, uOutNei), arma::span(b,b)); 
        }

        //std::cout << "backward recursion" << std::endl;
        /// Blocked backward sampling for Steps T-1,..., 1
        for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
        {  
          //std::cout << "Blocked backward sampling, Step " << t << ", Block " << j << std::endl;
          computeLocalWeights(logWeights, t, lOutNei, uOutNei); // calculates the local weights
          meanAux = A(arma::span(lOut, uOut), arma::span(lOutNei, uOutNei))*particlesFull.slice(t).rows(arma::span(lOutNei, uOutNei));
          
          //std::cout << "recursion for m" << std::endl;
          for (unsigned int m=0; m<M; m++)
          {  
            // Obtain self-normalised "local" backward sampling weights:
            backwardKernels = normaliseWeights(
                  logWeights + gaussian::evaluateDensityMultivariate(
                          xBackBlock.slice(t+1)(arma::span(lOut-lOutNei, sizeOutNei-1-(uOutNei-uOut)), arma::span(m,m)), 
                          meanAux,
                          BBT(arma::span(lOut, uOut), arma::span(lOut, uOut)), false, true, cores)); 
            
            b = sampleInt(backwardKernels);
            xBackBlock.slice(t)(arma::span::all, arma::span(m,m)) = 
            particlesFull.slice(t)(arma::span(lOutNei, uOutNei), arma::span(b,b));
          }
        }
        
        if (storeSuff)
        {
          suff_est_block(j);
        }
      }   
    }
    else if (M >= N && back == SMC_BACKWARD_KERNEL_STANDARD) // "forward" smoothing use standard backward kernels
    {
      if (filt == SMC_FILTER_BLOCK)
      {
        subsampleParticlesFromBpfApproximation();
        runStandardForwardSmoother(subsampledParticlesFromBpfApproximationFull, logWeightsForSubsampledBpfApproximationFull);
      }
      else
      {
        runStandardForwardSmoother(particlesFull, logWeightsFull);
      }

      
//       std::cout << "running standard FS" << std::endl;
      /*
      // Temporary quantities:
      std::vector<arma::cube>   suffHCompEstAuxOld(N);
      std::vector<arma::mat>    suffQCompEstAuxOld(N);
      std::vector<arma::colvec> suffRCompEstAuxOld(N);
      std::vector<arma::colvec> suffR1CompEstAuxOld(N);
      std::vector<arma::colvec> suffSCompEstAuxOld(N);
 
      std::vector<arma::cube>   suffHCompEstAuxNew(N);
      std::vector<arma::mat>    suffQCompEstAuxNew(N);
      std::vector<arma::colvec> suffRCompEstAuxNew(N);
      std::vector<arma::colvec> suffR1CompEstAuxNew(N);
      std::vector<arma::colvec> suffSCompEstAuxNew(N);

      // Time 0
      
//       std::cout << "M>=N, standard backward kernel, time: " << 0 << std::endl;
      for (unsigned int n=0; n<N; n++)
      { 
        
//         std::cout << "initialising the elements of each vector" << std::endl;
        suffRCompEstAuxNew[n].zeros(V);
        suffSCompEstAuxNew[n].zeros(V);
        suffHCompEstAuxNew[n].zeros(K+1,K+1,V);
        suffQCompEstAuxNew[n].zeros(K+1,V);
        suffR1CompEstAuxNew[n].zeros(V);
    
        suffRCompEstAuxOld[n].zeros(V);
        suffSCompEstAuxOld[n].zeros(V);
        suffHCompEstAuxOld[n].zeros(K+1,K+1,V);
        suffQCompEstAuxOld[n].zeros(K+1,V);  
        suffR1CompEstAuxOld[n].zeros(V);


//         std::cout << "computing initial suff. stat. R and S" << std::endl;
        for (unsigned int v=0; v<V; v++)
        {
          suffRCompEstAuxNew[n](v) += particlesFull(v,n,0) * particlesFull(v,n,0);
          suffSCompEstAuxNew[n](v) += particlesFull(v,n,0) * y(v,0);
        }
      }
      
//       std::cout << "assigning R1" << std::endl;
      suffR1CompEstAuxNew = suffRCompEstAuxNew;
      
      
//        std::cout << "initialising z" << std::endl;
      arma::mat z(2*K+V,N, arma::fill::zeros); // this pads the particles with zeros to avoid issues on the boundary of the state space
      double zk, zl;
    
      
      for (unsigned int t=1; t<T; t++)
      {
        meanAux = A*particlesFull.slice(t-1); // to ensure this only needs to be calculated once per (backward) time step
        z(arma::span(K,K+V-1), arma::span::all) = particlesFull.slice(t-1);
        
        // Temporary quantities:
        suffHCompEstAuxOld.swap(suffHCompEstAuxNew);
        suffQCompEstAuxOld.swap(suffQCompEstAuxNew);
        suffRCompEstAuxOld.swap(suffRCompEstAuxNew);
        suffR1CompEstAuxOld.swap(suffR1CompEstAuxNew);
        suffSCompEstAuxOld.swap(suffSCompEstAuxNew);

        for (unsigned int n=0; n<N; n++)
        { 
          
          suffRCompEstAuxNew[n].zeros(V);
          suffSCompEstAuxNew[n].zeros(V);
          suffHCompEstAuxNew[n].zeros(K+1,K+1,V);
          suffQCompEstAuxNew[n].zeros(K+1,V);
          suffR1CompEstAuxNew[n].zeros(V);
          
          // Backward kernel evaluated at all $N$ time-$t-1$ particles given the $n$th particle at time $t$
          logWeights = logWeightsFull.col(t-1) + gaussian::evaluateDensityMultivariate(
                  particlesFull.slice(t)(arma::span::all, arma::span(n,n)), 
                  meanAux, B, true, true, cores);
          backwardKernels = normaliseWeights(logWeights); // self-normalised backward kernels
          
          for (unsigned int m=0; m<N; m++)
          {
            for (unsigned int v=0; v<V; v++)
            {
              for (unsigned int k=0; k<K+1; k++)
              {
                if (k == 0)
                {
                  zk = z(K+v,m);
                }
                else
                {
                  zk = z(K+v-k,m) + z(K+v+k,m);
                }
                
                suffQCompEstAuxNew[n](k,v) += backwardKernels(m) * (suffQCompEstAuxOld[m](k,v) + particlesFull(v,n,t) * zk);
   
                for (unsigned int l=0; l<K+1; l++)
                {
                  if (l == 0)
                  {
                    zl = z(K+v,m);
                  }
                  else
                  {
                    zl = z(K+v-l,m) + z(K+v+l,m);
                  }
                  suffHCompEstAuxNew[n].slice(v)(k,l) += backwardKernels(m) * (suffHCompEstAuxOld[m].slice(v)(k,l) + zk * zl);
                }
              }
              suffR1CompEstAuxNew[n](v) += backwardKernels(m) * suffR1CompEstAuxOld[m](v);
              suffRCompEstAuxNew[n](v)  += backwardKernels(m) * (suffRCompEstAuxOld[m](v) + particlesFull(v,n,t) * particlesFull(v,n,t));
              suffSCompEstAuxNew[n](v)  += backwardKernels(m) * (suffSCompEstAuxOld[m](v) + particlesFull(v,n,t) * y(v,t));
              
            }
          }
        }
      }
      
      arma::colvec W = normaliseWeights(logWeightsFull.col(T-1));
      
      for (unsigned int n=0; n<N; n++) 
      {
        suffRCompEst  = suffRCompEst  + W(n) * suffRCompEstAuxNew[n];
        suffR1CompEst = suffR1CompEst + W(n) * suffR1CompEstAuxNew[n];
        suffSCompEst  = suffSCompEst  + W(n) * suffSCompEstAuxNew[n];
        suffHCompEst  = suffHCompEst  + W(n) * suffHCompEstAuxNew[n];
        suffQCompEst  = suffQCompEst  + W(n) * suffQCompEstAuxNew[n];
      }*/
    }
    else if (M >= N && back == SMC_BACKWARD_KERNEL_BLOCK) // "forward" smoothing using blocked backward kernels
    {
      
//       std::cout << "running blocked FS" << std::endl;
      
      // Temporary quantities:
      std::vector<arma::cube> suffHCompEstAuxOld(N);
      std::vector<arma::mat> suffQCompEstAuxOld(N);
      std::vector<arma::colvec> suffRCompEstAuxOld(N);
      std::vector<arma::colvec> suffR1CompEstAuxOld(N);
      std::vector<arma::colvec> suffSCompEstAuxOld(N);
      
      std::vector<arma::cube> suffHCompEstAuxNew(N);
      std::vector<arma::mat> suffQCompEstAuxNew(N);
      std::vector<arma::colvec> suffRCompEstAuxNew(N);
      std::vector<arma::colvec> suffR1CompEstAuxNew(N);
      std::vector<arma::colvec> suffSCompEstAuxNew(N);

      arma::mat z(2*K+V, N, arma::fill::zeros); // this pads the particles with zeros to avoid issues on the boundary of the state space
      double zk, zl;
      
      for (unsigned int j=0; j<nBlocks; j++)
      {
//         std::cout << "M>=N, blocked backward kernel, time: " << 0 << "; block: " << j << std::endl;
        computeBlockParameters(j);
        
        // Time 0
        for (unsigned int n=0; n<N; n++)
        {
          suffRCompEstAuxNew[n].zeros(blockSize);
          suffSCompEstAuxNew[n].zeros(blockSize);
          suffHCompEstAuxNew[n].zeros(K+1,K+1,blockSize);
          suffQCompEstAuxNew[n].zeros(K+1,blockSize);
          suffR1CompEstAuxNew[n].zeros(blockSize);
          
          suffRCompEstAuxOld[n].zeros(blockSize);
          suffSCompEstAuxOld[n].zeros(blockSize);
          suffHCompEstAuxOld[n].zeros(K+1,K+1,blockSize);
          suffQCompEstAuxOld[n].zeros(K+1,blockSize);
          suffR1CompEstAuxOld[n].zeros(blockSize);
        
          for (unsigned int v=0; v<blockSize; v++)
          {
            suffRCompEstAuxNew[n](v) += particlesFull(lInn+v,n,0) * particlesFull(lInn+v,n,0);
            suffSCompEstAuxNew[n](v) += particlesFull(lInn+v,n,0) * y(lInn+v,0);
          }
        }
        suffR1CompEstAuxNew = suffRCompEstAuxNew;
        
      
        for (unsigned int t=1; t<T; t++)
        {
          meanAux = A(arma::span(lOut, uOut), arma::span(lOutNei, uOutNei))*particlesFull.slice(t-1).rows(arma::span(lOutNei, uOutNei));
//           z.zeros();
          z(arma::span(K,K+V-1), arma::span::all) = particlesFull.slice(t-1);
 
          // Temporary quantities:
          suffHCompEstAuxOld.swap(suffHCompEstAuxNew);
          suffQCompEstAuxOld.swap(suffQCompEstAuxNew);
          suffRCompEstAuxOld.swap(suffRCompEstAuxNew);
          suffR1CompEstAuxOld.swap(suffR1CompEstAuxNew);
          suffSCompEstAuxOld.swap(suffSCompEstAuxNew);
//           suffHCompEstAuxOld  = suffHCompEstAuxNew;
//           suffQCompEstAuxOld  = suffQCompEstAuxNew;
//           suffRCompEstAuxOld  = suffRCompEstAuxNew;
//           suffR1CompEstAuxOld = suffR1CompEstAuxNew;
//           suffSCompEstAuxOld  = suffSCompEstAuxNew;
                    
          for (unsigned int n=0; n<N; n++)
          {  
            
            suffRCompEstAuxNew[n].zeros(blockSize);
            suffSCompEstAuxNew[n].zeros(blockSize);
            suffHCompEstAuxNew[n].zeros(K+1,K+1,blockSize);
            suffQCompEstAuxNew[n].zeros(K+1,blockSize);
            suffR1CompEstAuxNew[n].zeros(blockSize);
            
            // Backward kernel evaluated at all $N$ time-$t-1$ particles given the $n$th particle at time $t$
            computeLocalWeights(logWeights, t-1, lOutNei, uOutNei); // calculates the local weights
            logWeights = logWeights + 
              gaussian::evaluateDensityMultivariate(
                particlesFull.slice(t)(arma::span(lOut, uOut), arma::span(n,n)), 
                meanAux, BBT(arma::span(lOut, uOut), arma::span(lOut, uOut)), false, true, cores
              );
              
            backwardKernels = normaliseWeights(logWeights); // self-normalised backward kernels

            for (unsigned int m=0; m<N; m++)
            {
              for (unsigned int v=0; v<blockSize; v++)
              {
                for (unsigned int k=0; k<K+1; k++)
                {
                  if (k == 0)
                  {
                    zk = z(K+lInn+v,m);
                  }
                  else
                  {
                    zk = z(K+lInn+v-k,m) + z(K+lInn+v+k,m);
                  }

                  suffQCompEstAuxNew[n](k,v) += backwardKernels(m) * (suffQCompEstAuxOld[m](k,v) + particlesFull(lInn+v,n,t) * zk);

                  for (unsigned int l=0; l<K+1; l++)
                  {
                    if (l == 0)
                    {
                      zl = z(K+lInn+v,m);
                    }
                    else
                    {
                      zl = z(K+lInn+v-l,m) + z(K+lInn+v+l,m);
                    }

                    suffHCompEstAuxNew[n].slice(v)(k,l) += backwardKernels(m) * (suffHCompEstAuxOld[m].slice(v)(k,l) + zk * zl);
                  }
                }

                suffR1CompEstAuxNew[n](v) += backwardKernels(m) * suffR1CompEstAuxOld[m](v);
                suffRCompEstAuxNew[n](v)  += backwardKernels(m) * (suffRCompEstAuxOld[m](v) + particlesFull(lInn+v,n,t) * particlesFull(lInn+v,n,t));
                suffSCompEstAuxNew[n](v)  += backwardKernels(m) * (suffSCompEstAuxOld[m](v) + particlesFull(lInn+v,n,t) * y(lInn+v,t));
              }
            }
          }
        }
        
        computeLocalWeights(logWeights, T-1, lOut, uOut); // calculates the local weights
        arma::colvec W = normaliseWeights(logWeights);
        
        for (unsigned int n=0; n<N; n++) 
        {
          for (unsigned int v=0; v<blockSize; v++) 
          {
            suffRCompEst(lInn+v)       = suffRCompEst(lInn+v)       + W(n) * suffRCompEstAuxNew[n](v);
            suffR1CompEst(lInn+v)      = suffR1CompEst(lInn+v)      + W(n) * suffR1CompEstAuxNew[n](v);
            suffSCompEst(lInn+v)       = suffSCompEst(lInn+v)       + W(n) * suffSCompEstAuxNew[n](v);
            suffHCompEst.slice(lInn+v) = suffHCompEst.slice(lInn+v) + W(n) * suffHCompEstAuxNew[n].slice(v);
            suffQCompEst.col(lInn+v)   = suffQCompEst.col(lInn+v)   + W(n) * suffQCompEstAuxNew[n].col(v);
          }
        }
      }   
    }
  
    if (storeSuff)
    {
      if (M < N) 
      {
        suffHCompEst  = suffHCompEst  / M;
        suffQCompEst  = suffQCompEst  / M;
        suffRCompEst  = suffRCompEst  / M;
        suffR1CompEst = suffR1CompEst / M;
        suffSCompEst  = suffSCompEst  / M;
      }

      // Calculating the spatial component-wise gradient
      for (unsigned int v=0; v<V; v++)
      {
        if (T > 1)
        {
          for (unsigned int k=0; k<K+1; k++)
          {
            gradCompEst(k,v) = arma::as_scalar(exp(-2.0*theta(K+1)) * 
                               (suffQCompEst(k,v) - arma::trans(theta(arma::span(0,K))) * suffHCompEst.slice(v).col(k)));
          }
          gradCompEst(K+1,v) = arma::as_scalar(exp(-2.0*theta(K+1)) * 
                               (suffRCompEst(v) - suffR1CompEst(v) - 2.0 * arma::trans(theta(arma::span(0,K))) * suffQCompEst.col(v) +  
                               arma::trans(theta(arma::span(0,K))) * suffHCompEst.slice(v) * theta(arma::span(0,K))) - (T-1));
        }
        gradCompEst(K+2,v) = exp(-2.0*theta(K+2)) * 
                             (suffRCompEst(v) - 2.0 * suffSCompEst(v) + arma::accu(arma::pow(y.row(v), 2.0))) - T;
      }
      
      suffHEst.zeros(K+1, K+1);
      for (unsigned int v=0; v<V; v++)
      {
        suffHEst += suffHCompEst.slice(v);
      }
      suffQEst  = arma::sum(suffQCompEst, 1);
      suffREst  = arma::sum(suffRCompEst);
      suffR1Est = arma::sum(suffR1CompEst);
      suffSEst  = arma::sum(suffSCompEst);
      gradEst   = arma::sum(gradCompEst, 1);
    
    }
    
      
    /////////////////////////////
    /////////////////////////////
//     ProfilerStop();
    /////////////////////////////
    /////////////////////////////
  
  }
  /// Calculate component-wise sufficient statistics
  void computeSuffTrue( )
  {
    suffHCompTrue.zeros(K+1,K+1,V);
    suffQCompTrue.zeros(K+1,V);
    suffRCompTrue.zeros(V);
    suffR1CompTrue.zeros(V);
    suffSCompTrue.zeros(V);
    gradCompTrue.zeros(K+3,V);
  
    arma::mat CSAux(2*K + V, 2*K + V, arma::fill::zeros);
    arma::mat sigAux(2*V, 2*V);
    arma::mat sigAuxExt(2*(K+V), 2*(K+V), arma::fill::zeros);
    arma::colvec mExt(2*K + V, arma::fill::zeros);
      
    arma::mat HAux(K+1, K+1);
    arma::colvec QAux(K+1);
    double RAux, SAux;
      
    arma::mat Gk(V,V);
    arma::mat P2(V,V);
 
    for (unsigned int t=0; t<T; t++) 
    { 
        
      if (t > 0)
      {
        // Padding the covariance matrix of the marginal time-(t-1) smoothing distribution with zeros.
        CSAux(arma::span(K,V+K-1), arma::span(K,V+K-1)) = CS.slice(t-1);
            
        Gk = CU.slice(t-1) * A.t() * arma::inv(CP.slice(t)); 
        P2 = CU.slice(t-1) - Gk * CP.slice(t-1) * Gk.t(); 
          
        // The covariance matrix of the vector (x_t, x_{t-1}):
        sigAux(arma::span(0,V-1), arma::span(0,V-1))     = CS.slice(t); 
        sigAux(arma::span(V,2*V-1), arma::span(V,2*V-1)) = Gk.t() * CS.slice(t) * Gk.t() + P2; 
        sigAux(arma::span(V,2*V-1), arma::span(0,V-1))   = CS.slice(t) * Gk.t();
        sigAux(arma::span(0,V-1), arma::span(V,2*V-1))   = Gk.t() * CS.slice(t);
        
        sigAuxExt(arma::span(K,2*V+K-1), arma::span(K,2*V+K-1)) = sigAux; // padded by zeros
      
        // Smoothed mean padded by zeros
        mExt(arma::span(K,K+V-1)) = mS.col(t-1);
       
      }
          
      for (unsigned int v=0; v<V; v++) 
      { 
        if (t > 0)
        {
          for (unsigned int k=0; k<K+1; k++) 
          { 
            QAux(k) = sigAuxExt(K+v, K+V+v-k) + mExt(K+v-k) * mS(v,t);
            if (k > 0) 
            {
              QAux(k) += sigAuxExt(K+v, K+V+v+k) + mExt(K+v+k) * mS(v,t);
            }
              
            for (unsigned int l=0; l<K+1; l++) 
            { 
              HAux(k,l) = CSAux(K+v-k,K+v-l) + mExt(K+v-k) * mExt(K+v-l);
              if (k > 0 && l > 0)
              {
                HAux(k,l) += CSAux(K+v+k,K+v-l) + mExt(K+v+k) * mExt(K+v-l) +
                             CSAux(K+v+k,K+v+l) + mExt(K+v+k) * mExt(K+v+l) +
                             CSAux(K+v-k,K+v+l) + mExt(K+v-k) * mExt(K+v+l);
              }
              else if (k > 0 && l==0)
              {
                HAux(k,l) += CSAux(K+v+k,K+v-l) + mExt(K+v+k) * mExt(K+v-l);
              }
              else if (l > 0 && k==0)
              {
                HAux(k,l) += CSAux(K+v-k,K+v+l) + mExt(K+v-k) * mExt(K+v+l);
              }
                
            }
          }    
        }
         
        RAux = CS(v,v,t) + pow(mS(v,t), 2.0);
        SAux = mS(v,t) * y(v,t);
        
        if (t == 0)
        {
          suffR1CompTrue(v) = RAux;
        }
              
        // Storing the spatial component-wise sufficient statistics
        suffHCompTrue.slice(v) += HAux;
        suffQCompTrue.col(v)   += QAux;
        suffRCompTrue(v)       += RAux;
        suffSCompTrue(v)       += SAux;
        
      }
    }
    // Calculating the spatial component-wise gradient
    for (unsigned int v=0; v<V; v++)
    {
      if (T > 1)
      {
        for (unsigned int k=0; k<K+1; k++)
        {
          gradCompTrue(k,v) = arma::as_scalar(exp(-2.0*theta(K+1)) * 
                              (suffQCompTrue(k,v) - arma::trans(theta(arma::span(0,K))) * suffHCompTrue.slice(v).col(k)));
        }
        gradCompTrue(K+1,v) = arma::as_scalar(exp(-2.0*theta(K+1)) * 
                              (suffRCompTrue(v) - suffR1CompTrue(v) - 2.0 * arma::trans(theta(arma::span(0,K))) * suffQCompTrue.col(v) +  
                              arma::trans(theta(arma::span(0,K))) * suffHCompTrue.slice(v) * theta(arma::span(0,K))) - (T-1));
      }
      gradCompTrue(K+2,v) = exp(-2.0*theta(K+2)) * 
                            (suffRCompTrue(v) - 2.0 * suffSCompTrue(v) + arma::accu(arma::pow(y.row(v), 2.0))) - T;
    }

  
    suffHTrue.zeros(K+1, K+1);
    for (unsigned int v=0; v<V; v++) // TODO: this should be optimised
    {
      suffHTrue += suffHCompTrue.slice(v);
    }
    
    suffQTrue  = arma::sum(suffQCompTrue, 1);
    suffRTrue  = arma::sum(suffRCompTrue);
    suffR1True = arma::sum(suffR1CompTrue);
    suffSTrue  = arma::sum(suffSCompTrue);
    gradTrue   = arma::sum(gradCompTrue, 1);

  }
  
  /// Computes block-wise initial additive smoothing functional 
  /// used for approximating the gradient.
  arma::colvec initialAdditiveFunction(const arma::colvec& particle, unsigned int lb, unsigned int ub)
  {
    arma::colvec grad(K+3, arma::fill::zeros);
    
    if (lb <= ub)
    {
      grad(K+2) = arma::as_scalar(exp(-2.0*theta(K+2))*(
        arma::trans(particle(arma::span(lb, ub)))*particle(arma::span(lb, ub)) 
        - 2.0*arma::trans(particle(arma::span(lb, ub)))*y(arma::span(lb, ub), arma::span(0,0))
        + arma::trans(y(arma::span(lb, ub), arma::span(0,0)))*y(arma::span(lb, ub), arma::span(0,0))
      ) - (ub - lb + 1));
    }
    
    return grad;
  }
  /// Computes block-wise initial additive smoothing functional 
  /// used for approximating the gradient.
  /*
  arma::colvec initialAdditiveFunction2(const arma::colvec& particle, unsigned int lb, unsigned int ub)
  {
    arma::colvec suffStat((K+1)*(K+1)+(K+1)+2, arma::fill::zeros);
    
    if (lb <= ub)
    {
      double S3 = arma::accu(arma::pow(particle(arma::span(lb, ub)), 2.0));
      double S4 = arma::as_scalar(arma::trans(particle(arma::span(lb, ub))) * y(arma::span(lb, ub), arma::span(0,0)));
      
      suffStat((K+1)*(K+1)+(K+1)) = S3; //arma::trans(particle(arma::span(lb, ub))) * particle(arma::span(lb, ub));
      suffStat((K+1)*(K+1)+(K+1)+1) = S4; //arma::trans(particle(arma::span(lb, ub))) * y(arma::span(lb, ub), arma::span(0,0));
      
    }
    
    return suffStat;
  }
  */
  /// Computes block-wise additive smoothing functional 
  /// used for approximating the gradient.
  arma::colvec additiveFunction(const unsigned int t, const arma::colvec& particleNew, const arma::colvec& particleOld, unsigned int lb, unsigned int ub)
  {
    arma::colvec grad(K+3, arma::fill::zeros);
    arma::colvec S2(K+1, arma::fill::zeros);
    arma::mat S1(K+1, K+1, arma::fill::zeros);
       
    
    if (lb <= ub)
    {
      
//       std::cout << "S1 and S2 0" << std::endl;
      
      S2(0)   = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub)))     * particleOld(arma::span(lb+K, ub+K)));
      S1(0,0) = arma::as_scalar(arma::trans(particleOld(arma::span(lb+K, ub+K))) * particleOld(arma::span(lb+K, ub+K)));
      for (unsigned int k=1; k<K+1; k++)
      {
        S1(k,0) = arma::as_scalar(arma::trans(particleOld(arma::span(lb+K, ub+K)))*(particleOld(arma::span(lb+K-k, ub+K-k)) + particleOld(arma::span(lb+K+k, ub+K+k))));
        S1(0,k) = S1(k,0); // note that S1 is symmetric
      }
      
//       std::cout << "S1 and S2 k, l" << std::endl;
      for (unsigned int k=1; k<K+1; k++)
      {
        S2(k) = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub))) * (particleOld(arma::span(lb+K-k, ub+K-k)) + particleOld(arma::span(lb+K+k, ub+K+k))));
        for (unsigned int l=k; l<K+1; l++)
        {
          S1(k,l) = arma::as_scalar(arma::trans(particleOld(arma::span(lb+K-k, ub+K-k)) + particleOld(arma::span(lb+K+k, ub+K+k))) * (particleOld(arma::span(lb+K-l, ub+K-l)) + particleOld(arma::span(lb+K+l, ub+K+l))));
          if (l != k)
          {
            S1(l,k) = S1(k,l); // note that S1 is symmetric
          }
        }
      }
      
//       double S3 = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub))) * particleNew(arma::span(lb, ub)));
      
      double S3 = arma::accu(arma::pow(particleNew(arma::span(lb, ub)), 2.0));
      double S4 = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub))) * y(arma::span(lb, ub), arma::span(t,t)));
      
      double ySquared = arma::accu(arma::pow(y(arma::span(lb, ub), arma::span(t,t)), 2.0));
            
//       double ySquared = arma::as_scalar(arma::trans(y(arma::span(lb, ub), arma::span(t,t))) * y(arma::span(lb, ub), arma::span(t,t)));
      
//       std::cout << "grad(0:K)" << std::endl;
      grad(arma::span(0,K)) = exp(-2.0 * theta(K+1)) * (S2 - S1 * theta(arma::span(0,K)));
      
//           std::cout << "grad(K+1)" << std::endl;
      grad(K+1) = arma::as_scalar(
        exp(-2.0 * theta(K+1)) * (
          S3 
          - 2.0 * arma::trans(theta(arma::span(0,K))) * S2 
          + arma::trans(theta(arma::span(0,K))) * S1 * theta(arma::span(0,K))
        ) - (ub - lb + 1)
      );
      
//           std::cout << "grad(K+2)" << std::endl;
      grad(K+2) = arma::as_scalar(
        exp(-2.0 * theta(K+2)) * (S3 - 2.0 * S4 + ySquared) - (ub - lb + 1)
      );

    }
    return grad;
  }
  /// Computes block-wise additive smoothing functional 
  /// used for approximating the gradient.
  /*
  arma::colvec additiveFunction2(const unsigned int t, const arma::colvec& particleNew, const arma::colvec& particleOld, unsigned int lb, unsigned int ub)
  {
    arma::colvec suffStat((K+1)*(K+1)+(K+1)+2, arma::fill::zeros);
    arma::colvec S2(K+1, arma::fill::zeros);
    arma::mat S1(K+1, K+1, arma::fill::zeros);
    
    double S3, S4;
       
    
    if (lb <= ub)
    {
      
//       std::cout << "S1 and S2 0" << std::endl;
      
      S2(0)   = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub)))     * particleOld(arma::span(lb+K, ub+K)));
      S1(0,0) = arma::as_scalar(arma::trans(particleOld(arma::span(lb+K, ub+K))) * particleOld(arma::span(lb+K, ub+K)));
      for (unsigned int k=1; k<K+1; k++)
      {
        S1(k,0) = arma::as_scalar(arma::trans(particleOld(arma::span(lb+K, ub+K)))*(particleOld(arma::span(lb+K-k, ub+K-k)) + particleOld(arma::span(lb+K+k, ub+K+k))));
        S1(0,k) = S1(k,0); // note that S1 is symmetric
      }
      
//       std::cout << "S1 and S2 k, l" << std::endl;
      for (unsigned int k=1; k<K+1; k++)
      {
        S2(k) = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub))) * (particleOld(arma::span(lb+K-k, ub+K-k)) + particleOld(arma::span(lb+K+k, ub+K+k))));
        for (unsigned int l=k; l<K+1; l++)
        {
          S1(k,l) = arma::as_scalar(arma::trans(particleOld(arma::span(lb+K-k, ub+K-k)) + particleOld(arma::span(lb+K+k, ub+K+k))) * (particleOld(arma::span(lb+K-l, ub+K-l)) + particleOld(arma::span(lb+K+l, ub+K+l))));
          if (l != k)
          {
            S1(l,k) = S1(k,l); // note that S1 is symmetric
          }
        }
      }
      
//       double S3 = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub))) * particleNew(arma::span(lb, ub)));
      
      S3 = arma::accu(arma::pow(particleNew(arma::span(lb, ub)), 2.0));
      S4 = arma::as_scalar(arma::trans(particleNew(arma::span(lb, ub))) * y(arma::span(lb, ub), arma::span(t,t)));
         
//     std::cout << "storing S1" << std::endl;
    suffStat(arma::span(0,(K+1)*(K+1)-1)) = arma::vectorise(S1);
//     std::cout << "storing S2" << std::endl;
          
    suffStat(arma::span((K+1)*(K+1),(K+1)*(K+2)-1)) = S2;
//     std::cout << "storing S3" << std::endl;
    suffStat((K+1)*(K+2)) = S3;
//     std::cout << "storing S4" << std::endl;
    suffStat((K+1)*(K+2)+1) = S4;

    }
    

    
    return suffStat;
  }
  */
  /// Runs the blocked online gradient-ascent algorithm II.
  void runBlockedOnlineGradientAscent(arma::colvec& thetaInit, bool updateAfterEachBlock, unsigned int padding)
  {       
    arma::mat particlesOldResampled(V, N); // resampled particles from previous step
    arma::mat particlesOld(V, N); // particles from previous step
    arma::mat particlesNew(V, N); // particles from current step
    
    arma::mat logComponentWeightsOld(V, N); // local incremental weights from previous step
    arma::mat logComponentWeightsNew(V, N); // local incremental weights from current step
    
    arma::mat gradBlockNew(K+3, nBlocks); // block-wise gradient approximations
    arma::mat gradBlockOld(K+3, nBlocks); // block-wise gradient approximations
    
    arma::cube alphaNew(K+3, N, nBlocks); // for forward smoothing
    arma::cube alphaOld(K+3, N, nBlocks); // for forward smoothing
    
    arma::uvec parentIndices(N); // parent indices
    arma::colvec logWeights(N); // unnormalised log-weights
    logWeights.fill(-log(N)); // start with uniform weights
    
    arma::mat blockWeights(N, nBlocks); // self-normalised weights associated with blocks
    arma::mat blockNeighbourhoodWeightsNew(N, nBlocks); // self-normalised weights associated with block neighbourhoods
    arma::mat blockNeighbourhoodWeightsOld(N, nBlocks); // self-normalised weights associated with block neighbourhoods
        
    
    arma::colvec unnormalisedBackwardKernel(N);
    arma::colvec myZeros(K, arma::fill::zeros);

    double gradL2norm;
    unsigned int p; // iteration number of the gradient-ascent algorithm
    
    if (updateAfterEachBlock) // If we update the parameters once after each spatial block
    {
      thetaFull.set_size(K+3, T*nBlocks);
      p = 0;
    }
    else // If we update the parameters once after each time step
    {
      thetaFull.set_size(K+3, T);
    }
    theta = thetaInit;
    setAuxiliaryParameters();

    ///////////////////////////////////////////////////////////////////////////
    std::cout << "Step 1" << std::endl;
    ///////////////////////////////////////////////////////////////////////////
    
    for (unsigned int j=0; j<nBlocks; j++) 
    {
      computeBlockParameters(j);
      computeInteriorBlockParameters(padding);
      std::cout << "Block " << j << std::endl;
      
      /////////////////////////////////////////////////////////////////////////
      // std::cout << "Sample particles for the " << j << "th block and compute weights" << std::endl;
      /////////////////////////////////////////////////////////////////////////
      if (prop == SMC_PROPOSAL_PRIOR) 
      {
        particlesNew(arma::span(lInn, uInn), arma::span::all) = 
          gaussian::sampleMultivariate(N, m0(arma::span(lInn, uInn)), 
                  C0(arma::span(lInn, uInn), arma::span(lInn, uInn)), false); 
        logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all) = 
          gaussian::evaluateDensityUnivariate(arma::repmat(y(arma::span(lInn, uInn), arma::span(0,0)), 1, N), 
                particlesNew(arma::span(lInn, uInn), arma::span::all), 
                D(0,0), true, true);       
      }
      else if (prop == SMC_PROPOSAL_OPTIMAL)
      { 
        arma::mat sigma = arma::inv(1.0/(D(0,0)*D(0,0))*arma::eye(blockSize, blockSize) + arma::inv(C0(arma::span(lInn, uInn),arma::span(lInn, uInn))));
        arma::colvec mu = sigma*(1.0/(D(0,0) * D(0,0))*arma::eye(blockSize, blockSize)*y(arma::span(lInn, uInn), arma::span(0,0)) + arma::inv(C0(arma::span(lInn, uInn),arma::span(lInn, uInn)))*m0(arma::span(lInn, uInn)));
        
        particlesNew(arma::span(lInn, uInn), arma::span::all) = gaussian::sampleMultivariate(N, mu, sigma, false);
        for (unsigned int n=0; n<N; n++)
        {
          logComponentWeightsNew(arma::span(lInn, uInn), arma::span(n,n)) = gaussian::evaluateDensityUnivariate(y(arma::span(lInn, uInn), arma::span(0,0)), m0(arma::span(lInn, uInn)), C0(0,0) + D(0,0)*D(0,0), false, true);
        }
      }
      
      
      logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all)));
      blockWeights.col(j) = normaliseWeights(logWeights);
      logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lInnNei, uInnNei), arma::span::all)));
      blockNeighbourhoodWeightsNew.col(j) = normaliseWeights(logWeights);
      
      /////////////////////////////////////////////////////////////////////////
      // std::cout << "Compute gradient estimate for the " << j << "th block" << std::endl;
      /////////////////////////////////////////////////////////////////////////    
      alphaNew.zeros();
      gradBlockNew.zeros();

      if (updateAfterEachBlock)
      {
        for (unsigned int n=0; n<N; n++)
        {
          alphaNew.slice(j).col(n) += initialAdditiveFunction(particlesNew.col(n), lInnInt, uInnInt);
        }
        gradBlockNew.col(j) = alphaNew.slice(j) * blockWeights.col(j);

        gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(gradBlockNew.col(j), 2.0))));
        theta = theta + stepSizes(p) * gradBlockNew.col(j) / std::max(gradL2norm, 1.0);
        thetaFull.col(p) = theta;
        std::cout << arma::trans(theta) << std::endl;
        setParameters();
        p = p + 1;
      }
    }
    

    if (!updateAfterEachBlock)
    {  
      for (unsigned int j=0; j<nBlocks; j++)
      {
        computeBlockParameters(j);
        computeInteriorBlockParameters(padding);
        
        for (unsigned int n=0; n<N; n++)
        {
          alphaNew.slice(j).col(n) += initialAdditiveFunction(particlesNew.col(n), lInnInt, uInnInt);
        }
        gradBlockNew.col(j) = alphaNew.slice(j) * blockWeights.col(j);
      }
      
      gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(arma::sum(gradBlockNew, 1), 2.0))));
      theta = theta + stepSizes(1) * arma::sum(gradBlockNew, 1) / std::max(gradL2norm, 1.0); 
      thetaFull.col(0) = theta; 
      std::cout << arma::trans(theta) << std::endl;
      setParameters();
    }


    for (unsigned int t=1; t<T; t++)
    {  
      /////////////////////////////////////////////////////////////////////////
      std::cout << "Step " << t << std::endl;
      /////////////////////////////////////////////////////////////////////////
      gradBlockOld = gradBlockNew;
      particlesOld = particlesNew;
      alphaOld     = alphaNew;
      logComponentWeightsOld = logComponentWeightsNew;
      blockNeighbourhoodWeightsOld = blockNeighbourhoodWeightsNew;
      alphaNew.zeros();
      gradBlockNew.zeros();
      
      /////////////////////////////////////////////////////////////////////////
      std::cout << "Resampling" << std::endl;
      /////////////////////////////////////////////////////////////////////////
      if (filt == SMC_FILTER_STANDARD)
      {
        logWeights = arma::trans(arma::sum(logComponentWeightsNew, 0));
        resample::systematic(parentIndices, normaliseWeights(logWeights), N);  
        particlesOldResampled = particlesOld.cols(parentIndices); 
      }
      else if (filt == SMC_FILTER_BLOCK)
      {
        for (unsigned int j=0; j<nBlocks; j++) 
        {
          computeBlockParameters(j);
          computeInteriorBlockParameters(padding);
          logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all), 0));
          parentIndices = sampleInt(N, normaliseWeights(logWeights));
          for (unsigned int n=0; n<N; n++)
          {
            particlesOldResampled(arma::span(lInn,uInn), arma::span(n,n)) = particlesOld(arma::span(lInn,uInn), arma::span(parentIndices(n),parentIndices(n)));
          }
        }
      }
      
      // Proposing particles for each block
      for (unsigned int j=0; j<nBlocks; j++) 
      {
        computeBlockParameters(j);
        computeInteriorBlockParameters(padding);
        std::cout << "Block " << j << std::endl;

        ///////////////////////////////////////////////////////////////////////
        std::cout << "Sample particles for the "<< j << "th block and compute weights" << std::endl;
        ///////////////////////////////////////////////////////////////////////
        
        if (prop == SMC_PROPOSAL_PRIOR) 
        {
          particlesNew(arma::span(lInn, uInn), arma::span::all) = 
            gaussian::sampleMultivariate(N, 
              A(arma::span(lInn, uInn), arma::span(lInnNei, uInnNei)) * particlesOldResampled(arma::span(lInnNei, uInnNei), arma::span::all), 
              B(arma::span(lInn, uInn), arma::span(lInn, uInn)), true); 
          logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all) = 
            gaussian::evaluateDensityUnivariate(arma::repmat(y(arma::span(lInn, uInn), arma::span(t,t)), 1, N), 
                  particlesNew(arma::span(lInn, uInn), arma::span::all), 
                  D(0,0), true, true);
        }
        else if (prop == SMC_PROPOSAL_OPTIMAL)
        {      
          arma::mat sigma = 
            1.0/(1.0/(D(0,0)*D(0,0)) + 1.0/(B(0,0)*B(0,0))) * arma::eye(blockSize, blockSize);
            
          for (unsigned int n=0; n<N; n++)
          {
            
            arma::colvec mu = 
            sigma*(1.0/(D(0,0)*D(0,0)) * y(arma::span(lInn, uInn), arma::span(t,t)) + 
                    1.0/(B(0,0)*B(0,0)) * A(arma::span(lInn, uInn), arma::span(lInnNei, uInnNei)) * particlesOldResampled(arma::span(lInnNei, uInnNei), arma::span(n,n))
                  );
            
            particlesNew(arma::span(lInn, uInn), arma::span(n,n)) = gaussian::sampleMultivariate(1, mu, sigma, false); // was: mu.col(n)!
            
            logComponentWeightsNew(arma::span(lInn, uInn), arma::span(n,n)) = 
              gaussian::evaluateDensityUnivariate(y(arma::span(lInn, uInn), arma::span(t,t)), 
                A(arma::span(lInn, uInn), arma::span(lInnNei, uInnNei))*particlesOldResampled(arma::span(lInnNei, uInnNei), arma::span(n,n)), 
                B(0,0)*B(0,0) + D(0,0)*D(0,0), false, true);
          }
        }
        logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all), 0));
        blockWeights.col(j) = normaliseWeights(logWeights);
        logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lInnNei, uInnNei), arma::span::all), 0));
        blockNeighbourhoodWeightsNew.col(j) = normaliseWeights(logWeights);
        
        ///////////////////////////////////////////////////////////////////////
        std::cout << "Compute gradient estimate for the " << j << "th block" << std::endl;
        ///////////////////////////////////////////////////////////////////////
        
        if (updateAfterEachBlock)
        {
          for (unsigned int n=0; n<N; n++)
          {
            for (unsigned int m=0; m<N; m++)
            {
              //std::cout << "backward kernel:" << std::endl;
              
              unnormalisedBackwardKernel(m) = 
                blockNeighbourhoodWeightsOld(m, j) * 
                arma::as_scalar(gaussian::evaluateDensityMultivariate(particlesNew(arma::span(lInn, uInn), arma::span(n,n)), 
                  A(arma::span(lInn, uInn), arma::span(lInnNei, uInnNei)) * particlesOld(arma::span(lInnNei, uInnNei), arma::span(m,m)), B(0,0), true, false));
                
              //std::cout << "additive functional:" << std::endl;
              
              alphaNew.slice(j).col(n) += 
                unnormalisedBackwardKernel(m)*(alphaOld.slice(j).col(m) + 
                additiveFunction(t, particlesNew.col(n), arma::join_cols(arma::join_cols(myZeros, particlesOld.col(m)), myZeros), lInnInt, uInnInt)
              ); 
            }
            alphaNew.slice(j).col(n) = alphaNew.slice(j).col(n) / arma::accu(unnormalisedBackwardKernel);
          }
          gradBlockNew.col(j) = alphaNew.slice(j) * blockWeights.col(j);
        
          
          gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(gradBlockNew.col(j) - gradBlockOld.col(j), 2.0))));
          theta = theta + stepSizes(p) * (gradBlockNew.col(j) - gradBlockOld.col(j)) / std::max(gradL2norm, 1.0);
          std::cout << arma::trans(theta) << std::endl;
          thetaFull.col(p) = theta;
          setParameters();
          p = p + 1;
        }
      }
      
      if (!updateAfterEachBlock)
      {
        for (unsigned int j=0; j<nBlocks; j++)
        {
          computeBlockParameters(j);
          computeInteriorBlockParameters(padding);
                   
          for (unsigned int n=0; n<N; n++)
          {
            for (unsigned int m=0; m<N; m++)
            {
//               std::cout << "backward kernel:" << std::endl;
              
              // NOTE: is lOut and lOutNei here correct? Shouldn't it be lInn and lInnNei?
              unnormalisedBackwardKernel(m) = 
                blockNeighbourhoodWeightsOld(m, j) * 
                arma::as_scalar(gaussian::evaluateDensityMultivariate(particlesNew(arma::span(lOut, uOut), arma::span(n,n)), 
                  A(arma::span(lOut, uOut), arma::span(lOutNei, uOutNei)) * particlesOld(arma::span(lOutNei, uOutNei), arma::span(m,m)), B(0,0), true, false));
                
//               std::cout << "additive functional:" << std::endl;
              
              alphaNew.slice(j).col(n) += 
                unnormalisedBackwardKernel(m)*(alphaOld.slice(j).col(m) + 
                additiveFunction(t, particlesNew.col(n), arma::join_cols(arma::join_cols(myZeros, particlesOld.col(m)), myZeros), lInn, uInn)
              ); 
            }
            alphaNew.slice(j).col(n) = alphaNew.slice(j).col(n) / arma::accu(unnormalisedBackwardKernel);
          }
          gradBlockNew.col(j) = alphaNew.slice(j) * blockWeights.col(j);
        }

//         std::cout << "L2 norm of gradient" << std::endl;
        gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(arma::sum(gradBlockNew, 1) - arma::sum(gradBlockOld, 1), 2.0))));
        theta = theta + stepSizes(t) * (arma::sum(gradBlockNew, 1) - arma::sum(gradBlockOld, 1)) / std::max(gradL2norm, 1.0);
        std::cout << arma::trans(theta) << std::endl;
        thetaFull.col(t) = theta;
        setParameters();
      }
    }     
  }
  
  /// Runs the blocked online gradient-ascent algorithm with enlarged blocks.
  /// The algorithm updates the parameters only once per time step.
  void runBlockedOnlineGradientAscentEnlarged(arma::colvec& thetaInit, const bool normaliseByL2Norm, const bool estimateTheta, arma::mat& gradBlockOut) 
//                                               arma::mat& suffStatOut)
  {       
    
    arma::mat particlesOldResampled(V, N); // resampled particles from previous step
    arma::mat particlesOld(V, N); // particles from previous step
    arma::mat particlesNew(V, N); // particles from current step
    
    arma::mat logComponentWeightsOld(V, N); // local incremental weights from previous step
    arma::mat logComponentWeightsNew(V, N); // local incremental weights from current step
    
    arma::mat gradBlockNew(K+3, nBlocks); // block-wise gradient approximations
    arma::mat gradBlockOld(K+3, nBlocks); // block-wise gradient approximations
    
    arma::cube alphaNew(K+3, N, nBlocks); // for forward smoothing
    arma::cube alphaOld(K+3, N, nBlocks); // for forward smoothing
    
    arma::uvec parentIndices(N); // parent indices
    arma::colvec logWeights(N); // unnormalised log-weights
    logWeights.fill(-log(N)); // start with uniform weights
    
    arma::mat enlargedBlockWeights(N, nBlocks); // self-normalised weights associated with enlarged blocks
    arma::mat enlargedBlockNeighbourhoodWeightsNew(N, nBlocks); // self-normalised weights associated with neighbourhoods of enlarged blocks
    arma::mat enlargedBlockNeighbourhoodWeightsOld(N, nBlocks); // self-normalised weights associated with neighbourhoods of enlarged blocks
    
    arma::colvec unnormalisedBackwardKernel(N);
    arma::colvec myZeros(K, arma::fill::zeros);

    double gradL2norm;
//     unsigned int p; // iteration number of the gradient-ascent algorithm
    
    thetaFull.set_size(K+3, T);
    theta = thetaInit;
    setParameters();
    setAuxiliaryParameters(); // pre-computes some of the static parameters to avoid repeated calculations of these

    ///////////////////////////////////////////////////////////////////////////
    std::cout << "Step 0" << std::endl;
    ///////////////////////////////////////////////////////////////////////////
    
    if (filt == SMC_FILTER_STANDARD || filt == SMC_FILTER_BLOCK)
    {
      for (unsigned int j=0; j<nBlocks; j++) 
      {
        computeBlockParameters(j);
        std::cout << "Block " << j << std::endl;
        
        /////////////////////////////////////////////////////////////////////////
  //       std::cout << "Sample particles for the " << j << "th block and compute weights" << std::endl;
        /////////////////////////////////////////////////////////////////////////
      
        if (prop == SMC_PROPOSAL_PRIOR) 
        {
          particlesNew(arma::span(lInn, uInn), arma::span::all) = 
            gaussian::sampleMultivariate(N, m0(arma::span(lInn, uInn)), 
                    C0(arma::span(lInn, uInn), arma::span(lInn, uInn)), false); 
          logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all) = 
            gaussian::evaluateDensityUnivariate(arma::repmat(y(arma::span(lInn, uInn), arma::span(0,0)), 1, N), 
                  particlesNew(arma::span(lInn, uInn), arma::span::all), 
                  D(0,0), true, true);       
        }
        else if (prop == SMC_PROPOSAL_OPTIMAL)
        { 
          arma::mat sigma = arma::inv(1.0/(D(0,0)*D(0,0))*arma::eye(blockSize, blockSize) + arma::inv(C0(arma::span(lInn, uInn),arma::span(lInn, uInn))));
          arma::colvec mu = sigma*(1.0/(D(0,0) * D(0,0))*arma::eye(blockSize, blockSize)*y(arma::span(lInn, uInn), arma::span(0,0)) + arma::inv(C0(arma::span(lInn, uInn),arma::span(lInn, uInn)))*m0(arma::span(lInn, uInn)));
          
          particlesNew(arma::span(lInn, uInn), arma::span::all) = gaussian::sampleMultivariate(N, mu, sigma, false);
          for (unsigned int n=0; n<N; n++)
          {
            logComponentWeightsNew(arma::span(lInn, uInn), arma::span(n,n)) = gaussian::evaluateDensityUnivariate(y(arma::span(lInn, uInn), arma::span(0,0)), m0(arma::span(lInn, uInn)), C0(0,0) + D(0,0)*D(0,0), false, true);
          }
        }
        
        logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lOut, uOut), arma::span::all)));
        enlargedBlockWeights.col(j) = normaliseWeights(logWeights);
        logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lOutNei, uOutNei), arma::span::all)));
        enlargedBlockNeighbourhoodWeightsNew.col(j) = normaliseWeights(logWeights);
      
      
        /////////////////////////////////////////////////////////////////////////
        // std::cout << "Compute gradient estimate for the " << j << "th block" << std::endl;
        /////////////////////////////////////////////////////////////////////////
      }
    }
    else if (filt == SMC_FILTER_EXACT)
    {
      if (estimateTheta)
      {
        std::cout << "ERROR! Using IID samples from the exact filter is currently only implemented when keeping theta fixed!" << std::endl;
      }
      
      runKalman();
      
      particlesFull.set_size(V, N, T);
      enlargedBlockWeights.fill(1.0/N);
      enlargedBlockNeighbourhoodWeightsNew.fill(1.0/N);
      
      for (unsigned int t=0; t<T; t++)
      {  
        if (storeHistory)
        {
          particlesFull.slice(t) = gaussian::sampleMultivariate(N, mU.col(t), CU.slice(t), false);
        }
      }
      particlesNew = particlesFull.slice(0);
    }
    
    alphaNew.zeros();
    gradBlockNew.zeros();
    
    for (unsigned int j=0; j<nBlocks; j++)
    {
      computeBlockParameters(j);             
      for (unsigned int n=0; n<N; n++)
      {
        alphaNew.slice(j).col(n) += initialAdditiveFunction(particlesNew.col(n), lInn, uInn);
      }
      gradBlockNew.col(j) = alphaNew.slice(j) * enlargedBlockWeights.col(j);
    }

    if (estimateTheta)
    {
      if (normaliseByL2Norm)
      {
        gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(arma::sum(gradBlockNew, 1), 2.0))));
        std::cout << "gradL2Norm: " << gradL2norm << std::endl;
        theta = theta + stepSizes(0) * arma::sum(gradBlockNew, 1) / std::max(gradL2norm, 1.0); 
      }
      else 
      {
        theta = theta + stepSizes(0) * arma::sum(gradBlockNew, 1); 
      }
      
      thetaFull.col(0) = theta; 
      std::cout << arma::trans(theta) << std::endl;
    }  
    setParameters();

    for (unsigned int t=1; t<T; t++)
    {  
      /////////////////////////////////////////////////////////////////////////
      std::cout << "Step " << t << std::endl;
      /////////////////////////////////////////////////////////////////////////
      gradBlockOld = gradBlockNew;
      particlesOld = particlesNew;
      alphaOld     = alphaNew;
      logComponentWeightsOld = logComponentWeightsNew;
      enlargedBlockNeighbourhoodWeightsOld = enlargedBlockNeighbourhoodWeightsNew;
      alphaNew.zeros();
      gradBlockNew.zeros();
      
      /////////////////////////////////////////////////////////////////////////
//       std::cout << "Resampling" << std::endl;
      /////////////////////////////////////////////////////////////////////////
      if (filt == SMC_FILTER_STANDARD)
      {
        logWeights = arma::trans(arma::sum(logComponentWeightsNew, 0));
        resample::systematic(parentIndices, normaliseWeights(logWeights), N);  
        particlesOldResampled = particlesOld.cols(parentIndices); 
      }
      else if (filt == SMC_FILTER_BLOCK)
      {
        for (unsigned int j=0; j<nBlocks; j++) 
        {
          computeBlockParameters(j);
          logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all), 0));
          parentIndices = sampleInt(N, normaliseWeights(logWeights));
          for (unsigned int n=0; n<N; n++)
          {
            particlesOldResampled(arma::span(lInn,uInn), arma::span(n,n)) = particlesOld(arma::span(lInn,uInn), arma::span(parentIndices(n),parentIndices(n)));
          }
        }
      }
      else if (filt == SMC_FILTER_EXACT)
      {
        particlesOldResampled = particlesNew;
      }
      
      if (filt == SMC_FILTER_STANDARD || filt == SMC_FILTER_BLOCK)
      {
        // Proposing particles for each block
        for (unsigned int j=0; j<nBlocks; j++) 
        {
          computeBlockParameters(j);
          std::cout << "Block " << j << std::endl;

          ///////////////////////////////////////////////////////////////////////
          //std::cout << "Sample particles for the "<< j << "th block and compute weights" << std::endl;
          ///////////////////////////////////////////////////////////////////////
          
          if (prop == SMC_PROPOSAL_PRIOR) 
          {
            particlesNew(arma::span(lInn, uInn), arma::span::all) = 
              gaussian::sampleMultivariate(N, 
                A(arma::span(lInn, uInn), arma::span(lInnNei, uInnNei)) * particlesOldResampled(arma::span(lInnNei, uInnNei), arma::span::all), 
                B(arma::span(lInn, uInn), arma::span(lInn, uInn)), true); 
            logComponentWeightsNew(arma::span(lInn, uInn), arma::span::all) = 
              gaussian::evaluateDensityUnivariate(arma::repmat(y(arma::span(lInn, uInn), arma::span(t,t)), 1, N), 
                    particlesNew(arma::span(lInn, uInn), arma::span::all), 
                    D(0,0), true, true);
          }
          else if (prop == SMC_PROPOSAL_OPTIMAL)
          {      
            arma::mat sigma = 
              1.0/(1.0/(D(0,0)*D(0,0)) + 1.0/(B(0,0)*B(0,0))) * arma::eye(blockSize, blockSize);
                       
            for (unsigned int n=0; n<N; n++)
            {
              
              arma::colvec mu = 
              sigma*(1.0/(D(0,0)*D(0,0)) * y(arma::span(lInn, uInn), arma::span(t,t)) + 
                      1.0/(B(0,0)*B(0,0)) * A(arma::span(lInn, uInn), arma::span(lInnNei, uInnNei)) * particlesOldResampled(arma::span(lInnNei, uInnNei), arma::span(n,n))
                    );
              
              particlesNew(arma::span(lInn, uInn), arma::span(n,n)) = gaussian::sampleMultivariate(1, mu, sigma, false); // was: mu.col(n)!
              
              logComponentWeightsNew(arma::span(lInn, uInn), arma::span(n,n)) = 
                gaussian::evaluateDensityUnivariate(y(arma::span(lInn, uInn), arma::span(t,t)), 
                  A(arma::span(lInn, uInn), arma::span(lInnNei, uInnNei))*particlesOldResampled(arma::span(lInnNei, uInnNei), arma::span(n,n)), 
                  B(0,0)*B(0,0) + D(0,0)*D(0,0), false, true);
            }
          }
          logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lOut, uOut), arma::span::all), 0));
          enlargedBlockWeights.col(j) = normaliseWeights(logWeights);
          logWeights = arma::trans(arma::sum(logComponentWeightsNew(arma::span(lOutNei, uOutNei), arma::span::all), 0));
          enlargedBlockNeighbourhoodWeightsNew.col(j) = normaliseWeights(logWeights);
          
          ///////////////////////////////////////////////////////////////////////
          //std::cout << "Compute gradient estimate for the " << j << "th block" << std::endl;
          ///////////////////////////////////////////////////////////////////////
        }
      }
      else if (filt == SMC_FILTER_EXACT)
      {
        particlesNew = particlesFull.slice(t);
      }
      
      for (unsigned int j=0; j<nBlocks; j++)
      {
        computeBlockParameters(j);       
        for (unsigned int n=0; n<N; n++)
        {
          for (unsigned int m=0; m<N; m++)
          {
            unnormalisedBackwardKernel(m) = 
              enlargedBlockNeighbourhoodWeightsOld(m, j) * // TODO: shouldn't these be the previous weights?
              arma::as_scalar(gaussian::evaluateDensityMultivariate(particlesNew(arma::span(lOut, uOut), arma::span(n,n)), 
                A(arma::span(lOut, uOut), arma::span(lOutNei, uOutNei)) * particlesOld(arma::span(lOutNei, uOutNei), arma::span(m,m)), B(0,0), true, false));
              
            alphaNew.slice(j).col(n) += 
              unnormalisedBackwardKernel(m)*(alphaOld.slice(j).col(m) + 
              additiveFunction(t, particlesNew.col(n), arma::join_cols(arma::join_cols(myZeros, particlesOld.col(m)), myZeros), lInn, uInn)
            );    
          }
          alphaNew.slice(j).col(n) = alphaNew.slice(j).col(n) / arma::accu(unnormalisedBackwardKernel);
        }
        gradBlockNew.col(j) = alphaNew.slice(j) * enlargedBlockWeights.col(j);
      }

      if (estimateTheta)
      {
        if (normaliseByL2Norm)
        {
          gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(arma::sum(gradBlockNew, 1) - arma::sum(gradBlockOld, 1), 2.0))));
          theta = theta + stepSizes(t) * (arma::sum(gradBlockNew, 1) - arma::sum(gradBlockOld, 1)) / std::max(gradL2norm, 1.0);
        }
        else 
        {
          theta = theta + stepSizes(t) * (arma::sum(gradBlockNew, 1) - arma::sum(gradBlockOld, 1));
        }
        std::cout << arma::trans(theta) << std::endl;
        thetaFull.col(t) = theta;
        setParameters();
      }
    }
    
    if (!estimateTheta)
    {
      gradBlockOut = gradBlockNew;
      ////////////////////////////////
//       suffStatOut = suffStatBlockNew;
      ////////////////////////////////
    }
  }
  /// Runs offline stochastic (blocked) gradient algorithm using FFBS
  void run_offline_stochastic_gradient_ascent( )
  {
    unsigned int G = stepSizes.n_rows; // number of iterations
    thetaFull.set_size(K+3, G);
    thetaFull.col(0) = thetaInit;
    theta = thetaInit;
    double gradL2norm;
    storeHistory = true;
    storeSuff = true;
    
    for (unsigned int g=0; g<G; g++) 
    { 
      //std::cout << "Stochastic gradient-ascent, Step: " << g << "; back: " << static_cast<unsigned int>(back) << std::endl;
      runSmc();
      runParticleSmoother();
      // Gradient estimate is normalised to have unit L2-norm:
      gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(gradEst, 2.0))));
      if (arma::as_scalar(stepSizes(g))*gradL2norm > 0.1)
      {
        theta += arma::as_scalar(stepSizes(g)) * gradEst/std::max(gradL2norm, 1.0);
      }
      else
      {
        theta += arma::as_scalar(stepSizes(g)) * gradEst;
      }
      thetaFull.col(g) = theta;
      setParameters();
      setAuxiliaryParameters();
    }
  }
  /// Runs offline gradient algorithm using exactly calculated gradients
  void run_offline_gradient_ascent( )
  {
    unsigned int G = stepSizes.n_rows; // number of iterations
    thetaFull.set_size(K+3, G);
    thetaFull.col(0) = thetaInit;
    theta = thetaInit;
    double gradL2norm;
    
    for (unsigned int g=0; g<G; g++) 
    { 
//       std::cout << "Gradient-ascent, Step: " << g << std::endl;
      runKalman();
      runKalmanSmoother();
      computeSuffTrue();
      gradL2norm = sqrt(arma::as_scalar(arma::sum(arma::pow(gradTrue, 2.0))));
//       if (arma::as_scalar(stepSizes(g))*gradL2norm > 0.5)
//       {
//         std::cout << "gradL2norm: " << gradL2norm << std::endl;
        theta += arma::as_scalar(stepSizes(g)) * gradTrue / std::max(gradL2norm, 1.0);
//       }
//       else
//       {
//         theta += arma::as_scalar(stepSizes(g)) * gradTrue;
//       }
//       theta += arma::as_scalar(stepSizes(g)) * gradTrue;
//       std::cout << theta << std::endl;
      thetaFull.col(g) = theta;
      setParameters();
      setAuxiliaryParameters();

    }
  }
  /// Runs offline approximate blocked gradient algorithm using FFBS.
  void run_offline_stochastic_em( )
  {
    unsigned int G = stepSizes.n_rows; // number of iterations
    thetaFull.set_size(K+3, G);
    thetaFull.col(0) = thetaInit;
    theta = thetaInit;
    storeSuff = true;
    storeHistory = true;
    
    for (unsigned int g=0; g<G; g++) 
    { 
      //std::cout << "Stochastic offline EM, Step: " << g << "; back: " << static_cast<unsigned int>(back) << std::endl;
      runSmc();
      runParticleSmoother();
      calculate_argmax_Q(suffHEst, suffQEst, suffREst, suffR1Est, suffSEst);
      thetaFull.col(g) = theta;
      setParameters();
      setAuxiliaryParameters();
    }
  }
  
  /// Runs offline gradient algorithm using exactly calculated gradients
  void run_offline_em( )
  {
    unsigned int G = stepSizes.n_rows; // number of iterations
    thetaFull.set_size(K+3, G);
    thetaFull.col(0) = thetaInit;
    theta = thetaInit;
    
    for (unsigned int g=0; g<G; g++) 
    { 
      // std::cout << "Offline EM, Step: " << g << std::endl;
      
      runKalman();
      
      runKalmanSmoother();
      
      computeSuffTrue();
      
      calculate_argmax_Q(suffHTrue, suffQTrue, suffRTrue, suffR1True, suffSTrue);
      thetaFull.col(g) = theta;
      setParameters();
      setAuxiliaryParameters();
    }
  }
  /// Checking whether the matrix A is such that 
  /// the latent process is stationary.
  bool checkStationarity(const arma::colvec& theta)
  {
    arma::mat A_;
    // Creating the banded diagonal matrix A from theta
    if (K == 0)
    {
      A_ = arma::as_scalar(theta(0)) * arma::eye(V, V);
    }
    else // assuming that K < V
    {
      arma::colvec firstCol(V, arma::fill::zeros);
      firstCol(arma::span(0,K)) = theta(arma::span(0,K));
      A_ = arma::toeplitz(firstCol);
    }
    return(arma::all(arma::abs(arma::eig_sym(A_)) < 1.0)); 
  }
  /// Samples a value of theta from the prior.
  void samplePriorTheta(arma::colvec& theta)
  {
    theta.set_size(K+3);
    
    // Uniform prior over $(-1, 1)$, for $a_0, \dotsc, a_K$
    // but further restricted to the subspace on which the 
    // transition matrix A is stationary  
    bool isStationary = false;
    while (isStationary == false)
    {
      theta(arma::span(0,K)) = 2.0 * arma::randu<arma::colvec>(K+1) - 1;
      isStationary = checkStationarity(theta);
    }
    
    std::cout << "sample b,d" << std::endl;
    
    // Inverse-Gamma priors for the diagonal entries of B*B.t() and of D*D.t():
    double b = 1.0 / std::sqrt(arma::as_scalar(arma::randg(1, arma::distr_param(hyperParameters(0), 1.0 / hyperParameters(1)))));
    double d = 1.0 / std::sqrt(arma::as_scalar(arma::randg(1, arma::distr_param(hyperParameters(2), 1.0 / hyperParameters(3)))));   
    
    std::cout << "store b,d" << std::endl;
        
    theta(K+1) = repar(b);
    theta(K+2) = repar(d);
    
  }
  /// Evaluates the log-unnormalised prior density.
  double evaluateLogPriorDensityTheta(const arma::colvec& theta_)
  {
    double logDensity = 0; 
    
    // Checking whether the transition matrix is stationary: 
    bool isStationary = checkStationarity(theta_);
 
    if (isStationary)
    {
      switch (para)
      {
        case SMC_PARAMETRISATION_UNBOUNDED:        
          logDensity =  - 2.0 * hyperParameters(0) * theta_(K+1) - hyperParameters(1) * exp(-2.0 * theta_(K+1));
          logDensity += - 2.0 * hyperParameters(2) * theta_(K+2) - hyperParameters(3) * exp(-2.0 * theta_(K+2));
          break;
        case SMC_PARAMETRISATION_NATURAL:
          logDensity =  - (2.0 * hyperParameters(0) + 1.0) * log(theta_(K+1)) - hyperParameters(1) / pow(theta_(K+1), 2.0);
          logDensity += - (2.0 * hyperParameters(2) + 1.0) * log(theta_(K+2)) - hyperParameters(3) / pow(theta_(K+2), 2.0);
          break; 
      } 
    }
    else
    {
      logDensity = - arma::datum::inf;
    }
    
    return logDensity;
  }
  /// Proposes a new value of theta for use in a random-walk
  /// Metropolis--Hastings update.
  void proposeTheta(arma::colvec& thetaProp, const arma::colvec& thetaOld, const double proposalScale)
  {
    thetaProp.set_size(K+3); 
    switch (para)
    {
      
      case SMC_PARAMETRISATION_UNBOUNDED:      
        thetaProp(arma::span(K+1,K+2)) = thetaOld(arma::span(K+1,K+2)) + proposalScale * rwmhSd(arma::span(K+1,K+2)) % arma::randn<arma::colvec>(2);          
        break;
      case SMC_PARAMETRISATION_NATURAL:
        thetaProp(K+1) = gaussian::rtnorm(0.0, arma::datum::inf, thetaOld(K+1), proposalScale * rwmhSd(K+1));
        thetaProp(K+2) = gaussian::rtnorm(0.0, arma::datum::inf, thetaOld(K+2), proposalScale * rwmhSd(K+2));
        break; 

    } 
    for (unsigned int k=0; k<K+1; k++)
    {
      thetaProp(k) = gaussian::rtnorm(-1.0, 1.0, thetaOld(k), proposalScale * rwmhSd(k));
    }
  }
  /// Evaluates the log-unnormalised proposal density
  /// for use in a random-walk Metropolis--Hastings update.
  double evaluateLogProposalDensityTheta(const arma::colvec& thetaProp, const arma::colvec& theta, const double proposalScale)
  {
    double logDensity = 0; 
    switch (para)
    {
      case SMC_PARAMETRISATION_NATURAL:
        logDensity += gaussian::dtnorm(thetaProp(K+1), 0.0, arma::datum::inf, theta(K+1), proposalScale * rwmhSd(K+1), true, true) + 
                      gaussian::dtnorm(thetaProp(K+2), 0.0, arma::datum::inf, theta(K+2), proposalScale * rwmhSd(K+2), true, true);
        break;    
      case SMC_PARAMETRISATION_UNBOUNDED:       
        logDensity += gaussian::evaluateDensityUnivariate(thetaProp(K+1), theta(K+1), proposalScale * rwmhSd(K+1), true, true) + 
                      gaussian::evaluateDensityUnivariate(thetaProp(K+2), theta(K+2), proposalScale * rwmhSd(K+2), true, true);           
        break;
    } 
    for (unsigned int k=0; k<K+1; k++)
    {
      logDensity += gaussian::dtnorm(thetaProp(k), -1.0, 1.0, theta(k), proposalScale * rwmhSd(k), true, true);
    }
    return logDensity;
  }
  /// Evaluates the log-marginal likelihood
  double evaluateLogMarginalLikelihood(const arma::colvec& theta_)
  {
    theta = theta_;
    setParameters();
    setAuxiliaryParameters();
    runKalman();
    return logLikeTrue;
  }
  /// Evaluates the log-complete likelihood; also includes 
  /// an inverse-temperature parameter beta to raise the conditional likelihood
  /// (the log-density of the observations given the parameters and
  /// given the latent variables) to some power.
  double evaluateLogCompleteLikelihood(const arma::colvec& theta_, const arma::mat& particlePath_, const double beta_)
  {
    theta = theta_;
    
    theta(K+1) = repar(inverseRepar(theta(K+1)) / sqrt(beta_));
    theta(K+2) = repar(inverseRepar(theta(K+2)) / sqrt(beta_));

    setParameters();
    setAuxiliaryParameters();
    
    double logCompLike = arma::as_scalar(gaussian::evaluateDensityMultivariate(particlePath_.col(0), m0, C0, false, true)) + 
                         arma::accu(gaussian::evaluateDensityMultivariate(particlePath_(arma::span::all, arma::span(1,T-1)), 
                           A*particlePath_(arma::span::all, arma::span(0,T-2)), B, true, true)) +
                         arma::accu(gaussian::evaluateDensityMultivariate(y, C*particlePath_, D, true, true));
    return logCompLike;
  }
  /// Runs an SMC algorithm to sample a likelihood approximation.
  double runSmc(const unsigned int N_, const arma::colvec& theta_)
  {
    N = N_;
    theta = theta_;
    storePath = false;
    storeHistory = false;
    setParameters();
    setAuxiliaryParameters();
    runSmc();
    return logLikeEst;
  }
  /// Runs an SMC algorithm to sample a likelihood approximation
  /// and also samples a single particle path.
  double runSmc(const unsigned int N_, const arma::colvec& theta_, arma::mat& particlePath_)
  {
    N = N_;
    theta = theta_;
    storePath = true;
    storeHistory = true;
    setParameters();
    setAuxiliaryParameters();
    runSmc(particlePath_);
    return logLikeEst;
  }
  /// Runs an SMC algorithm to sample a likelihood approximation
  /// and also samples a single particle path. Here, we also include
  /// an inverse temperature parameter beta which raises the conditional
  /// likelihood to some power.
  double runSmc(unsigned int N_, const arma::colvec& theta_, arma::mat& particlePath_, const double beta_)
  {
    N = N_;
    theta = theta_;
    theta(K+1) = repar(inverseRepar(theta(K+1)) / sqrt(beta_));
    theta(K+2) = repar(inverseRepar(theta(K+2)) / sqrt(beta_));
    storePath = true;
    storeHistory = true;
    setParameters();
    setAuxiliaryParameters();
    runSmc(particlePath_);
    return logLikeEst;
  }
  /// Runs a CSMC algorithm to sample a likelihood approximation
  /// and also returns a single particle path, possibly 
  /// obtained via backward or ancestor sampling
  double runCsmc(const unsigned int N_, const arma::colvec& theta_, arma::mat& particlePath_, const CsmcType csmc_)
  {
    N = N_;
    theta = theta_;
    csmc = csmc_;
    setParameters();
    setAuxiliaryParameters();
    runCsmc(particlePath_);
    return logLikeEst;
  }
  /// Runs a CSMC algorithm to sample a likelihood approximation
  /// and also returns a single particle path, possibly 
  /// obtained via backward or ancestor sampling; here, we also include
  /// an inverse temperature parameter beta which raises the conditional
  /// likelihood to some power.
  double runCsmc(const unsigned int N_, const arma::colvec& theta_, arma::mat& particlePath_, const CsmcType csmc_, const double beta_)
  {
    N = N_;
    theta = theta_;
    theta(K+1) = repar(inverseRepar(theta(K+1)) / sqrt(beta_));
    theta(K+2) = repar(inverseRepar(theta(K+2)) / sqrt(beta_));
    csmc = csmc_;
    setParameters();
    setAuxiliaryParameters();
    runCsmc(particlePath_);
    return logLikeEst;
  }
  /// Obtains a sample path from the joint smoothing distribution.
  void runGibbs(arma::mat& particlePath_)
  {
    runKalman();
    arma::mat sigmaAux(V, V);
    arma::colvec muAux(V);
    particlePath_.set_size(V,T);
    particlePath_.col(T-1) = gaussian::sampleMultivariate(1, mU.col(T-1), CU.slice(T-1), false);
    
    for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
    {
      sigmaAux = arma::inv(A.t() * arma::inv(BBT) * A + arma::inv(CU.slice(t)));
      muAux = sigmaAux * (arma::inv(CU.slice(t)) * mU.col(t) + A.t() * arma::inv(BBT) * particlePath_.col(t+1));
      particlePath_.col(t) = gaussian::sampleMultivariate(1, muAux, sigmaAux, false);
    }
  }
  /// Obtains a sample path from the joint smoothing distribution
  /// given some parameter vector.
  void runGibbs(const arma::colvec& theta_, arma::mat& particlePath_)
  {
    theta = theta_;
    setParameters();
    setAuxiliaryParameters();
    runGibbs(particlePath_); 
  }
  /// Obtains a sample path from the joint smoothing distribution
  /// given some parameter vector. Here, we also include
  /// an inverse temperature parameter beta which raises the conditional
  /// likelihood to some power.
  void runGibbs(const arma::colvec& theta_, arma::mat& particlePath_, const double beta_)
  {
    theta = theta_;
    theta(K+1) = repar(inverseRepar(theta(K+1)) / sqrt(beta_));
    theta(K+2) = repar(inverseRepar(theta(K+2)) / sqrt(beta_));
    setParameters();
    setAuxiliaryParameters();
    runGibbs(particlePath_); 
  }


private:
  
  /// Turns the elements on the diagonals of B and D into the last two elements of theta.
  double repar(const double x) 
  {
    double out = 0;
    switch (para)
    {
      case SMC_PARAMETRISATION_NATURAL:
        out = x;
        break; 
      case SMC_PARAMETRISATION_UNBOUNDED:        
        out = log(x);
        break;
    }  
    return out;
  }
  /// Turns the last two elements of theta into the elements on the the diagonals of B and D.
  double inverseRepar(const double x) 
  {
    double out = 0;
    switch (para)
    {
      case SMC_PARAMETRISATION_NATURAL:
        out = x;
        break; 
      case SMC_PARAMETRISATION_UNBOUNDED:        
        out = exp(x);
        break;
    }  
    return out;
  }
  /// Determines theta from explicit model parameters.
  void set_theta() 
  {
    theta(arma::span(0,K)) = A(arma::span(0,0),arma::span(0,K)).t();
    theta(K+1) = repar(arma::as_scalar(B(1,1)));
    theta(K+2) = repar(arma::as_scalar(D(1,1)));
  }
  /// Determines explicit model parameters from theta.
  void setParameters() 
  {
    // Creating the banded diagonal matrix A from theta
    if (K == 0)
    {
      A = arma::as_scalar(theta(0)) * arma::eye(V, V);
    }
    else // assuming that K < V
    {
      arma::colvec firstCol(V, arma::fill::zeros);
      firstCol(arma::span(0,K)) = theta(arma::span(0,K));
      A = arma::toeplitz(firstCol);
    }
    
    if (isSimple)
    {
      C = arma::diagmat(arma::eye(dy, dy));
      B = arma::diagmat(inverseRepar(arma::as_scalar(theta(K+1))) * arma::eye(V, V));
      D = arma::diagmat(inverseRepar(arma::as_scalar(theta(K+2))) * arma::eye(dy, dy));
    }
    else
    {
      C = arma::eye(dy, dy);
      B = inverseRepar(arma::as_scalar(theta(K+1))) * arma::eye(V, V);
      D = inverseRepar(arma::as_scalar(theta(K+2))) * arma::eye(dy, dy);
    }
  } 
  /// Determines auxiliary model parameters from theta, 
  /// i.e. these are often used functions of the model parameters.
  void setAuxiliaryParameters() 
  {
    
    /*
    if (isSimple)
    {
      DDT = arma::diagmat(arma::as_scalar(pow(D(0,0), 2.0)) * arma::eye<arma::mat>(V,V));
      BBT = arma::diagmat(arma::as_scalar(pow(B(0,0), 2.0)) * arma::eye<arma::mat>(V,V));
    }
    else
    {
      DDT = D*D.t();
      BBT = B*B.t();
    }
    */
    
    DDT = D*D.t();
    BBT = B*B.t();
    arma::mat invBBT = arma::inv(BBT);
     
    switch (prop) 
    {
      case SMC_PROPOSAL_PRIOR:
        break;
      case SMC_PROPOSAL_OPTIMAL:
        
        if (isSimple)
        {
          
          /// THIS IS THE NEW VERSION
          /*
          //arma::mat invBBT = arma::inv(arma::diagmat(BBT));
          
          CTinvDDT      = arma::inv(arma::diagmat(DDT));
          sigmaProp     = arma::inv(arma::diagmat(CTinvDDT + invBBT));
          cholSigmaProp = arma::trimatu(arma::chol(sigmaProp)); 
          sigmaW        = arma::diagmat(BBT + DDT);
          cholSigmaW    = arma::trimatu(arma::chol(sigmaW));
          muProp1       = arma::diagmat(sigmaProp*CTinvDDT);
          muProp2       = arma::diagmat(sigmaProp*invBBT*A);
          CA            = A;
          sigmaW0       = arma::diagmat(C0 + DDT);
          meanW0        = m0;
          */
          
         CTinvDDT      = C.t()*arma::inv(arma::diagmat(DDT));
         sigmaProp     = arma::inv(arma::diagmat(CTinvDDT*C + invBBT));
         cholSigmaProp = arma::trimatu(arma::chol(sigmaProp)); 
         sigmaW        = C*BBT*C.t() + DDT;
         cholSigmaW    = arma::trimatu(arma::chol(sigmaW));
         muProp1       = sigmaProp*CTinvDDT;
         muProp2       = sigmaProp*invBBT*A;
         CA            = C*A;
         meanW0        = C*m0;
         sigmaW0       = C*C0*C.t() + DDT;
       
        }
        else
        {
          CTinvDDT      = C.t()*arma::inv(DDT);
          sigmaProp     = arma::inv(CTinvDDT*C + invBBT);
          cholSigmaProp = arma::trimatu(arma::chol(sigmaProp)); 
          sigmaW        = C*BBT*C.t() + DDT;
          cholSigmaW    = arma::trimatu(arma::chol(sigmaW));
          muProp1       = sigmaProp*CTinvDDT;
          muProp2       = sigmaProp*invBBT*A;
          CA            = C*A;
          sigmaW0       = C*C0*C.t() + DDT;
          meanW0        = C*m0;
        }

        break;
    }
  }
  /// Performs sampling and weighting in the first SMC step.
  void initialiseSmc(
    arma::mat& particlesNew,       // Particles from current SMC step
    arma::colvec& logWeights,
    const arma::mat& particlePath_)    // Log-unnormalised weights
  { 
    
    //std::cout << "initialiseSmc()" << std::endl;
    switch (prop) 
    {
      case SMC_PROPOSAL_PRIOR:
        
        particlesNew = gaussian::sampleMultivariate(N, m0, C0, false);
        
        if (isConditional)
        {
          particlesNew.col(particleIndices(0)) = particlePath_.col(0);
        }

        if (isSimple)
        {
          logComponentWeightsFull.slice(0) = gaussian::evaluateDensityUnivariate(arma::repmat(y.col(0), 1, N), C*particlesNew, D(0,0), true, true);
          logWeights += arma::trans(arma::sum(logComponentWeightsFull.slice(0)));
        }
        else
        {
          logWeights += gaussian::evaluateDensityMultivariate(y.col(0), C*particlesNew, D, true, true, cores);
        }
          
        break;
        
      case SMC_PROPOSAL_OPTIMAL:
        
        //arma::mat sigma = arma::inv(C.t()*arma::inv(D*D.t())*C + arma::inv(C0));
        //arma::colvec mu = sigma*(C.t()*arma::inv(D*D.t())*y.col(0) + arma::inv(C0)*m0);
        arma::mat sigma = arma::inv(CTinvDDT*C + arma::inv(C0));
        arma::colvec mu = sigma*(CTinvDDT*y.col(0) + arma::inv(C0)*m0);
          
        particlesNew = gaussian::sampleMultivariate(N, mu, sigma, false);
  
        if (isConditional)
        {
          particlesNew.col(particleIndices(0)) = particlePath_.col(0);
        }
        if (isSimple)
        {
          logComponentWeightsFull.slice(0).each_col() = gaussian::evaluateDensityUnivariate(y.col(0), meanW0, sigmaW0(0,0), false, true);
          logWeights += arma::trans(arma::sum(logComponentWeightsFull.slice(0)));
        }
        else
        {
          logWeights += arma::as_scalar(gaussian::evaluateDensityMultivariate(y.col(0), meanW0, sigmaW0, false, true, cores));
        }
       
        break;
    } 
  }
  
  /// Performs sampling and weighting at SMC Step t, t > 1.
  void iterateSmc(
    const unsigned int t,  // SMC step
    arma::mat& particlesNew,       // Particles from current SMC step
    const arma::mat& particlesOld, // Particles from previous SMC step
    arma::colvec& logWeights,
    const arma::mat& particlePath_)    // Log-unnormalised weights
  {
    
    switch (prop) 
    {
      case SMC_PROPOSAL_PRIOR:
        
        particlesNew = gaussian::sampleMultivariate(N, A*particlesOld, B, true);
        
        if (isConditional)
        {
          particlesNew.col(particleIndices(t)) = particlePath_.col(t);
        }
        
        if (isSimple)
        {
          logComponentWeightsFull.slice(t) = gaussian::evaluateDensityUnivariate(repmat(y.col(t), 1, N), C*particlesNew, D(0,0), true, true);
          logWeights += arma::trans(arma::sum(logComponentWeightsFull.slice(t)));
        }
        else
        {
          logWeights += gaussian::evaluateDensityMultivariate(y.col(t), C*particlesNew, D, true, true, cores);
        }
        
        break;
        
      case SMC_PROPOSAL_OPTIMAL:
        
        // Original version:
        //arma::mat sigma = arma::inv(C.t()*arma::inv(D*D.t())*C + arma::inv(B*B.t()));
        //arma::mat mu = sigma*(arma::repmat(C.t()*arma::inv(D*D.t())*y.col(t), 1, N) + arma::inv(B*B.t())*A*particlesOld);
        
        //particlesNew = gaussian::sampleMultivariate_ms_arma(N, mu, sigma, false);
        //logWeights += gaussian::evaluateDensityMultivariate_sms_arma(y.col(t), C*A*particlesOld, C*B*B.t()*C.t() + D*D.t(), false, true, cores);
        
        // (Hopefully) faster version which avoids repeated calculations of certain parameters:
        arma::mat muProp = arma::repmat(muProp1*y.col(t), 1, N) + muProp2*particlesOld;
        particlesNew = gaussian::sampleMultivariate(N, muProp, cholSigmaProp, true);
        
        if (isConditional)
        {
          particlesNew.col(particleIndices(t)) = particlePath_.col(t);
        }
        
        if (isSimple)
        {
          logComponentWeightsFull.slice(t) = gaussian::evaluateDensityUnivariate(repmat(y.col(t), 1, N), CA*particlesOld, cholSigmaW(0,0), true, true);
          logWeights += arma::trans(arma::sum(logComponentWeightsFull.slice(t)));
        }
        else
        {
          logWeights += gaussian::evaluateDensityMultivariate(y.col(t), CA*particlesOld, cholSigmaW, true, true, cores);
        }

        break;
    }
  }
  
  /// Obtains the log of the local weights of a specific block.
  void computeLocalWeights(
    arma::colvec& logWeights,
    unsigned int t, 
    unsigned int lb, 
    unsigned int ub)
  {
    logWeights = arma::trans(arma::sum(logComponentWeightsFull.slice(t)(arma::span(lb,ub), arma::span::all)));
  }
  /// Calculates component-wise approximations of sufficient  
  /// statistics and the gradient, based on particle trajectories 
  /// obtained via standard backward sampling
  void suff_est()
  {
    arma::mat z(2*K+V,M, arma::fill::zeros); // pads the particles with zeros to avoid issues on the boundary of the state space
    double zk, zl;
    
    // Time 1:
    for (unsigned int m=0; m<M; m++)
    {
      for (unsigned int v=0; v<V; v++)
      {
        suffRCompEst(v) += xBack(v,m,0) * xBack(v,m,0);
        suffSCompEst(v) += xBack(v,m,0) * y(v,0);
      }
    }
    
    suffR1CompEst = suffRCompEst;
    
    // Time t, t > 1:
    for (unsigned int t=1; t<T; t++)
    {
      z(arma::span(K,K+V-1), arma::span::all) = xBack.slice(t-1);
      for (unsigned int m=0; m<M; m++)
      {
        for (unsigned int v=0; v<V; v++)
        {
          for (unsigned int k=0; k<K+1; k++)
          {
            if (k == 0)
            {
              zk = z(K+v,m);
            }
            else
            {
              zk = z(K+v-k,m) + z(K+v+k,m);
            }
            suffQCompEst(k,v) += xBack(v,m,t) * zk;
            
            for (unsigned int l=0; l<K+1; l++)
            {
              if (l == 0)
              {
                zl = z(K+v,m);
              }
              else
              {
                zl = z(K+v-l,m) + z(K+v+l,m);
              }
              suffHCompEst.slice(v)(k,l) += zk * zl;
            }
          }
          suffRCompEst(v) += xBack(v,m,t) * xBack(v,m,t);
          suffSCompEst(v) += xBack(v,m,t) * y(v,t);
        }
      }
    }
   
  }
  /// Calculates component-wise approximations of sufficient  
  /// statistics and the gradient, based on particle trajectories (associated with the jth block)
  /// obtained via backward sampling
  void suff_est_block(unsigned int j) 
  {
    
    //std::cout << "Block: " << j << std::endl;
    unsigned int sizeInn = uInn-lInn+1; // number of components in the inner block
    arma::mat z(sizeInn + 2*K, M, arma::fill::zeros); // the number of columns in z is the inner block + padding of size K on each side
    double zk, zl;
    
    unsigned int minZ = 0;
    unsigned int minX = 0;
    unsigned int maxZ = z.n_rows - 1;
    unsigned int maxX = xBackBlock.n_rows - 1;
    
    if (lInn < K)
    {
      minZ = K - lInn;
    }
    if (uInn + K + 1 > V)
    {
      maxZ = z.n_rows - 1 - (uInn + K + 1 - V);
    }
    if (lInn - lOutNei > K)
    {
      minX = lInn - lOutNei - K;
    }
    if (uOutNei - uInn > K)
    {
      maxX = xBackBlock.n_rows - 1 - (uOutNei - uInn - K);
    }
    // Time 1:
    //std::cout << "time-0 smoothing functional" << std::endl;
    for (unsigned int m=0; m<M; m++)
    {
      for (unsigned int v=0; v<sizeInn; v++)
      {
        suffRCompEst(v+lInn) += xBackBlock(v+(lInn-lOutNei),m,0) * xBackBlock(v+(lInn-lOutNei),m,0);
        suffSCompEst(v+lInn) += xBackBlock(v+(lInn-lOutNei),m,0) * y(v+lInn,0);
      }
    }
    
    suffR1CompEst(arma::span(lInn, uInn)) = suffRCompEst(arma::span(lInn, uInn));
    
    // Time t, t > 1:
    //std::cout << "time-t smoothing functional" << std::endl;
    for (unsigned int t=1; t<T; t++)
    {
      z(arma::span(minZ, maxZ), arma::span::all) = xBackBlock.slice(t-1).rows(arma::span(minX, maxX));
      for (unsigned int m=0; m<M; m++)
      {
        for (unsigned int v=0; v<sizeInn; v++)
        {
          for (unsigned int k=0; k<K+1; k++)
          {
            if (k == 0)
            {
              zk = z(K+v,m);
            }
            else
            {
              zk = z(K+v-k,m) + z(K+v+k,m);
            }
            suffQCompEst(k,v+lInn) += xBackBlock(v+(lInn-lOutNei),m,t) * zk;
            
            for (unsigned int l=0; l<K+1; l++)
            {
              if (l == 0)
              {
                zl = z(K+v,m);
              }
              else
              {
                zl = z(K+v-l,m) + z(K+v+l,m);
              }
              suffHCompEst.slice(v+lInn)(k,l) += zk * zl;
            }
          }        
          suffRCompEst(v+lInn) += xBackBlock(v+(lInn-lOutNei),m,t) * xBackBlock(v+(lInn-lOutNei),m,t);
          suffSCompEst(v+lInn) += xBackBlock(v+(lInn-lOutNei),m,t) * y(v+lInn, t);          
        }
      }
    }
  }
  
  /// Calculates component-wise approximations of sufficient  
  /// statistics and the gradient, based on particle trajectories (associated with the jth block)
  /// obtained via blocked backward sampling; to be used for SMC_BACKWARD_KERNEL_BLOCK_ALT_B
  /*
  void suff_est_block(unsigned int j, arma::ucube& b) 
  {
    
    //std::cout << "Block: " << j << std::endl;
    unsigned int sizeInn = uInn-lInn+1; // number of components in the inner block
    arma::colvec z(sizeInn + 2*K, arma::fill::zeros); // the number of columns in z is the inner block + padding of size K on each side
    double zk, zl;
    
    unsigned int minZ = 0;
    unsigned int minX = 0;
    unsigned int maxZ = z.n_rows - 1;
    unsigned int maxX = V - 1;
    
    if (lInn < K)
    {
      minZ = K - lInn;
    }
    if (uInn + K + 1 > V)
    {
      maxZ = z.n_rows - 1 - (uInn + K + 1 - V);
    }
    
    if (lInn > K)
    {
      minX = lInn - K;
    }
    if (uInn + K < V)
    {
      maxX = uInn + K;
    }
    
    //std::cout << "X: " << minX << " " << maxX << " " << maxX-minX << std::endl;
    //std::cout << "Z: " << minZ << " " << maxZ << " " << maxZ-minZ << std::endl;
    
    // Time 1:
    //std::cout << "time-0 smoothing functional" << std::endl;
    for (unsigned int m=0; m<M; m++)
    {
      for (unsigned int v=0; v<sizeInn; v++)
      {
        suffRCompEst(v+lInn) += particlesFull.slice(0)(v+lInn,b(j,m,0)) * particlesFull.slice(0)(v+lInn,b(j,m,0));
        suffSCompEst(v+lInn) += particlesFull.slice(0)(v+lInn,b(j,m,0)) * y(v+lInn,0);
      }
    }
    
    suffR1CompEst(arma::span(lInn, uInn)) = suffRCompEst(arma::span(lInn, uInn)); ////////////// check this!!!!
    
    // Time t, t > 1:
    //std::cout << "time-t smoothing functional" << std::endl;
    for (unsigned int t=1; t<T; t++)
    {
      //z(arma::span(minZ, maxZ), arma::span::all) = xBackBlock.slice(t-1).rows(arma::span(minX, maxX));
      
      for (unsigned int m=0; m<M; m++)
      {
        
        z(arma::span(minZ, maxZ)) = particlesFull.slice(t-1)(arma::span(minX, maxX), arma::span(b(j,m,t-1),b(j,m,t-1)));

        for (unsigned int v=0; v<sizeInn; v++)
        {
          for (unsigned int k=0; k<K+1; k++)
          {
            if (k == 0)
            {
              zk = z(K+v);
            }
            else
            {
              zk = z(K+v-k) + z(K+v+k);
            }
            suffQCompEst(k,v+lInn) += xBack(v+lInn,m,t) * zk;
            
            for (unsigned int l=0; l<K+1; l++)
            {
              if (l == 0)
              {
                zl = z(K+v);
              }
              else
              {
                zl = z(K+v-l) + z(K+v+l);
              }
              suffHCompEst.slice(v+lInn)(k,l) += zk * zl;
            }
          }        
          suffRCompEst(v+lInn) += xBack(v+lInn,m,t) * xBack(v+lInn,m,t);
          suffSCompEst(v+lInn) += xBack(v+lInn,m,t) * y(v+lInn, t);          
        }
      }
    }
  }
  */
  
  /// Calculates some auxiliary quantities associated with a particular block
  void computeBlockParameters(unsigned int j)
  {
    // Borders of the current block:
    lInn = blockInn(0,j); // index of the first component in the inner block
    uInn = blockInn(1,j); // index of the last component in the inner block
    lOut = blockOut(0,j); // index of the first component in the extended block
    uOut = blockOut(1,j); // index of the last component in the extended block
        
    // index of the first component in the neighbourhood of the extended block
    if (lOut > K)
    {
      lOutNei = lOut - K;
    }
    else 
    {
      lOutNei = 0;
    }
    uOutNei = std::min(uOut + K, V-1); // index of the last component in the neighbourhood of the extended block
    
    if (lInn > K)
    {
      lInnNei = lInn - K;
    }
    else 
    {
      lInnNei = 0;
    }
    uInnNei = std::min(uInn + K, V-1); // index of the last component in the neighbourhood of the extended block
    
    sizeOutNei = uOutNei - lOutNei + 1;
    sizeInnNei = uInnNei - lInnNei + 1;
    blockSize = uInn-lInn +1;
  }
  /// Computes the borders of the interior of blocks (only needed for the
  /// online parameter-estimation algorithms)
  void computeInteriorBlockParameters(unsigned int padding)
  {
    lInnInt = std::min(lInn+padding, uInn); // index of the first component in the interior of the inner block
    if (uInn > padding)
    {
      uInnInt = std::max(uInn-padding, lInn); // index of the last component in the interior of the inner block
    }
    else
    {
      uInnInt = lInn; // index of the last component in the interior of the inner block
    }  
    if (lInnInt > K)
    {
      lInnIntNei = lInnInt - K;
    }
    else
    {
      lInnIntNei = 0;
    }
    uInnIntNei = std::min(uInnInt+K, V-1);
    
    if (lInnInt <= uInnInt)
    {
      blockSizeInterior = uInnInt-lInnInt +1;
    }
    else
    {
      blockSizeInterior = 0;
    }
  }
  /// Performs maximisation step necessary for offline-EM algorithms
  void calculate_argmax_Q(const arma::mat& h, const arma::colvec& q, double r, double r1, double s)
  {
    theta(arma::span(0,K)) = arma::inv(h) * q;

    theta(K+1) = arma::as_scalar(log( (r - r1 - 2.0 * arma::trans(theta(arma::span(0,K))) * q +
                       arma::trans(theta(arma::span(0,K))) * h * theta(arma::span(0,K)) ) / (V*(T-1)) 
                    ) / 2.0);
    

    theta(K+2) = log( (r  - 2.0 * s + arma::accu(arma::pow(y, 2.0))) / (V*T) 
                    ) / 2.0;
  }
  /// Subsamples N particles from the blocking approximation at each time step.
  void subsampleParticlesFromBpfApproximation()
  {
    subsampledParticlesFromBpfApproximationFull.set_size(V, N, T);
    arma::colvec W;
    arma::uvec auxiliaryParticleIndices(N);
    logWeightsForSubsampledBpfApproximationFull.ones(N, T);
    logWeightsForSubsampledBpfApproximationFull.fill(-std::log(N)); 
    arma::colvec logWeights(N);
    
    for (unsigned int t=0; t<T; t++) 
    {
      for (unsigned int j=0; j<nBlocks; j++)
      {  
        computeBlockParameters(j);
        computeLocalWeights(logWeights, t, lInn, uInn);
            
        W = normaliseWeights(logWeights); // self-normalised weights for the jth block
        auxiliaryParticleIndices = sampleInt(N, W);
        
        for (unsigned int n=0; n<N; n++)
        {
          subsampledParticlesFromBpfApproximationFull.slice(t)(arma::span(lInn,uInn), arma::span(n,n)) = particlesFull.slice(t)(arma::span(lInn,uInn), arma::span(auxiliaryParticleIndices(n),auxiliaryParticleIndices(n)));
        }
      }
    }
  }
  /// Performs standard backward sampling:
  void runStandardBackwardSampler(arma::cube particlesFullAux, arma::mat logWeightsFullAux)
  {
    
    arma::colvec backwardKernels(N);
    arma::colvec logWeights(N); // log-unnormalised backward weights (randomDiscrete() needs the weights-argument to be an STL vector)
    unsigned int b;    // auxiliary particle index
    arma::mat meanAux; // auxiliary mean vectors used for standard and blocked backward sampling
    
    if (storeSuff)
    {
      suffHCompEst.zeros(K+1,K+1,V);
      suffQCompEst.zeros(K+1,V);
      suffRCompEst.zeros(V);
      suffR1CompEst.zeros(V);
      suffSCompEst.zeros(V);
      gradCompEst.zeros(K+3,V);
    }
    
    std::cout << "running standard BS" << std::endl;
      
    xBack.set_size(V, M, T); // sample paths obtained via backward sampling
    backwardKernels = normaliseWeights(logWeightsFull.col(T-1));
    
    //std::cout << "Backward sampling, Step T" << std::endl;
    for (unsigned int m=0; m<M; m++)
    {  
      b = sampleInt(backwardKernels);
      xBack.slice(T-1)(arma::span::all, arma::span(m,m)) = 
      particlesFullAux.slice(T-1)(arma::span::all, arma::span(b,b));
    }
    
    for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
    {  
      //std::cout << "Backward sampling, Step " << t << std::endl;
      meanAux = A*particlesFullAux.slice(t); // only needs to be calculated once per (backward) time step
      for (unsigned int m=0; m<M; m++)
      {  
        logWeights = logWeightsFullAux.col(t) + gaussian::evaluateDensityMultivariate(
                xBack.slice(t+1)(arma::span::all, arma::span(m,m)), 
                meanAux, B, true, true, cores);
                
        backwardKernels = normaliseWeights(logWeights);
        b = sampleInt(backwardKernels); 
        xBack.slice(t)(arma::span::all, arma::span(m,m)) = 
        particlesFullAux.slice(t)(arma::span::all, arma::span(b,b));
      }
    }
    
    if (storeSuff)
    {
      suff_est();
    }
    
  }
  /// Performs standard forward smoothing.
  void runStandardForwardSmoother(arma::cube particlesFullAux, arma::mat logWeightsFullAux)
  {
    
    arma::colvec backwardKernels(N);
    arma::colvec logWeights(N); // log-unnormalised backward weights (randomDiscrete() needs the weights-argument to be an STL vector)
    arma::mat meanAux; // auxiliary mean vectors used for standard and blocked backward sampling
    
    if (storeSuff)
    {
      suffHCompEst.zeros(K+1,K+1,V);
      suffQCompEst.zeros(K+1,V);
      suffRCompEst.zeros(V);
      suffR1CompEst.zeros(V);
      suffSCompEst.zeros(V);
      gradCompEst.zeros(K+3,V);
    }
    
    
    // Temporary quantities:
    std::vector<arma::cube>   suffHCompEstAuxOld(N);
    std::vector<arma::mat>    suffQCompEstAuxOld(N);
    std::vector<arma::colvec> suffRCompEstAuxOld(N);
    std::vector<arma::colvec> suffR1CompEstAuxOld(N);
    std::vector<arma::colvec> suffSCompEstAuxOld(N);

    std::vector<arma::cube>   suffHCompEstAuxNew(N);
    std::vector<arma::mat>    suffQCompEstAuxNew(N);
    std::vector<arma::colvec> suffRCompEstAuxNew(N);
    std::vector<arma::colvec> suffR1CompEstAuxNew(N);
    std::vector<arma::colvec> suffSCompEstAuxNew(N);

    // Time 0
    for (unsigned int n=0; n<N; n++)
    { 
      suffRCompEstAuxNew[n].zeros(V);
      suffSCompEstAuxNew[n].zeros(V);
      suffHCompEstAuxNew[n].zeros(K+1,K+1,V);
      suffQCompEstAuxNew[n].zeros(K+1,V);
      suffR1CompEstAuxNew[n].zeros(V);
  
      suffRCompEstAuxOld[n].zeros(V);
      suffSCompEstAuxOld[n].zeros(V);
      suffHCompEstAuxOld[n].zeros(K+1,K+1,V);
      suffQCompEstAuxOld[n].zeros(K+1,V);  
      suffR1CompEstAuxOld[n].zeros(V);

      for (unsigned int v=0; v<V; v++)
      {
        suffRCompEstAuxNew[n](v) += particlesFullAux(v,n,0) * particlesFullAux(v,n,0);
        suffSCompEstAuxNew[n](v) += particlesFullAux(v,n,0) * y(v,0);
      }
    }

    suffR1CompEstAuxNew = suffRCompEstAuxNew;
    
    arma::mat z(2*K+V,N, arma::fill::zeros); // this pads the particles with zeros to avoid issues on the boundary of the state space
    double zk, zl;
  
    
    for (unsigned int t=1; t<T; t++)
    {
      meanAux = A*particlesFullAux.slice(t-1); // to ensure this only needs to be calculated once per (backward) time step
      z(arma::span(K,K+V-1), arma::span::all) = particlesFullAux.slice(t-1);
      
      // Temporary quantities:
      suffHCompEstAuxOld.swap(suffHCompEstAuxNew);
      suffQCompEstAuxOld.swap(suffQCompEstAuxNew);
      suffRCompEstAuxOld.swap(suffRCompEstAuxNew);
      suffR1CompEstAuxOld.swap(suffR1CompEstAuxNew);
      suffSCompEstAuxOld.swap(suffSCompEstAuxNew);

      for (unsigned int n=0; n<N; n++)
      { 
        suffRCompEstAuxNew[n].zeros(V);
        suffSCompEstAuxNew[n].zeros(V);
        suffHCompEstAuxNew[n].zeros(K+1,K+1,V);
        suffQCompEstAuxNew[n].zeros(K+1,V);
        suffR1CompEstAuxNew[n].zeros(V);
        
        // Backward kernel evaluated at all $N$ time-$t-1$ particles given the $n$th particle at time $t$
        logWeights = logWeightsFullAux.col(t-1) + gaussian::evaluateDensityMultivariate(
                particlesFullAux.slice(t)(arma::span::all, arma::span(n,n)), 
                meanAux, B, true, true, cores);
        backwardKernels = normaliseWeights(logWeights); // self-normalised backward kernels
        
        for (unsigned int m=0; m<N; m++)
        {
          for (unsigned int v=0; v<V; v++)
          {
            for (unsigned int k=0; k<K+1; k++)
            {
              if (k == 0)
              {
                zk = z(K+v,m);
              }
              else
              {
                zk = z(K+v-k,m) + z(K+v+k,m);
              }
              
              suffQCompEstAuxNew[n](k,v) += backwardKernels(m) * (suffQCompEstAuxOld[m](k,v) + particlesFullAux(v,n,t) * zk);
  
              for (unsigned int l=0; l<K+1; l++)
              {
                if (l == 0)
                {
                  zl = z(K+v,m);
                }
                else
                {
                  zl = z(K+v-l,m) + z(K+v+l,m);
                }
                suffHCompEstAuxNew[n].slice(v)(k,l) += backwardKernels(m) * (suffHCompEstAuxOld[m].slice(v)(k,l) + zk * zl);
              }
            }
            suffR1CompEstAuxNew[n](v) += backwardKernels(m) * suffR1CompEstAuxOld[m](v);
            suffRCompEstAuxNew[n](v)  += backwardKernels(m) * (suffRCompEstAuxOld[m](v) + particlesFullAux(v,n,t) * particlesFullAux(v,n,t));
            suffSCompEstAuxNew[n](v)  += backwardKernels(m) * (suffSCompEstAuxOld[m](v) + particlesFullAux(v,n,t) * y(v,t));
            
          }
        }
      }
    }
    
    arma::colvec W = normaliseWeights(logWeightsFullAux.col(T-1));
    
    for (unsigned int n=0; n<N; n++) 
    {
      suffRCompEst  = suffRCompEst  + W(n) * suffRCompEstAuxNew[n];
      suffR1CompEst = suffR1CompEst + W(n) * suffR1CompEstAuxNew[n];
      suffSCompEst  = suffSCompEst  + W(n) * suffSCompEstAuxNew[n];
      suffHCompEst  = suffHCompEst  + W(n) * suffHCompEstAuxNew[n];
      suffQCompEst  = suffQCompEst  + W(n) * suffQCompEstAuxNew[n];
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Variables:
  /////////////////////////////////////////////////////////////////////////////

  // Random number generation:
  Rng* rng;
  
  // Parameters of the model:
  unsigned int V;            // number of components in the state vectors
  unsigned int dy;           // number of components in the observation vectors
  unsigned int T;            // number of time steps
  unsigned int K;            // K-1 is the number of non-zero off-diagonal entries in the matrix A
  arma::colvec theta;        // static parameters in vectorised form (a vector of length K+1)
  arma::mat A, B, C, D;      // explicit model parameters
  arma::mat C0;              // Step-1 covariance matrix
  arma::colvec m0;           // Step-1 mean vector
  const arma::mat& y;        // observations
  double logLikeTrue;        // exact evaluation of the log of the marginal likelihood of theta 
  
  // Parameters of the SMC algorithm:
  unsigned int N;            // number of forward particles
  double H;                  // ESS resampling threshold
  FilterType filt;           // type of SMC filter to use
  ProposalType prop;         // type of proposal kernel to use
  bool storeHistory;         // should the entire particle system be stored?
  ParametrisationType para;  // type of reparametrisation of the static parameters
  double logLikeEst;         // estimate of the normalising constant
  unsigned int cores;        // number of cores to use where parallel computation is possible
  
  // Variables for storing the entire particle system:
  arma::cube particlesFull;          // holds all the particles obtained from one run of the SMC algorithm
  arma::cube subsampledParticlesFromBpfApproximationFull; // holds all the particles obtained from one run of the SMC algorithm
  arma::cube logComponentWeightsFull; // holds all "local" unnormalised log-weights
  arma::umat parentIndicesFull;          // holds all the parent indices from one run of the SMC algorithm
  arma::mat logWeightsFull;        // holds all the log-unnormalised weights from one run of the SMC algorithm
  arma::mat logWeightsForSubsampledBpfApproximationFull;  // holds all the log-unnormalised weights from one run of the SMC algorithm
  
  // Auxiliary variables for better performance:
  arma::mat sigmaProp, cholSigmaProp, sigmaW, sigmaW0, cholSigmaW, CTinvDDT, muProp1, muProp2, CA, BBT, DDT; 
  arma::colvec meanW0;
  bool isSimple; // TRUE, if B and D are scaled identity matrices and C is the identity matrix
  
  // Backward-sympling type algorithms:
  BackwardKernelType back; // the type of backward kernels to use (i.e. standard or blocked)
  unsigned int M;            // number of "particles" used for backward sampling
  arma::cube xBack;          // holds all the particles obtained from backward sampling
  arma::cube xBackBlock;     // holds all the particles obtained from backward sampling, associated with some part of the space
  
  // Blocking approximations:
  arma::mat blockInn;        // matrix with 2 rows giving the borders of the partition of the space
  arma::mat blockOut;        // matrix with 2 rows giving the borders of (potentially overlapping) spatial blocks
  unsigned int nBlocks;      // number of spatial blocks
  unsigned int sizeInnNei, sizeOutNei; // number of components in some (extended) block
  unsigned int blockSize, blockSizeInterior; // number of components in the inner block / in the interior of the inner block
  unsigned int lInn, uInn, lInnInt, uInnInt, lOut, uOut, lInnNei, uInnNei, lInnIntNei, uInnIntNei, lOutNei, uOutNei; // borders of some inner, some outer and neighbourhood of some outer block
  
  // Gradient-ascent algorithms:
  arma::colvec stepSizes;    // static parameters in vectorised form (a vector of length K+1)
  arma::colvec thetaInit;    // initial value for the static parameters
  arma::mat thetaFull;       // initial value for the static parameters

  // Blocked smoothing distributions
  //bool storeMarginals;
  //arma::cube marginals;   // (evenly-weighted) particle approximations of all marginals of the smoothing distribution
  
  // Mean vectors and covariance matrices (smoothed, predicted, updated)
  arma::mat mS, mP, mU;
  arma::cube CS, CP, CU;
  
  // Sufficient statistics for EM algorithms (stored for each component) and
  // gradient approximations
  // In each case, the last dimension represents the components of the state space.
  bool storeSuff;
  
  // Spatial component-wise estimates and true values of sufficient statistics and gradient
  arma::cube suffHCompEst, suffHCompTrue;
  arma::mat suffQCompEst, suffQCompTrue;
  arma::colvec suffRCompEst, suffR1CompEst, suffSCompEst, suffRCompTrue, suffR1CompTrue, suffSCompTrue;
  arma::mat gradCompEst, gradCompTrue;

  // Estimates and true values of sufficient statistics and gradient
  arma::mat suffHEst, suffHTrue;
  arma::colvec suffQEst, suffQTrue;
  double suffREst, suffR1Est, suffSEst, suffRTrue, suffR1True, suffSTrue;
  arma::colvec gradEst, gradTrue;
  
  // Parameters for Bayesian inference:
  arma::colvec hyperParameters; // Parameters of prior distributions: hyperParameters = [a1, b1, a2, b2], where b^2 ~ IG(a1,b1) and d^2 ~ IG(a2,b2)
  arma::colvec rwmhSd; // vector of standard deviation for the Gaussian random-walk MH proposal kernel for theta
  
  // Parameters for obtaining particle paths and for CSMC algorithms:
  bool storePath; // should the SMC algorithm also generate a single particle path (and thus store all parent indices?)
  bool isConditional; // use Conditional SMC (rather than "standard" SMC)?
  CsmcType csmc; // type of backward-simulation method to use with the CSMC algorithm
  arma::mat particlePath; // a single particle path
  arma::uvec particleIndices; // particle indices associated with the single particle path
  
};

#endif
