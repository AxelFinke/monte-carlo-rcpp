/// \file
/// \brief Some helper functions for Kalman filtering and smoothing.
///
/// This file contains the functions for calculating the log-marginal
/// likelihood and (marginal) smoothing distributions in linear
/// Gaussian state-space models based around the Kalman filter.

#ifndef __KALMAN_H
#define __KALMAN_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <omp.h>
#include <random>
#include <vector>
#include "main/rng/Rng.h"
#include "main/rng/gaussian.h"
#include "main/helperFunctions/helperFunctions.h"

namespace kalman
{
  /// Returns the log-complete likelihood
  /// (overload for univariate case).
  double evaluateLogCompleteLikelihood(
    const arma::colvec& X,
    const double A, 
    const double B, 
    const double C, 
    const double D, 
    const double m0, 
    const double C0, 
    const arma::colvec& y
  )
  {
    unsigned int T = y.size();
    
    double logLikelihood = gaussian::evaluateDensityUnivariate(X(0), m0, C0, false, true) +
      arma::accu(gaussian::evaluateDensityUnivariate(y, C*X, D, true, true)) +
      arma::accu(gaussian::evaluateDensityUnivariate(X(arma::span(1,T-1)), A*X(arma::span(0,T-2)), B, true, true));
      
    return logLikelihood;
    
  }
  /// Returns the log-complete likelihood
  /// (overload for multivariate case).
  double evaluateLogCompleteLikelihood(
    const arma::mat& X,
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int T = y.n_cols;
    
    double logLikelihood = 
      arma::as_scalar(gaussian::evaluateDensityMultivariate(X.col(0), m0, C0, false, true)) + 
      arma::accu(gaussian::evaluateDensityMultivariate(X.cols(arma::span(1,T-1)), A*X.cols(arma::span(0,T-2)), B, true, true)) +
      arma::accu(gaussian::evaluateDensityMultivariate(y, C*X, D, true, true));
      
    return logLikelihood;
    
  }
  /// Runs a Kalman filter and returns the log-marginal likelihood
  /// (overload for univariate case).
  double evaluateLogMarginalLikelihood(
    const double A, 
    const double B, 
    const double C, 
    const double D, 
    const double m0, 
    const double C0, 
    const arma::colvec& y
  )
  {
    unsigned int T = y.size();
    
    // Predictive mean and covariance matrix
    double mP, CP;
    
    // Updated mean and covariance matrix
    double mU, CU;
    
    // Mean and covariance matrix of the incremental likelihood
    double mY, CY;
    
    // Log-marginal likelihood
    double logLikelihood = 0;
    
    // Auxiliary quantities
    double Q = B * B;
    double R = D * D;
    double kg;
    
    for (unsigned int t=0; t<T; t++)
    {
      // Prediction step
      if (t > 0) 
      {
        mP = A * mU;
        CP = A * CU * A + Q; 
      } 
      else 
      {
        mP = m0;
        CP = C0; 
      }
      
      // Likelihood step
      mY = C * mP;
      CY = C * CP * C + R;

      // Update step
      kg = C * CP / CY;
      mU = mP + kg * (y(t) - mY);
      CU = CP - kg * C * CP;

      // Adding the incremental log-marginal likelihood
      logLikelihood += gaussian::evaluateDensityUnivariate(y(t), mY, CY, false, true);
    }
    return logLikelihood;
  }
  /// Runs a Kalman filter and returns the log-marginal likelihood
  /// (overload for multivariate case).
  double evaluateLogMarginalLikelihood(
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int dimX = A.n_rows;
    unsigned int dimY = y.n_rows;
    unsigned int T    = y.n_cols;
    
    // Predictive mean and covariance matrix
    arma::colvec mP(dimX);
    arma::mat    CP(dimX, dimX);
    
    // Updated mean and covariance matrix
    arma::colvec mU(dimX);
    arma::mat    CU(dimX, dimX);
    
    // Mean and covariance matrix of the incremental likelihood
    arma::colvec mY(dimY);
    arma::mat    CY(dimY, dimY);
    
    // Log-marginal likelihood
    double logLikelihood = 0;
    
    // Auxiliary quantities
    arma::mat Q = B * B.t();
    arma::mat R = D * D.t();
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
      logLikelihood += arma::as_scalar(gaussian::evaluateDensityMultivariate(y.col(t), mY, CY, false, true));
    }
    return logLikelihood;
  }
  /// Runs a Kalman filter
  /// (overload for univariate case).
  void runFilter(
    arma::colvec& mU,
    arma::colvec& CU,
    const double A, 
    const double B, 
    const double C, 
    const double D, 
    const double m0, 
    const double C0, 
    const arma::colvec& y
  )
  {
    unsigned int T = y.size(); // number of time steps
    
    // Predictive mean and covariance matrix
    double mP, CP;
    
    // Updated mean and covariance matrix
    mU.set_size(T);
    CU.set_size(T);
    
    // Mean and covariance matrix of the incremental likelihood
    double mY, CY;
    
    // Auxiliary quantities
    double kg;
    double Q = B * B;
    double R = D * D;

    for (unsigned int t=0; t<T; t++) {
      
      // Prediction step
      if (t > 0) {
        mP = A * mU(t-1);
        CP = A * CU(t-1) * A + Q; 
      } 
      else 
      {
        mP = m0;
        CP = C0; 
      }
      
      // Likelihood step
      mY = C * mP;
      CY = C * CP * C + R;

      // Update step
      kg = C * CP / CY;
      mU(t) = mP + kg * (y(t) - mY);
      CU(t) = CP - kg * C * CP;
    } 
  }
  /// Runs a Kalman filter
  /// (overload for multivariate case).
  void runFilter(
    arma::mat& mU,
    arma::cube& CU,
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int T    = y.n_cols; // number of time steps
    unsigned int dimX = A.n_rows;
    unsigned int dimY = y.n_rows;
    
    // Predictive mean and covariance matrix
    arma::colvec mP(dimX);
    arma::mat CP(dimX, dimX);
    
    // Updated mean and covariance matrix
    mU.set_size(dimX, T);
    CU.set_size(dimX, dimX, T);
    
    // Mean and covariance matrix of the incremental likelihood
    arma::colvec mY(dimY);
    arma::mat CY(dimY, dimY);
    
    // Auxiliary quantities
    arma::mat kg;
    arma::mat Q = B * B.t();
    arma::mat R = D * D.t();

    for (unsigned int t=0; t<T; t++) {
      
      // Prediction step
      if (t > 0) {
        mP = A * mU.col(t-1);
        CP = A * CU.slice(t-1) * A.t() + Q; 
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
      mU.col(t)   = mP + kg * (y.col(t) - mY);
      CU.slice(t) = CP - kg * C * CP;
    } 
  }
  /// Runs a Kalman filter
  /// (overload for univariate case).
  void runFilter(
    arma::colvec& mP,
    arma::colvec& CP,
    arma::colvec& mU,
    arma::colvec& CU,
    const double A, 
    const double B, 
    const double C, 
    const double D, 
    const double m0, 
    const double C0, 
    const arma::colvec& y
  )
  {
    unsigned int T = y.size(); // number of time steps
    
    // Predictive means and variances.
    mP.set_size(T);
    CP.set_size(T);
    
    // Updated mean and variances
    mU.set_size(T);
    CU.set_size(T);
    
    // Mean and variances of the incremental likelihood
    double mY, CY;
    
    // Auxiliary quantities
    double kg;
    double Q = B * B;
    double R = D * D;

    for (unsigned int t=0; t<T; t++) {
      
      // Prediction step
      if (t > 0) {
        mP(t) = A * mU(t-1);
        CP(t) = A * CU(t-1) * A + Q; 
      } 
      else 
      {
        mP(0) = m0;
        CP(0) = C0; 
      }
      
      // Likelihood step
      mY = C * mP(t);
      CY = C * CP(t) * C + R;

      // Update step
      kg = C * CP(t) / CY;
      mU(t) = mP(t) + kg * (y(t) - mY);
      CU(t) = CP(t) - kg * C * CP(t);
    } 
  }
  /// Runs a Kalman filter
  /// (overload for multivariate case).
  void runFilter(
    arma::mat& mP,
    arma::cube& CP,
    arma::mat& mU,
    arma::cube& CU,
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int T    = y.n_cols; // number of time steps
    unsigned int dimX = A.n_rows;
    unsigned int dimY = y.n_rows;
    
    // Predictive mean and covariance matrix
    mP.set_size(dimX, T);
    CP.set_size(dimX, dimX, T);
    
    // Updated mean and covariance matrix
    mU.set_size(dimX, T);
    CU.set_size(dimX, dimX, T);
    
    // Mean and covariance matrix of the incremental likelihood
    arma::colvec mY(dimY);
    arma::mat CY(dimY, dimY);
    
    // Auxiliary quantities
    arma::mat kg;
    arma::mat Q = B * B.t();
    arma::mat R = D * D.t();

    for (unsigned int t=0; t<T; t++) {
      
      // Prediction step
      if (t > 0) {
        mP.col(t) = A * mU.col(t-1);
        CP.slice(t) = A * CU.slice(t-1) * A.t() + Q; 
      } 
      else 
      {
        mP.col(0) = m0;
        CP.slice(0) = C0; 
      }
      
      // Likelihood step
      mY = C * mP.col(t);
      CY = C * CP.slice(t) * C.t() + R;

      // Update step
      kg = (arma::solve(CY.t(), C * arma::trans(CP.slice(t)))).t();
      mU.col(t)   = mP.col(t)   + kg * (y.col(t) - mY);
      CU.slice(t) = CP.slice(t) - kg * C * CP.slice(t);
    } 
  }
  /// Performs forward filtering--backward smoothing in a 
  /// linear Gaussian state-space model
  /// (overload for univariate case).
  void runSmoother(
    arma::colvec& mS,
    arma::colvec& CS,
    const arma::colvec& mP,
    const arma::colvec& CP,
    const arma::colvec& mU,
    const arma::colvec& CU,
    const double A, 
    const double B, 
    const double C, 
    const double D, 
    const double m0, 
    const double C0, 
    const arma::colvec& y
  )
  {
    unsigned int T = y.size(); // number of time steps
    
    // Smoothed means and variances
    mS.set_size(T);
    CS.set_size(T);

    double Ck;
    mS(T-1) = mU(T-1);
    CS(T-1) = CU(T-1);
    
    if (T > 1) 
    {
      for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
      {
        Ck    = CU(t) * A / CP(t+1);
        mS(t) = mU(t) + Ck * (mS(t+1) - mP(t+1));
        CS(t) = CU(t) + Ck * (CS(t+1) - CP(t+1)) * Ck;
      }
    }
  }
  /// Performs forward filtering--backward smoothing in a 
  /// linear Gaussian state-space model
  /// (overload for multivariate case).
  void runSmoother(
    arma::mat& mS,
    arma::cube& CS,
    const arma::mat& mP,
    const arma::cube& CP,
    const arma::mat& mU,
    const arma::cube& CU,
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int T    = y.n_cols; // number of time steps
    unsigned int dimX = A.n_rows;

    // Smoothed mean and covariance matrix
    mS.set_size(dimX, T);
    CS.set_size(dimX, dimX, T);

    arma::mat Ck(dimX, dimX);
    mS.col(T-1)   = mU.col(T-1);
    CS.slice(T-1) = CU.slice(T-1);
    
    for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
    {
      Ck          = CU.slice(t) * A.t() * arma::inv(CP.slice(t+1));
      mS.col(t)   = mU.col(t)   + Ck * (mS.col(t+1)   - mP.col(t+1));
      CS.slice(t) = CU.slice(t) + Ck * (CS.slice(t+1) - CP.slice(t+1)) * Ck.t();
    }
  }
  /// Performs forward filtering--backward smoothing in a 
  /// linear Gaussian state-space model
  /// (overload for univariate case).
  void runSampler(
    arma::colvec& X,
    const arma::colvec& mU,
    const arma::colvec& CU,
    const double A, 
    const double B
  )
  {
    unsigned int T = mU.size(); // number of time steps
    double Q = B * B;
    double sigmaAux, muAux;
    
    X.set_size(T);
    X(T-1) = mU(T-1) + std::sqrt(CU(T-1)) * arma::randn();
    
    for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
    {
      sigmaAux = 1.0 / (A / Q * A + 1.0 / CU(t));
      muAux    = sigmaAux * (1.0 / CU(t) * mU(t) + A / Q * X(t+1));
      X(t)     = muAux + std::sqrt(sigmaAux) * arma::randn();
    }
  }
  /// Performs forward filtering--backward smoothing in a 
  /// linear Gaussian state-space model
  /// (overload for multivariate case).
  void runSampler(
    arma::mat& X,
    const arma::mat& mU,
    const arma::cube& CU,
    const arma::mat& A, 
    const arma::mat& B
  )
  {
    unsigned int T    = mU.n_cols; // number of time steps
    unsigned int dimX = A.n_rows;

    arma::mat Q = B * B.t();
    arma::mat sigmaAux(dimX, dimX);
    arma::colvec muAux(dimX);
    
    X.set_size(dimX,T);
    X.col(T-1) = gaussian::sampleMultivariate(1, mU.col(T-1), CU.slice(T-1), false);
    
    for (unsigned int t=T-2; t != static_cast<unsigned>(-1); t--)
    {
      sigmaAux = arma::inv(A.t() * arma::inv(Q) * A + arma::inv(CU.slice(t)));
      muAux    = sigmaAux * (arma::inv(CU.slice(t)) * mU.col(t) + A.t() * arma::inv(Q) * X.col(t+1));
      X.col(t) = gaussian::sampleMultivariate(1, muAux, sigmaAux, false);
    }
  }
  /// Performs forward filtering--backward smoothing in a 
  /// linear Gaussian state-space model
  /// (overload for univariate case).
  void runForwardFilteringBackwardSmoothing(
    arma::colvec& mP,
    arma::colvec& CP,
    arma::colvec& mU,
    arma::colvec& CU,
    arma::colvec& mS,
    arma::colvec& CS,
    const double A, 
    const double B, 
    const double C, 
    const double D, 
    const double m0, 
    const double C0, 
    const arma::colvec& y
  )
  { 
    runFilter(mP, CP, mU, CU, A, B, C, D, m0, C0, y);
    runSmoother(mS, CS, mP, CP, mU, CU, A, B, C, D, m0, C0, y);  
  }
  /// Performs forward filtering--backward smoothing in a 
  /// linear Gaussian state-space model
  /// (overload for multivariate case).
  void runForwardFilteringBackwardSmoothing(
    arma::mat& mP,
    arma::cube& CP,
    arma::mat& mU,
    arma::cube& CU,
    arma::mat& mS,
    arma::cube& CS,
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  { 
    runFilter(mP, CP, mU, CU, A, B, C, D, m0, C0, y);
    runSmoother(mS, CS, mP, CP, mU, CU, A, B, C, D, m0, C0, y);  
  }
  /// Performs forward filtering--backward sampling in a 
  /// linear Gaussian state-space model 
  /// (overload for univariate case).
  void runForwardFilteringBackwardSampling(
    arma::colvec& X,
    const double A, 
    const double B, 
    const double C, 
    const double D, 
    const double m0, 
    const double C0, 
    const arma::colvec& y
  )
  {
    unsigned int T    = y.size(); // number of time steps

    arma::colvec mU(T);
    arma::colvec CU(T);

    runFilter(mU, CU, A, B, C, D, m0, C0, y);
    runSampler(X, mU, CU, A, B);  
  }
  /// Performs forward filtering--backward sampling in a 
  /// linear Gaussian state-space model 
  /// (overload for multivariate case).
  void runForwardFilteringBackwardSampling(
    arma::mat& X,
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int T    = y.n_cols; // number of time steps
    unsigned int dimX = A.n_rows;

    arma::mat  mU(dimX, T);
    arma::cube CU(dimX, dimX, T);

    runFilter(mU, CU, A, B, C, D, m0, C0, y);
    runSampler(X, mU, CU, A, B);  
  }
  
} // end of namespace kalman
#endif
