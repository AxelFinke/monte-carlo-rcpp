/// \file
/// \brief Some helper functions for dealing with a high-dimensional SSM
///
/// This file contains the functions for dealing with a potentially 
/// high-dimensional linear Gaussian state-space model .

#ifndef __HIGHDIM_H
#define __HIGHDIM_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <omp.h>
#include <vector>
#include "main/rng/Rng.h"
#include "main/rng/gaussian.h"
#include "main/helperFunctions/helperFunctions.h"
#include "main/helperFunctions/kalman.h"

namespace highDim
{
  /// Type of parametrisation of the covariance-matrix parameters.
  enum ParametrisationType 
  { 
    HIGHDIM_PARAMETRISATION_UNBOUNDED = 0, // parametrisation in terms of components of A, and log(B(0,0)), log(D(0,0))
    HIGHDIM_PARAMETRISATION_NATURAL // parametrisation in terms of components of A, B(0,0), D(0,0) 
  };
  /// Constructs the model parameters from the vector theta.
  void computeModelParameters(
    arma::mat& A,
    arma::mat& B,
    arma::mat& C,
    arma::mat& D,
    const unsigned int K,
    const unsigned int dimX,
    const unsigned int dimY,
    const arma::colvec& theta,
    const ParametrisationType param
  )
  {
    
//         std::cout << "Started setting parameters!" << std::endl;
        
    if (K == 0)
    {
      A = arma::as_scalar(theta(0)) * arma::eye(dimX, dimX);
    }
    else
    {
      arma::colvec firstCol(dimX, arma::fill::zeros);
      firstCol(arma::span(0,K)) = theta(arma::span(0,K));
      A = arma::toeplitz(firstCol);
    }
    if (dimX == dimY)
    {
      C = arma::eye(dimY, dimX);
    }
    else if (dimY == 1)
    {
      C = 1.0 / (static_cast<double>(dimX)) * arma::ones<arma::mat>(1,dimX);
      
//       C.ones(1, dimX);
//       C = C * arma::as_scalar(1.0 / (static_cast<double>(dimX)));
    }
    else
    {
      std::cout << "ERROR: dimY must either be 1 or equal to dimX!" << std::endl;
    }

    switch (param)
    {
      case HIGHDIM_PARAMETRISATION_UNBOUNDED:
        B = exp(theta(K + 1)) * arma::eye(dimX, dimX);
        D = exp(theta(K + 2)) * arma::eye(dimY, dimY);
        break;
      case HIGHDIM_PARAMETRISATION_NATURAL:
        B = theta(K + 1) * arma::eye(dimX, dimX);
        D = theta(K + 2) * arma::eye(dimY, dimY);
        break;
    }
    
//     std::cout << "Finished setting parameters!" << std::endl;
  }
  /// Calculates the spatial-componentwise sufficient statistics.
  void computeSuffComp(
    arma::cube& suffS1Comp,
    arma::mat& suffS2Comp,
    arma::colvec& suffS3Comp,
    arma::colvec& suffS3FirstComp,
    arma::colvec& suffS4Comp,
    const arma::mat& mP,
    const arma::cube& CP,
    const arma::mat& mU,
    const arma::cube& CU,
    const arma::mat& mS,
    const arma::cube& CS,
    const arma::mat& A, 
    const arma::mat& B, 
    const arma::mat& C, 
    const arma::mat& D, 
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int K = suffS2Comp.n_rows - 1; // number of non-zero off-diagonals of A on each side of the main diagonal
    unsigned int V = y.n_rows; // dimension of the states
    unsigned int T = y.n_cols; // number of time steps
    
    arma::mat CSAux(2*K + V, 2*K + V, arma::fill::zeros);
    arma::mat sigAux(2*V, 2*V);
    arma::mat sigAuxExt(2*(V + K), 2*(V + K), arma::fill::zeros);
    arma::colvec mExt(2*K + V, arma::fill::zeros);
      
    arma::mat HAux(K+1, K+1);
    arma::colvec QAux(K+1);
    double RAux, SAux;
      
    arma::mat Gk(V, V);
    arma::mat P2(V, V);

    for (unsigned int t=0; t<T; t++) 
    { 
        
      if (t > 0)
      {
        // Padding the covariance matrix of the marginal time-(t-1) smoothing distribution with zeros.
        CSAux(arma::span(K, V+K-1), arma::span(K, V+K-1)) = CS.slice(t-1);
            
        Gk = CU.slice(t-1) * A.t() * arma::inv(CP.slice(t)); 
        P2 = CU.slice(t-1) - Gk * CP.slice(t-1) * Gk.t(); 
          
        // The covariance matrix of the vector (x_t, x_{t-1}):
        sigAux(arma::span(0, V-1), arma::span(0, V-1))   = CS.slice(t); 
        sigAux(arma::span(V,2*V-1), arma::span(V,2*V-1)) = Gk.t() * CS.slice(t) * Gk.t() + P2; 
        sigAux(arma::span(V,2*V-1), arma::span(0, V-1))  = CS.slice(t) * Gk.t();
        sigAux(arma::span(0, V-1), arma::span(V,2*V-1))  = Gk.t() * CS.slice(t);
        
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
          suffS3FirstComp(v) = RAux;
        }
              
        // Storing the spatial component-wise sufficient statistics
        suffS1Comp.slice(v) += HAux;
        suffS2Comp.col(v)   += QAux;
        suffS3Comp(v)       += RAux;
        suffS4Comp(v)       += SAux;
        
      }
    }
  }
  /// Computes the score vector from the smoothed sufficient statistics.
  void computeScoreCompFromSuffComp(
    arma::mat& scoreComp,
    const arma::colvec& theta,
    const ParametrisationType param,
    const arma::cube& suffS1Comp,
    const arma::mat& suffS2Comp, 
    const arma::colvec& suffS3Comp, 
    const arma::colvec& suffS3FirstComp, 
    const arma::colvec& suffS4Comp,
    const arma::mat y
  )
  {
    unsigned int K = suffS2Comp.size() - 1; // number of non-zero off-diagonals of A on each side of the main diagonal
    unsigned int V = y.n_rows; // dimension of the states
    unsigned int T = y.n_cols; // number of time steps
    scoreComp.zeros(K+3, V);
    
    // Calculating the spatial component-wise score
    switch (param)
    {
      case HIGHDIM_PARAMETRISATION_UNBOUNDED:
        
        for (unsigned int v=0; v<V; v++)
        {
          if (T > 1)
          {
            for (unsigned int k=0; k<K+1; k++)
            {
              scoreComp(k,v) = arma::as_scalar(exp(-2.0*theta(K+1)) * 
                                  (suffS2Comp(k,v) - arma::trans(theta(arma::span(0,K))) * suffS1Comp.slice(v).col(k)));
            }
            scoreComp(K+1,v) = arma::as_scalar(exp(-2.0*theta(K+1)) * 
                                  (suffS3Comp(v) - suffS3FirstComp(v) - 2.0 * arma::trans(theta(arma::span(0,K))) * suffS2Comp.col(v) +  
                                  arma::trans(theta(arma::span(0,K))) * suffS1Comp.slice(v) * theta(arma::span(0,K))) - (T-1));
          }
          scoreComp(K+2,v) = exp(-2.0*theta(K+2)) * 
                                (suffS3Comp(v) - 2.0 * suffS4Comp(v) + arma::accu(arma::pow(y.row(v), 2.0))) - T;
        }
        break;
        
      case HIGHDIM_PARAMETRISATION_NATURAL:
        
        for (unsigned int v=0; v<V; v++)
        {
          if (T > 1)
          {
            for (unsigned int k=0; k<K+1; k++)
            {
              scoreComp(k,v) = arma::as_scalar(exp(-2.0*log(theta(K+1))) * 
                                  (suffS2Comp(k,v) - arma::trans(theta(arma::span(0,K))) * suffS1Comp.slice(v).col(k)));
            }
            scoreComp(K+1,v) = arma::as_scalar(exp(-2.0*log(theta(K+1))) * 
                                  (suffS3Comp(v) - suffS3FirstComp(v) - 2.0 * arma::trans(theta(arma::span(0,K))) * suffS2Comp.col(v) +  
                                  arma::trans(theta(arma::span(0,K))) * suffS1Comp.slice(v) * theta(arma::span(0,K))) - (T-1)) / theta(K+1);
          }
          scoreComp(K+2,v) = arma::as_scalar(exp(-2.0*log(theta(K+2))) * 
                                (suffS3Comp(v) - 2.0 * suffS4Comp(v) + arma::accu(arma::pow(y.row(v), 2.0))) - T) / theta(K+2);
        }
        break;
    }
  }
  /// Sums the spatial sufficient-statistic components
  void sumSuffComp(
    arma::mat& suffS1,
    arma::colvec& suffS2,
    double& suffS3,
    double& suffS3First,
    double& suffS4,
    const arma::cube& suffS1Comp, 
    const arma::mat& suffS2Comp, 
    const arma::colvec& suffS3Comp, 
    const arma::colvec& suffS3FirstComp,
    const arma::colvec& suffS4Comp
  )
  {
    suffS1.zeros();
    for (unsigned int v=0; v<suffS3Comp.size(); v++)
    {
      suffS1 += suffS1Comp.slice(v);
    }
    suffS2      = arma::sum(suffS2Comp, 1);
    suffS3      = arma::sum(suffS3Comp);
    suffS3First = arma::sum(suffS3FirstComp);
    suffS4      = arma::sum(suffS4Comp);
  }
  /// Computes the score vector from the smoothed sufficient statistics.
  void computeScoreFromSuff(
    arma::colvec& score,
    const arma::colvec& theta,
    const ParametrisationType param,
    const arma::mat& suffS1,
    const arma::colvec& suffS2,
    const double suffS3,
    const double suffS3First,
    const double suffS4,
    const arma::mat& y
  )
  {
    unsigned int K = suffS2.size() - 1; // number of non-zero off-diagonals of A on each side of the main diagonal
    unsigned int V = y.n_rows; // dimension of the states
    unsigned int T = y.n_cols; // number of time steps
    
    // Calculating the score
    switch (param)
    {
      case HIGHDIM_PARAMETRISATION_UNBOUNDED:

        score(arma::span(0,K)) = exp(-2.0 * theta(K+1)) * (suffS2 - suffS1 * theta(arma::span(0,K)));

        score(K+1) = arma::as_scalar(exp(-2.0 * theta(K+1)) * 
          (suffS3 - suffS3First - 2.0 * arma::trans(theta(arma::span(0,K))) * suffS2 + arma::trans(theta(arma::span(0,K))) * suffS1 * theta(arma::span(0,K))) - V*(T-1));
        
        score(K+2) = arma::as_scalar(exp(-2.0 * theta(K+2)) * (suffS3 - 2.0 * suffS4 + arma::accu(arma::pow(y, 2.0))) - V*T);
        break;
        
      case HIGHDIM_PARAMETRISATION_NATURAL:
        
        score(arma::span(0,K)) = exp(-2.0 * log(theta(K+1))) * (suffS2 - suffS1 * theta(arma::span(0,K)));

        score(K+1) = arma::as_scalar(exp(-2.0 * log(theta(K+1))) * 
          (suffS3 - suffS3First - 2.0 * arma::trans(theta(arma::span(0,K))) * suffS2 + arma::trans(theta(arma::span(0,K))) * suffS1 * theta(arma::span(0,K))) - V*(T-1)) / theta(K+1);
        
        score(K+2) = arma::as_scalar(exp(-2.0 * log(theta(K+2))) * (suffS3 - 2.0 * suffS4 + arma::accu(arma::pow(y, 2.0))) - V*T) / theta(K+1);
        break;
    }
  }
  /// Sums the spatial score components.
  void sumScoreComp(arma::colvec& score, const arma::mat& scoreComp)
  {
    score = arma::sum(scoreComp, 1);
  }
  /// Computes the score in a particular linear Gaussian state-space model.
  void computeScore(
    arma::colvec& score,
    const arma::colvec& theta,
    const ParametrisationType& param,
    const arma::mat& m0, 
    const arma::mat& C0, 
    const arma::mat& y
  )
  {
    unsigned int K = theta.size() - 3; // number of non-zero off-diagonals on each side of the main diagonal
    unsigned int V = y.n_rows; // dimension of the states
        
    // Model parameters:
    arma::mat A(V,V);
    arma::mat B(V,V);
    arma::mat C(V,V);
    arma::mat D(V,V);
    
    // Kalman-filter and smoother quantities:
    arma::mat mP, mU, mS;
    arma::cube CP, CU, CS;
    
    computeModelParameters(A, B, C, D, V, V, K, theta, param); 
    kalman::runSmoother(mP, CP, mU, CU, mS, CS, A, B, C, D, m0, C0, y);

    // Component-wise sufficient statistics:
    arma::cube suffS1Comp(K+1,K+1, V, arma::fill::zeros);
    arma::mat suffS2Comp(K+1, V, arma::fill::zeros);
    arma::colvec suffS3Comp(V, arma::fill::zeros);
    arma::colvec suffS3FirstComp(V, arma::fill::zeros);
    arma::colvec suffS4Comp(V, arma::fill::zeros);
    arma::mat scoreComp(K+3, V, arma::fill::zeros);
    
    arma::mat suffS1(K+1, K+1);
    arma::colvec suffS2(K+1);
    double suffS3;
    double suffS3First;
    double suffS4;
    
    computeSuffComp(suffS1Comp, suffS2Comp, suffS3Comp, suffS3FirstComp, suffS4Comp, 
                    mP, CP, mU, CU, mS, CS, A, B, C, D, m0, C0, y);
    
    sumSuffComp(suffS1, suffS2, suffS3, suffS3First, suffS4, 
                suffS1Comp, suffS2Comp, suffS3Comp, suffS3FirstComp, suffS4Comp);
        
    computeScoreFromSuff(score, theta, param, suffS1, suffS2, suffS3, suffS3First, suffS4, y);
    
    /*
    computeScoreComp(
      scoreComp, theta, param, suffS1Comp, suffS2Comp, suffS3Comp, suffS3FirstComp, suffS4Comp, y
    );
    
    // Summing over the spatial components of the score.
    sumScoreComp(score, scoreComp);
    */
  }
  /// Checking whether the matrix A is such that 
  /// the latent process is stationary.
  bool checkStationarity(
    const unsigned int dimX, 
    const arma::colvec& theta, 
    const unsigned int nOffDiagonals
  )
  {
    arma::mat A;
    if (nOffDiagonals == 0)
    {
      A = arma::as_scalar(theta(0)) * arma::eye(dimX, dimX);
    }
    else
    {
      arma::colvec firstCol(dimX, arma::fill::zeros);
      firstCol(arma::span(0,nOffDiagonals)) = theta(arma::span(0,nOffDiagonals));
      A = arma::toeplitz(firstCol);
    }
    return(arma::all(arma::abs(arma::eig_sym(A)) < 1.0)); 
  }
  
} // end of namespace highDim
#endif
