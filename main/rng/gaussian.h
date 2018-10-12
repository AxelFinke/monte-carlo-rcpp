/// \file
/// \brief Some helper functions for dealing with Gaussian distributions.
///
/// This file contains the functions for evaluating the density of and sampling
/// from a multivariate Gaussian distribution. Not for export to R.

#ifndef __GAUSSIAN_H
#define __GAUSSIAN_H

#include <RcppArmadillo.h>
#include <omp.h>
#include <random>
#include <vector>

const double log2pi = std::log(2.0 * M_PI);


namespace gaussian
{
  ////////////////////////////////////////////////////////////////////////////////
  // Density of a univariate normal distribution
  ////////////////////////////////////////////////////////////////////////////////

  // Multiple evaluations/multiple mean vectors/multiple variances
  arma::mat evaluateDensityUnivariate(
    const arma::mat& x,  
    const arma::mat& mean,  
    const arma::mat& sigma,
    bool is_sd = false,
    bool logd = false) 
  { 
    arma::mat out(x.n_rows, x.n_cols);
    
    if (is_sd == false)
    {
      out = - 0.5 * (log2pi + arma::log(sigma) + arma::pow(x - mean, 2)/sigma);
    } 
    else
    {
      out = - arma::log(sigma) - 0.5 * (log2pi + arma::pow(x - mean, 2)/arma::pow(sigma,2));
    }
  
    if (logd == false) 
    {
      out = arma::exp(out);
    }
      
    return out;
  }

  // Multiple evaluations/single mean vector/multiple variances
  arma::mat evaluateDensityUnivariate(
    const arma::mat& x,  
    const double mean,  
    const arma::mat& sigma,
    bool is_sd = false,
    bool logd = false) 
  { 
    arma::mat out(x.n_rows, x.n_cols);
    
    if (is_sd == false)
    {
      out = - 0.5 * (log2pi + arma::log(sigma) + arma::pow(x - mean, 2)/sigma);
    } 
    else
    {
      out = - arma::log(sigma) - 0.5 * (log2pi + arma::pow(x - mean, 2)/arma::pow(sigma,2));
    }
  
    if (logd == false) 
    {
      out = arma::exp(out);
    }
      
    return out;
  }

  // Multiple evaluations/multiple mean vectors/single variance
  arma::mat evaluateDensityUnivariate(
    const arma::mat& x,  
    const arma::mat& mean,  
    const double sigma,
    bool is_sd = false,
    bool logd = false) 
  { 
    arma::mat out(x.n_rows, x.n_cols);
    
    if (is_sd == false)
    {
      out = - 0.5 * (log2pi + log(sigma) + arma::pow(x - mean, 2)/sigma);
    } 
    else
    {
      out = - log(sigma) - 0.5 * (log2pi + arma::pow(x - mean, 2)/pow(sigma,2));
    }
  
    if (logd == false) 
    {
      out = arma::exp(out);
    }
      
    return out;
  }

  // Multiple evaluations/single mean vector/single variance
  arma::mat evaluateDensityUnivariate(
    const arma::mat& x,  
    const double mean,  
    const double sigma,
    bool is_sd = false,
    bool logd = false) 
  { 
    arma::mat out(x.n_rows, x.n_cols);
    
    if (is_sd == false)
    {
      out = - 0.5 * (log2pi + log(sigma) + arma::pow(x - mean, 2)/sigma);
    } 
    else
    {
      out = - log(sigma) - 0.5 * (log2pi + arma::pow(x - mean, 2)/pow(sigma,2));
    }
  
    if (logd == false) 
    {
      out = arma::exp(out);
    }
      
    return out;
  }

  // Single evaluation/single mean vector/single variance
  double evaluateDensityUnivariate(
    const double x,  
    const double mean,  
    const double sigma,
    bool is_sd = false,
    bool logd = false) 
  { 
    double out;
    
    if (is_sd == false)
    {
      out = - 0.5 * (log2pi + log(sigma) + pow(x - mean, 2)/sigma);
    } 
    else
    {
      out = - log(sigma) - 0.5 * (log2pi + pow(x - mean, 2)/pow(sigma,2));
    }
  
    if (logd == false) 
    {
      out = exp(out);
    }
      
    return out;
  }




  ////////////////////////////////////////////////////////////////////////////////
  // Density of a multivariate normal distribution
  ////////////////////////////////////////////////////////////////////////////////
  
  // Multiple evaluations/multiple mean vectors/general covariance matrix
  arma::colvec evaluateDensityMultivariate(
    const arma::mat& x,  
    const arma::mat& mean,  
    const arma::mat& sigma,
    bool is_chol = false,
    bool logd = false,
    unsigned int cores = 1) 
  { 
    //omp_set_num_threads(cores);
    
    unsigned int nx = x.n_cols;
    unsigned int nm = mean.n_cols;
    unsigned int dx = x.n_rows;
    unsigned int n  = std::max(nx,nm);
    
    arma::colvec out(n);
    arma::mat rooti; // Inverse root of the covariance matrix
      
    if (is_chol == false) 
    {
      rooti = arma::trans(arma::inv(arma::trimatu(arma::chol(sigma))));
    } 
    else 
    {
      rooti = arma::trans(arma::inv(arma::trimatu(sigma)));
    }
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(static_cast<double>(dx)/2.0) * log2pi;
    
    //# pragma omp parallel for schedule(static)
    for (unsigned int i=0; i<n; i++) 
    {
      arma::vec z = rooti * (x.col(std::min(i,nx-1)) - mean.col(std::min(i,nm-1)));    
      out(i)      = constants - 0.5 * arma::sum(z%z) + rootisum;     
    }  
        
    if (logd == false) 
    {
      out = arma::exp(out);
    }
      
    return out;
  }

  // Multiple evaluations/multiple mean vectors/covariance matrix is scaled identity matrix
  arma::colvec evaluateDensityMultivariate(
    const arma::mat& x,  
    const arma::mat& mean,  
    const double sigma,
    bool is_chol = false,
    bool logd = false,
    unsigned int cores = 1) 
  { 
    //omp_set_num_threads(cores);
    
    unsigned int nx = x.n_cols;
    unsigned int nm = mean.n_cols;
    unsigned int dx = x.n_rows;
    unsigned int n  = std::max(nx,nm);
    
    arma::colvec out(n);
    double rooti; // Inverse root of the covariance matrix
      
    if (is_chol == false)
    {
      rooti = 1.0 / sqrt(sigma);
    } 
    else 
    {
      rooti = 1.0 / sigma;
    }
    
    double rootisum = static_cast<double>(dx)*log(rooti);
    double constants = -(static_cast<double>(dx)/2.0) * log2pi;
    
    //# pragma omp parallel for schedule(static)
    for (unsigned int i=0; i<n; i++)
    {
      arma::vec z = rooti * (x.col(std::min(i,nx-1)) - mean.col(std::min(i,nm-1)));      
      out(i)      = constants - 0.5 * arma::sum(z%z) + rootisum;     
    }  
        
    if (logd == false) {
      out = arma::exp(out);
    }
      
    return(out);
  }


  ////////////////////////////////////////////////////////////////////////////////
  // Generating multivariate normal random numbers
  ////////////////////////////////////////////////////////////////////////////////

  // Normal random numbers: general covariance matrix
  arma::mat sampleMultivariate(
    unsigned int n, 
    const arma::mat& mean, 
    const arma::mat& sigma, 
    bool is_chol = false)
  {
    unsigned int nm = mean.n_cols;
    unsigned int dx = mean.n_rows;
    arma::mat x = arma::randn(dx, n);
    
    if (is_chol == false)
    {
      return(arma::repmat(mean, 1, std::max(n-nm+1,static_cast<unsigned int>(1))) + arma::chol(sigma) * x);
    }
    else 
    {
      return(arma::repmat(mean, 1, std::max(n-nm+1,static_cast<unsigned int>(1))) + sigma * x);
    }
  }

  // Normal random numbers: covariance matrix is a scaled identity matrix
  arma::mat sampleMultivariate(
    unsigned int n, 
    const arma::mat& mean, 
    double sigma, 
    bool is_chol = false)
  {
    unsigned int nm = mean.n_cols;
    unsigned int dx = mean.n_rows;
    arma::mat x = arma::randn(dx, n);
    
    if (is_chol == false)
    {
      return(arma::repmat(mean, 1, std::max(n-nm+1,static_cast<unsigned int>(1))) + sqrt(sigma) * x);
    }
    else 
    {
      return(arma::repmat(mean, 1, std::max(n-nm+1,static_cast<unsigned int>(1))) + sigma * x);
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  // Generating (univariate) truncated normal random variables
  ////////////////////////////////////////////////////////////////////////////////

  // Normal random numbers: general covariance matrix
  double rtnorm(
    const double lower,
    const double upper,
    const double mean,
    const double sigma,
    bool is_sd = false)
  {
    double x;
    
    if (is_sd == true)
    {
      x = R::qnorm(arma::as_scalar(R::pnorm(lower, mean, sigma, true, false) + arma::randu(1) * 
            (R::pnorm(upper, mean, sigma, true, false) - R::pnorm(lower, mean, sigma, true, false))),
            mean, sigma, true, false);
    }
    else 
    {
      x = R::qnorm(arma::as_scalar(R::pnorm(lower, mean, sqrt(sigma), true, false) + arma::randu(1) * 
            (R::pnorm(upper, mean, sqrt(sigma), true, false) - R::pnorm(lower, mean, sqrt(sigma), true, false))),
            mean, sqrt(sigma), true, false);
    }
    
    return(x);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Generating (univariate) truncated normal random variables
  ////////////////////////////////////////////////////////////////////////////////

  // Normal random numbers: general covariance matrix
  double dtnorm(
    const double x,
    const double lower,
    const double upper,
    const double mean,
    const double sigma,
    bool is_sd = false,
    bool logd = false)
  {
    double logDensity;
    if (is_sd == true)
    {
      logDensity = R::dnorm(x, mean, sigma, 1) - 
                  log(R::pnorm(upper, mean, sigma, 1, 0) - R::pnorm(lower, mean, sigma, 1, 0));
    }
    else 
    {
      logDensity = R::dnorm(x, mean, sqrt(sigma), 1) - 
                  log(R::pnorm(upper, mean, sqrt(sigma), 1, 0) - R::pnorm(lower, mean, sqrt(sigma), 1, 0));
    }
    
    if (logd == false)
    {
      logDensity = exp(logDensity);
    }
    
    return(logDensity);
  }
}
#endif
