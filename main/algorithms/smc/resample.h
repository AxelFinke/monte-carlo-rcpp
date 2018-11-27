/// \file
/// \brief Some helper functions for resampling and dealing particle weights.
///
/// This file contains the functions for performing resampling in SMC algorithms
/// as well as some auxiliary functions for normalising the weights in log-space.

#ifndef __RESAMPLE_H
#define __RESAMPLE_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <random>
#include <iostream>
#include <vector>
#include "main/helperFunctions/helperFunctions.h"

namespace resample
{
  
  ////////////////////////////////////////////////////////////////////////////////
  // Standard multinomial resampling
  ////////////////////////////////////////////////////////////////////////////////
  
  /// \brief Performs multinomial resampling given a particular
  /// uniform random number.
  void multinomialBase
  (
    arma::uvec& parentIndices,   // stores the post-resampling particle labels
    const arma::colvec& w,       // self-normalised particle weights
    unsigned int N               // total number of offspring
  )        
  {
    parentIndices = sampleInt(N, w);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Conditional multinomial resampling
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Performs conditional systematic resampling given a particular
  /// uniform random number.
  void conditionalMultinomialBase
  (
    arma::uvec& parentIndices, // stores the post-resampling particle labels
    unsigned int& b,       // particle index of the distinguished particle
    const arma::colvec& w, // particle weights
    const unsigned int N,  // total number of offspring
    const unsigned int a   // parent index
  )
  { 
    parentIndices = sampleInt(N, w);
    b = 0; // NOTE: for simplicity, we set the index of the reference input path to 0 when using multinomial resampling!
    parentIndices(b) = a;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Standard systematic resampling
  ////////////////////////////////////////////////////////////////////////////////
  
  /// \brief Performs systematic resampling given a particular
  /// uniform random number.
  void systematicBase
  (
    const double u,              // unform random variable supplied by the user
    arma::uvec& parentIndices, // stores the post-resampling particle labels
    const arma::colvec& w,       // self-normalised particle weights
    unsigned int N               // total number of offspring
  )        
  {
    arma::colvec T = (arma::linspace(0, N-1, N) + u) / N;
    arma::colvec Q = arma::cumsum(w);

    unsigned int i = 0;
    unsigned int j = 0;
      
    while (j < N) 
    {
      if (T(j) <= Q(i)) 
      {
        parentIndices(j) = i;
        ++j;
      } 
      else 
      {
        ++i;
      }
    }
  }
  /// \brief Performs systematic resampling.
  void systematic
  (
    arma::uvec& parentIndices, // stores the post-resampling particle labels
    const arma::colvec& w,       // self-normalised particle weights
    const unsigned int N               // total number of offspring
  )
  {
    systematicBase(arma::randu(), parentIndices, w, N);
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // Conditional systematic resampling
  ////////////////////////////////////////////////////////////////////////////////
  
  /// \brief Performs conditional systematic resampling given a particular
  /// uniform random number.
  void conditionalSystematicBase
  (
    double u,              // uniform random variable
    arma::uvec& parentIndices, // stores the post-resampling particle labels
    unsigned int& b,       // particle index of the distinguished particle
    const arma::colvec& w, // particle weights
    const unsigned int N,  // total number of offspring
    const unsigned int a   // parent index
  )
  { 
    arma::colvec Q = arma::cumsum(N*w);
    
    // determines the highest posstible stratum for each cumulated particle weight
    arma::uvec bins = arma::conv_to<arma::uvec>::from(arma::ceil(Q) - 1);
    bins.elem(arma::find(bins == N)).fill(N-1);
    arma::colvec wAux; // stores the probabilities of sampling b from a particular stratum
      
    // First step: obtain the index of the distinguished particle at 
    // the current step given the entire history of the particle system
    // and in particular, given that the parent of this particle has index a.
    
    if (a == 0 || bins(a) == bins(a-1)) 
    {
      b = bins(a);
    }
    else
    {
      wAux.zeros(N); 
      if (bins(a) > bins(a-1) + 1)
      {
        // Assigning equal weights to all possible strata associated with b:
        wAux(arma::span(bins(a-1)+1, bins(a)-1)).fill(1);
      }
        
      // Dealing with the last possibly stratum:
      wAux(bins(a)) = Q(a) - bins(a); 
      
      // Dealing with the first possible stratum:
      wAux(bins(a-1)) = bins(a-1) - Q(a-1) + 1;
      
      // Self-normalising the weights:
      wAux = arma::normalise(wAux, 1);

      // Determining the index of the distinguished particle:
      b = sampleInt(wAux);
    }
    
    // Second step: obtain a single uniform random variable compatible
    // with the indices a and b.
    
    double lb = 0; // lower bound for u
    double ub = 1; // upper bound for u
    if (a > 0 && b == bins(a-1)) // i.e. if b falls into the first possible stratum
    {
      lb = Q(a-1) - bins(a-1);
    } 
    if (b == bins(a)) // i.e. if b falls into the last possible stratum
    {
      ub = Q(a) - bins(a);
    } 
    
    u = lb + (ub - lb)*u;
    
    // Third step: perform standard systematic resampling given u.
    arma::colvec T = (arma::linspace(0, N-1, N) + u);

    unsigned int i = 0;
    unsigned int j = 0;  
      
    while (j <= b) 
    {
      if (T(j) <= Q(i)) 
      {
        parentIndices(j) = i;
        ++j;
      } 
      else 
      {
        ++i;
      }
    }
    
    if (parentIndices(b) != a) 
    {
      std::cout << "Warning: conditional systematic resampling did not set the parent index of the conditioning path correctly!" << std::endl;
      std::cout << "Most likely cause: weight of the conditioning path is numerically zero!" << std::endl;
      //std::cout << "lb, ub, u, b, a, W(a), parentIndices(b): " << lb << ", " << ub << ", " << u << ", " << b << ", " << a << ", " << w(a) << ", " << parentIndices(b) << std::endl;          
      parentIndices(b) = a;
    }
    
    i = a;
    j = b+1;  
      
    while (j < N) 
    {
      /////////////////////////////////////////////////////////////////////////
      // This is only used to catch errors resulting from w(a) = 0 (which shouldn't normally happen)
      if (i == N) 
      {
        std::cout << "Warning: i = N!" << std::endl;
        std::cout << "########## Skipped resampling step! ##########" << std::endl;
        b = a;
        parentIndices = arma::linspace<arma::uvec>(0, N-1, N);
        break;
      }
      /////////////////////////////////////////////////////////////////////////
      
      if (T(j) <= Q(i)) 
      {
        parentIndices(j) = i;
        ++j;
      } 
      else 
      {
        ++i;
      }
    }
    
  }
  /// \brief Performs conditional systematic resampling.
  void conditionalSystematic
  ( 
    arma::uvec& parentIndices, // stores the post-resampling particle labels
    unsigned int& b,       // particle index of the distinguished particle
    const arma::colvec& w, // particle weights
    const unsigned int N,  // total number of offspring
    const unsigned int a   // parent index
  )
  { 
    conditionalSystematicBase(arma::randu(), parentIndices, b, w, N, a);
  }
}
#endif
