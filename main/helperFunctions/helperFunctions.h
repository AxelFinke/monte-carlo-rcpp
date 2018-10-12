/// \file
/// \brief Some auxiliary functions.
///
/// This file contains a number of auxiliary functions for 
/// use in other C++ programmes.

#ifndef __HELPERFUNCTIONS_H
#define __HELPERFUNCTIONS_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <iostream>
#include <string>
#include <vector>
#include <time.h> 
#include <omp.h>
#include "rng/Rng.h"
#include "rng/gaussian.h"


/// Samples a single value from a multinomial distribution (with size $1$)
/// (i.e. outputs a value in {0, ..., length(W)-1}).
unsigned int sampleInt(const arma::colvec& W) 
{ 
  unsigned int x = arma::conv_to<unsigned int>::from(arma::find(arma::cumsum(W) >= arma::randu(), 1, "first"));
  return x;
}
/// Samples a single value from a multinomial distribution (with size $1$)
/// (i.e. outputs a value in {0, ..., length(W)-1}).
unsigned int sampleInt(const arma::rowvec& W) 
{ 
  unsigned int x = arma::conv_to<unsigned int>::from(arma::find(arma::cumsum(W) >= arma::randu(), 1, "first"));
  return x;
}
/// Samples multiple values from a multinomial distribution (with size $1$)
/// (i.e. outputs N values in {0, ..., length(W)-1})
arma::uvec sampleInt(unsigned int N, const arma::colvec& W) 
{ 
  arma::colvec cumW = arma::cumsum(W);
  arma::uvec x(N);
  for (unsigned int n=0; n<N; n++)
  {
    x(n) =  arma::conv_to<unsigned int>::from(arma::find(cumW >= arma::randu(), 1, "first"));
  }
  return x;
}
/// Normalise a single distribution in log-space (returns 
/// normalised weights in log space).
arma::vec normaliseExp(const arma::vec& logW) 
{ 
  double logWMax = arma::max(logW);
  double logZ = logWMax + log(arma::sum(arma::exp(logW - logWMax)));
  return(logW - logZ);
}
/// Normalises a single distribution in log-space (returns 
/// normalised weights and normalising constant in log space).
arma::vec normaliseExp(const arma::vec& logW, double& logZ) 
{ 
  double logWMax = arma::max(logW);
  logZ = logWMax + log(arma::sum(arma::exp(logW - logWMax)));
  return(logW - logZ);
}
/// Normalises a single distribution in log-space (returns 
/// normalised weights in normal space).
arma::colvec normaliseWeights(const arma::colvec& logW) 
{ 
  double logWMax = arma::max(logW);
  double logZ = logWMax + log(arma::sum(arma::exp(logW - logWMax)));
  // return the 1-unit norm (to make sure the elements of the vector sum to 1)
  return(arma::normalise(arma::exp(logW - logZ), 1)); 
}
arma::colvec normaliseWeights(const arma::colvec& logW, double& logZ) 
{ 
  double logWMax = arma::max(logW);
  logZ = arma::as_scalar(logWMax + log(arma::sum(arma::exp(logW - logWMax))));
  // return the 1-unit norm (to make sure the elements of the vector sum to 1)
  return(arma::normalise(arma::exp(logW - logZ), 1)); 
}
/// Normalises a single distribution in log-space and overwrites the vector of weights
void normaliseWeightsInplace(arma::colvec& logW) 
{ 
  double logWMax = arma::max(logW);
  double logZ = logWMax + log(arma::sum(arma::exp(logW - logWMax)));
  // return the 1-unit norm (to make sure the elements of the vector sum to 1)
  logW = arma::normalise(arma::exp(logW - logZ), 1); 
}
////////////////////////////////////////////////////////////////////////////////
// Converts between std::vector<arma::colvec> and arma::mat
////////////////////////////////////////////////////////////////////////////////
/// Converts a std::vector<arma::colvec> of length T 
/// (in which each element is an N-dimensional arma::colvec)
/// to an (N,T)-dimensional arma::mat.
void convertStdVecToArmaMat(const std::vector<arma::colvec>& x, arma::mat& y)
{
  unsigned int N = x[0].n_rows;
  unsigned int T = x.size();
  
  y.set_size(N, T);
  for (unsigned int t=0; t<T; t++)
  {
    y.col(t) = x[t];
  }
}
/// Converts a std::vector<arma::colvec> of length T 
/// (in which each element is an N-dimensional arma::colvec)
/// to an (N,T)-dimensional arma::mat.
void convertStdVecToArmaMat(const std::vector<arma::uvec>& x, arma::umat& y)
{
  unsigned int N = x[0].n_rows;
  unsigned int T = x.size();
  
  y.set_size(N, T);
  for (unsigned int t=0; t<T; t++)
  {
    y.col(t) = x[t];
  }
}
/// Converts an (N,T)-dimensional arma::mat to 
/// a std::vector<arma::colvec> of length T.
void convertArmaMatToStdVec(const arma::mat& y, std::vector<arma::colvec>& x)
{
  unsigned int T = y.n_cols;
  x.resize(T);
  for (unsigned int t=0; t<T; t++)
  {
    x[t] = y.col(t);
  }
}
/// Converts an (N,T)-dimensional arma::umat to 
/// a std::vector<arma::uvec> of length T.
void convertArmaMatToStdVec(const arma::umat& y, std::vector<arma::uvec>& x)
{
  unsigned int T = y.n_cols;
  x.resize(T);
  for (unsigned int t=0; t<T; t++)
  {
    x[t] = y.col(t);
  }
}
/// Converts an arma::colvec to a matrix
arma::mat convertArmaVecToArmaMat(const arma::colvec& v, const unsigned int nRows, const unsigned int nCols) {
  arma::mat M;
  M.insert_cols(0, v);
  M.reshape(nRows, nCols);
  return(M);
}
/// Returns the log of the unnormalised(!) density of a multinomial distribution.
double logUnnormalisedMultinomialDensity(const arma::uvec& x, const arma::colvec& p)
{
  
  // TODO: need to figure out why simply writing
  // arma::accu(x % arma::log(p) % (p>0))
  // doesn't work! This would be much more efficient than the current loop:
  
  double sumAux = 0.0;
  for (unsigned int i=0; i<p.size(); i++)
  {
    if (p(i)>0.0) {
      sumAux += x(i)*std::log(p(i));
    }
  }
  return sumAux;
  //   return arma::accu(x % arma::log(p) % (p>0));
}
/// Returns the log of the normalised density of a multinomial distribution.
double logMultinomialDensity(const arma::uvec& x, const unsigned int n, const arma::colvec& p)
{
  // TODO: need to figure out why simply writing
  // arma::accu(x % arma::log(p) % (p>0))
  // doesn't work! This would be much more efficient than the current loop:
  
  double sumAux = 0.0;
  for (unsigned int i=0; i<p.size(); i++)
  {
    if (p(i)>0.0) {
      sumAux += x(i)*std::log(p(i));
    }
  }
  return sumAux + std::lgamma(n+1) - arma::accu(arma::lgamma(x+1)); 
  //   return arma::accu(x % arma::log(p) % (p>0)) + std::lgamma(n+1) - arma::accu(arma::lgamma(x+1));
}
/// Returns the log of the normalised density of a multinomial distribution.
double logMultinomialDensity(const arma::urowvec& x, const unsigned int n, const arma::rowvec& p)
{
//   std::cout << "=======================================================" << std::endl;
//   arma::rowvec logP = arma::log(p);
  
  // TODO: need to figure out why simply writing
  // arma::accu(x % arma::log(p) % (p>0))
  // doesn't work! This would be much more efficient than the current loop:
  
  double sumAux = 0.0;
  for (unsigned int i=0; i<p.size(); i++)
  {
    if (p(i)>0.0) {
      sumAux += x(i)*std::log(p(i));
    }
  }

  return sumAux + std::lgamma(n+1) - arma::accu(arma::lgamma(x+1)); 
}

/// Inverse of the logistic transform:
arma::colvec inverseLogit(const arma::colvec& x)
{
  return 1.0 / (1.0 + arma::exp((-1) * x));
}
/// Inverse of the logistic transform:
double inverseLogit(const double x)
{
  return 1.0 / (1.0 + std::exp((-1) * x));
}
/// Returns the log of the density of the inverse-gamma distribution with
/// shape/scale parametrisation.
double dInverseGamma(const double x, const double shape, const double scale)
{
  return - shape * std::log(scale) - std::lgamma(shape) - (shape + 1.0) * std::log(x) - 1.0 / (x * scale);
}
/// Returns the log of the density of the gamma distribution with
/// shape/scale parametrisation.
double dGamma(const double x, const double shape, const double scale)
{
  return - shape * std::log(scale) - std::lgamma(shape) + (shape - 1.0) * std::log(x) - x / scale;
}
/// Base function for sorting the elements of x into bins defined by edges as
/// [0, edges(0)], (edges(0), edges(1)], ... (edges(n-1), edges(n)],
/// where edges is a vector of length n. NOTE: all elements of x must be
/// between edges(0) and edges(n)! The function computes a vector binIndices,
/// where binIndices(i) contains the bin-number of the $i$th element of $x$. 
/// If x is empty the function returns an empty vector.
void computeBinIndicesBase(arma::uvec& binIndices, const arma::colvec& x, const arma::colvec& edges)
{
  binIndices.set_size(x.size());
  unsigned int i = 0;
  unsigned int j = 0;
    
  while (j < x.size()) 
  {
    if (x(j) <= edges(i)) 
    {
      binIndices(j) = i;
      ++j;
    } 
    else 
    {
      ++i;
    }
  }
}
/// Function for sorting the elements of x into bins defined by edges as
/// [0, edges(0)], (edges(0), edges(1)], ... (edges(n-1), edges(n)],
/// where edges is a vector of length n. NOTE: all elements of x must be
/// between edges(0) and edges(n)! The function computes a vector binIndices,
/// where binIndices(i) contains the bin-number of the $i$th element of $x$. 
/// If x is empty the function returns an empty vector.
arma::uvec computeBinIndicesBase(const arma::colvec& x, const arma::colvec& edges, const bool isIncreasing)
{
  arma::uvec binIndices;
  if (!x.is_empty())
  {
    if (isIncreasing)
    {
      computeBinIndicesBase(binIndices, x, edges);
    }
    else 
    {
      computeBinIndicesBase(binIndices, arma::sort(x), edges);
    }
  }
  else
  {
    binIndices.reset();
  }
  return binIndices;
}
/// Base function for sorting the elements of x into bins defined by edges as
/// [0, edges(0)], (edges(0), edges(1)], ... (edges(n-1), edges(n)],
/// where edges is a vector of length n. NOTE: all elements of x must be
/// between edges(0) and edges(n)! The function computes a vector of vector binContents,
/// where the vector binContents[i] contains the indices of the elements of
/// x which fall into the corresponding bin.
/// If x is empty the function returns an empty vector.
void computeBinContentsBase(std::vector<std::vector<unsigned int>>& binContents, const arma::colvec& x, const arma::colvec& edges)
{
  binContents.resize(edges.size());
  binContents[0].clear();
  unsigned int i = 0;
  unsigned int j = 0;
    
  while (j < x.size()) 
  {
    if (x(j) <= edges(i)) 
    {
      binContents[i].push_back(j);
      ++j;
    } 
    else 
    {
      ++i;
      binContents[i].clear();
    }
  }
}
/// Function for sorting the elements of x into bins defined by edges as
/// [0, edges(0)], (edges(0), edges(1)], ... (edges(n-1), edges(n)],
/// where edges is a vector of length n. NOTE: all elements of x must be
/// between edges(0) and edges(n)! The function computes a vector binIndices,
/// where binIndices(i) contains the bin-number of the $i$th element of $x$. 
/// If x is empty the function returns an empty vector.
std::vector<std::vector<unsigned int>> computeBinContents(const arma::colvec& x, const arma::colvec& edges, const bool isIncreasing)
{
  std::vector<std::vector<unsigned int>> binContents(edges.size());
  if (isIncreasing)
  {
    computeBinContentsBase(binContents, x, edges);
  }
  else 
  {
    computeBinContentsBase(binContents, arma::sort(x), edges);
  }
  return binContents;
}
#endif
