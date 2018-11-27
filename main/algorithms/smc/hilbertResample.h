/// \file
/// \brief Some helper functions for resampling and dealing particle weights.
///
/// This file contains the functions for performing resampling in SMC algorithms
/// as well as some auxiliary functions for normalising the weights in log-space.

#ifndef __HILBERTRESAMPLE_H
#define __HILBERTRESAMPLE_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <random>
#include <iostream>
#include <vector>
#include "main/helperFunctions/helperFunctions.h"
#include "main/algorithms/smc/resample.h"

// The following are needed for the Hilbert sort:
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/hilbert_sort.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/spatial_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>
#include <boost/iterator/counting_iterator.hpp>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Homogeneous_d.h>
#include <stdio.h>


namespace resample
{
  ////////////////////////////////////////////////////////////////////////////////
  // Systematic resampling based on particles sorted using the Hilbert curve
  ////////////////////////////////////////////////////////////////////////////////
  
  /// \brief Sorts particle indices using the Hilbert curve, assuming that
  /// each particle has type arma::colvec.
  /// (adapted from Mathieu Gerber's SQMC libary:
  /// https://bitbucket.org/mgerber/sqmc).
  void hilbertSort(const std::vector<arma::colvec> particles, arma::uvec& I, const double lb, const double ub)
  {
    
    typedef CGAL::Cartesian_d<double> Kernel;
    typedef Kernel::Point_d Point_d;
    typedef CGAL::Spatial_sort_traits_adapter_d<Kernel,Point_d*> Search_traits_d;
    
    unsigned int N = particles.size(); // number of particles
    unsigned int d = particles[0].size(); // dimension
    
    // Logistic transform
    std::vector<arma::colvec> uniforms(N);
    for (unsigned int n=0; n<N; n++)
    {
      uniforms[n].set_size(d);
      for (unsigned int i=0; i<d; i++)
      {
        uniforms[n](i) = 1.0/(1 + std::exp(- (particles[n](i) - lb)/(ub - lb)));
      }
    }
  
    std::vector<Point_d> points; 
    for (unsigned int n=0; n<N; n++)
    {
      points.push_back(Point_d(uniforms[0].size(), uniforms[n].begin(), uniforms[n].end()));
    }
    
    std::vector<std::ptrdiff_t> indices;
    indices.reserve(points.size());
    std::copy(boost::counting_iterator<std::ptrdiff_t>(0),
              boost::counting_iterator<std::ptrdiff_t>(points.size()),
              std::back_inserter(indices));

    CGAL::hilbert_sort(indices.begin(), indices.end(), Search_traits_d(&(points[0])));
    
    I.set_size(N); // TODO: this is probably not needed
     
    for (unsigned int n=0; n<N; n++)
    {
      I(n) = indices[n];
    }
  }
  /// \brief Sorts particle indices using the Hilbert curve, assuming that
  /// each particle has type arma::uvec.
  /// (adapted from Mathieu Gerber's SQMC libary:
  /// https://bitbucket.org/mgerber/sqmc).
  void hilbertSort(const std::vector<arma::uvec> particles, arma::uvec& I, const double lb, const double ub)
  {
    
    typedef CGAL::Cartesian_d<double> Kernel;
    typedef Kernel::Point_d Point_d;
    typedef CGAL::Spatial_sort_traits_adapter_d<Kernel,Point_d*> Search_traits_d;
    
    unsigned int N = particles.size(); // number of particles
    unsigned int d = particles[0].size(); // dimension
    
    // Logistic transform
    std::vector<arma::colvec> uniforms(N);
    for (unsigned int n=0; n<N; n++)
    {
      uniforms[n].set_size(d);
      for (unsigned int i=0; i<d; i++)
      {
        uniforms[n](i) = 1.0/(1 + std::exp(- (particles[n](i) - lb)/(ub - lb)));
      }
    }
  
    std::vector<Point_d> points; 
    for (unsigned int n=0; n<N; n++)
    {
      points.push_back(Point_d(uniforms[0].size(), uniforms[n].begin(), uniforms[n].end()));
    }
    
    std::vector<std::ptrdiff_t> indices;
    indices.reserve(points.size());
    std::copy(boost::counting_iterator<std::ptrdiff_t>(0),
              boost::counting_iterator<std::ptrdiff_t>(points.size()),
              std::back_inserter(indices));

    CGAL::hilbert_sort(indices.begin(), indices.end(), Search_traits_d(&(points[0])));
    
    I.set_size(N); // TODO: this is probably not needed
     
    for (unsigned int n=0; n<N; n++)
    {
      I(n) = indices[n];
    }
  }
  /// \brief Overloads hilbertSort() in the case that each particle has
  /// type double.
  void hilbertSort(const std::vector<double>& particles, arma::uvec& I, const double lb, const double ub)
  {
    unsigned int N = particles.size(); // number of particles
    
    // Logistic transform
    std::vector<double> uniforms(N);
    for (unsigned int n=0; n<N; n++)
    {
      uniforms[n] = 1.0/(1 + std::exp(- (particles[n] - lb)/(ub - lb)));
    }

    std::vector<unsigned int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    auto comparator = [&uniforms](unsigned int a, unsigned int b) 
    {
      return uniforms[a] < uniforms[b];
    };
    std::sort(indices.begin(), indices.end(), comparator);
    
    I.set_size(N); // TODO: this is probably not needed
    for (unsigned int n=0; n<N; n++)
    {
      I(n) = indices[n];
    }
  }
  /// \brief Performs systematic resampling (given a particular uniform random number)
  /// after sorting the transformed particles using the Hilbert curve.
  /// \param L type of each particle: arma::colvec or double.
  template <class L> void hilbertBase
  (
    const double u,                 // unform random variable supplied by the user
    arma::uvec& parentIndices,      // stores the post-resampling particle labels
    const std::vector<L> particles, // particles
    const arma::colvec& w,          // self-normalised particle weights
    const unsigned int N,                  // total number of offspring
    const double lb, const double ub
  )        
  {
    arma::uvec sortedIndices(w.size()); // sorted particle indices
    hilbertSort(particles, sortedIndices, lb, ub);
    
    arma::colvec T = (arma::linspace(0, N-1, N) + u) / N;
    arma::colvec Q = arma::cumsum(w.rows(sortedIndices));

    unsigned int i = 0;
    unsigned int j = 0;
      
    while (j < N) 
    {
      if (T(j) <= Q(i)) 
      {
        parentIndices(j) = sortedIndices(i);
        ++j;
      } 
      else 
      {
        ++i;
      }
    }
  }
  /// \brief Performs systematic resampling after sorting the transformed particles
  /// using the Hilbert curve.
  /// \param L type of each particle: arma::colvec or double.
  template <class L> void hilbert
  (
    arma::uvec& parentIndices,   // stores the post-resampling particle labels
    const std::vector<L> particles, // particles transformed to take values in the unit cube
    const arma::colvec& w,         // self-nomralised particle weights
    const unsigned int N,                 // total number of offspring
    const double lb, const double ub           
  )
  {
    hilbertBase<L>(arma::randu(), parentIndices, particles, w, N, lb, ub);
  }
}
#endif
