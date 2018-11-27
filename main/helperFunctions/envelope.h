/// \file
/// \brief Some helper functions for adaptively approximating a univariate density.
///
/// This file contains the functions for adaptively approximating a univariate, 
/// unimodal density on a compact set (with known location of the mode) by
/// a piecewise-uniform distribution.

#ifndef __ENVELOPE_H
#define __ENVELOPE_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <random>
#include <iostream>
#include <vector>
#include <functional>
#include <math.h>
#include <limits>

#include "main/helperFunctions/helperFunctions.h"

namespace envelope
{
  /// Samples from the envelope.
  double sample(
    const arma::colvec& points, 
    const arma::colvec& levels,
    const arma::colvec& probabilities
  )
  {
    
//     std::cout << "started sample()" <<std::endl;
    
    // TODO: we could probably use randomised QMC point sets (a la systematic resampling) here. To do this,
    // we need to make this function take a single uniform as an extra argument and then wrap it in 
    // another function which creates a systematic-resampling-like set of N QMC points. The marginal
    // proposal distributions will remain unaffected
    
    unsigned int idx = sampleInt(probabilities);
    double a  = points(idx);
    double b  = points(idx+1);
    double Z  = (b-a)*(levels(idx) + (levels(idx+1)-levels(idx))/2.0); // normalising constant for the specific interval
    double fa = levels(idx) / Z;
    double fb = levels(idx+1) / Z;
    double slope = (fb - fa) / (b - a);

    double x = a + fa/slope + std::sqrt(2.0/slope*arma::randu() + std::pow((fa/slope), 2.0));
    
    if (arma::randu() <= std::min(fa,fb)*(b-a))
    {
      x = a + (b-a)*arma::randu();
    }
    else 
    {
      if (fa <= fb)
      {
        x = a + std::sqrt(arma::randu()) * (b-a);
      }
      else
      {
        x = b - std::sqrt(arma::randu()) * (b-a);
      }
    }
    /////////////////////
          if (x<a) 
          {
            std::cout << "################## Warning: sampled x < a " << x << " < " << a << "  #####################" << std::endl;      
          }
          if (x>b) 
          {
            std::cout << "################## Warning: sampled x > b  " << x << " > " << b << "  #####################" << std::endl;
          }   
    ////////////////////
//     std::cout << "finished sample()" <<std::endl;
    return x;
  }
  /// Normalises the envelope.
  void normaliseLevels(
    const unsigned int nPoints, // number of current points
    const arma::colvec& points, 
    const arma::colvec& unnormalisedLogLevels,
    arma::colvec& levels,
    arma::colvec& probabilities
  )
  {
//         std::cout << "started normaliseLevels()" <<std::endl;
        
    // Normalising the levels in log-space:
    arma::colvec aux = arma::exp(normaliseExp(unnormalisedLogLevels));
    
    for (unsigned int g=0; g<(nPoints-1); g++)
    {
      probabilities(g) = (points(g+1) - points(g)) * (aux(g) + (aux(g+1) - aux(g)) / 2.0);
    }
     
    double normalisingConstant = arma::accu(probabilities(arma::span(0,nPoints-2)));
    probabilities(arma::span(0,nPoints-2)) = probabilities(arma::span(0,nPoints-2)) / normalisingConstant;
    levels(arma::span(0,nPoints-1)) = aux(arma::span(0,nPoints-1)) / normalisingConstant;
    
//      std::cout << "finished normaliseLevels()" <<std::endl;
  }
  /// Generates an envelope for the unnormalised target density associated with 
  /// a single observation
  void create(
    arma::colvec& points, 
    arma::colvec& levels, 
    arma::colvec& probabilities,
    const std::function<double(const double)>& logDensity,
    const unsigned int nSegments,
    const double lb, 
    const double ub,
    const double mode,
    const bool isBracketing
  )
  {
    
//     std::cout << "started create()" <<std::endl;
     
    levels.zeros(nSegments + 1);

    unsigned int idx;
    double newPoint, newLogDensity;
    arma::uvec sortedIndices;

    points.zeros(nSegments+1);
    arma::colvec unnormalisedLogLevels(nSegments+1);
    unnormalisedLogLevels.fill(std::log(0));
    probabilities.zeros(nSegments); 
    
    points(0) = lb;
    points(1) = mode;
    points(2) = ub;
    unnormalisedLogLevels(0) = std::log(0);
    unnormalisedLogLevels(1) = logDensity(mode);
    unnormalisedLogLevels(2) = std::log(0);
    
    normaliseLevels(3, points, unnormalisedLogLevels, levels, probabilities);
    
    /*
    std::cout << "lb: " << lb << "; mode: " << mode << "; ub: " << ub << std::endl;
    std::cout << "generating points" << std::endl;
    std::cout << "old points: " << points(arma::span(0,2)).t() << std::endl;
    std::cout << "old levels: " << unnormalisedLogLevels(arma::span(0,2)).t() << std::endl;
    */
    
    for (unsigned int g=3; g<nSegments+1; g++)
    {
      // Sample new point:
      newPoint = sample(points(arma::span(0,g-1)), levels(arma::span(0,g-1)), probabilities(arma::span(0,g-2)));
      newLogDensity = logDensity(newPoint); // calculating associated unnormalised target density:
   
      /*
      // NOTE: the problem seems to be that some sampled points are NANs (which shows up as zero in the output!)
      std::cout << "old points: " << points(arma::span(0,g-1)).t() << std::endl;
      std::cout << "old unnormalisedLevels: " << unnormalisedLevels(arma::span(0,g-1)).t() << std::endl;
      std::cout << "old probabilities: " << probabilities(arma::span(0,g-2)).t() << std::endl;
      std::cout << "sum of old probabilities: " << arma::accu(probabilities(arma::span(0,g-2))) << std::endl; 
      std::cout << "newPoint: " << newPoint << "; newDensity: " << newDensity << std::endl;
      std::cout << "finding the first element of points that is bigger than newPoint" << std::endl;
      std::cout <<arma::as_scalar(arma::find(points(arma::span(0,g-1)) > newPoint, 1, "first"))<< std::endl;
     */

      idx = arma::as_scalar(arma::find(points(arma::span(0,g-1)) > newPoint, 1, "first"));
      
      // Inserting new point:
      points(arma::span(idx+1,g)) = points(arma::span(idx,g-1));
      points(idx) = newPoint;
      unnormalisedLogLevels(arma::span(idx+1,g)) = unnormalisedLogLevels(arma::span(idx,g-1));
      unnormalisedLogLevels(idx) = newLogDensity;
      
      // Normalise the "envelope":
      normaliseLevels(g+1, points, unnormalisedLogLevels, levels, probabilities);
    }
      /*
      std::cout << "old points: " << points.t() << std::endl;
      std::cout << "old levels: " << unnormalisedLevels.t() << std::endl;
      std::cout << "old probabilities: " << probabilities.t() << std::endl;
      std::cout << "pMax: " << pMax << "; newPoint: " << newPoint << "; newDensity: " << newDensity << std::endl;
      std::cout << "finding the first element of points that is bigger than newPoint" << std::endl;
      */
    
//      std::cout << "finished create()" <<std::endl;

  }
  /// Evaluates the piecewise-constant density function associated
  /// with the envelope.
  double evaluateLogDensity(
    const double x,
    const arma::colvec& points, 
    const arma::colvec& levels
  )
  {
    // Determining inverval index:
    
//     std::cout << "started evaluateLogDensity()" <<std::endl;
//     std::cout << "points in evalLogDens(): " << points.t() << std::endl;
//     std::cout << " " << "x in evalLogDens(): " << x << "   ";
//     

    

//        std::cout << "x: " << x <<  "; points: " << points.t()  << std::endl;
//       std::cout << "min: " << arma::min(points) << "; max: " << arma::max(points) << std::endl;
//        std::cout << arma::find(points > x, 1, "first") << std::endl;
      
    
      unsigned int idx = arma::as_scalar(arma::find(points > x, 1, "first")) - 1;
      
              
//       std::cout << "finished evaluateLogDensity()" <<std::endl;
               
      return std::log(levels(idx) + (levels(idx+1) - levels(idx)) / (points(idx+1) - points(idx)) * (x - points(idx)));
      

  }
}
#endif
