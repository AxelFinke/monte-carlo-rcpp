/// \file
/// \brief Some helper functions for using the Newton method.
///
/// This file contains the functions for finding a root of real-valued function
/// with bounded support

#ifndef __NEWTON_H
#define __NEWTON_H

#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <random>
#include <iostream>
#include <vector>
#include <functional>
#include <math.h>
#include "main/helperFunctions/helperFunctions.h"

namespace rootFinding
{
  /// Finds the root of a real-valued function 
  /// using a safeguarded Newton approach (NOTE: it assumes tha a solution
  /// exists in the given interval).
  double saveGuardedNewton(
    bool& isBracketing,
    const std::function<double(const double)>& f,
    const std::function<double(const double)>& f1,
    const double lb, 
    const double ub,
    const double tolX, // tolerance: interval boundaries
    const double tolF, // tolerance: values of f
    const unsigned int nIterations
  )
  {
    
    double x1, fx;
    double a = lb;
    double b = ub;
    double fa = f(lb);
    double fb = f(ub);
    unsigned int i = 0;
    double x = a;
  
    if (fa * fb > 0) 
    { // CASE I: interval not bracketing
      isBracketing = false;
      std::cout << "Error: interval not bracketing!" << std::endl;
    }
    else
    { // CASE II: interval is bracketing
      isBracketing = true;

      fx  = f(x); 
      
      while ((i == 0) || ((std::abs(a - b) > tolX) && (std::abs(fx) > tolF) && (i < nIterations)))
      {
        x1 = x - fx / f1(x);
        
        if ((((fa * f(x1) < 0) || (fb * f(x1) < 0)) && (a < x1) && (x1 < b))) 
        {
          x = x1;
        }
        else
        {
          x = (a + b) / 2.0;
        }
        
        fx  = f(x);
        i++;

        if (fa * fx <= 0)
        {
          b  = x;
          fb = fx;
        }
        else 
        {
          a  = x;
          fa = fx;
        }
      }
    }
    return x;
  }


  /// The bisection method.
  double bisection(
    bool& isBracketing,
    const std::function<double(const double)>& f,
    const double lb, 
    const double ub,
    const double tolX, // tolerance: interval boundaries
    const double tolF, // tolerance: values of f
    const unsigned int nIterations
  )
  {
    double a = lb;
    double b = ub;
    double fa = f(lb);
    double fb = f(ub);
    unsigned int i = 0;
    double x = a;
    double fx = fa;
  
    if (fa * fb > 0) 
    { // CASE I: interval not bracketing
      isBracketing = false;
      std::cout << "Error: interval not bracketing!" << std::endl;
    }
    else
    { // CASE II: interval is bracketing
      isBracketing = true;

      while ((i == 0) || ((std::abs(a - b) > tolX) && (std::abs(fx) > tolF) && (i < nIterations)))
      {
        x = (a + b) / 2.0; // new midpoint
        fx = f(x);
        i++;
       
        if (fa * fx <= 0) // i.e. the root must be in [a,x]
        {
          b  = x;
//           fb = fx;
        }
        else // i.e. the root must be in (x,b]
        {
          a  = x;
          fa = fx;
        }
      }
    }
    if (i == nIterations)
    {
      std::cout << "Error: maximum number of iterations exceeded!" << std::endl;
    }
    return x;
  }
  
  
}
#endif
