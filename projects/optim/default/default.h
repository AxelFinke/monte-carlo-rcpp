/// \file
/// \brief Functions for implementing the optim class in the absence of reparametrisation.
///
/// This file contains the functions for the optim class
/// if no non-centred parametrisation is used.

#ifndef __DEFAULT_H
#define __DEFAULT_H

#include "projects/optim/Optim.h"

// [[Rcpp::depends("RcppArmadillo")]]

///////////////////////////////////////////////////////////////////////////////
/// Member functions of the class <<Optim>>
///////////////////////////////////////////////////////////////////////////////

/// Reparametrises latent variables from the standard (centred) parametrisation
/// to a (partially) non-centred parametrisation.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>  
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::convertLatentPathToLatentPathRepar(const arma::colvec& theta, const LatentPath& latentPath, LatentPathRepar& latentPathRepar)
{
  latentPathRepar = latentPath;
}
/// Reparametrises latent variables from (partially) non-centred parametrisation
/// to the standard (centred) parametrisation
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>  
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::convertLatentPathReparToLatentPath(const arma::colvec& theta, LatentPath& latentPath, const LatentPathRepar& latentPathRepar)
{
  latentPath = latentPathRepar;
}

#endif
