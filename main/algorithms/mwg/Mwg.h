/// \file
/// \brief Performing Metropolis-within-Gibbs updates for latent variables. 
///
/// This file contains the functions associated with the Mwg class.

#ifndef __MWG_H
#define __MWG_H

#include <time.h> 
#include "main/model/Model.h"

// [[Rcpp::depends("RcppArmadillo")]]

/// Class template for running (conditional) SMC algorithms or other forms 
/// of importance sampling.
template<class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class MwgParameters> class Mwg
{
public:
  
  /// Initialises the class.
  Mwg
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    const unsigned int nCores
  ) : 
    rng_(rng), 
    model_(model),
    nCores_(nCores)
  {
    // Empty!
  }
  
  /// Returns the the number of updates.
  unsigned int getNUpdates() const {return nUpdates_;}
  /// Specifies the number of updates.
  void setNUpdates(const unsigned int nIterations) {nUpdates_ = nUpdates;}
  /// Returns the probabilities for each type of move.
  void setMoveProbabilities(const arma::colve& moveProbabilities) {moveProbabilities_ = moveProbabilities;}
  /// Randomly draws a move.
  unsigned int randomlySelectMove()
  {
    return sampleInt(moveProbabilities_);
  }
  /// Runs (a series of) Metropolis-within-Gibbs updates.
  void runSampler(arma::colvec& theta, LatentPath& latentPath)
  {
    runSamplerBase(theta, latentPath);
  }
  
private:
 
  /// Applies the MCMC kernels.
  void runSamplerBase(
    const arma::colvec& theta,
    LatentPath& latentPath
  );
  
  Rng& rng_; // random number generation.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // the targeted model.
  unsigned int nUpdates_; // number of Metropolis-within-Gibbs steps
  MwgParameters mwgParameters_; // holds some additional auxiliary parameters for the SMC algorithm.
  arma::colvec moveProbabilities_; // probabilities for the different types of MCMC moves
  unsigned int nCores_; // number of cores to use (not currently used)
};

#endif
