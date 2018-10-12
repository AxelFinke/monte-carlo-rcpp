#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
// #include <gperftools/profiler.h>
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <time.h> 
#include <omp.h>
#include "rng/Rng.h"
#include "rng/gaussian.h"
#include "smc/resample.h"
#include "helperFunctions.h"
#include "examples/linearGaussianHmm/linearGaussianHmm.h"

// TODO: disable range checks (by using at() for indexing elements of cubes/matrices/vectors)
// once the code is tested; 
// To that end, compile the code with ARMA_NO_DEBUG defined.

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

////////////////////////////////////////////////////////////////////////////////
// Create Toeplitz matrix by passing the first column as an argument
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::mat toeplitzCpp(const arma::colvec& firstCol)
{ 
  arma::mat A = arma::toeplitz(firstCol);
  return A;
}
////////////////////////////////////////////////////////////////////////////////
// Obtain log-marginal likelihood via the Kalman filter
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double kalmanCpp(
  const arma::mat& A,  // must be a (V, V)-matrix
  const arma::mat& B,  // must be a (V, V)-matrix
  const arma::mat& C,  // must be a (dy, V)-matrix
  const arma::mat& D,  // must be a (dy, dy)-matrix
  const arma::vec& m0, // must be a (V, 1)-vector
  const arma::mat& C0, // must be a (V, V)-matrix
  const arma::mat& y)  // must be a (dy, T)-matrix
{
  Kalman K(A, B, C, D, m0, C0, y);
  K.runKalman();
  return (K.get_logLike());
}
////////////////////////////////////////////////////////////////////////////////
// Simulates data in a multivariate linear Gaussian HMM
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List simulateDataCpp(
  const unsigned int T, // number of time steps
  const arma::mat& A, // must be a (V, V)-matrix
  const arma::mat& B, // must be a (V, V)-matrix
  const arma::mat& C, // must be a (dy, V)-matrix
  const arma::mat& D, // must be a (dy, dy)-matrix
  const arma::colvec& m0, // must be a (V, 1)-vector
  const arma::mat& C0 // must be a (V, V)-matrix
)
{
  const unsigned int V = A.n_rows;
  const unsigned int dy = C.n_rows;
  arma::mat x(V, T); // states
  arma::mat y(dy, T); // observations
  
  x.col(0) = gaussian::sampleMultivariate(1, m0, C0, false);
  y.col(0) = gaussian::sampleMultivariate(1, C*x.col(0), D, true);
  
  for (unsigned int t=1; t<T; t++) {
    x.col(t) = gaussian::sampleMultivariate(1, A*x.col(t-1), B, true);
    y.col(t) = gaussian::sampleMultivariate(1, C*x.col(t), D, true);
  }
  return Rcpp::List::create(Rcpp::Named("x") = x, Rcpp::Named("y") = y);
}





////////////////////////////////////////////////////////////////////////////////
// Perform inference on the states in a multivariate linear Gaussian HMMs
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double smcCpp(
  const unsigned int N, // number of particles
  const double H, // ESS resampling threshold
  unsigned int prop, // use locally optimal proposal kernel?
  arma::mat& A,  // must be a (V, V)-matrix
  arma::mat& B,  // must be a (V, V)-matrix
  arma::mat& C,  // must be a (dy, V)-matrix
  arma::mat& D,  // must be a (dy, dy)-matrix
  arma::colvec& m0, // must be a (V, 1)-vector
  arma::mat& C0, // must be a (V, V)-matrix
  arma::mat& y,  // must be a (dy, T)-matrix
  unsigned int cores)
{
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;

  bool isSimple = true;
  Model S(rng, N,H,
          static_cast<FilterType>(0), 
          static_cast<ProposalType>(prop),
          static_cast<ParametrisationType>(0),
          A,B,C,D,m0,C0,y,isSimple,cores);
  S.runSmc();
  return S.get_logLikeEst();
}

////////////////////////////////////////////////////////////////////////////////
// Perform inference on the states in a multivariate linear Gaussian HMMs
// using block particle filters
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double bpfCpp(
   unsigned int N, // number of particles
   unsigned int prop, // use locally optimal proposal kernel?
   arma::mat& blockInn,
   arma::mat& blockOut,
   arma::mat& A,  // must be a (V, V)-matrix
   arma::mat& B,  // must be a (V, V)-matrix
   arma::mat& C,  // must be a (dy, V)-matrix
   arma::mat& D,  // must be a (dy, dy)-matrix
   arma::colvec& m0, // must be a (V, 1)-vector
   arma::mat& C0, // must be a (V, V)-matrix
   arma::mat& y,  // must be a (dy, T)-matrix
   unsigned int cores)
{
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;

  double H = 1.0;
  bool isSimple = true;
  Model S(rng, N, H,
          static_cast<FilterType>(1), 
          static_cast<ProposalType>(prop),
          static_cast<ParametrisationType>(0),
          A,B,C,D,m0,C0,y,isSimple,cores);
  S.setBlocks(blockInn, blockOut);
  S.runSmc();
  return S.get_logLikeEst();
}

////////////////////////////////////////////////////////////////////////////////
// Perform inference on the parameters in a multivariate linear Gaussian HMMs
// using a PMMH algorithm (with likelihood given/approximated by
// standard SMC/BPF/exact methods.
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runPmmhCpp(
   const unsigned int N, // number of particles
   const unsigned int filt,
   const unsigned int prop, // use locally optimal proposal kernel?
   const unsigned int nIterations,
   arma::colvec& hyperParameters,
   arma::colvec& rwmhSd,
   arma::mat& blockInn,
   arma::mat& blockOut,
   arma::colvec& m0, // must be a (V, 1)-vector
   arma::mat& C0, // must be a (V, V)-matrix
   const arma::mat& y,  // must be a (dy, T)-matrix
   const unsigned int cores)
{
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;

  double H = 1.0;
  arma::colvec stepSizes(10);
  arma::colvec thetaInit(4);
  std::vector<arma::colvec> thetaFull(nIterations);

  Model model(rng,N, H,static_cast<FilterType>(filt), static_cast<ProposalType>(prop),static_cast<ParametrisationType>(0),thetaInit,stepSizes,m0,C0,y,true,cores);
  
  model.setBlocks(blockInn, blockOut);
  model.setHyperParameters(hyperParameters);
  model.setRwmhSd(rwmhSd);
  arma::colvec thetaProp(model.getDimTheta());
  arma::colvec theta(model.getDimTheta());
  double logLikeProp = 0;
  double logLike = 0;
  double logAlpha = 0;
  model.samplePriorTheta(theta);
  thetaFull[0] = theta;
  

  
  // TODO
  if (filt == 0 || filt == 1) // Standard SMC/BPF approximation of the likelihood
  {
    model.runSmc(N, theta);
    logLike = model.get_logLikeEst();
  }
  else if (filt == 2) // exact computation of the likelihood
  {
    logLike = model.evaluateLogMarginalLikelihood(theta);
  }
  
  for (unsigned int g=1; g<nIterations; g++)
  {
    std::cout << "Iteration " << g << " of PMMH algorithm with lower-level algorithnm " << filt << std::endl;
    
    model.proposeTheta(thetaProp, theta, 1.0);
    
//     std::cout << "finished proposing theta" << std::endl;
    
    logAlpha  = model.evaluateLogProposalDensityTheta(theta, thetaProp, 1.0) - model.evaluateLogProposalDensityTheta(thetaProp, theta, 1.0);
    logAlpha += model.evaluateLogPriorDensityTheta(thetaProp) - model.evaluateLogPriorDensityTheta(theta);
    
//         std::cout << "finished calculating logAlpha" << std::endl;

    if (std::isfinite(logAlpha))
    {
      if (filt == 0 || filt == 1)
      {
//         std::cout << "runSmc()" << std::endl;
            
        model.runSmc(N, thetaProp);
        logLikeProp = model.get_logLikeEst();
        
//         std::cout << "finished runSmc()" << std::endl;
      }
      else if (filt == 2)
      {
        logLikeProp = model.evaluateLogMarginalLikelihood(thetaProp);
      }
      logAlpha += logLikeProp - logLike;
    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }
    std::cout << "logAlpha: " << logAlpha << std::endl;
              
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      std::cout << "################### ACCEPTANCE ###################" << std::endl;
      theta = thetaProp;
      logLike = logLikeProp;
      
    }
    thetaFull[g] = theta;
  }

  return Rcpp::List::create(Rcpp::Named("theta") = thetaFull);
}
////////////////////////////////////////////////////////////////////////////////
// Returns approximate (component-wise) smoothed expectations for the 
// sufficient statistics and gradient components
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List suffCpp(
  const unsigned int N,    // number of particles
  const unsigned int M,    // number of "backward" particles
  const double H,          // ESS resampling threshold
  arma::mat& blockInn,
  arma::mat& blockOut,
  unsigned int filt,
  unsigned int prop,       // use locally optimal proposal kernel?
  unsigned int back,       // type of backward-sampling scheme to use
  arma::colvec& thetaInit, 
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{
  


  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;

  bool isSimple = true;
  arma::colvec stepSizes(1, arma::fill::zeros);
  
  Model S(rng,N,H,static_cast<FilterType>(filt),static_cast<ProposalType>(prop),static_cast<ParametrisationType>(0),thetaInit,stepSizes,m0,C0,y,isSimple,cores);
  
  S.setStoreHistory(true); // force storage of the entire particle system
  S.setBlocks(blockInn, blockOut);
  S.setM(M);
  S.setStoreSuff(true);
  S.set_back(static_cast<BackwardKernelType>(back));
  
//   /////////////////////////////
//   /////////////////////////////
//   ProfilerStart("/home/axel/Dropbox/phd/code/cpp/generic/Scratch/output/blockedSmoothing/linearGaussianHmm/profile_output.log");
//   /////////////////////////////
//   /////////////////////////////
  
  S.runSmc();
  S.runParticleSmoother();
  
//   /////////////////////////////
//   /////////////////////////////
//   ProfilerStop();
//   /////////////////////////////
//   /////////////////////////////

  return Rcpp::List::create(Rcpp::Named("suffHCompEst") = S.get_suffHCompEst(),
                            Rcpp::Named("suffQCompEst") = S.get_suffQCompEst(),
                            Rcpp::Named("suffRCompEst") = S.get_suffRCompEst(),
                            Rcpp::Named("suffSCompEst") = S.get_suffSCompEst(),
                            Rcpp::Named("gradCompEst")  = S.get_gradCompEst(),
                            Rcpp::Named("suffHEst")     = S.get_suffHEst(),
                            Rcpp::Named("suffQEst")     = S.get_suffQEst(),
                            Rcpp::Named("suffREst")     = S.get_suffREst(),
                            Rcpp::Named("suffSEst")     = S.get_suffSEst(),
                            Rcpp::Named("gradEst")      = S.get_gradEst()
                           );
}
////////////////////////////////////////////////////////////////////////////////
// Returns (component-wise) smoothed expectations for the sufficient statistics
// and gradient components
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List suffExactCpp(
  arma::colvec& thetaTrue,
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;
  
  arma::colvec aux(1, arma::fill::zeros);

  bool isSimple = true;
  Model S(rng,10,1,static_cast<FilterType>(0), static_cast<ProposalType>(0),static_cast<ParametrisationType>(0),thetaTrue,aux,m0,C0,y,isSimple,cores);
  
 
  S.runKalman();
  S.runKalmanSmoother();
  S.computeSuffTrue();
  
  return Rcpp::List::create(Rcpp::Named("suffHCompTrue") = S.get_suffHCompTrue(),
                            Rcpp::Named("suffQCompTrue") = S.get_suffQCompTrue(),
                            Rcpp::Named("suffRCompTrue") = S.get_suffRCompTrue(),
                            Rcpp::Named("suffSCompTrue") = S.get_suffSCompTrue(),
                            Rcpp::Named("gradCompTrue")  = S.get_gradCompTrue(),
                            Rcpp::Named("suffHTrue")     = S.get_suffHTrue(),
                            Rcpp::Named("suffQTrue")     = S.get_suffQTrue(),
                            Rcpp::Named("suffRTrue")     = S.get_suffRTrue(),
                            Rcpp::Named("suffSTrue")     = S.get_suffSTrue(),
                            Rcpp::Named("gradTrue")      = S.get_gradTrue()
                           );
}
////////////////////////////////////////////////////////////////////////////////
// Perform parameter-inference in a multivariate linear Gaussian HMMs
// via (approximate) stochastic gradient-ascent algorithms
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runOsgaCpp(
  const unsigned int N,    // number of particles
  const unsigned int M,    // number of "backward" particles
  const double H,          // ESS resampling threshold
  arma::mat& blockInn,
  arma::mat& blockOut,
  unsigned int filt,
  unsigned int prop,       // use locally optimal proposal kernel?
  unsigned int back,       // type of backward-sampling scheme to use
  arma::colvec& thetaInit, 
  arma::colvec& stepSizes,
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;

  //unsigned int hist = 1;
  bool isSimple = true;
  Model S(rng,N,H,static_cast<FilterType>(filt), static_cast<ProposalType>(prop),static_cast<ParametrisationType>(0),thetaInit,stepSizes,m0,C0,y,isSimple,cores);
  
  S.set_back(static_cast<BackwardKernelType>(back));
  S.setM(M);
  S.setStoreSuff(true);
  S.setBlocks(blockInn, blockOut);
  S.run_offline_stochastic_gradient_ascent( );
  
  return Rcpp::List::create(Rcpp::Named("theta") = S.get_thetaFull());
}
////////////////////////////////////////////////////////////////////////////////
// Perform parameter-inference in a multivariate linear Gaussian HMMs
// via a gradient-ascent algorithm
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runOgaCpp(
  arma::colvec& thetaInit, 
  arma::colvec& stepSizes,
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;
  
  bool isSimple = true;
  Model S(rng,0,0,static_cast<FilterType>(0), static_cast<ProposalType>(0),static_cast<ParametrisationType>(0),thetaInit,stepSizes,m0,C0,y,isSimple,cores);
  S.setStoreSuff(true);
  S.run_offline_gradient_ascent( );
  
  return Rcpp::List::create(Rcpp::Named("theta") = S.get_thetaFull());
}
////////////////////////////////////////////////////////////////////////////////
// Perform parameter-inference in a multivariate linear Gaussian HMMs
// via (approximate) stochastic EM algorithms
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runOsemCpp(
  const unsigned int N,    // number of particles
  const unsigned int M,    // number of "backward" particles
  const double H,          // ESS resampling threshold
  arma::mat& blockInn,
  arma::mat& blockOut,
  unsigned int filt,
  unsigned int prop,       // use locally optimal proposal kernel?
  unsigned int back,       // type of backward-sampling scheme to use
  arma::colvec& thetaInit, 
  arma::colvec& stepSizes,
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;
  
  bool isSimple = true;
  Model S(rng,N,H,static_cast<FilterType>(filt), static_cast<ProposalType>(prop), static_cast<ParametrisationType>(0), thetaInit,stepSizes,m0,C0,y,isSimple,cores);
  
  S.set_back(static_cast<BackwardKernelType>(back));
  S.setM(M);
  S.setStoreSuff(true);
  S.setBlocks(blockInn, blockOut);
  S.run_offline_stochastic_em( );
  
  return Rcpp::List::create(Rcpp::Named("theta") = S.get_thetaFull());
}

////////////////////////////////////////////////////////////////////////////////
// Perform parameter-inference in a multivariate linear Gaussian HMMs
// via an EM algorithm
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runOemCpp(
  arma::colvec& thetaInit, 
  arma::colvec& stepSizes,
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;
  
  bool isSimple = true;
  Model S(rng,0,0,static_cast<FilterType>(0), static_cast<ProposalType>(0),static_cast<ParametrisationType>(0),thetaInit,stepSizes,m0,C0,y,isSimple,cores);
  S.setStoreSuff(true);
  S.run_offline_em( );
  
  return Rcpp::List::create(Rcpp::Named("theta") = S.get_thetaFull());
}

////////////////////////////////////////////////////////////////////////////////
// Perform parameter-inference in a multivariate linear Gaussian HMMs
// via blocked online gradient-ascent algorithm
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runOnlineStochasticGradientAscentCpp(
  const unsigned int N,    // number of particles
  arma::mat& blockInn,
  arma::mat& blockOut,
  unsigned int padding,
  unsigned int filt,
  unsigned int prop,       // use locally optimal proposal kernel?
  arma::colvec& thetaInit, 
  arma::colvec& stepSizes,
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;
  
  std::cout << "set up model" << std::endl;
  Model S(rng,N,1.0,static_cast<FilterType>(filt), static_cast<ProposalType>(prop), static_cast<ParametrisationType>(0), thetaInit,stepSizes,m0,C0,y,true,cores);
  
  bool updateAfterEachBlock; // should the parameters be updated after each spatial block
  if (stepSizes.size() == y.n_cols)
  {
    updateAfterEachBlock = false;
  }
  else
  {
    updateAfterEachBlock = true;
  }
  
  S.setBlocks(blockInn, blockOut);
  
    
  std::cout << "run online algorithm" << std::endl;
  S.runBlockedOnlineGradientAscent(thetaInit, updateAfterEachBlock, padding);
  
  return Rcpp::List::create(Rcpp::Named("theta") = S.get_thetaFull());
}

////////////////////////////////////////////////////////////////////////////////
// Perform parameter-inference in a multivariate linear Gaussian HMMs
// via a blocked online gradient-ascent algorithm
// which makes use of enlarged blocks
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
Rcpp::List runOnlineStochasticGradientAscentEnlargedCpp(
  const unsigned int N,    // number of particles
  const bool normaliseByL2Norm, // should the gradient approximation be normalised by the L2 norm?
  const bool estimateTheta, // should theta be updated after each time step?
  arma::mat& blockInn,
  arma::mat& blockOut,
  unsigned int filt,
  unsigned int back,
  unsigned int prop,       // use locally optimal proposal kernel?
  arma::colvec& thetaInit, 
  arma::colvec& stepSizes,
  arma::colvec& m0,        // must be a (V, 1)-vector
  arma::mat& C0,           // must be a (V, V)-matrix
  arma::mat& y,            // must be a (dy, T)-matrix
  unsigned int cores)
{

  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  Rng* rng = &rngDerived;
  
  std::cout << "set up model" << std::endl;
  Model S(rng,N,1.0,static_cast<FilterType>(filt), static_cast<ProposalType>(prop), static_cast<ParametrisationType>(0), thetaInit,stepSizes,m0,C0,y,true,cores);
  
  S.setBlocks(blockInn, blockOut);
  S.set_back(static_cast<BackwardKernelType>(back));

  
  arma::mat gradBlockOut;
  //////////////////////
//   arma::mat suffStatBlockOut;
  ///////////////////////
    
  std::cout << "run online algorithm" << std::endl;
  //S.runBlockedOnlineGradientAscentEnlarged(thetaInit, normaliseByL2Norm, estimateTheta, gradBlockOut);
  S.runBlockedOnlineGradientAscentEnlarged(thetaInit, normaliseByL2Norm, estimateTheta, gradBlockOut); //, suffStatBlockOut);
  
  std::cout << estimateTheta << std::endl;
  
  if (estimateTheta)
  {
    return Rcpp::List::create(Rcpp::Named("theta") = S.get_thetaFull());
  }
  else
  {
    return Rcpp::List::create(Rcpp::Named("gradient") = gradBlockOut); //, Rcpp::Named("suffStat") = suffStatBlockOut);
  }
}
