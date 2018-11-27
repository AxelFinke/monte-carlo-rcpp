/// \file
/// \brief Some auxiliary functions.
///
/// This file contains a number of auxiliary functions for 
/// use in other C++ programmes.

#ifndef __OPTIM_H
#define __OPTIM_H

#include "main/model/Model.h"
#include "main/algorithms/smc/Smc.h"
#include "main/algorithms/mcmc/Mcmc.h"

// TODO: 
// - implement adaptive resamping at the upper-level SMC sampler
// - implement CESS-adaptive tempering schedule (problem: this leads to a random number of steps)
// - implement adaptively tuning the proposal scale for \theta
// - make sure we are not missing any normalising constants in the upper-level SMC sampler
// as this would change the model evidence

// use std::reference_wrapper !!!

/// Container for holding all the (relevant) random variables in the state-space 
/// of the pseudo-marginal MCMC-based optimisation algorithm and related methods.
template <class LatentPath, class Aux> class Opt 
{
public:
  
  /// Initialises the class.
  Opt() {}
  /// Initialises the class and specifies number of components 
  /// and length of the parameter vector.
  Opt(const unsigned int dimTheta, const unsigned int k) 
  { 
    this->resize(dimTheta, k);
  }
  /// Destructor.
  ~Opt() {};
  /// Resizes the vectors in the class.
  void resize(const unsigned int dimTheta, const unsigned int k)
  {
    theta.set_size(dimTheta);
    logLikelihoods.resize(k);
    latentPath.resize(k);
    aux.resize(k);
    gradients.resize(k);
    for (unsigned int l=0; l<k; l++)
    {
      gradients[l].set_size(dimTheta);
    }
  }
  /// Returns a weighted sum of the the first coefficients.size() logLikelihoods.
  double sumLogLikelihoods(const arma::colvec& coefficients)
  {
    double logLike = 0;
    for (unsigned int k=0; k<coefficients.size(); k++)
    {
      logLike += logLikelihoods[k] * coefficients(k);
    }
    return logLike;
  }
  /// Returns a weighted sum of the the first coefficients.size() gradients.
  arma::colvec sumGradients(const arma::colvec& coefficients)
  {
    arma::colvec grad(gradients[0].size(), arma::fill::zeros);
    for (unsigned int k=1; k<coefficients.size(); k++)
    {
      grad = grad + gradients[k] * coefficients(k);
    }
    return grad;
  }
  /// Returns a weighted sum of the the first gradients.
  arma::colvec sumGradients(const unsigned int maxNumber)
  {
    arma::colvec grad(gradients[0].size(), arma::fill::zeros);
    for (unsigned int k=1; k<maxNumber; k++)
    {
      grad = grad + gradients[k];
    }
    return grad;
  }
  /// Returns the number of loglikelihoods/latent variable sets/auxiliary
  /// variable sets/gradients.
  unsigned int nComponents() {return logLikelihoods.size();}
  /// Returns the length of the parameter vector.
  unsigned int dimTheta() {return theta.size();}
  
  arma::colvec theta;
  std::vector<double> logLikelihoods;
  std::vector<LatentPath> latentPath;
  std::vector<AuxFull<Aux>> aux;
  std::vector<arma::colvec> gradients;
};

/// Swaps the elements of two Opt classes.
// template <class Par, class Lat, class Aux> void swapOpt(Opt<LatentPath, Aux>& opt, Opt<LatentPath, Aux>& optProp)
// {
//   opt.theta = optProp.theta;
//   opt.logLikelihoods.swap(optProp.logLikelihoods);
//   opt.latentPath.swap(optProp.latentPath);
//   opt.aux.swap(optProp.aux);
//   opt.gradients.swap(optProp.gradients);
// }

// [[Rcpp::depends("RcppArmadillo")]]

/// Upper-level Monte-Carlo algorithm.
enum OptimUpperType 
{  
  OPTIM_UPPER_MCMC = 0, // perform optimisation via a single MCMC chain
  OPTIM_UPPER_SMC // perform optimisation via a population-based (i.e. SMC-sampler) approach
};
/// Lower-level Monte-Carlo algorithm.
enum OptimLowerType 
{  
  OPTIM_LOWER_PSEUDO_MARGINAL_SAME = 0, // pseudo-marginal SAME algorithm
  OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED, // correlated pseudo-marginal SAME algorithm
  OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY, // pseudo-marginal SAME algorithm (noisy)
  OPTIM_LOWER_RUBENTHALER, // naive algorithm
  OPTIM_LOWER_RUBENTHALER_CORRELATED, // naive algorithm with correlated pseudo-marginal updates
  OPTIM_LOWER_RUBENTHALER_NOISY, // naive algorithm (noisy)
  OPTIM_LOWER_SIMULATED_ANNEALING, // simulated annealing
  OPTIM_LOWER_PSEUDO_GIBBS_SAME, // pseudo-Gibbs sampler implementation of the SAME algorithm
  OPTIM_LOWER_GIBBS_SAME // Gibbs-sampler implementation of the SAME algorithm
};

/// Class for performing optimisation in latent variale models 
/// and more generally.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters> class Optim
{
  
public:
 
  /// Initialises the class for use with upper-level MCMC algorithm.
  Optim(
    Rng& rng, 
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc,
    Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>& smc,
    const OptimLowerType lower, 
    const arma::colvec& beta, 
    bool areInverseTemperaturesIntegers,
    const arma::uvec& nParticlesLower,
    const unsigned int nThetaUpdates,
    const double proportionCorrelated,
    const unsigned int nCores
    ) : 
    rng_(rng), 
    model_(model),
    mcmc_(mcmc),
    smc_(smc),
    upper_(OPTIM_UPPER_MCMC), 
    lower_(lower),
    beta_(beta), 
    areInverseTemperaturesIntegers_(areInverseTemperaturesIntegers),
    nParticlesLower_(nParticlesLower),
    nThetaUpdates_(nThetaUpdates),
    proportionCorrelated_(proportionCorrelated),
    nCores_(nCores)
  {
    nIterations_ = beta_.n_rows;
    betaMin_ = beta_(0);
    betaMax_ = beta_(nIterations_-1);
    storeHistory_ = true; // TODO: make this optional later
    setUseGaussianParametrisation();
  }
  /// Initialises the class for use with upper-level SMC algorithm.
  Optim(
    Rng& rng, 
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc,
    Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>& smc,
    const OptimLowerType lower, 
    const arma::colvec& beta, 
    bool areInverseTemperaturesIntegers,
    const unsigned int nParticlesUpper,
    const arma::uvec& nParticlesLower,
    const double essResamplingThreshold,
    const unsigned int nThetaUpdates,
    const double proportionCorrelated,
    const unsigned int nCores
    ) : 
    rng_(rng), 
    model_(model),
    mcmc_(mcmc),
    smc_(smc),
    upper_(OPTIM_UPPER_SMC), 
    lower_(lower),
    beta_(beta), 
    areInverseTemperaturesIntegers_(areInverseTemperaturesIntegers),
    nParticlesUpper_(nParticlesUpper),
    nParticlesLower_(nParticlesLower),
    essResamplingThreshold_(essResamplingThreshold),
    nThetaUpdates_(nThetaUpdates),
    proportionCorrelated_(proportionCorrelated),
    nCores_(nCores)
  {
    nIterations_ = beta_.n_rows;
    betaMin_ = beta_(0);
    betaMax_ = beta_(nIterations_-1);
    storeHistory_ = true; // TODO: make this optional later
    setUseGaussianParametrisation();
  }
  /// Runs an MCMC-based optimisation algorithm.
  void runMcmc();
  /// Runs an SMC-based optimisation algorithm.
  void runSmc();
  /// Runs an SMC-based optimisation algorithm (with adaptive tempering schedule).
  void runAdaptiveSmc();
  /// Determines the number Metropolis--Hastings updates for the parameters
  /// (given the latent variables) within each Gibbs-sampler
  /// or pseudo-Gibbs sampler update.
  void setNParameterUpdates(const unsigned int nThetaUpdates) {nThetaUpdates_ = nThetaUpdates;}
  /// Determines the number of particles used within each lower_-level
  /// (C)SMC algorithm.
  void setNParticlesLower(const arma::uvec& nParticlesLower) {nParticlesLower_ = nParticlesLower;}
  /// Determines the number of particles used by the upper_-level SMC sampler.
  void setNParticlesUpper(const unsigned int nParticlesUpper) {nParticlesUpper = nParticlesUpper_;}
  /// Returns the parameters sampled over all MCMC iterations.
  std::vector<arma::colvec> getThetaFullMcmc() const {return thetaFullMcmc_;}
  /// Returns the parameters sampled over all MCMC iterations.
  void getThetaFullMcmc(std::vector<arma::colvec>& thetaFullMcmc) const {thetaFullMcmc = thetaFullMcmc_;}
  /// Returns the parameters sampled over all SMC steps.
  std::vector<std::vector<arma::colvec> > getThetaFullSmc() const {return thetaFullSmc_;}
  /// Returns the parameters sampled over all Smc steps.
  void getThetaFullSmc(std::vector< std::vector<arma::colvec> >& thetaFullSmc) const {thetaFullSmc = thetaFullSmc_;}
  /// Determines whether the current lower-level algorithm needs to 
  /// calculate and store Gaussian auxiliary variables.
  void setUseGaussianParametrisation()
  {
    if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED ||
        lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
    {
      smc_.setUseGaussianParametrisation(true);
    }
    else
    {
      smc_.setUseGaussianParametrisation(false);
    }
  }
  /// Specifies whether the (peudo-)Gibbs samplers should make use of a
  /// non-centred parametrisation.
  void setUseNonCentredParametrisation(const bool useNonCentredParametrisation) 
  {
    useNonCentredParametrisation_ = useNonCentredParametrisation;
  }
  /// Specifies the probability of switching to the non-centred
  /// parametrisation (if useNonCentredParametrisation == true).
  void setNonCentringProbability(const double nonCentringProbability)
  {
    nonCentringProbability_ = nonCentringProbability;
  }
  
private:
  
  /// Determines some auxiliary variables used at each iteration/step of the 
  /// MCMC algorithm/SMC sampler.
  void setAuxiliaryParameters(const unsigned int g);
  /// Resamples in the upper_-level SMC sampler.
  void resampleUpper(std::vector<Opt<LatentPath, Aux>>& opt, arma::colvec& logWeights);
  /// Initialises the initial state of the MCMC chain.
  void initialise(Opt<LatentPath, Aux>& opt);
  /// Initialises the initial state of a single particle and computes the log-Weight.
  void initialise(Opt<LatentPath, Aux>& opt, double& logWeight);
  /// Performs a change move.
  void change(const unsigned int g, Opt<LatentPath, Aux>& opt);
  /// Performs a change move and also updates the log-weight.
  void change(const unsigned int g, Opt<LatentPath, Aux>& opt, double& logWeight);
  /// Performs an extend move.
  void extend(const unsigned int g, Opt<LatentPath, Aux>& opt);
  /// Performs an extend move and also updates the log-weight.
  void extend(const unsigned int g, Opt<LatentPath, Aux>& opt, double& logWeight);
  /// Performs an update move.
  void update(const unsigned int g, Opt<LatentPath, Aux>& opt);
  /// Wrapper for SMC algorithms.
  void runSmcLower(Opt<LatentPath, Aux>& opt, const unsigned int k, const double inverseTemperature)
  {
    opt.logLikelihoods[k] = smc_.runSmc(nParticlesNew_, opt.theta, opt.latentPath[k], opt.aux[k], opt.gradients[k], inverseTemperature);
  }
  /// Wrapper for CSMC algorithms.
  void runCsmcLower(Opt<LatentPath, Aux>& opt, const unsigned int k, const double inverseTemperature)
  {
    opt.logLikelihoods[k] = smc_.runCsmc(nParticlesNew_, opt.theta, opt.latentPath[k], opt.aux[k], opt.gradients[k], inverseTemperature);
  }
  /// Wrapper for Gibbs-sampling updates.
  void runGibbs(Opt<LatentPath, Aux>& opt, const unsigned int k, const double inverseTemperature)
  {
    model_.setUnknownParameters(opt.theta);
    model_.setInverseTemperature(inverseTemperature);
    model_.runGibbs(opt.latentPath[k]);
  }
  /// Evaluates the log-marginal likelihood.
  double evaluateLogMarginalLikelihood(const arma::colvec& theta)
  {
    model_.setUnknownParameters(theta);
    return model_.evaluateLogMarginalLikelihood();
  }
  /// Evaluates the score.
  void evaluateGradient(const arma::colvec& theta, arma::colvec& score)
  {
    model_.setUnknownParameters(theta);
    model_.evaluateScore(score);
    model_.addGradLogPriorDensity(score);
  }
  /// Evaluates the log of the likelihood of the parameters given the latent 
  /// variables.
  double evaluateLogCompleteLikelihood(const arma::colvec& theta, const LatentPath& latentPath, const double inverseTemperature)
  {
    model_.setUnknownParameters(theta);
    model_.setInverseTemperature(inverseTemperature);
    return model_.evaluateLogCompleteLikelihood(latentPath);
  }
  /// Evaluates the log of the likelihood of the parameters given the latent 
  /// variables using a (partially) non-centred parametrisation.
  double evaluateLogCompleteLikelihoodRepar(const arma::colvec& theta, const LatentPathRepar& latentPathRepar, const double inverseTemperature)
  {
    model_.setUnknownParameters(theta);
    model_.setInverseTemperature(inverseTemperature);
    return model_.evaluateLogCompleteLikelihoodRepar(latentPathRepar);
  }
  /// Evaluates the log-prior density.
  double evaluateLogPriorDensity(arma::colvec theta)
  {
    model_.setUnknownParameters(theta);
    return model_.evaluateLogPriorDensity();
  }
  /// Reparametrises latent variables from the standard (centred) parametrisation
  /// to a (partially) non-centred parametrisation.
  void convertLatentPathToLatentPathRepar(const arma::colvec& theta, const LatentPath& latentPath, LatentPathRepar& latentPathRepar);
  /// Reparametrises latent variables from (partially) non-centred parametrisation
  /// to the standard (centred) parametrisation
  void convertLatentPathReparToLatentPath(const arma::colvec& theta, LatentPath& latentPath, const LatentPathRepar& latentPathRepar);
  
  Rng& rng_; // Class for handling random number generation
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // needs to hold the observations and to provide a number of model-related functions.
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc_; //  class for performing MCMC updates
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>& smc_; // class for performing SMC and importance sampling
  OptimUpperType upper_; // type of upper_-level Monte Carlo scheme
  OptimLowerType lower_; // type of lower_-level Monte Carlo scheme
  
  unsigned int nIterations_; // number of MCMC iterations/(upper_-level) SMC-sampler steps
  double betaMin_, betaMax_; // minimum and maximum inverse temperature
  arma::colvec beta_; // vector of inverse temperatures
  bool areInverseTemperaturesIntegers_; // are we only using integer-valued inverse temperatures?

  unsigned int nParticlesUpper_; // number of upper-level particles
  arma::uvec nParticlesLower_; // (nIterations_, 1)-vector containing the number of lower-level particles
  double essResamplingThreshold_; // proportion of ESS/nParticlesUpper below which resampling is triggered (for upper-level SMC).
  
  std::vector<arma::colvec> thetaFullMcmc_; // length-nIterations_ vector of stored parameter values (for upper-level MCMC)
  std::vector<std::vector<arma::colvec> > thetaFullSmc_; // (length(opt.theta), nParticlesUpper_, nIterations_)-matrix of stored parameter values (for upper-level SMC)
  
  // Auxiliary parameters used at each iteration/step of the MCMC algorithm/SMC sampler:
  double betaNew_, betaOld_, betaSharpNew_, betaSharpOld_;
  unsigned int betaCeilingNew_, betaCeilingOld_; // ceiling of inverse temperatures
  unsigned int nParticlesNew_, nParticlesOld_; // number of lower-level particles
  arma::colvec exponentsNew_, exponentsOld_;

  /// Number of applications of the MH kernel for updating par 
  /// conditional on the latent variables in (pseudo-)Gibbs samplers:
  unsigned int nThetaUpdates_; 
  
  /// For correlated pseudo-marginal (CPM) kernels: proportion of iterations 
  /// that use CPM updates (as opposed to PMMH updates):
  double proportionCorrelated_; 
  
  /// Should (peudo-)Gibbs samplers should make use of a non-centred parametrisation?
  bool useNonCentredParametrisation_;
  
  /// Probability of switching to the non-centred parametrisation (if useNonCentredParametrisation == true):
  double nonCentringProbability_;
  
  bool storeHistory_; // should all the parameter estimates be saved? (currently always set to TRUE)
  unsigned int nCores_; // number of cores to use (currently unused)
  
};


///////////////////////////////////////////////////////////////////////////////
// Public functions
///////////////////////////////////////////////////////////////////////////////

/// Runs an MCMC-based optimisation algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::runMcmc() // TODO
{
  std::cout << "set auxiliary parameters" << std::endl;
  setAuxiliaryParameters(0);
  Opt<LatentPath, Aux> opt;
  
  std::cout << "initialise" << std::endl;
  initialise(opt);

  std::cout << "storeHistory" << std::endl;
  if (storeHistory_)
  {
    thetaFullMcmc_.resize(nIterations_);
    thetaFullMcmc_[0] = opt.theta;
  }
  
  
  
  for (unsigned int g=1; g<nIterations_; g++)
  {
    std::cout << "##################### Iteration " << g << " of MCMC algorithm with kernel " << lower_ << "##################" << std::endl;
    
    setAuxiliaryParameters(g);

    
//     std::cout << "change" << std::endl;
    change(g, opt);
    
    
//     std::cout << "extend" << std::endl;
    extend(g, opt);
    
    
//     std::cout << "update" << std::endl;
    update(g, opt);   

//     std::cout << "store history" << std::endl;
    if (storeHistory_)
    {
      thetaFullMcmc_[g] = opt.theta;
    }
    
  }
  
  
  
//   std::cout << "end of runMcmc()" << std::endl;
}
/// Runs an SMC-based optimisation algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::runSmc()
{
//   std::cout << "setting auxiliary parameters" << std::endl;
  setAuxiliaryParameters(0);

  // The vectors holding the particles:
//   std::cout << "setting up opt" << std::endl;
  std::vector<Opt<LatentPath, Aux>> opt(nParticlesUpper_); 
  arma::colvec logWeights(nParticlesUpper_);
  
  
//   std::cout << "store history" << std::endl;
  if (storeHistory_)
  {
    thetaFullSmc_.resize(nIterations_);
    for (unsigned int g=0; g<nIterations_; g++)
    {
      thetaFullSmc_[g].resize(nParticlesUpper_);
    }
  }
  
//   std::cout << "runSmc initialise()" << std::endl;
  for (unsigned int m=0; m<nParticlesUpper_; m++)
  {
    initialise(opt[m], logWeights(m));
  }
  
//   std::cout << "finished initialise" << std::endl;
  
  for (unsigned int g=0; g<nIterations_; g++)
  {
    
    std::cout << "Step " << g << " of SMC Algorithm with kernel " << lower_ << std::endl;
    
    if (g > 0)
    {
      setAuxiliaryParameters(g); 
      
          
//    std::cout << "smc change move"<< std::endl;
      for (unsigned int m=0; m<nParticlesUpper_; m++)
      {
        change(g, opt[m], logWeights(m));
      }
      
          
//     std::cout << "smc resample upper" <<std::endl;
      resampleUpper(opt, logWeights);
      
          
//     std::cout << "smc extend"<<std::endl;
      for (unsigned int m=0; m<nParticlesUpper_; m++)
      {
        extend(g, opt[m], logWeights(m));
      }
    }
   
       
//     std::cout << "smc resample upper"<< std::endl;
    resampleUpper(opt, logWeights);
   
        
//    std::cout << "update move" <<std::endl;
    for (unsigned int m=0; m<nParticlesUpper_; m++)
    {
      update(g, opt[m]);
    }
    
//    std::cout << "smc store history" <<std::endl;
    if (storeHistory_)
    { 
      for (unsigned int m=0; m<nParticlesUpper_; m++)
      {
        thetaFullSmc_[g][m] = opt[m].theta;
      }
    }
//    std::cout << "finished smc store history"<< std::endl;
  }
//   std::cout << "finished runSmc()" << std::endl;
}

/// Runs an SMC-based optimisation algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::runAdaptiveSmc()
{
  // TODO: make sure that the "standard" SMC sampler uses adaptive resampling correctly (i.e. make sure
  // the weight updates and evidence approximations are carried out correctly).
  // the only difference with runAdaptiveSmc() is then that the latter uses an adaptive tempering schedule
  
  // NOTE: for the adaptive schedule, all we need is the min. and max. (inverse) temperature
  
  // NOTE: adaptive tempering is more complicated with adaptive tempering: we can only use this to decrease the temperature between two integer temperatures. If the CESS criterium leads to the next integer temperature, we have to select that one. Of course, in the PM-SAME/PM-Gibbs case, the CESS criterium only depends on one of the lower-level SMC algorithms.
  
  // NOTE: this function should allow us to both do Bayesian model selection and optimisation
  
  // NOTE: need to implement importance tempering as in my thesis (can only be used if the minimum temperature is 1!)
  
//   std::cout << "setting auxiliary parameters" << std::endl;
  setAuxiliaryParameters(0); // TODO: need to modify this function

  // The vectors holding the particles:
//   std::cout << "setting up opt" << std::endl;
  std::vector<Opt<LatentPath, Aux>> opt(nParticlesUpper_); 
  arma::colvec logWeights(nParticlesUpper_);
  
  
//   std::cout << "store history" << std::endl;
  if (storeHistory_)
  {
    thetaFullSmc_.resize(nIterations_);
    for (unsigned int g=0; g<nIterations_; g++)
    {
      thetaFullSmc_[g].resize(nParticlesUpper_);
    }
  }
  
//   std::cout << "runSmc initialise()" << std::endl;
  for (unsigned int m=0; m<nParticlesUpper_; m++)
  {
    initialise(opt[m], logWeights(m));
  }
  
//   std::cout << "finished initialise" << std::endl;
  
  for (unsigned int g=0; g<nIterations_; g++) // TODO: needs to be a WHILE loop
  {
    
    std::cout << "Step " << g << " of SMC Algorithm with kernel " << lower_ << std::endl;
   
    if (g > 0) // NOTE: this part will not be active unless we perform optimisation!
    {
      // TODO: potentially adapt temperature here
      
      
      setAuxiliaryParameters(g);  // TODO: needs to be modified
      
          
//    std::cout << "smc change move"<< std::endl;
      for (unsigned int m=0; m<nParticlesUpper_; m++)
      {
        change(g, opt[m], logWeights(m));
      }
      
          
//     std::cout << "smc resample upper" <<std::endl;

      // TODO: need to sort out the weight updates if we make this adaptive!
      resampleUpper(opt, logWeights);
      
          
//     std::cout << "smc extend" <<std::endl;
      for (unsigned int m=0; m<nParticlesUpper_; m++)
      {
        extend(g, opt[m], logWeights(m));
      }
    }
   
       
//     std::cout << "smc resample upper"<< std::endl;
      // TODO: need to sort out the weight updates if we make this adaptive!
    resampleUpper(opt, logWeights);
   
        
//    std::cout << "update move" <<std::endl;
    for (unsigned int m=0; m<nParticlesUpper_; m++)
    {
      update(g, opt[m]);
    }
    
//    std::cout << "smc store history" <<std::endl;
    if (storeHistory_)
    { 
      for (unsigned int m=0; m<nParticlesUpper_; m++)
      {
        thetaFullSmc_[g][m] = opt[m].theta;
      }
    }
//    std::cout << "finished smc store history"<< std::endl;
  }
//   std::cout << "finished runSmc()" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// Private functions
///////////////////////////////////////////////////////////////////////////////

/// Determines some auxiliary variables used at each iteration/step of the 
/// MCMC algorithm/SMC sampler.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::setAuxiliaryParameters(const unsigned int g)
{
  betaNew_        = beta_(g);
  betaSharpNew_   = betaNew_ - floor(betaNew_);
  betaCeilingNew_ = ceil(betaNew_);
  nParticlesNew_  = nParticlesLower_(g);
  exponentsNew_.ones(betaCeilingNew_);

  if (!areInverseTemperaturesIntegers_ && betaSharpNew_ > 0.0)
  {
    exponentsNew_(betaCeilingNew_-1) = betaSharpNew_;
  }

  if (g == 0) 
  {
    betaCeilingOld_ = 0;
    betaOld_        = 0.0;
    nParticlesOld_  = 0;
  }
  else
  {
    betaOld_        = beta_(g-1);
    betaCeilingOld_ = ceil(betaOld_);
    nParticlesOld_  = nParticlesLower_(g-1);
    betaSharpOld_   = betaOld_ - floor(betaOld_);
    exponentsOld_.ones(betaCeilingOld_);
    
    if (!areInverseTemperaturesIntegers_ && betaSharpOld_ > 0.0)
    {
      exponentsOld_(betaCeilingOld_-1) = betaSharpOld_;
    }
  }
}
/// Resamples in the upper_-level SMC sampler.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::resampleUpper
(
  std::vector<Opt<LatentPath, Aux>>& opt,
  arma::colvec& logWeights
)
{
  arma::colvec Weights(nParticlesUpper_);
  arma::uvec parentIndices(nParticlesUpper_);
  std::vector<Opt<LatentPath, Aux>> optOld(nParticlesUpper_); 
  
  /////////////////////////////////////////////////////////////////////////////
  for (unsigned int m=0; m<nParticlesUpper_; m++)
  {
    if (!std::isfinite(logWeights(m)))
    {
      std::cout << "WARNING: logWeights that are NaN have been set to -infinity in the upper-level SMC algorithm!" << std::endl;
      logWeights(m) = - std::numeric_limits<double>::infinity();
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  
  Weights = normaliseWeights(logWeights);
  double ess = 1.0 / arma::dot(Weights, Weights); // effective sample size
  
  if (ess < essResamplingThreshold_*nParticlesUpper_) // checking ESS resampling threshold
  {
    resample::systematic(parentIndices, Weights, nParticlesUpper_);
    logWeights.fill(-log(nParticlesUpper_));
    
    for (unsigned int m=0; m<nParticlesUpper_; m++)
    {
      optOld[m] = opt[parentIndices(m)];
    }
    
    // TODO: maybe there is a more efficient way to do this without
    // copying everything
    opt.swap(optOld);
  }
}
/// Initialises the initial state of the MCMC chain.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::initialise
(
  Opt<LatentPath, Aux>& opt
)
{
  // Resizing the vectors that store the auxiliary variables:
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME || 
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED || 
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY || 
      lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME ||
      lower_ == OPTIM_LOWER_GIBBS_SAME)
  {
    opt.resize(model_.getDimTheta(), ceil(betaMax_));
  }
  else
  {
    opt.resize(model_.getDimTheta(), 1);
  }
  
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED || 
      lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
  {
    // Here, we need to first sample the auxiliary Gaussian variables.
    smc_.setDetermineParticlesFromGaussians(false); 
  }
  
  model_.sampleFromPrior(opt.theta);
  
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME || 
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
  {
    for (unsigned int k=0; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, 1.0);
    }
  }
//   else if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
//   {
//     
//     for (unsigned int k=0; k<betaCeilingNew_; k++)
//     {
//       //opt.aux[k].resizeOuter(smc_.getNSteps());
//       runSmcLower(opt, k, 1.0);
//     }
//   }
  else if (lower_ == OPTIM_LOWER_RUBENTHALER || 
           lower_ == OPTIM_LOWER_RUBENTHALER_NOISY ||
           lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
  {
    runSmcLower(opt, 0, 1.0);
  }
//   else if (lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
//   {
//     //opt.aux[0].resizeOuter(smc_.getNSteps());
//     runSmcLower(opt, 0, 1.0);
//   }
  else if (lower_ == OPTIM_LOWER_SIMULATED_ANNEALING)
  {
    opt.logLikelihoods[0] = evaluateLogMarginalLikelihood(opt.theta);
    if (mcmc_.getUseGradients())
    {
      evaluateGradient(opt.theta, opt.gradients[0]);
    }
  }
  else if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
  {
    for (unsigned int k=0; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, exponentsNew_(k));
    }
  } 
  else if (lower_ == OPTIM_LOWER_GIBBS_SAME)
  { 
    for (unsigned int k=0; k<betaCeilingNew_; k++)
    {
      runGibbs(opt, k, exponentsNew_(k));
    }
  }
}
/// Initialises the initial state of the MCMC chain/
/// SMC sampler and also calculates the weights.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::initialise
(
  Opt<LatentPath, Aux>& opt,
  double& logWeight
)
{
  // Resizing the vectors that store the auxiliary variables:
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED || 
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY || 
      lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME ||
      lower_ == OPTIM_LOWER_GIBBS_SAME)
  {
    opt.resize(model_.getDimTheta(), ceil(betaMax_));
  }
  else
  {
    opt.resize(model_.getDimTheta(), 1);
  }
  
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED || 
      lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
  {
    // Here, we need to first sample the auxiliary Gaussian variables.
    smc_.setDetermineParticlesFromGaussians(false); 
  }
  
  model_.sampleFromPrior(opt.theta);
  
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
  {
    for (unsigned int k=0; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, 1.0);
    }
    logWeight = opt.sumLogLikelihoods(exponentsNew_);
  }
//   else if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
//   {
//     for (unsigned int k=0; k<betaCeilingNew_; k++)
//     {
//       opt.aux[k].resizeOuter(smc_.getNSteps());
//       runSmcLower(opt, k, 1.0);
//     }
//     logWeight = opt.sumLogLikelihoods(exponentsNew_);
//   }
  else if (lower_ == OPTIM_LOWER_RUBENTHALER ||
           lower_ == OPTIM_LOWER_RUBENTHALER_NOISY ||
           lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
  {
    runSmcLower(opt, 0, 1.0);
    logWeight = betaNew_ * opt.logLikelihoods[0];
  }
//   else if (lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
//   {
//     opt.aux[0].resizeOuter(smc_.getNSteps());
//     runSmcLower(opt, 0, 1.0);
//     logWeight = betaNew_ * opt.logLikelihoods[0];
//   }
  else if (lower_ == OPTIM_LOWER_SIMULATED_ANNEALING)
  {
    opt.logLikelihoods[0] = evaluateLogMarginalLikelihood(opt.theta);
    logWeight = betaNew_ * opt.logLikelihoods[0];
    
    if (mcmc_.getUseGradients())
    {
      evaluateGradient(opt.theta, opt.gradients[0]);
    }
  }
  else if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME || 
           lower_ == OPTIM_LOWER_GIBBS_SAME)
  {
    logWeight = 0;
    for (unsigned int k=0; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, exponentsNew_(k));
      logWeight += logWeight = opt.sumLogLikelihoods(exponentsNew_);
    }
  }
}
/// Performs a change move.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::change
(
  const unsigned int g, 
  Opt<LatentPath, Aux>& opt
)
{
  if (nParticlesNew_ > nParticlesOld_)
  {
//     std::cout << "performing a change move" << std::endl;
    if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME || 
        lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED || 
        lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY)
    {
      //NOTE: if nParticlesUpper_ changes regularly then this will non-negligibly increase the computational cost!
      //if (floor(betaOld_) > 0)
      //{
      for (unsigned int k=0; k<betaCeilingOld_; k++)
      {
        runCsmcLower(opt, k, 1.0);
      }       
      //}     
    }
    else if (lower_ == OPTIM_LOWER_RUBENTHALER || 
             lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)
    {
      runSmcLower(opt, 0, 1.0); // TODO: check whether we really need this...
    }
    else if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
    {
      /*
      for (unsigned int k=0; k<betaCeilingOld_; k++)
      {
        opt.logLikelihoods[k] = smc_.runCsmc(nParticlesNew_, opt.theta, opt.latentPath[k], exponentsOld_(k));
      }
      */
    }   
  }
}
/// Performs a change move and also
/// computes the associated incremental weight.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::change
(
  const unsigned int g, 
  Opt<LatentPath, Aux>& opt,
  double& logWeight
)
{            
  if (nParticlesNew_ > nParticlesOld_)
  {   
    if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME ||
        lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED || 
        lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY)
    {
      /// NOTE: if nParticlesUpper_ changes regularly then this will non-negligibly increase the computational cost!
      //if (floor(betaOld_) > 0)
      //{
      for (unsigned int k=0; k<floor(betaOld_); k++)
      {
        runCsmcLower(opt, k, 1.0);
      }
      //}
      
      if (floor(betaOld_) < betaCeilingOld_)
      {
        // In this case, the incremental weight is the ratio of
        // the new and old final normalising constant.
        logWeight -= betaSharpOld_ * opt.logLikelihoods[betaCeilingOld_-1];
        runSmcLower(opt, betaCeilingOld_-1, 1.0);
        logWeight += betaSharpOld_ * opt.logLikelihoods[betaCeilingOld_-1];
      }
    }
    else if (lower_ == OPTIM_LOWER_RUBENTHALER ||
             lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED ||
             lower_ == OPTIM_LOWER_RUBENTHALER_NOISY)
    {
      logWeight -= betaOld_ * opt.logLikelihoods[0];
      runSmcLower(opt, 0, 1.0);
      logWeight += betaOld_ * opt.logLikelihoods[0];
    }
    else if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
    {
      // We don't need this!
      /*
      for (unsigned int k=0; k<betaCeilingOld_; k++)
      {
        runCsmcLower(opt, k, exponentsOld_(k));
      }
      */
    }
  }
}
/// Performs an extend move.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::extend
(
  const unsigned int g, 
  Opt<LatentPath, Aux>& opt
)
{
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
  {
    for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, 1.0);
    }
  }
//   else if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED) 
//   {
//     for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
//     {
//       opt.aux[k].resizeOuter(smc_.getNSteps());
//       runSmcLower(opt, k, 1.0);
//     }
//   }
  else if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
  {
    for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, exponentsNew_(k));
    }
  }
  else if (lower_ == OPTIM_LOWER_GIBBS_SAME)
  {  
    for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
    {
      runGibbs(opt, k, exponentsNew_(k));
    }
  }
}
/// Performs an extend move and also
/// computes the associated incremental weight.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::extend
(
  const unsigned int g, 
  Opt<LatentPath, Aux>& opt,
  double& logWeight
)
{
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY ||
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
  {
    logWeight -= opt.sumLogLikelihoods(exponentsOld_);
    for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, 1.0);
    }
    logWeight += opt.sumLogLikelihoods(exponentsNew_);
  }
//   else if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
//   {
//     logWeight -= opt.sumLogLikelihoods(exponentsOld_);
//     for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
//     {
//       //opt.aux[k].resizeOuter(smc_.getNSteps());
//       runSmcLower(opt, k, 1.0);
//     }
//     logWeight += opt.sumLogLikelihoods(exponentsNew_);
//   }
  else if (lower_ == OPTIM_LOWER_RUBENTHALER ||
           lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED ||
           lower_ == OPTIM_LOWER_RUBENTHALER_NOISY ||
           lower_ == OPTIM_LOWER_SIMULATED_ANNEALING)
  {
    logWeight += (betaNew_ - betaOld_) * arma::as_scalar(opt.logLikelihoods[0]);
  }
  /*
  else if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
  {
    if (floor(betaOld_) < betaCeilingOld_) // i.e. if the previous inverse temperature was not an integer
    {
      logWeight -= evaluateLogCompleteLikelihood(opt.theta, opt.latentPath[betaCeilingOld_-1], exponentsOld_(betaCeilingOld_-1));
      logWeight += evaluateLogCompleteLikelihood(opt.theta, opt.latentPath[betaCeilingOld_-1], exponentsNew_(betaCeilingOld_-1));
    }
    for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
    {
      runSmcLower(opt, k, exponentsNew_(k));
      logWeight += opt.logLikelihoods[k];
    }
  }
  */
  else if (lower_ == OPTIM_LOWER_GIBBS_SAME || lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
  {  
    // TODO: some redundant calculations here!
    logWeight -= evaluateLogCompleteLikelihood(opt.theta, opt.latentPath[betaCeilingOld_-1], exponentsOld_(betaCeilingOld_-1));
    logWeight += evaluateLogCompleteLikelihood(opt.theta, opt.latentPath[betaCeilingOld_-1], exponentsNew_(betaCeilingOld_-1));
    for (unsigned int k=betaCeilingOld_; k<betaCeilingNew_; k++)
    {
      //runGibbs(opt, k, exponentsNew_(k));
      runSmcLower(opt, k, exponentsNew_(k));
      logWeight += opt.logLikelihoods[k];
    }
  }
}
/// Performs an update move for the pseudo-marginal SAME algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::update
(
  const unsigned int g, 
  Opt<LatentPath, Aux>& opt
)
{
  double logAlpha = 0;
 
  if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME || 
      lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_CORRELATED)
  {     
    
    Opt<LatentPath, Aux> optProp(model_.getDimTheta(), opt.logLikelihoods.size()); // NOTE: the vectors in optProp are longer than needed! maybe grow dynamically!
    
    mcmc_.sampleProposalScale(1.0 / std::sqrt(betaNew_));
    if (mcmc_.getUseGradients())
    {
      arma::colvec gradient = opt.sumGradients(exponentsNew_);
      mcmc_.proposeTheta(optProp.theta, opt.theta, gradient);
      logAlpha = - mcmc_.evaluateLogProposalDensity(optProp.theta, opt.theta, gradient);
    }
    else
    {
      mcmc_.proposeTheta(optProp.theta, opt.theta);
      logAlpha = - mcmc_.evaluateLogProposalDensity(optProp.theta, opt.theta);
    }
    
    logAlpha += evaluateLogPriorDensity(optProp.theta) -
                evaluateLogPriorDensity(opt.theta);

    if (std::isfinite(logAlpha))
    {
      if (lower_== OPTIM_LOWER_PSEUDO_MARGINAL_SAME || 
          g <= (1.0 - proportionCorrelated_) * nIterations_)
      {
        for (unsigned int k = 0; k<betaCeilingNew_; k++)
        {
          runSmcLower(optProp, k, 1.0);
        }
      }
      else
      {
        smc_.setDetermineParticlesFromGaussians(true);
        for (unsigned int k = 0; k<betaCeilingNew_; k++)
        {
          // TODO: make this more efficient
          optProp.aux[k] = opt.aux[k];
          optProp.aux[k].addCorrelatedGaussianNoise(mcmc_.getCrankNicolsonScale(nParticlesNew_, betaNew_));
         
          runSmcLower(optProp, k, 1.0);
        }
        smc_.setDetermineParticlesFromGaussians(false);
      }

      logAlpha += optProp.sumLogLikelihoods(exponentsNew_) - opt.sumLogLikelihoods(exponentsNew_);
            
      /////////////////////////////////////////////////
        std::cout << "logLikeNum: " << optProp.sumLogLikelihoods(exponentsNew_) << "; logLikeDen: " << opt.sumLogLikelihoods(exponentsNew_) << std::endl;
      /////////////////////////////////////////////////
      

      if (mcmc_.getUseGradients())
      {
        logAlpha += mcmc_.evaluateLogProposalDensity(opt.theta, optProp.theta, optProp.sumGradients(exponentsNew_));
      }
      else
      {
        logAlpha += mcmc_.evaluateLogProposalDensity(opt.theta, optProp.theta);
      }

    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }
    
    std::cout << "logAlpha after logProposalDensity: " << logAlpha << std::endl;
          
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      std::cout << "################### ACCEPTANCE ###################" << std::endl;
      opt = optProp;
    }
  }
  else if (lower_ == OPTIM_LOWER_PSEUDO_MARGINAL_SAME_NOISY)
  {  
    arma::colvec logLikeNum(betaCeilingNew_);
    arma::colvec logLikeDen(betaCeilingNew_);
    std::vector<LatentPath> latNum(betaCeilingNew_);
    std::vector<LatentPath> latDen(betaCeilingNew_);
    
    arma::colvec thetaProp;
    mcmc_.sampleProposalScale(1.0 / std::sqrt(betaNew_));
    mcmc_.proposeTheta(thetaProp, opt.theta);   
    logAlpha = mcmc_.computeLogAlpha(thetaProp, opt.theta);
    
    if (std::isfinite(logAlpha))
    {
      for (unsigned int k = 0; k<betaCeilingNew_; k++)
      {
        logLikeNum(k) = smc_.runSmc(nParticlesNew_, thetaProp, latNum[k], opt.aux[k], 1.0);
        logLikeDen(k) = smc_.runSmc(nParticlesNew_, opt.theta, latDen[k], opt.aux[k], 1.0);
      }
      logAlpha += arma::accu(exponentsNew_ % (logLikeNum - logLikeDen));
    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }
    
    /////////////////////////////////////////////////
    
    std::cout << "logLikeNum: " << arma::accu(exponentsNew_ % logLikeNum) << "; logLikeDen: " << arma::accu(exponentsNew_ % logLikeDen) << "; logAlpha: " << logAlpha << std::endl;
//     std::cout << " is finite? " << std::isfinite(logAlpha) << " " << std::isfinite(arma::accu(exponentsNew_ % (logLikeNum - logLikeDen))) << std::endl;
    /////////////////////////////////////////////////
    
    
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      std::cout << "################### ACCEPTANCE ###################" << std::endl;
      opt.theta = thetaProp;  
      for (unsigned int k=0; k<betaCeilingNew_; k++) // TODO: can this be made more efficient?
      {
        opt.logLikelihoods[k] = logLikeNum(k);
        opt.latentPath[k] = latNum[k];
      }
    }
  }
  else if (lower_ == OPTIM_LOWER_RUBENTHALER || 
           lower_ == OPTIM_LOWER_RUBENTHALER_CORRELATED)         
  {
    Opt<LatentPath, Aux> optProp(opt.dimTheta(), 1); // NOTE: the vectors in optProp are longer than needed! this might be inefficient
    mcmc_.sampleProposalScale(1.0 / std::sqrt(betaNew_));
    
    if (mcmc_.getUseGradients())
    {
      mcmc_.proposeTheta(optProp.theta, opt.theta, opt.gradients[0]);
      logAlpha = - mcmc_.evaluateLogProposalDensity(optProp.theta, opt.theta, opt.gradients[0]);
    }
    else
    {
      mcmc_.proposeTheta(optProp.theta, opt.theta);
      logAlpha = - mcmc_.evaluateLogProposalDensity(optProp.theta, opt.theta);
    }
    logAlpha += evaluateLogPriorDensity(optProp.theta) - evaluateLogPriorDensity(opt.theta);

    if (std::isfinite(logAlpha))
    {
      if (lower_== OPTIM_LOWER_RUBENTHALER || 
          g <= (1.0 - proportionCorrelated_) * nIterations_)
      {
        runSmcLower(optProp, 0, 1.0);
      }
      else
      {
        smc_.setDetermineParticlesFromGaussians(true);
        optProp.aux[0] = opt.aux[0];
        optProp.aux[0].addCorrelatedGaussianNoise(mcmc_.getCrankNicolsonScale(nParticlesNew_, betaNew_));
        runSmcLower(optProp, 0, 1.0);
        smc_.setDetermineParticlesFromGaussians(false);
      }
      logAlpha += betaNew_ * (optProp.logLikelihoods[0] - opt.logLikelihoods[0]);
      if (mcmc_.getUseGradients())
      {
        logAlpha += mcmc_.evaluateLogProposalDensity(opt.theta, optProp.theta, optProp.gradients[0]);
      }
      else
      {
        logAlpha += mcmc_.evaluateLogProposalDensity(opt.theta, optProp.theta);
      }
    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }
    std::cout << "logAlpha: " << logAlpha << std::endl;
              
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      std::cout << "################### ACCEPTANCE ###################" << std::endl;
      opt = optProp;
    }
  }
  else if (lower_ == OPTIM_LOWER_RUBENTHALER_NOISY)        
  { 
    double logLikeNum = 0;
    double logLikeDen = 0;
    
    arma::colvec thetaProp;
    mcmc_.sampleProposalScale(1.0 / std::sqrt(betaNew_));
    mcmc_.proposeTheta(thetaProp, opt.theta);
    logAlpha = mcmc_.computeLogAlpha(thetaProp, opt.theta);

    if (std::isfinite(logAlpha))
    {
      logLikeNum = smc_.runSmc(nParticlesNew_, thetaProp, opt.latentPath[0], opt.aux[0], 1.0);
      logLikeDen = smc_.runSmc(nParticlesNew_, opt.theta, opt.latentPath[0], opt.aux[0], 1.0);
      logAlpha += betaNew_ * (logLikeNum - logLikeDen);
    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }  
    
    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      std::cout << "################### ACCEPTANCE ###################" << std::endl;
      opt.theta = thetaProp;
      opt.logLikelihoods[0] = logLikeNum;
    }
  }
  else if (lower_ == OPTIM_LOWER_SIMULATED_ANNEALING)
  {
    Opt<LatentPath, Aux> optProp(opt.dimTheta(), 1);
    optProp.logLikelihoods[0] = 0;
    mcmc_.sampleProposalScale(1.0 / std::sqrt(betaNew_));
    
    if (mcmc_.getUseGradients())
    {
      mcmc_.proposeTheta(optProp.theta, opt.theta, betaNew_*opt.gradients[0]);
      evaluateGradient(optProp.theta, optProp.gradients[0]);
      logAlpha = mcmc_.evaluateLogProposalDensity(opt.theta, optProp.theta, optProp.gradients[0]) -
                 mcmc_.evaluateLogProposalDensity(optProp.theta, opt.theta, opt.gradients[0]);
    }
    else
    {
      mcmc_.proposeTheta(optProp.theta, opt.theta);
      logAlpha = mcmc_.evaluateLogProposalDensity(opt.theta, optProp.theta) -
                 mcmc_.evaluateLogProposalDensity(optProp.theta, opt.theta);
    }
    logAlpha += evaluateLogPriorDensity(optProp.theta) - evaluateLogPriorDensity(opt.theta);

    if (std::isfinite(logAlpha))
    {
      optProp.logLikelihoods[0] = evaluateLogMarginalLikelihood(optProp.theta);
      logAlpha += betaNew_ * (optProp.logLikelihoods[0] - opt.logLikelihoods[0]);
    }
    else
    {
      std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
    }  

    if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
    {
      std::cout << "################### ACCEPTANCE ###################" << std::endl;
      opt = optProp;
    }
  }
  else if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME ||
           lower_ == OPTIM_LOWER_GIBBS_SAME) // i.e. if we use one of the (pseudo-)Gibbs kernels
  {
    
    arma::colvec thetaProp;
    arma::colvec logCompleteLikeNum(betaCeilingNew_);
    arma::colvec logCompleteLikeDen(betaCeilingNew_);

    if (useNonCentredParametrisation_ && arma::randu() < nonCentringProbability_) // i.e. use non-centred parametrisation
    {
      
      std::vector<LatentPathRepar> latentPathRepar(betaCeilingNew_);
      
      for (unsigned int k=0; k<betaCeilingNew_; k++)
      {
        // Convert to non-centred parametrisation.
        convertLatentPathToLatentPathRepar(opt.theta, opt.latentPath[k], latentPathRepar[k]);
      }
      for (unsigned int k=0; k<betaCeilingNew_; k++)
      {
        logCompleteLikeDen(k) = evaluateLogCompleteLikelihoodRepar(opt.theta, latentPathRepar[k], exponentsNew_(k));
      }

      for (unsigned int n=0; n<nThetaUpdates_; n++)
      {
        // Adjusting the scale of the proposal kernel.
        mcmc_.sampleProposalScale(1.0 / std::sqrt(betaNew_));
        
        // Propose parameters.
        mcmc_.proposeTheta(thetaProp, opt.theta);
    
        logAlpha = mcmc_.computeLogAlpha(thetaProp, opt.theta);
        
        if (std::isfinite(logAlpha))
        {
          for (unsigned int k=0; k<betaCeilingNew_; k++)
          {
            logCompleteLikeNum(k) = evaluateLogCompleteLikelihoodRepar(thetaProp, latentPathRepar[k], exponentsNew_(k));
          }
          logAlpha += arma::accu(logCompleteLikeNum - logCompleteLikeDen);
        }
        else
        {
          std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
        }  
        
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
                    ////////////////////////////
//           std::cout << "===================ACCEPTANCE=================="<< std::endl;
//           std::cout << "logAlpha: " << logAlpha << std::endl;
//           std::cout << "thetaProp: " << thetaProp.t() << std::endl;
          ///////////////////////////////
          opt.theta = thetaProp;
          logCompleteLikeDen = logCompleteLikeNum;
        }
      }
      for (unsigned int k=0; k<betaCeilingNew_; k++)
      {
        // Convert back to centred parametrisation.
        convertLatentPathReparToLatentPath(opt.theta, opt.latentPath[k], latentPathRepar[k]);
      }

      if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
      {
        for (unsigned int k=0; k<betaCeilingNew_; k++)
        {
//           std::cout << "started runCsmc" << std::endl;
          opt.logLikelihoods[k] = smc_.runCsmc(nParticlesNew_, opt.theta, opt.latentPath[k], opt.aux[k], exponentsNew_(k));
//              std::cout << "started runCsmc" << std::endl;
        }
      }
      else // i.e. if we use the exact Gibbs update for the latent variables
      {
        for (unsigned int k=0; k<betaCeilingNew_; k++)
        {
          runGibbs(opt, k, exponentsNew_(k));
        }
      }
    }
    else // i.e. use standard parametrisation
    {
      for (unsigned int k=0; k<betaCeilingNew_; k++)
      {
        logCompleteLikeDen(k) = evaluateLogCompleteLikelihood(opt.theta, opt.latentPath[k], exponentsNew_(k));
      }
      
      /////////////////
//             std::cout << "theta before parameter update: " << std::endl;
//             std::cout << opt.theta.t() << std::endl;
//       std::cout << "latent path before parameter update: " << std::endl;
//          std::cout << opt.latentPath[0].t() << std::endl;
      ////////////////
        
      for (unsigned int n=0; n<nThetaUpdates_; n++)
      {
        // Adjusting the scale of the proposal kernel.
        mcmc_.sampleProposalScale(1.0 / std::sqrt(betaNew_));
        
        // Propose parameters.
        mcmc_.proposeTheta(thetaProp, opt.theta);
    
        logAlpha = mcmc_.computeLogAlpha(thetaProp, opt.theta);
        
        if (std::isfinite(logAlpha))
        {
          for (unsigned int k=0; k<betaCeilingNew_; k++)
          {
            logCompleteLikeNum(k) = evaluateLogCompleteLikelihood(thetaProp, opt.latentPath[k], exponentsNew_(k));
          }
          logAlpha += arma::accu(logCompleteLikeNum - logCompleteLikeDen);
        }
        else
        {
          std::cout << "--------- WARNING: skipped logLikeNum  --------- " << logAlpha << std::endl;  
        }  
        
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
          //////////////////////////////
//           std::cout << "===================ACCEPTANCE=================="<< std::endl;
//           std::cout << "thetaProp: " << thetaProp.t() << std::endl;
          ///////////////////////////////
          opt.theta = thetaProp;
          logCompleteLikeDen = logCompleteLikeNum;
        }
      }
      
        /////////////////
//                     std::cout << "theta after parameter update: " << std::endl;
//             std::cout << opt.theta.t() << std::endl;
//       std::cout << "latent path after parameter update: " << std::endl;
//          std::cout << opt.latentPath[0].t() << std::endl;
      ////////////////
      
      if (lower_ == OPTIM_LOWER_PSEUDO_GIBBS_SAME)
      {
        for (unsigned int k=0; k<betaCeilingNew_; k++)
        {
          opt.logLikelihoods[k] = smc_.runCsmc(nParticlesNew_, opt.theta, opt.latentPath[k], opt.aux[k], exponentsNew_(k));
        }
      }
      else // i.e. if we use the exact Gibbs update for the latent variables
      {
        for (unsigned int k=0; k<betaCeilingNew_; k++)
        {
          runGibbs(opt, k, exponentsNew_(k));
        }
      }
    }
  }
}
/// Runs an MCMC-based optimisation algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void optimMcmc
(
  std::vector<arma::colvec>& output,         // parameter values obtained from the algorithm
  const unsigned int lower,                  // type of lower-level Monte Carlo algorithm to use
  const arma::colvec& beta,                  // vector of inverse temperatures
  const bool areInverseTemperaturesIntegers, // are we only considering integer-valued inverse temperatures?
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperparameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (theta.size(), 2)-matrix containing the bound of the support of each parameter
  const Observations& observations,          // observations
  const double proposalDownScaleProbability, // probability of using an RWMH proposal whose scale decreases in the inverse temperature
  const unsigned int kern,                   // type of proposal kernel for the random-walk Metropolis--Hastings kernels
  const bool useGradients,                   // are we using gradient information in the parameter proposals?
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const unsigned int fixedLagSmoothingOrder, // lag-order for fixed-lag smoothing (currently only used to approximate gradients).
  const double crankNicolsonScaleParameter,  // correlation parameter for Crank--Nicolson proposals
  const double proportionCorrelated,         // for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  const unsigned int smcProposalType,        // type of proposal kernel within the lower-level SMC sampler
  const unsigned int nThetaUpdates,          // number of parameter updates per iteration of Gibbs samplers of CSMC-based algorithms
  const bool onlyTemperObservationDensity,   // should only the observation densities be tempered?
  const unsigned int nSmcStepsLower,         // number of lower-level SMC steps
  const arma::uvec& nParticlesLower,         // number of particles per MCMC iteration within each lower-level SMC algorithm
  const double essResamplingThresholdLower,  // ESS-based resampling threshold for the lower-level SMC algorithms
  const unsigned int smcBackwardSamplingType,   // type of backward-sampling scheme to use with the lower-level conditional SMC kernels
  const bool useNonCentredParametrisation,   // should Gibbs-sampling type algorithms use an NCP?
  const double nonCentringProbability,       // probability of using an NCP (if useNonCentredParametrisation == true)
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
  // Class for dealing with random number generation.
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperparameters, observations, nCores);
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
  // Class for running MCMC algorithms.
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, 
    static_cast<McmcKernelType>(kern), 
    useGradients,
    proposalDownScaleProbability, 
    rwmhSd, crankNicolsonScaleParameter, nCores
  );
  
 
  // Class for running SMC algorithms.
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSmcStepsLower,
    static_cast<SmcProposalType>(smcProposalType), 
    essResamplingThresholdLower, 
    static_cast<SmcBackwardSamplingType>(smcBackwardSamplingType),
    useGradients,
    fixedLagSmoothingOrder,
    nCores
  );
  
 
  // Optimisation algorithms.
  Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters> optim(
    rngDerived, model, mcmc, smc,
    static_cast<OptimLowerType>(lower),
    beta, 
    areInverseTemperaturesIntegers, 
    nParticlesLower,
    nThetaUpdates, proportionCorrelated, nCores
  );
  optim.setUseNonCentredParametrisation(useNonCentredParametrisation);
  optim.setNonCentringProbability(nonCentringProbability);
  
  optim.runMcmc();
  optim.getThetaFullMcmc(output);
}
/// Runs an SMC-based optimisation algorithm.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void optimSmc
(
  std::vector< std::vector<arma::colvec> >& output,   // parameter values obtained from the algorithm
  const unsigned int lower,                  // type of lower-level Monte Carlo algorithm to use
  const arma::colvec& beta,                  // vector of inverse temperatures
  const bool areInverseTemperaturesIntegers, // are we only considering integer-valued inverse temperatures?
  const unsigned int dimTheta,               // length of the parameter vector
  const arma::colvec& hyperparameters,       // hyperparameters and other auxiliary model parameters
  const arma::mat& support,                  // (theta.size(), 2)-matrix containing the bound of the support of each parameter
  const Observations& observations,          // observations
  const double proposalDownScaleProbability, // probability of using an RWMH proposal whose scale decreases in the inverse temperature
  const unsigned int kern,                   // type of proposal kernel for the random-walk Metropolis--Hastings kernels
  const bool useGradients,                   // are we using gradient information in the parameter proposals?
  const arma::colvec& rwmhSd,                // scaling of the random-walk Metropolis--Hastings proposals
  const unsigned int fixedLagSmoothingOrder, // lag-order for fixed-lag smoothing (currently only used to approximate gradients).
  const double crankNicolsonScaleParameter,  // correlation parameter for Crank--Nicolson proposals
  const double proportionCorrelated,         // for correlated pseudo-marginal (CPM) kernels: proportion of iterations that use CPM updates (as opposed to PMMH updates). 
  const unsigned int smcProposalType,        // type of proposal kernel within the lower-level SMC sampler
  const unsigned int nThetaUpdates,          // number of parameter updates per iteration of Gibbs samplers of CSMC-based algorithms
  const bool onlyTemperObservationDensity,   // should only the observation densities be tempered?
  const unsigned int nSmcStepsLower,         // number of lower-level SMC steps
  const arma::uvec& nParticlesLower,         // number of particles per MCMC iteration within each lower-level SMC algorithm
  const double essResamplingThresholdLower,  // ESS-based resampling threshold for the lower-level SMC algorithms
  const unsigned int smcBackwardSamplingType,   // type of backward-sampling scheme to use with the lower-level conditional SMC kernels
  const unsigned int nParticlesUpper,        // number of particles in the upper-level SMC sampler
  const double essResamplingThresholdUpper,  // ESS-based resampling threshold for the lower-level SMC algorithms
  const bool useNonCentredParametrisation,   // should Gibbs-sampling type algorithms use an NCP?
  const double nonCentringProbability,       // probability of using an NCP (if useNonCentredParametrisation == true)
  const unsigned int nCores                  // number of nCores used (currently, this is not implemented)
)
{
  // Class for dealing with random number generation.
  std::mt19937 engine; 
  RngDerived<std::mt19937> rngDerived(engine);
  
  // Model class.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations> model(rngDerived, hyperparameters,  observations, nCores); 
  model.setSupport(support);
  model.setDimTheta(dimTheta);
  
  // Class for running MCMC algorithms.
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters> mcmc(
    rngDerived, model, 
    static_cast<McmcKernelType>(kern),
    useGradients,
    proposalDownScaleProbability, 
    rwmhSd, crankNicolsonScaleParameter, nCores
  );
  
 
     
  // Class for running SMC algorithms.
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters> smc(
    rngDerived, model, nSmcStepsLower,
    static_cast<SmcProposalType>(smcProposalType), 
    essResamplingThresholdLower,
    static_cast<SmcBackwardSamplingType>(smcBackwardSamplingType),
    useGradients,
    fixedLagSmoothingOrder,
    nCores
  );
  

  // Optimisation algorithms.
  Optim<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters> optim(
    rngDerived, model, mcmc, smc,
    static_cast<OptimLowerType>(lower),
    beta, 
    areInverseTemperaturesIntegers, 
    nParticlesUpper, nParticlesLower,
    essResamplingThresholdUpper,
    nThetaUpdates, proportionCorrelated, nCores
  );
  optim.setUseNonCentredParametrisation(useNonCentredParametrisation);
  optim.setNonCentringProbability(nonCentringProbability);
  
  optim.runSmc();
  optim.getThetaFullSmc(output);
  
}

#endif
