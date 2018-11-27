/// \file
/// \brief Running an SMC sampler (e.g. for approximating the model evidence) 
///
/// This file contains the functions associated with the SmcSampler class template.
/// The SMC sampler uses tempering to approximate the model evidence and
/// estimate the posterior distribution of the model parameters.

#ifndef __SMCSAMPLER_H
#define __SMCSAMPLER_H

// #include <time.h> 

#include "main/model/Model.h"
#include "main/algorithms/smc/Smc.h"
// #include "smc/default/single.h"
#include "main/helperFunctions/rootFinding.h"

// [[Rcpp::depends("RcppArmadillo")]]

/// Type of lower-level algorithm.
enum SmcSamplerLowerType 
{ 
  SMC_SAMPLER_LOWER_PSEUDO_MARGINAL = 0, // pseudo-marginal algorithm
  SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_CORRELATED, // correlated pseudo-marginal algorithm
  SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_NOISY, // pseudo-marginal algorithm (noisy)
  SMC_SAMPLER_LOWER_MARGINAL // idealised marginal algorithm
};

/*
// TODO: this should be part of the MCMC class
/// Type of proposal kernel for the lower-level Metropolis--Hastings updates.
/// NOTE: most of these are not implemented!
enum LowerProposalType 
{ 
  LOWER_PROPOSAL_GAUSSIAN_RANDOM_WALK = 0, // Gaussian random-walk proposal
  LOWER_PROPOSAL_GAUSSIAN_INDEPENDENT, // Gaussian independent Metropolis proposal
  LOWER_PROPOSAL_TRUNCATED_GAUSSIAN_RANDOM_WALK, // truncated Gaussian random-walk proposal
  LOWER_PROPOSAL_TRUNCATED_GAUSSIAN_INDEPENDENT, // truncated Gaussian independent Metropolis proposal
  LOWER_PROPOSAL_ADAPTIVE_GAUSSIAN_RANDOM_WALK // see e.g. Gareth Peters et al. (2010)
};
*/

/// Container for holding a single particle of the SMC sampler.
template <class LatentPath, class Aux> class ParticleUpper 
{
public:
  
  /// Initialises the class.
  ParticleUpper(){};
  /// Destructor.
  ~ParticleUpper() {};
  
  /// Initialises the class.
  void initialise(const unsigned int dimTheta)
  {
    theta_.set_size(dimTheta);
    logLikelihoodFirst_  = 0.0; // part of the (marginal) likelihood that can be evaluated analytically
    logLikelihoodSecond_ = 0.0; // part of the marginal likelihood that is approximated using a lower-level SMC algorithm
  }
  /// Returns the length of the parameter vector.
  unsigned int dimTheta() const {return theta_.size();}
  /// Returns the log-likelihood.
  double getlogLikelihood() const {return logLikelihoodFirst_ + logLikelihoodSecond_;}
  /// Returns the part of the log-marginal likelihood that can be evaluated analytically.
  void setlogLikelihoodFirst(const double logLikelihoodFirst) {logLikelihoodFirst_ = logLikelihoodFirst;}
  /// Returns the part of the log-marginal likelihood that is approximated using a lower-level SMC algorithm.
  void setlogLikelihoodSecond(const double logLikelihoodSecond) {logLikelihoodSecond_ = logLikelihoodSecond;}
  
  /// Data members:
  arma::colvec theta_;
  double logLikelihoodFirst_;
  double logLikelihoodSecond_;
  LatentPath latentPath_;
  AuxFull<Aux> aux_;
  arma::colvec gradient_;
  
};



/// Class template for running an SMC sampler.
template<class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class ParticleLower, class Aux, class SmcParameters, class McmcParameters> class SmcSampler
{
public:
  
  /// Initialises the class.
  SmcSampler
  (
    Rng& rng,
    Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model,
    Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, ParticleLower, Aux, SmcParameters>& smc,
    Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc,
    const unsigned int nCores
  ) : 
    rng_(rng), 
    model_(model),
    smc_(smc),
    mcmc_(mcmc),
    nCores_(nCores)
  {
    storeHistory_ = true; // TODO: make this accessible from the outside 
    nMetropolisHastingsUpdates_ = 1;
    useAdaptiveCessTarget_ = false;
  }
  
  /// Returns the estimate of the evidence.
  double getLogEvidenceEstimate() const {return logEvidenceEstimate_;}
  /// Returns the estimate of the evidence based on importance tempering using the 
  /// unconditional(!) effective sample sizes to weight particle sets from different 
  /// steps of the SMC sampler.
  double getLogEvidenceEstimateEss() const {return logEvidenceEstimateEss_;}
  /// Returns the evidence estimate computed by weighting the evidence estimate
  /// from each particle generation according to the ESS.
  double getLogEvidenceEstimateEssAlternate() const {return logEvidenceEstimateEssAlternate_;}
  /// Returns the estimate of the evidence based on importance tempering using the 
  /// effective sample sizes to weight particle sets from different 
  /// steps of the SMC sampler. As in Nguyen, Septier, Peters et al. (2016), an additional
  /// resampling step is applied if resampling is not performed at a particular step
  /// of the SMC sampler.
  double getLogEvidenceEstimateEssResampled() const {return logEvidenceEstimateEssResampled_;}
  /// Returns the evidence estimate computed by weighting the evidence estimate
  /// from each particle generation according to the ESS.
  double getLogEvidenceEstimateEssResampledAlternate() const {return logEvidenceEstimateEssResampledAlternate_;}
  /// Returns the estimate of the evidence based on importance tempering using the 
  /// conditional effective sample sizes to weight particle sets from different 
  /// steps of the SMC sampler.
  double getLogEvidenceEstimateCess() const {return logEvidenceEstimateCess_;}
  /// Returns the evidence estimate computed by weighting the evidence estimate
  /// from each particle generation according to the CESS.
  double getLogEvidenceEstimateCessAlternate() const {return logEvidenceEstimateCessAlternate_;}
  /// Returns the number of SMC steps.
  unsigned int getNSteps() const {return nSteps_;}
  /// Returns the tempering schedule.
  arma::colvec getAlpha() const 
  {
    return arma::conv_to<arma::colvec>::from(alpha_);
  }
  /// Returns the number of particles.
  unsigned int getNParticles() const {return nParticles_;}
  /// Returns all the particles generated by the algorithm.
  std::vector<std::vector<arma::colvec>> getThetaFull() const {return thetaFull_;}
  /// Returns the particle weights.
  arma::mat getSelfNormalisedWeightsFull() const {return selfNormalisedWeightsFull_;}
  /// Returns the log-unnormalised weights.
  arma::mat getLogUnnormalisedWeightsFull() const {return logUnnormalisedWeightsFull_;}
  /// Returns the vector of full particles from the final step of the SMC sampler.
  void getFinalParticles(std::vector<ParticleUpper<LatentPath, Aux>>& finalParticles) {finalParticles = finalParticles_;}
  
  // The following nine functions are only needed for importance tempering schemes:
  
  /// Returns the ESS associated with each particle generation reweighted to target the posterior.
  arma::colvec getEss() const {return ess_;}
  /// Returns the ESS associated with each particle generation reweighted to target the posterior
  /// in the case that the approach from Nguyen et al. (2016) is used.
  arma::colvec getEssResampled() const {return essResampled_;}
  /// Returns the CESS associated with each particle generation reweighted to target the posterior.
  arma::colvec getCess() const {return cess_;}
  /// Returns the log-unnormalised weights re-weighted 
  /// so that particles from each generation target the posterior.
  arma::mat getLogUnnormalisedReweightedWeights() const {return logUnnormalisedReweightedWeights_;}
  /// Returns the log-unnormalised weights (constructed after potentially adding 
  /// another resampling step as in Nguyen et al. (2016)) re-weighted 
  /// so that particles from each generation target the posterior; 
  arma::mat getLogUnnormalisedReweightedWeightsResampled() const {return logUnnormalisedReweightedWeightsResampled_;}
  /// Returns the self-nnormalised weights re-weighted 
  /// so that particles from each generation target the posterior.
  arma::mat getSelfNormalisedReweightedWeights() const {return selfNormalisedReweightedWeights_;}
  /// Returns the self-normalised weights (constructed after potentially adding 
  /// another resampling step as in Nguyen et al. (2016)) re-weighted 
  /// so that particles from each generation target the posterior; 
  arma::mat getSelfNormalisedReweightedWeightsResampled() const {return selfNormalisedReweightedWeightsResampled_;}
  /// Returns the self-normalised weights (for targeting the posterior) weighted by the ESS.
  arma::mat getSelfNormalisedWeightsEss() const {return selfNormalisedWeightsEss_;}
  /// Returns the self-normalised weights for importance tempering as in Nguyen et al. (2016)
  arma::mat getSelfNormalisedWeightsEssResampled() const {return selfNormalisedWeightsEssResampled_;}
  /// Returns the self-normalised weights (for targeting the posterior) weighted by the CESS.
  arma::mat getSelfNormalisedWeightsCess() const {return selfNormalisedWeightsCess_;}
  /// Returns the resampled particles needed by the approach from Nguyen et al. (2016)
  std::vector<std::vector<arma::colvec>> getThetaFullResampled() const {return thetaFullResampled_;}
  /// Returns the overall mean-acceptance rates of the MH updates.
  std::vector<double> getAcceptanceRates() const {return acceptanceRates_;}
  /// Returns the first-stage mean-acceptance rates of the MH updates (if delayed acceptance is used).
  std::vector<double> getAcceptanceRatesFirst() const {return acceptanceRatesFirst_;}
  /// Returns mean-autocorrelations between the particles before and after applying the the MH updates.
  std::vector<double> getMaxParticleAutocorrelations() const {return maxParticleAutocorrelations_;}
  
  /// Specifies the number of particles.
  void setNParticles(const unsigned int nParticles) {nParticles_ = nParticles;}
  /// Specifies the number of MH updates per SMC step.
  void setNMetropolisHastingsUpdates(const unsigned int nMetropolisHastingsUpdates) {nMetropolisHastingsUpdates_ = nMetropolisHastingsUpdates;}
  /// Specifies the number of MH updates per SMC step in the first stage of the SMC sampler when using dual tempering.
  void setNMetropolisHastingsUpdatesFirst(const unsigned int nMetropolisHastingsUpdatesFirst) {nMetropolisHastingsUpdatesFirst_ = nMetropolisHastingsUpdatesFirst;}
  /// Specifies the lower-level algorithm used for approximating (part of) the marginal likelihood.
  void setLower(const SmcSamplerLowerType lower) {lower_ = lower;}
  /// Returns whether the tempering schedule is adaptively determined according to the conditional ESS.
  bool getUseAdaptiveTempering() const {return useAdaptiveTempering_;}
  /// Specifies whether the tempering schedule should be adaptively determined according to the conditional ESS.
  void setUseAdaptiveTempering(const bool useAdaptiveTempering) {useAdaptiveTempering_ = useAdaptiveTempering;}
  /// Specifies whether the CESS target used for adaptive tempering should be adaptively determined according to the autocorrelation of the MCMC kernels.
  void setUseAdaptiveCessTarget(const bool useAdaptiveCessTarget) {useAdaptiveCessTarget_ = useAdaptiveCessTarget;}
  /// Returns whether both likelihood terms are tempered separately.
  bool getUseDoubleTempering() const {return useDoubleTempering_;}
  /// Specifies whether both likelihood terms should be tempered separately.
  void setUseDoubleTempering(const bool useDoubleTempering) {useDoubleTempering_ = useDoubleTempering;}
  /// Specifies the ESS resampling threshold.
  void setEssResamplingThreshold(const double essResamplingThreshold) {essResamplingThreshold_ = essResamplingThreshold;}
  /// Specifies the CESS target.
  void setCessTarget(const double cessTarget) {cessTarget_ = cessTarget;}
  /// Specifies the CESS target for the first part in a double tempering approach.
  void setCessTargetFirst(const double cessTargetFirst) {cessTargetFirst_ = cessTargetFirst;}
  /// Specifies the tempering schedule.
  void setAlpha(const arma::colvec& alpha) 
  {
    alpha_ = arma::conv_to<std::vector<double>>::from(alpha);
    nSteps_ = alpha_.size();
    useAdaptiveTempering_ = false;
  }
  /// Runs the SMC sampler and returns the evidence estimate.
  void runSmcSampler()
  {
    if (useDoubleTempering_)
    {
      std::cout << "using double tempering!" << std::endl;
      runSmcSamplerDoubleTemperingBase();
    }
    else 
    {
      runSmcSamplerBase();
    }
  }
  /// Computes various quantities for recycling all particles in an
  /// importance-tempering like approach. 
  void computeImportanceTemperingWeights();
  
private:
  
  /////////////////////////////////////////////////////////////////////////////
  // Private member functions
  /////////////////////////////////////////////////////////////////////////////
  
  /// Runs the SMC sampler.
  void runSmcSamplerBase(); 
  /// Runs the SMC sampler for a model with two likelihood terms which are 
  /// tempered separately.
  void runSmcSamplerDoubleTemperingBase();
  /// Calculates the effective sample size.
  double computeEss(const arma::colvec& selfNormalisedWeights) 
  {
    return 1.0 / arma::dot(selfNormalisedWeights, selfNormalisedWeights);
  }
  /// Calculates the effective sample size (used for importance tempering)
  double computeEss(const double alphaNew, const double alphaOld, const arma::colvec& logUnnormalisedWeights, const arma::colvec& logLikelihood)
  {
    arma::colvec logLikeNorm = normaliseExp(logLikelihood);
    return 
      std::pow(arma::accu(arma::exp(logUnnormalisedWeights + (alphaNew-alphaOld)*logLikeNorm)), 2.0) / 
      arma::accu(arma::exp(2.0*logUnnormalisedWeights + 2.0*(alphaNew-alphaOld)*logLikeNorm));

  }
  /// Calculates the conditional effective sample size.
  double computeCess(const double alphaNew, const double alphaOld, const arma::colvec& selfNormalisedWeights, const arma::colvec& logLikelihood)
  {
    if (alphaNew - alphaOld > 0)
    {
      arma::colvec logLikeNorm = normaliseExp(logLikelihood);
      return nParticles_ * 
        std::pow(arma::dot(selfNormalisedWeights, arma::exp((alphaNew-alphaOld)*logLikeNorm)), 2.0) / 
        arma::dot(selfNormalisedWeights, arma::exp(2*(alphaNew-alphaOld)*logLikeNorm));
    }
    else
    {
      return nParticles_;
    }
  }
  /// Computes the mean for a weighted sample.
  void computeSampleMoments(const std::vector<ParticleUpper<LatentPath, Aux>>& particles, const arma::colvec selfNormalisedWeights)
  {
    sampleMean_.zeros();
    sampleCovarianceMatrix_.zeros();
    
    for (unsigned int n=0; n<nParticles_; n++)
    {
      sampleMean_ = sampleMean_ + selfNormalisedWeights(n)*particles[n].theta_;
    }
    
    for (unsigned int n=0; n<nParticles_; n++)
    {
      sampleCovarianceMatrix_ = sampleCovarianceMatrix_ + 
        selfNormalisedWeights(n) * (particles[n].theta_ - sampleMean_) * arma::trans(particles[n].theta_ - sampleMean_);
    }
    mcmc_.setSampleCovarianceMatrix(sampleCovarianceMatrix_ + 0.0001 * arma::eye(sampleCovarianceMatrix_.n_rows, sampleCovarianceMatrix_.n_cols));
  }
  /// Initialises a particle by sampling theta from the prior
  /// and potentially running a lower-level SMC algorithm.
  void initialise(ParticleUpper<LatentPath, Aux>& particle)
  {
    particle.initialise(model_.getDimTheta());
    sampleFromPrior(particle);
    
    ////////////////////////////////
    std::cout << "particle: theta " << particle.theta_.t() << std::endl;
//     std::cout << "dimTheta: " << particle.theta_.size() << std::endl;
    ////////////////////////////////
    
    if (lower_ == SMC_SAMPLER_LOWER_MARGINAL)
    {
      evaluateLogMarginalLikelihoodFirst(particle);
      evaluateLogMarginalLikelihoodSecond(particle);
    }
    else 
    {
      evaluateLogMarginalLikelihoodFirst(particle);
      runSmcLower(particle);
    }
    
//     std::cout << "particle: logLikelihoodFirst_ " << particle.logLikelihoodFirst_ << "; " << "logLikelihoodSecond_ " << particle.logLikelihoodSecond_ << std::endl;
  }
  /// Initialises a particle by sampling theta from the prior. Only used
  /// if we temper both likelihood terms separately; assumes that 
  /// computing the first likelihood term does not require running a
  /// particle filter.
  void initialiseFirst(ParticleUpper<LatentPath, Aux>& particle)
  {
    particle.initialise(model_.getDimTheta());
    particle.logLikelihoodSecond_ = 0.0;
    sampleFromPrior(particle);
    evaluateLogMarginalLikelihoodFirst(particle);
  }
  /// Initialises the SMC approximation of the (second part of the) 
  /// marginal likelihood for each particle. Only used
  /// if we temper both likelihood terms separately.
  void initialiseSecond(ParticleUpper<LatentPath, Aux>& particle)
  {
    // TODO: sample those parameters from the prior here which do not depend on the first part of the likelihood!
    
//     ParticleUpper<LatentPath, Aux> particleAux;
//     sampleFromPrior(particleAux);
//     particle.theta_.rows(model_.getModelParameters().getThetaIndicesSecond()) 
//       = particleAux.theta_.rows(model_.getModelParameters().getThetaIndicesSecond());
//     
    if (lower_ == SMC_SAMPLER_LOWER_MARGINAL)
    {
      evaluateLogMarginalLikelihoodSecond(particle);
    }
    else 
    {
      runSmcLower(particle);
    }
  }
  /// Updates a particle using some suitable MCMC kernel.
  void update(unsigned int t, ParticleUpper<LatentPath, Aux>& particleNew, ParticleUpper<LatentPath, Aux>& particleOld, const double alpha)
  {
    
    double logAlpha = 0.0;
    ParticleUpper<LatentPath, Aux> particleProp;
    particleProp.initialise(model_.getDimTheta());
    
    if (mcmc_.getUseGradients())
    {
      std::cout << "WARNING: use of gradient information has not yet been implemented!" << std::endl;
    }
    
    if (mcmc_.getUseDelayedAcceptance())
    {
      
      /////////////////////////////////////////////////////////////////////////////
//           clock_t t1,t2; // timing
//           t1 = clock(); // start timer
     /////////////////////////////////////////////////////////////////////////////

      mcmc_.proposeTheta(t, particleProp.theta_, particleOld.theta_);
      evaluateLogMarginalLikelihoodFirst(particleProp);
      logAlpha = mcmc_.evaluateLogProposalDensity(t, particleOld.theta_, particleProp.theta_) -
        mcmc_.evaluateLogProposalDensity(t, particleProp.theta_, particleOld.theta_) +
        evaluateLogPriorDensity(particleProp.theta_) - 
        evaluateLogPriorDensity(particleOld.theta_) + 
        alpha * (particleProp.logLikelihoodFirst_ - particleOld.logLikelihoodFirst_);
        
         /////////////////////////////////////////////////////////////////////////////
//     t2 = clock(); // stop timer 
//     double seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//     std::cout << "before smcLower(): " << seconds1 << " sec." << " ";
    /////////////////////////////////////////////////////////////////////////////
        
      if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
      {
//         std::cout << "################### ACCEPTANCE AT STAGE 1 ###################" << std::endl;
        nAcceptedMovesFirst_++;
        
        if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL)
        {
          
           /////////////////////////////////////////////////////////////////////////////
//           clock_t t1,t2; // timing
//           t1 = clock(); // start timer
     /////////////////////////////////////////////////////////////////////////////

          runSmcLower(particleProp);
          
          /////////////////////////////////////////////////////////////////////////////
//           t2 = clock(); // stop timer 
//           double seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//           std::cout << "during smcLower(): " << seconds1 << " sec." << " ";
          /////////////////////////////////////////////////////////////////////////////
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_CORRELATED)
        {
          std::cout << "WARNING: lower-level SMC samplers using correlated pseudo-marginal approaches have not yet been implemented" << std::endl;
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_NOISY)
        {
          runSmcLower(particleProp);
          runSmcLower(particleOld);
        }
        else // i.e. lower_ == SMC_SAMPLER_LOWER_MARGINAL
        {
          evaluateLogMarginalLikelihoodSecond(particleProp);
//           std::cout << "WARNING: using delayed acceptance kernels in the case that the entire marginal likelihood can be evaluated analytically have not yet been implemented" << std::endl;
        }
        logAlpha = alpha * (particleProp.logLikelihoodSecond_ - particleOld.logLikelihoodSecond_);
                  
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
//           std::cout << "################### ACCEPTANCE AT STAGE 2 ###################" << std::endl;
          nAcceptedMoves_++;
          particleNew = particleProp;
        }
        else
        {
          particleNew = particleOld;
        }
      }
      else
      {
        particleNew = particleOld;
      }
    }
    else // i.e. if we do /not/ use the delayed acceptance approach
    {
      mcmc_.proposeTheta(t, particleProp.theta_, particleOld.theta_);
      logAlpha = 
        mcmc_.evaluateLogProposalDensity(t, particleOld.theta_, particleProp.theta_) -
        mcmc_.evaluateLogProposalDensity(t, particleProp.theta_, particleOld.theta_) +
        evaluateLogPriorDensity(particleProp.theta_) - 
        evaluateLogPriorDensity(particleOld.theta_);
        
      if (std::isfinite(logAlpha))
      {
        if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL)
        {
          evaluateLogMarginalLikelihoodFirst(particleProp);
          runSmcLower(particleProp);
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_CORRELATED)
        {
          std::cout << "WARNING: lower-level SMC samplers using correlated pseudo-marginal approaches have not yet been implemented" << std::endl;
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_NOISY)
        {
          evaluateLogMarginalLikelihoodFirst(particleProp);
          runSmcLower(particleProp);
          runSmcLower(particleOld);
        }
        else // i.e. lower_ == SMC_SAMPLER_LOWER_MARGINAL
        {
          evaluateLogMarginalLikelihoodFirst(particleProp);
          evaluateLogMarginalLikelihoodSecond(particleProp);
        }
        logAlpha += alpha * (particleProp.getlogLikelihood() - particleOld.getlogLikelihood());
                  
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
//           std::cout << "################### ACCEPTANCE ###################" << std::endl;
          nAcceptedMoves_++;
          particleNew = particleProp;
        }
        else
        {
          particleNew = particleOld;
        }
      }
      else
      {
        particleNew = particleOld;
      }
    }
  }
  /// Updates a particle in the first stage of the dual-tempering
  /// SMC sampler using some suitable MCMC kernel.
  void updateFirst(unsigned int t, ParticleUpper<LatentPath, Aux>& particleNew, ParticleUpper<LatentPath, Aux>& particleOld, const double alpha)
  {
    
    double logAlpha = 0.0;
    ParticleUpper<LatentPath, Aux> particleProp;
    particleProp.initialise(model_.getDimTheta());
    
    if (mcmc_.getUseGradients())
    {
      std::cout << "WARNING: use of gradient information has not yet been implemented!" << std::endl;
    }
    
    mcmc_.proposeTheta(t, particleProp.theta_, particleOld.theta_);
    logAlpha = 
      mcmc_.evaluateLogProposalDensity(t, particleOld.theta_, particleProp.theta_) -
      mcmc_.evaluateLogProposalDensity(t, particleProp.theta_, particleOld.theta_) +
      evaluateLogPriorDensity(particleProp.theta_) - 
      evaluateLogPriorDensity(particleOld.theta_);
      
    if (std::isfinite(logAlpha))
    {
      if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL)
      {
        evaluateLogMarginalLikelihoodFirst(particleProp);       
      }
      else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_CORRELATED)
      {
        std::cout << "WARNING: lower-level SMC samplers using correlated pseudo-marginal approaches have not yet been implemented" << std::endl;
      }
      else 
      {
        evaluateLogMarginalLikelihoodFirst(particleProp);
      }

      logAlpha += alpha * (particleProp.getlogLikelihood() - particleOld.getlogLikelihood());
                
      if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
      {
        nAcceptedMovesFirst_++;
        nAcceptedMoves_++;
//           std::cout << "################### ACCEPTANCE ###################" << std::endl;
        particleNew = particleProp;
      }
      else
      {
        particleNew = particleOld;
      }
    }
    else
    {
      particleNew = particleOld;
    }
  }
  /// Updates a particle using some suitable MCMC kernel.
  void updateSecond(unsigned int t, ParticleUpper<LatentPath, Aux>& particleNew, ParticleUpper<LatentPath, Aux>& particleOld, const double alpha)
  {
    
    double logAlpha = 0.0;
    ParticleUpper<LatentPath, Aux> particleProp;
    particleProp.initialise(model_.getDimTheta());
    
    if (mcmc_.getUseGradients())
    {
      std::cout << "WARNING: use of gradient information has not yet been implemented!" << std::endl;
    }
    
    if (mcmc_.getUseDelayedAcceptance())
    {
      mcmc_.proposeTheta(t, particleProp.theta_, particleOld.theta_);
      evaluateLogMarginalLikelihoodFirst(particleProp);
      logAlpha = mcmc_.evaluateLogProposalDensity(t, particleOld.theta_, particleProp.theta_) -
        mcmc_.evaluateLogProposalDensity(t, particleProp.theta_, particleOld.theta_) +
        evaluateLogPriorDensity(particleProp.theta_) - 
        evaluateLogPriorDensity(particleOld.theta_) + 
        1.0 * (particleProp.logLikelihoodFirst_ - particleOld.logLikelihoodFirst_);
        
      if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
      {
//         std::cout << "################### ACCEPTANCE AT STAGE 1 ###################" << std::endl;
        nAcceptedMovesFirst_++;
             
        if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL)
        {
          runSmcLower(particleProp);
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_CORRELATED)
        {
          std::cout << "WARNING: lower-level SMC samplers using correlated pseudo-marginal approaches have not yet been implemented" << std::endl;
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_NOISY)
        {
          runSmcLower(particleProp);
          runSmcLower(particleOld);
        }
        else // i.e. lower_ == SMC_SAMPLER_LOWER_MARGINAL
        {
          evaluateLogMarginalLikelihoodSecond(particleProp);
          
//           std::cout << particleProp.logLikelihoodSecond_ << " ";
//           std::cout << "WARNING: using delayed acceptance kernels in the case that the entire marginal likelihood can be evaluated analytically have not yet been implemented" << std::endl;
        }
        logAlpha = alpha * (particleProp.logLikelihoodSecond_ - particleOld.logLikelihoodSecond_);
                  
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
          nAcceptedMoves_++;
//           std::cout << "################### ACCEPTANCE AT STAGE 2 ###################" << std::endl;
          particleNew = particleProp;
        }
        else
        {
          particleNew = particleOld;
        }
      }
      else
      {
        particleNew = particleOld;
      }
    }
    else // i.e. if we do /not/ use the delayed acceptance approach
    {
      mcmc_.proposeTheta(t, particleProp.theta_, particleOld.theta_);
      logAlpha = 
        mcmc_.evaluateLogProposalDensity(t, particleOld.theta_, particleProp.theta_) -
        mcmc_.evaluateLogProposalDensity(t, particleProp.theta_, particleOld.theta_) +
        evaluateLogPriorDensity(particleProp.theta_) - 
        evaluateLogPriorDensity(particleOld.theta_);
        
      if (std::isfinite(logAlpha))
      {
        if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL)
        {
          evaluateLogMarginalLikelihoodFirst(particleProp);
          runSmcLower(particleProp);
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_CORRELATED)
        {
          std::cout << "WARNING: lower-level SMC samplers using correlated pseudo-marginal approaches have not yet been implemented" << std::endl;
        }
        else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_NOISY)
        {
          evaluateLogMarginalLikelihoodFirst(particleProp);
          runSmcLower(particleProp);
          runSmcLower(particleOld);
        }
        else // i.e. lower_ == SMC_SAMPLER_LOWER_MARGINAL
        {
          evaluateLogMarginalLikelihoodFirst(particleProp);
          evaluateLogMarginalLikelihoodSecond(particleProp);
        }
        logAlpha += 1.0 * (particleProp.logLikelihoodFirst_ - particleOld.logLikelihoodFirst_) + alpha * (particleProp.logLikelihoodSecond_ - particleOld.logLikelihoodSecond_);
                  
        if (std::isfinite(logAlpha) && log(arma::randu()) < logAlpha)
        {
//           std::cout << "################### ACCEPTANCE ###################" << std::endl;
          nAcceptedMoves_++;
          particleNew = particleProp;
        }
        else
        {
          particleNew = particleOld;
        }
      }
      else
      {
        particleNew = particleOld;
      }
    }
  }
  /// Evaluates the part of the log-marginal likelihood normally approximated by SMC.
  void evaluateLogMarginalLikelihoodSecond(ParticleUpper<LatentPath, Aux>& particle)
  {
    particle.logLikelihoodSecond_ = model_.evaluateLogMarginalLikelihoodSecond(particle.theta_, particle.latentPath_);
  }
  /// Evaluates the the part of the log-marginal likelihood that can be evaluated analytically.
  void evaluateLogMarginalLikelihoodFirst(ParticleUpper<LatentPath, Aux>& particle)
  {
    particle.logLikelihoodFirst_ = model_.evaluateLogMarginalLikelihoodFirst(particle.theta_, particle.latentPath_);
  }
  /// Evaluates the log-prior density.
  double evaluateLogPriorDensity(arma::colvec theta)
  {
    model_.setUnknownParameters(theta);
    return model_.evaluateLogPriorDensity();
  }
  /// Wrapper for the lower-level SMC filter.
  void runSmcLower(ParticleUpper<LatentPath, Aux>& particle)
  {
    particle.logLikelihoodSecond_ = smc_.runSmc(particle.theta_, particle.latentPath_, particle.aux_, particle.gradient_); // TODO: need to implement this function in the smc class
  }
  /// Wrapper for sampling the parameters from their prior
  void sampleFromPrior(ParticleUpper<LatentPath, Aux>& particle)
  {
    model_.sampleFromPrior(particle.theta_);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Private data members:
  /////////////////////////////////////////////////////////////////////////////
  
  Rng& rng_; // random number generation.
  Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>& model_; // class for dealing with the targeted model
  Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, ParticleLower, Aux, SmcParameters>& smc_; // class for dealing with the lower-level SMC filter
  Mcmc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, McmcParameters>& mcmc_; // class for dealing with the mcmc updates
  
  SmcSamplerLowerType lower_; // type of lower-level algorithm
  
  std::vector<std::vector<arma::colvec>> thetaFull_;
  
  unsigned int nParticles_; // number of particles.
  unsigned int nSteps_; // number of SMC steps (unless useAdaptiveTempering_ == true)
  unsigned int nMetropolisHastingsUpdates_; // number of MH steps applied to each particle per SMC step
  unsigned int nMetropolisHastingsUpdatesFirst_; // number of MH steps applied to each particle per SMC step in the first stage when using the dual-tempering approach
  double essResamplingThreshold_; // proportion of ESS/nParticles below which resampling is triggered.
  
  double logEvidenceEstimate_; // estimate of the normalising constant.
  std::vector<double> logPartialEvidenceEstimates_; // log of the estimate of the normalising constant after intermediate steps
  double logEvidenceEstimateEss_, logEvidenceEstimateEssAlternate_; // estimate of the normalising constant obtained via importance tempering using the ESS.
  double logEvidenceEstimateEssResampled_, logEvidenceEstimateEssResampledAlternate_; // estimate of the normalising constant obtained via importance tempering using the ESS but including additional resampling steps to obtain evenly weighted particles.
  double logEvidenceEstimateCess_, logEvidenceEstimateCessAlternate_; // estimate of the normalising constant obtained via importance tempering using the ESS.
  
  double alphaNew_, alphaOld_, alphaInc_; // current and previous inverse temperatures and the increment
  bool useAdaptiveTempering_; // should the tempering schedule be adaptively determined according to the CESS?
  bool useAdaptiveCessTarget_; // should the CESS target be adapted according the the autocorrelation of the MCMC kernels?
  
  double cessTarget_; // value of conditional ESS targeted if adaptive tempering is used
  double cessTargetFirst_; // value of conditional ESS targeted if adaptive tempering is used (in the first stage of a double tempering approach)
  bool storeHistory_; // store history of the particle system?
  bool useDoubleTempering_; // should we temper the two likelihood terms separately?
  
  arma::colvec sampleMean_; // mean vector for the Gaussian proposal.
  arma::mat sampleCovarianceMatrix_; // covariance matrix of the Gaussian proposal
  std::vector<double> alpha_; // inverse temperatures
  std::vector<bool> isResampled_; // indicates whether resampling took place at a specific step
  std::vector<double> acceptanceRates_; // empirical acceptance rate at each step of the algorithm
  std::vector<double> acceptanceRatesFirst_; // empirical acceptance rate for the first stage of a delayed-acceptance MH update at each step of the algorithm
  std::vector<double> maxParticleAutocorrelations_; // mean-empirical autocorrelations between particles before and after applying the MH update
  unsigned int nAcceptedMoves_; // number of accepted MH proposals in the current step of the SMC sampler
  unsigned int nAcceptedMovesFirst_; // number of accepted MH proposals in the first stage of a delayed-acceptance MH update at each step of the algorithm
  
  std::vector<ParticleUpper<LatentPath, Aux>> finalParticles_; // the set vector of full particles from the final step of the SMC sampler.
    
  arma::mat logLikelihoodsFull_; // quantities needed for the incremental weights
  arma::uvec particleIndices_; // particle indices associated with the single particle path 
  arma::umat parentIndicesFull_; // (nParticles_, nSteps_)-dimensional: holds all parent indices
  arma::mat logUnnormalisedWeightsFull_; // (nParticles_, nSteps_)-dimensional: holds all log-unnormalised weight
  arma::mat selfNormalisedWeightsFull_; // (nParticles_, nSteps_)-dimensional: holds all self-normalised weights 
  
  // The following nine quantities are only required for importance-tempering type schemes:
  
  arma::colvec ess_; // holds the ESS for each step (only needed for importance tempering)
  arma::colvec essResampled_; // holds the ESS for each step (only needed for importance tempering using the approach from Nguyen et al. (2016))
  arma::colvec cess_; // holds the CESS for each step (only needed for importance tempering)
  arma::mat logUnnormalisedReweightedWeights_, selfNormalisedReweightedWeights_; // log-unnormalised/self-normalised weights (for targeting the posterior)
  arma::mat selfNormalisedWeightsEss_; // self-normalised weights (for targeting the posterior) weighted by the ESS
  arma::mat logUnnormalisedReweightedWeightsResampled_, selfNormalisedReweightedWeightsResampled_; // log-unnormalised/self-normalised weights (for targeting the posterior) (but after adding another resampling step as in Nguyen et al. (2016))
  arma::mat selfNormalisedWeightsEssResampled_; // self-normalised weights (for targeting the posterior) weighted by the ESS (but after adding another resampling step as in Nguyen et al. (2016))
  arma::mat selfNormalisedWeightsCess_; // self-normalised weights (for targeting the posterior) weighted by the ESS
  std::vector<std::vector<arma::colvec>> thetaFullResampled_; // resampled particles needed for the approach from Nguyen et al. (2016).
  
  unsigned int nCores_; // number of cores to use (not currently used)
  
};

/// Runs the SMC sampler.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void SmcSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::runSmcSamplerBase
(
//   std::vector<std::vector<arma::colvec>>& thetaFull_ // (nSteps_, nParticles_)-dimensional: holds all particles
)
{
  
  // TODO: make these accessible from the outside:
  double cessTargetMin_ = 0.5;
  double cessTargetMax_ = 0.9999;
  double adaptiveCessTargetScale_ = 10.0;
  
  
//   double maxParticleAutocorrelation = 0;
  arma::colvec esjd(model_.getDimTheta()); // holds the expected squared jumping distance for each component of theta
  
  /////////////////////////////////////////////////////////////////////////////
  // Initialisation
  /////////////////////////////////////////////////////////////////////////////
  bool isBracketing;
  arma::uvec parentIndices(nParticles_);
  std::vector<ParticleUpper<LatentPath, Aux>> particlesNew(nParticles_);
  std::vector<ParticleUpper<LatentPath, Aux>> particlesOld(nParticles_);
  
  ParticleUpper<LatentPath, Aux> singleParticle;
  
  sampleMean_.zeros(model_.getDimTheta());
  sampleCovarianceMatrix_.zeros(model_.getDimTheta(), model_.getDimTheta());
  
  logEvidenceEstimate_ = 0; // estimate of the model evidence
  
  nAcceptedMoves_ = 0; // number of accepted MH moves in each SMC step
  if (mcmc_.getUseDelayedAcceptance()) 
  {
    nAcceptedMovesFirst_ = 0; // number of accepted moves at the first stage of a delayed acceptance MH update
  }
  
  alphaNew_ = 0; // current inverse temperature
  alphaOld_ = 0; // previous inverse temperature
  alphaInc_ = 0; // increment of the inverse temperature
  
  arma::colvec logLikelihoods(nParticles_);
//   arma::colvec logUnnormalisedWeights = - std::log(nParticles_) * arma::ones(nParticles_); // log-unnormalised weights
  
  arma::colvec logUnnormalisedWeights = - std::log(nParticles_)*arma::ones(nParticles_); // log-unnormalised weights
    
  arma::colvec selfNormalisedWeights  = arma::ones(nParticles_) / nParticles_; // self-normalised weights
  
  unsigned int t = 0; // step counter
  
  // TODO: initialise the "...Full_" quantities and figure out how to best grow these matrices if adaptive tempering is used 
  
  if (useAdaptiveTempering_)
  { 
    thetaFull_.reserve(200);
    thetaFull_.resize(1);
    logUnnormalisedWeightsFull_.set_size(nParticles_, 1);
    selfNormalisedWeightsFull_.set_size(nParticles_, 1);
    logLikelihoodsFull_.set_size(nParticles_, 1);
    alpha_.reserve(200);
    alpha_.resize(1);
    alpha_[0] = alphaNew_;
    isResampled_.reserve(200);
    logPartialEvidenceEstimates_.reserve(200);
    acceptanceRates_.reserve(200);
    maxParticleAutocorrelations_.reserve(200);
//     maxParticleAutocorrelations_.push_back(0);
//     acceptanceRates_.push_back(0); // because there is no MH update at the initial step
    if (mcmc_.getUseDelayedAcceptance()) 
    {
      acceptanceRatesFirst_.reserve(200);
      acceptanceRatesFirst_.push_back(0); // because there is no MH update at the initial step
    }
  }
  else
  { 
    thetaFull_.resize(nSteps_);
    logUnnormalisedWeightsFull_.set_size(nParticles_, nSteps_);
    selfNormalisedWeightsFull_.set_size(nParticles_, nSteps_);
    logLikelihoodsFull_.set_size(nParticles_, nSteps_);
    isResampled_.resize(nSteps_);
    logPartialEvidenceEstimates_.resize(nSteps_);
    acceptanceRates_.resize(nSteps_);
//     acceptanceRates_.push_back(0); // because there is no MH update at the initial step
    maxParticleAutocorrelations_.resize(nSteps_);
    if (mcmc_.getUseDelayedAcceptance()) 
    {
      acceptanceRatesFirst_.resize(nSteps_);
//       acceptanceRatesFirst_.push_back(0); // because there is no MH update at the initial step
    }
  }
  
  
   
//   std::cout << "start: initialise SMC sampler" << std::endl;
  for (unsigned int n=0; n<nParticles_; n++)
  {
    
        ///////////////////////
    std::cout << "n: " << n << std::endl;
    ////////////////////////////
    
    initialise(particlesNew[n]); 
    

    
    logLikelihoods(n) = particlesNew[n].getlogLikelihood();
    std::cout << particlesNew[n].getlogLikelihood() << std::endl;
  }
  
//   std::cout << "end: initialise SMC sampler" << std::endl;
  
//   std::cout << "loglikelihoods at step 0: " << std::endl;
//   std::cout << logLikelihoods.t() << std::endl;
//   std::cout << "acceptance rate at step " << 0 << ": " << acceptanceRates_[acceptanceRates_.size()-1] << std::endl;
//   std::cout << "finished initialise SMC sampler" << std::endl;

  if (storeHistory_)
  { 
    thetaFull_[0].resize(nParticles_);
    for (unsigned int n=0; n<nParticles_; n++)
    {
      thetaFull_[0][n] = particlesNew[n].theta_;
    }
    logUnnormalisedWeightsFull_.col(0) = logUnnormalisedWeights;
    selfNormalisedWeightsFull_.col(0)  = selfNormalisedWeights;
    logLikelihoodsFull_.col(0)         = logLikelihoods;
  }
  

  t++; // step counter 
  
  /////////////////////////////////////////////////////////////////////////////
  // Iterations
  /////////////////////////////////////////////////////////////////////////////
  
  while (alphaNew_ < 1.0)
  {

    std::cout << "Step " << t << " of SMC Sampler " << std::endl;
    
    //////////////////////////////////////////////
//     std::cout << "logLikelihoods: " << logLikelihoods.t() << std::endl;
//     for (unsigned int n=0; n<nParticles_; n++)
//     {
//       std::cout << particlesNew[n].theta_.t() << std::endl;
//     }
    //////////////////////////////////////////////
//     if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL)
//     {
//       std::cout << "Using PMMH kernels to move the particles" << std::endl;
//     } 
//     else if (lower_ == SMC_SAMPLER_LOWER_PSEUDO_MARGINAL_NOISY)
//     {
//       std::cout << "Using MCWM kernels to move the particles" << std::endl;
//     }

    
    // --------------------------------------------------------------------- //
    // Determine the inverse temperature, 
    // --------------------------------------------------------------------- //
    
    alphaOld_ = alphaNew_;

    if (useAdaptiveTempering_)
    { // Adaptive tempering, 
      // i.e. numerically solve CESS(alpha) = CESS* for alpha:
      
      // adaptively setting CESS* based on the mixing of the MCMC kernels
      if (t > 1 && useAdaptiveCessTarget_)
      {
//         double adaptedCessTarget = adaptiveCessTargetScale_ * maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1];
        
        double adaptedCessTarget = 1.0 - (1.0 - maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1]) / adaptiveCessTargetScale_;
        if (adaptedCessTarget < cessTargetMin_)
        {
          adaptedCessTarget = cessTargetMin_;
        } 
        else if (adaptedCessTarget > cessTargetMax_)
        {
          adaptedCessTarget = cessTargetMax_;
        }
        std::cout << "adapted CESS Target: " << adaptedCessTarget << std::endl;
        auto f = [&] (double alpha) {return computeCess(alpha, alphaOld_, selfNormalisedWeights, logLikelihoods) - nParticles_* adaptedCessTarget;};
        alphaNew_ = rootFinding::bisection(isBracketing, f, alphaOld_, 2.0, 0.000000001, 0.000000001, 1000);
        std::cout << "next inverse temperature found by bisection: " << alphaNew_ << std::endl;
        
        ////////////
        auto gg = [&] (double alpha) {return computeCess(alpha, alphaOld_, selfNormalisedWeights, logLikelihoods) - nParticles_* cessTarget_;};
        double alphaNewTest = rootFinding::bisection(isBracketing, gg, alphaOld_, 2.0, 0.000000001, 0.000000001, 1000);
        std::cout << "next inverse temperature if we hadn't adapted the CESS target: " << alphaNewTest << std::endl;
        ///////////
      }
      else if (t == 1 && useAdaptiveCessTarget_)
      {
        alphaNew_ = std::numeric_limits<double>::epsilon();
        isBracketing = true;
        std::cout << "alphaNew_ set to epsilon at t=1:" << alphaNew_ << std::endl;
      }
      else
      {
        // Computes CESS(alpha) for given weights:
        auto f = [&] (double alpha) {return computeCess(alpha, alphaOld_, selfNormalisedWeights, logLikelihoods) - nParticles_* cessTarget_;};
        alphaNew_ = rootFinding::bisection(isBracketing, f, alphaOld_, 2.0, 0.000000001, 0.000000001, 1000);
        std::cout << "next inverse temperature found by bisection: " << alphaNew_ << std::endl;
      }


      
      if (!isBracketing) { std::cout << "Warning: interval is not bracketing!" << std::endl; }
      
      if (alphaNew_ > 1.0 || !isBracketing)
      {
        alphaNew_ = 1.0;
      }
      alpha_.push_back(alphaNew_);
    }
    else 
    { // Manual tempering, 
      // i.e. user-specified schedule:
      alphaNew_ = alpha_[t];
    }
    alphaInc_ = alphaNew_ - alphaOld_; // inverse temperature increment
    
    // --------------------------------------------------------------------- //
    // Update weights
    // --------------------------------------------------------------------- //
    
    logUnnormalisedWeights = logUnnormalisedWeights + alphaInc_ * logLikelihoods;
    
    // self-normalise the weights:
    selfNormalisedWeights = normaliseWeights(logUnnormalisedWeights);
    
    if (!arma::is_finite(selfNormalisedWeights))
    {
      std::cout << "WARNING: W (in SMC sampler) contains NaNs!" << std::endl;
    }
    
    // --------------------------------------------------------------------- //
    // Compute proposal scale
    // --------------------------------------------------------------------- //

    if (mcmc_.getUseAdaptiveProposal())
    {
      // Computes the first two moments of the weighted sample.
      computeSampleMoments(particlesNew, selfNormalisedWeights);
      if (t > 1 && mcmc_.getUseAdaptiveProposalScaleFactor1())
      {
        // Adapts the additional scale factor by which the sample covariance matrix
        // is multiplied according to the empirical
        // acceptance rate at the previous SMC step.
        mcmc_.adaptProposalScaleFactor1(acceptanceRates_[acceptanceRates_.size()-1]);
      }
    }
    
    
    // --------------------------------------------------------------------- //
    // Adaptive systematic resampling
    // --------------------------------------------------------------------- //
 
    logPartialEvidenceEstimates_.push_back(logEvidenceEstimate_ + std::log(arma::sum(arma::exp(logUnnormalisedWeights)))); 
  
    if (computeEss(selfNormalisedWeights) < nParticles_ * essResamplingThreshold_)
    {
      std::cout << "Resampling at Step " << t << std::endl;
      isResampled_.push_back(1);
      // Update estimate of the model evidence:
//       logEvidenceEstimate_ += std::log(arma::sum(arma::exp(logUnnormalisedWeights)));    
      
      logEvidenceEstimate_ = logPartialEvidenceEstimates_[t-1];

      // Systematic resampling:
      resample::systematicBase(arma::randu(), parentIndices, selfNormalisedWeights, nParticles_);
      
//       std::cout << parentIndices.t() << std::endl;
      
      // Re-setting the weights:
      logUnnormalisedWeights.fill(-std::log(nParticles_)); // resetting the weights
      
//       std::cout << "the following line makes the evidence estimates wrong! (the commented out line is correct instead)" << std::endl;
//       logUnnormalisedWeights.fill(- std::log(nParticles_));
//       logUnnormalisedWeights.fill(- std::log(nParticles_) + std::log(arma::accu(arma::exp(logUnnormalisedWeights))));
      
//       selfNormalisedWeights = normaliseWeights(logUnnormalisedWeights);
      selfNormalisedWeights.fill(1.0 / nParticles_);
    } 
    else // i.e. no resampling:
    {
      
      std::cout << "No resampling at Step " << t << std::endl;
      isResampled_.push_back(0);

      parentIndices = arma::linspace<arma::uvec>(0, nParticles_-1, nParticles_);
    }
    // Determining the parent particles based on the parent indices: 
    for (unsigned int n=0; n<nParticles_; n++)
    {
      particlesOld[n] = particlesNew[parentIndices(n)];
    }

    
//     std::cout << logUnnormalisedWeights.t() << std::endl;
//     std::cout << parentIndices.t() << std::endl;

    
    // --------------------------------------------------------------------- //
    // Apply MCMC kernel
    // --------------------------------------------------------------------- //
  
    if (nMetropolisHastingsUpdates_ > 1)
    {
      for (unsigned int n=0; n<nParticles_; n++)
      {
        
        /////////////////////////////////////////////////////////////////////////////
//           clock_t t1,t2; // timing
//           t1 = clock(); // start timer
     /////////////////////////////////////////////////////////////////////////////
        
        singleParticle = particlesOld[n];
        for (unsigned int k=0; k<nMetropolisHastingsUpdates_; k++)
        {
          update(t, particlesNew[n], singleParticle, alphaNew_); 
          singleParticle = particlesNew[n];
        }
        logLikelihoods(n) = particlesNew[n].getlogLikelihood();
        
         /////////////////////////////////////////////////////////////////////////////
//           t2 = clock(); // stop timer 
//           double seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//           std::cout << "update(more than one MH update): " << seconds1 << " sec." << " ";
          /////////////////////////////////////////////////////////////////////////////
      }
    }
    else
    {
      for (unsigned int n=0; n<nParticles_; n++)
      {
        /////////////////////////////////////////////////////////////////////////////
//           clock_t t1,t2; // timing
//           t1 = clock(); // start timer
     /////////////////////////////////////////////////////////////////////////////
          
        update(t, particlesNew[n], particlesOld[n], alphaNew_); 
        logLikelihoods(n) = particlesNew[n].getlogLikelihood();
          
          /////////////////////////////////////////////////////////////////////////////
//           t2 = clock(); // stop timer 
//           double seconds1 = ((double)t2-(double)t1) / CLOCKS_PER_SEC;
//           std::cout << "update(one MH update): " << seconds1 << " sec." << " ";
          /////////////////////////////////////////////////////////////////////////////

      }
    }
    
    
    ///////////////////////////////// START: computing the autocorrelation (only needed for adapting cessTarget)
//     if (useAdaptiveCessTarget_)
//     {
      maxParticleAutocorrelations_.push_back(0);
      double componentParticleAutocovariance, componentParticleAutocorrelation;
      double dimTheta = particlesNew[0].theta_.size();
      double meanOld, meanNew;
      double varOld, varNew, sdProd;
      
      for (unsigned int j=0; j<dimTheta; j++)
      {
        componentParticleAutocovariance = 0;
        meanOld = 0;
        meanNew = 0;
        varOld  = 0;
        varNew  = 0;
        
        // Computes the means.
        for (unsigned int n=0; n<nParticles_; n++)
        {
          meanNew += selfNormalisedWeights(n) * particlesNew[n].theta_(j);
          meanOld += selfNormalisedWeights(n) * particlesOld[n].theta_(j);
        }
        // Computes variances.
        for (unsigned int n=0; n<nParticles_; n++)
        {
          varNew += selfNormalisedWeights(n) * particlesNew[n].theta_(j) * particlesNew[n].theta_(j);
          varOld += selfNormalisedWeights(n) * particlesOld[n].theta_(j) * particlesOld[n].theta_(j);
        }
        varNew -= meanNew * meanNew;
        varOld -= meanOld * meanOld;
        sdProd = std::sqrt(varOld) * std::sqrt(varNew);
        // Computes the auto-covariances
        for (unsigned int n=0; n<nParticles_; n++)
        {
          componentParticleAutocovariance += selfNormalisedWeights(n) * particlesOld[n].theta_(j) * particlesNew[n].theta_(j);
        }
        componentParticleAutocovariance -= meanNew * meanOld;
        componentParticleAutocorrelation = componentParticleAutocovariance / sdProd;
//         maxParticleAutocorrelations += componentParticleAutocorrelation / dimTheta;
        
        // Option 1: Using the mean over the autocorrelations of the particle components:
//         maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] 
//           = maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] 
//             + componentParticleAutocorrelation / dimTheta;
            
        // Option 2: Using the maximum over the autocorrelations of the particle components:
        maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] 
          = std::max(maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1], componentParticleAutocorrelation);
        
//         std::cout << "; meanOld: " << meanOld << "; meanNew;" << meanNew;
//         std::cout << "; varOld: " << varOld << "; varNew: " << varNew;
//         std::cout << "; component particle autocorrelation: " << componentParticleAutocorrelation;
      }
//       std::cout << "" << std::endl;
      std::cout << "maxParticleAutocorrelation: " << maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] << std::endl;
//     }
    ///////////////////////////////// END: computing the autocorrelation (only needed for adapting cessTarget)
    
    // Computing the expected squared jumping distance:
    esjd.zeros();
    for (unsigned int n=0; n<nParticles_; n++) 
    {
      esjd = esjd + arma::pow(particlesNew[n].theta_ - particlesOld[n].theta_, 2.0);
    }
    esjd = arma::sqrt(esjd / nParticles_);
//     std::cout << "ESJD: " << esjd.t() << std::endl;
    std::cout << "Mean ESJD: " << arma::accu(esjd)/esjd.size() << std::endl;
    
    if (mcmc_.getUseDelayedAcceptance()) {
      acceptanceRatesFirst_.push_back(static_cast<double>(nAcceptedMovesFirst_) / (nParticles_ * nMetropolisHastingsUpdates_)); 
      std::cout << "First-stage acceptance rate: " << acceptanceRatesFirst_[acceptanceRatesFirst_.size()-1] << std::endl;
      nAcceptedMovesFirst_ = 0;
    }
    
    acceptanceRates_.push_back(static_cast<double>(nAcceptedMoves_) / (nParticles_ * nMetropolisHastingsUpdates_)); 
    std::cout << "Overall acceptance rate: " << acceptanceRates_[acceptanceRates_.size()-1] << std::endl;
    nAcceptedMoves_ = 0;
    

    // --------------------------------------------------------------------- //
    // Store output
    // --------------------------------------------------------------------- //
    
    if (storeHistory_)
    { 
      if (useAdaptiveTempering_)
      {
        std::vector<arma::colvec> thetaNew(nParticles_);
        
        for (unsigned int n=0; n<nParticles_; n++)
        {
          //thetaFull_[t][n] = particlesNew[n].theta_; // TODO
          thetaNew[n] = particlesNew[n].theta_; // TODO
        }
        thetaFull_.push_back(thetaNew);

        
        // NOTE: std::cout << "NOTE: logUnnormalisedWeightsFull_ is now defined differently than logUnnormalisedWeights!" << std::endl;
        logUnnormalisedWeightsFull_ = arma::join_rows(logUnnormalisedWeightsFull_, logUnnormalisedWeights + logEvidenceEstimate_);
        selfNormalisedWeightsFull_  = arma::join_rows(selfNormalisedWeightsFull_,  selfNormalisedWeights);
        logLikelihoodsFull_         = arma::join_rows(logLikelihoodsFull_,         logLikelihoods);
      }
      else 
      {

        std::vector<arma::colvec> thetaNew(nParticles_);
        for (unsigned int n=0; n<nParticles_; n++)
        {
          thetaNew[n] = particlesNew[n].theta_; // TODO
        }
                
        thetaFull_[t] = thetaNew;
        
        /// NOTE: std::cout << "NOTE: logUnnormalisedWeightsFull_ is now defined differently than logUnnormalisedWeights!" << std::endl;
        logUnnormalisedWeightsFull_.col(t) = logUnnormalisedWeights + logEvidenceEstimate_;
        selfNormalisedWeightsFull_.col(t)  = selfNormalisedWeights;
        logLikelihoodsFull_.col(t)         = logLikelihoods;
      }
    }
    
    t++;
    
  } // end of while loop
  if (useAdaptiveTempering_) { nSteps_ = t; } // determines the number of SMC steps
//   logEvidenceEstimate_ += std::log(arma::sum(arma::exp(logUnnormalisedWeights))); 
  
  // The following is numerically unstable and assumes that logUnnormalisedWeights is defined as in the Ph.D. thesis of A. Finke (2015),
  // i.e. it assumes that the product of the sums of the unnormalised weights are included in the weights:
//   if (isResampled_[isResampled_.size()-1])
//   {
//     logEvidenceEstimate_ = logUnnormalisedWeights(0);
//     //- std::log(nParticles_) + std::log(arma::accu(arma::exp(logUnnormalisedWeights)));
//   }
//   else 
//   {
//     logEvidenceEstimate_ = std::log(arma::accu(arma::exp(logUnnormalisedWeights)));
//   }
  
  // The following assume that the product of the sums of the unnormalised weights are not included in the weights:
  if (!isResampled_[isResampled_.size()-1])
  {
//     logEvidenceEstimate_ += std::log(arma::accu(arma::exp(logUnnormalisedWeights)));
    logEvidenceEstimate_ = logPartialEvidenceEstimates_[logPartialEvidenceEstimates_.size()-1];
  }
  
  finalParticles_ = particlesNew;
}
/// Runs the SMC sampler for a model with two likelihood terms which are 
/// tempered separately.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void SmcSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::runSmcSamplerDoubleTemperingBase
(
//   std::vector<std::vector<arma::colvec>>& thetaFull_ // (nSteps_, nParticles_)-dimensional: holds all particles
)
{
  
    // TODO: make these accessible from the outside:
//   double cessTargetMin_ = 0.5;
//   double cessTargetMax_ = 0.9999;
//   double adaptiveCessTargetScale_ = 10.0;
  
  
//   double maxParticleAutocorrelation = 0;
  arma::colvec esjd(model_.getDimTheta()); // holds the expected squared jumping distance for each component of theta
  
  /////////////////////////////////////////////////////////////////////////////
  // Initialisation
  /////////////////////////////////////////////////////////////////////////////

  bool isBracketing;
  bool isDoubleTemperingFirst = true;
  double cessTarget = cessTargetFirst_;
  
  arma::uvec parentIndices(nParticles_);
  std::vector<ParticleUpper<LatentPath, Aux>> particlesNew(nParticles_);
  std::vector<ParticleUpper<LatentPath, Aux>> particlesOld(nParticles_);
  
  ParticleUpper<LatentPath, Aux> singleParticle;
  
  sampleMean_.zeros(model_.getDimTheta());
  sampleCovarianceMatrix_.zeros(model_.getDimTheta(), model_.getDimTheta());
  
  logEvidenceEstimate_ = 0; // estimate of the model evidence
  
  nAcceptedMoves_ = 0; // number of accepted MH moves in each SMC step
  if (mcmc_.getUseDelayedAcceptance()) 
  {
    nAcceptedMovesFirst_ = 0; // number of accepted moves at the first stage of a delayed acceptance MH update
  }
  
  alphaNew_ = 0; // current inverse temperature
  alphaOld_ = 0; // previous inverse temperature
  alphaInc_ = 0; // increment of the inverse temperature
  
  arma::colvec logLikelihoods(nParticles_);
//   arma::colvec logLikelihoodsA.zeros(nParticles_);
//   arma::colvec logLikelihoodsB.zeros(nParticles_);
    
  arma::colvec logUnnormalisedWeights = - std::log(nParticles_) * arma::ones(nParticles_); // log-unnormalised weights
  arma::colvec selfNormalisedWeights  = arma::ones(nParticles_) / nParticles_; // self-normalised weights
  
  unsigned int t = 0; // step counter
  
  // TODO: initialise the "...Full_" quantities and figure out how to best grow these matrices if adaptive tempering is used 
  
if (useAdaptiveTempering_)
  { 
    thetaFull_.reserve(200);
    thetaFull_.resize(1);
    logUnnormalisedWeightsFull_.set_size(nParticles_, 1);
    selfNormalisedWeightsFull_.set_size(nParticles_, 1);
    logLikelihoodsFull_.set_size(nParticles_, 1);
    alpha_.reserve(200);
    alpha_.resize(1);
    alpha_[0] = alphaNew_;
    isResampled_.reserve(200);
    logPartialEvidenceEstimates_.reserve(200);
    acceptanceRates_.reserve(200);
    acceptanceRates_.push_back(0); // because there is no MH update at the initial step
    if (mcmc_.getUseDelayedAcceptance()) 
    {
      acceptanceRatesFirst_.reserve(200);
      acceptanceRatesFirst_.push_back(0); // because there is no MH update at the initial step
    }
  }
  else
  { 
    thetaFull_.resize(nSteps_);
    logUnnormalisedWeightsFull_.set_size(nParticles_, nSteps_);
    selfNormalisedWeightsFull_.set_size(nParticles_, nSteps_);
    logLikelihoodsFull_.set_size(nParticles_, nSteps_);
    isResampled_.resize(nSteps_);
    logPartialEvidenceEstimates_.resize(nSteps_);
    acceptanceRates_.resize(nSteps_);
//     acceptanceRates_.push_back(0); // because there is no MH update at the initial step
    if (mcmc_.getUseDelayedAcceptance()) 
    {
      acceptanceRatesFirst_.resize(nSteps_);
      acceptanceRatesFirst_.push_back(0); // because there is no MH update at the initial step
    }
  }
  
  std::cout << "initialise SMC sampler" << std::endl;
  for (unsigned int n=0; n<nParticles_; n++)
  {
    initialiseFirst(particlesNew[n]); // TODO: implement this!
    logLikelihoods(n) = particlesNew[n].logLikelihoodFirst_;
  }
  
//   std::cout << "acceptance rate at step" << 0 << ": " << acceptanceRates_[acceptanceRates_.size()-1] << std::endl;
  
  std::cout << "finished initialise SMC sampler" << std::endl;

  if (storeHistory_)
  { 
    thetaFull_[0].resize(nParticles_);
    for (unsigned int n=0; n<nParticles_; n++)
    {
      thetaFull_[0][n] = particlesNew[n].theta_;
    }
    logUnnormalisedWeightsFull_.col(0) = logUnnormalisedWeights;
    selfNormalisedWeightsFull_.col(0)  = selfNormalisedWeights;
    logLikelihoodsFull_.col(0)         = logLikelihoods;
  }
  
  t++; // step counter 
  
  /////////////////////////////////////////////////////////////////////////////
  // Iterations
  /////////////////////////////////////////////////////////////////////////////
  
  while (alphaNew_ < 1.0)
  {

    std::cout << "Step " << t << " of SMC Sampler " << std::endl;

    
    // --------------------------------------------------------------------- //
    // Determine the inverse temperature, 
    // --------------------------------------------------------------------- //
    
    alphaOld_ = alphaNew_;

    if (useAdaptiveTempering_)
    { // Adaptive tempering, 
      // i.e. numerically solve CESS(alpha) = CESS* for alpha:
      
      // Computes CESS(alpha) for given weights:
      
      auto f = [&] (double alpha) {return computeCess(alpha, alphaOld_, selfNormalisedWeights, logLikelihoods) - nParticles_* cessTarget;};

      alphaNew_ = rootFinding::bisection(isBracketing, f, alphaOld_, 2.0, 0.000000001, 0.000000001, 1000);

      std::cout << "next inverse temperature found by bisection: " << alphaNew_ << std::endl;
      
      if (!isBracketing) { std::cout << "Warning: interval is not bracketing!" << std::endl; }
      
      if (alphaNew_ > 1.0 || !isBracketing)
      {
        alphaNew_ = 1.0;
      }
      alpha_.push_back(alphaNew_);
    }
    else 
    { // Manual tempering, 
      // i.e. user-specified schedule:
      alphaNew_ = alpha_[t];
    }
    alphaInc_ = alphaNew_ - alphaOld_; // inverse temperature increment
    
    // --------------------------------------------------------------------- //
    // Update weights
    // --------------------------------------------------------------------- //
    
    logUnnormalisedWeights = logUnnormalisedWeights + alphaInc_ * logLikelihoods;
    
    // self-normalise the weights:
    selfNormalisedWeights = normaliseWeights(logUnnormalisedWeights);
    
    if (!arma::is_finite(selfNormalisedWeights))
    {
      std::cout << "WARNING: W (in the SMC sampler) contains NaNs!" << std::endl;
    }
    
    // --------------------------------------------------------------------- //
    // Compute proposal scale
    // --------------------------------------------------------------------- //

    // Computing the first two moments of the weighted sample:
//     if (mcmc_.getUseAdaptiveProposal())
//     {
//       computeSampleMoments(particlesNew, selfNormalisedWeights);
//     }

    if (mcmc_.getUseAdaptiveProposal())
    {
      // Computes the first two moments of the weighted sample.
      computeSampleMoments(particlesNew, selfNormalisedWeights);
      if (t > 1 && mcmc_.getUseAdaptiveProposalScaleFactor1())
      {
        // Adapts the additional scale factor by which the sample covariance matrix
        // is multiplied according to the empirical
        // acceptance rate at the previous SMC step.
        mcmc_.adaptProposalScaleFactor1(acceptanceRates_[acceptanceRates_.size()-1]);
      }
    }
    
    // --------------------------------------------------------------------- //
    // Adaptive systematic resampling
    // --------------------------------------------------------------------- //
    
    logPartialEvidenceEstimates_.push_back(logEvidenceEstimate_ + std::log(arma::sum(arma::exp(logUnnormalisedWeights)))); 
    
    if (computeEss(selfNormalisedWeights) < nParticles_ * essResamplingThreshold_)
    {
      std::cout << "Resampling at Step " << t << std::endl;
      isResampled_.push_back(1);
      // Update estimate of the model evidence:
      logEvidenceEstimate_ = logPartialEvidenceEstimates_[t-1];
      
      // Systematic resampling:
      resample::systematicBase(arma::randu(), parentIndices, selfNormalisedWeights, nParticles_);
      
//       std::cout << parentIndices.t() << std::endl;
      
      // Re-setting the weights:
      logUnnormalisedWeights.fill(-std::log(nParticles_)); // resetting the weights
//       selfNormalisedWeights = normaliseWeights(logUnnormalisedWeights);
      selfNormalisedWeights.fill(1.0 / nParticles_);
    } 
    else // i.e. no resampling:
    {
      std::cout << "No resampling at Step " << t << std::endl;
      isResampled_.push_back(0);
      parentIndices = arma::linspace<arma::uvec>(0, nParticles_-1, nParticles_);
    }
    // Determining the parent particles based on the parent indices: 
    for (unsigned int n=0; n<nParticles_; n++)
    {
      particlesOld[n] = particlesNew[parentIndices(n)]; // TODO: inefficient? Can we resample "in place"?
    }

    
    // --------------------------------------------------------------------- //
    // Apply MCMC kernel
    // --------------------------------------------------------------------- //
    
    if (isDoubleTemperingFirst)
    {
      if (nMetropolisHastingsUpdatesFirst_ > 1)
      {
        for (unsigned int n=0; n<nParticles_; n++)
        {
          singleParticle = particlesOld[n];
          for (unsigned int k=0; k<nMetropolisHastingsUpdatesFirst_; k++)
          {
            updateFirst(t, particlesNew[n], singleParticle, alphaNew_); // TODO: implement this!
            singleParticle = particlesNew[n];
          }
          logLikelihoods(n) = particlesNew[n].logLikelihoodFirst_;
        }
      }
      else
      {
        for (unsigned int n=0; n<nParticles_; n++)
        {
          updateFirst(t, particlesNew[n], particlesOld[n], alphaNew_);  // TODO: implement this!
          logLikelihoods(n) = particlesNew[n].logLikelihoodFirst_;
        }
      }
    }
    else // i.e. if we also update the latent variables in the 2nd part of the algorithm
    {
      if (nMetropolisHastingsUpdates_ > 1)
      {
        for (unsigned int n=0; n<nParticles_; n++)
        {
          singleParticle = particlesOld[n];
          for (unsigned int k=0; k<nMetropolisHastingsUpdates_; k++)
          {
            updateSecond(t, particlesNew[n], singleParticle, alphaNew_); 
            singleParticle = particlesNew[n];
          }
          logLikelihoods(n) = particlesNew[n].logLikelihoodSecond_; // TODO: is this correct?
        }
      }
      else
      {
        for (unsigned int n=0; n<nParticles_; n++)
        {
          updateSecond(t, particlesNew[n], particlesOld[n], alphaNew_); 
          logLikelihoods(n) = particlesNew[n].logLikelihoodSecond_;// TODO: is this correct?
        }
      }
    }
    
     ///////////////////////////////// START: computing the autocorrelation (only needed for adapting cessTarget)
//     if (useAdaptiveCessTarget_)
//     {
      maxParticleAutocorrelations_.push_back(0);
      double componentParticleAutocovariance, componentParticleAutocorrelation;
      double dimTheta = particlesNew[0].theta_.size();
      double meanOld, meanNew;
      double varOld, varNew, sdProd;
      
      for (unsigned int j=0; j<dimTheta; j++)
      {
        componentParticleAutocovariance = 0;
        meanOld = 0;
        meanNew = 0;
        varOld  = 0;
        varNew  = 0;
        
        // Computes the means.
        for (unsigned int n=0; n<nParticles_; n++)
        {
          meanNew += selfNormalisedWeights(n) * particlesNew[n].theta_(j);
          meanOld += selfNormalisedWeights(n) * particlesOld[n].theta_(j);
        }
        // Computes variances.
        for (unsigned int n=0; n<nParticles_; n++)
        {
          varNew += selfNormalisedWeights(n) * particlesNew[n].theta_(j) * particlesNew[n].theta_(j);
          varOld += selfNormalisedWeights(n) * particlesOld[n].theta_(j) * particlesOld[n].theta_(j);
        }
        varNew -= meanNew * meanNew;
        varOld -= meanOld * meanOld;
        sdProd = std::sqrt(varOld) * std::sqrt(varNew);
        // Computes the auto-covariances
        for (unsigned int n=0; n<nParticles_; n++)
        {
          componentParticleAutocovariance += selfNormalisedWeights(n) * particlesOld[n].theta_(j) * particlesNew[n].theta_(j);
        }
        componentParticleAutocovariance -= meanNew * meanOld;
        componentParticleAutocorrelation = componentParticleAutocovariance / sdProd;
//         maxParticleAutocorrelations += componentParticleAutocorrelation / dimTheta;
        
        // Option 1: Using the mean over the autocorrelations of the particle components:
//         maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] 
//           = maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] 
//             + componentParticleAutocorrelation / dimTheta;
            
        // Option 2: Using the maximum over the autocorrelations of the particle components:
        maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] 
          = std::max(maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1], componentParticleAutocorrelation);
        
//         std::cout << "; meanOld: " << meanOld << "; meanNew;" << meanNew;
//         std::cout << "; varOld: " << varOld << "; varNew: " << varNew;
//         std::cout << "; component particle autocorrelation: " << componentParticleAutocorrelation;
      }
//       std::cout << "" << std::endl;
      std::cout << "maxParticleAutocorrelation: " << maxParticleAutocorrelations_[maxParticleAutocorrelations_.size()-1] << std::endl;
//     }
    ///////////////////////////////// END: computing the autocorrelation (only needed for adapting cessTarget)
    
    // Computing the expected squared jumping distance:
    esjd.zeros();
    for (unsigned int n=0; n<nParticles_; n++) 
    {
      esjd = esjd + arma::pow(particlesNew[n].theta_ - particlesOld[n].theta_, 2.0);
    }
    esjd = arma::sqrt(esjd / nParticles_);
//     std::cout << "ESJD: " << esjd.t() << std::endl;
    std::cout << "Mean ESJD: " << arma::accu(esjd)/esjd.size() << std::endl;
    
    if (mcmc_.getUseDelayedAcceptance()) {
      acceptanceRatesFirst_.push_back(static_cast<double>(nAcceptedMovesFirst_) / (nParticles_ * nMetropolisHastingsUpdates_)); 
      std::cout << "First-stage acceptance rate at Step " << t << ": " << acceptanceRatesFirst_[acceptanceRatesFirst_.size()-1] << std::endl;
      nAcceptedMovesFirst_ = 0;
    }
    
    acceptanceRates_.push_back(static_cast<double>(nAcceptedMoves_) / (nParticles_ * nMetropolisHastingsUpdates_)); 
    std::cout << "Overall acceptance rate at Step     " << t << ": " << acceptanceRates_[acceptanceRates_.size()-1] << std::endl;
    nAcceptedMoves_ = 0;
    

    // --------------------------------------------------------------------- //
    // Store output
    // --------------------------------------------------------------------- //
    
//     std::cout << "loglikelihoods at step " << t << ": " << logLikelihoods.t() << std::endl;
    
    if (storeHistory_)
    { 
      if (useAdaptiveTempering_)
      {
        std::vector<arma::colvec> thetaNew(nParticles_);
        
        for (unsigned int n=0; n<nParticles_; n++)
        {
          //thetaFull_[t][n] = particlesNew[n].theta_; // TODO
          thetaNew[n] = particlesNew[n].theta_; // TODO
        }
        thetaFull_.push_back(thetaNew);

        logUnnormalisedWeightsFull_ = arma::join_rows(logUnnormalisedWeightsFull_, logUnnormalisedWeights);
        selfNormalisedWeightsFull_  = arma::join_rows(selfNormalisedWeightsFull_,  selfNormalisedWeights);
        logLikelihoodsFull_         = arma::join_rows(logLikelihoodsFull_,         logLikelihoods);
      }
      else 
      {

        std::vector<arma::colvec> thetaNew(nParticles_);
        for (unsigned int n=0; n<nParticles_; n++)
        {
          thetaNew[n] = particlesNew[n].theta_; // TODO
        }
                
        thetaFull_[t] = thetaNew;
        
        logUnnormalisedWeightsFull_.col(t) = logUnnormalisedWeights;
        selfNormalisedWeightsFull_.col(t)  = selfNormalisedWeights;
        logLikelihoodsFull_.col(t)         = logLikelihoods;
      }
    }
    
    t++;
    
    if (alphaNew_ >= 1 && isDoubleTemperingFirst)
    {
      isDoubleTemperingFirst = false;
      cessTarget = cessTarget_;
      for (unsigned int n=0; n<nParticles_; n++)
      {
        initialiseSecond(particlesNew[n]); // TODO: implement this!
        logLikelihoods(n) = particlesNew[n].getlogLikelihood();
      }
      alphaNew_ = 0.0;
      alphaOld_ = 0.0;
    }
    
    /////////////////////////////
    //////////
//     logLikelihoods.replace(arma::datum::nan, -arma::datum::inf); 
    //////////
  
//     std::cout << "loglikelihoods at step " << t << ": " << logLikelihoods.t() << std::endl;

//  std::cout << "loglikelihoods at step " << t << "has nans? " << logLikelihoods.has_nan() << std::endl;
//     for (unsigned int n=0; n<nParticles_; n++)
//     {
//       std::cout << " " << logLikelihoods(n) << " ";
//     }
    ///////////////////////////
    
  } // end of while loop
  
  
  if (useAdaptiveTempering_) { nSteps_ = t; }
  if (!isResampled_[isResampled_.size()-1])
  {
//     logEvidenceEstimate_ += std::log(arma::accu(arma::exp(logUnnormalisedWeights)));
    logEvidenceEstimate_ = logPartialEvidenceEstimates_[logPartialEvidenceEstimates_.size()-1];
  }
  
  finalParticles_ = particlesNew;

}

/// Computes various quantities for recycling all particles in an
/// importance-tempering like approach
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters, class McmcParameters>
void SmcSampler<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters, McmcParameters>::computeImportanceTemperingWeights()
{

  ///////////////////////////////////////////////////////////////////////////
  // Reweighting standard particle weights so that particles from
  // each generation target the posterior
  ///////////////////////////////////////////////////////////////////////////
  
  std::cout << "Reweighting standard particle weights" << std::endl;
  
  logUnnormalisedReweightedWeights_.set_size(nParticles_, nSteps_);
  
  // TODO: logUnnormalisedReweightedWeights now includes the products of sums of unnormalised weights
  // but logUnnormalisedWeights does not; this is inconsistent and may lead to wrong results when 
  // normalising constants are computed in R using the output of the SMC sampler; this needs to be fixed.

//   arma::colvec partialLogEvidenceEstimate(nSteps_, arma::fill::zeros);
// 
//   for (unsigned int t=1; t<nSteps_; t++)
//   {
//     if (!isResampled_[t-1])
//     {
//       partialLogEvidenceEstimate(t) = arma::accu(partialLogEvidenceEstimate(arma::span(0,t-1))) 
//         + std::log(arma::accu(arma::exp(logUnnormalisedWeightsFull_.col(t))));
//     }
//   }
  
  for (unsigned int t=0; t<nSteps_; t++)
  {
    // log-unnormalised weights reweighted to target the unnormalised posterior
    logUnnormalisedReweightedWeights_.col(t) = 
      logUnnormalisedWeightsFull_.col(t) + (1.0 - alpha_[t]) * logLikelihoodsFull_.col(t);   
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // Applying additional resampling steps (these are used in the approach
  // from Nguyen, Septier, Peters & Delignon (2016) in order to directly
  // apply the importance tempering approach from Grammacy, Samworth & King
  // (2010) which assumes that the particles within each generation are
  // already equally weighted (i.e. have been resampled)
  ///////////////////////////////////////////////////////////////////////////
  
  std::cout << "Applying additional resampling steps" << std::endl;
  
  arma::uvec resampledIndices(nParticles_);  
  thetaFullResampled_ = thetaFull_;
  logUnnormalisedReweightedWeightsResampled_ = logUnnormalisedReweightedWeights_;
  
  // NOTE: the particles are evenly weighted at Step 0!
  
  for (unsigned int t=1; t<nSteps_; t++)
  {
    if (!isResampled_[t-1]) 
    {
      resampledIndices = sampleInt(nParticles_, selfNormalisedWeightsFull_.col(t));
      
      for (unsigned int n=0; n<nParticles_; n++)
      {
        thetaFullResampled_[t][n] = thetaFull_[t][resampledIndices(n)];
        logUnnormalisedReweightedWeightsResampled_(n,t) = (1 - alpha_[t]) * logLikelihoodsFull_(resampledIndices(n),t);
      }
//       logUnnormalisedReweightedWeightsResampled_.col(t) = logUnnormalisedReweightedWeightsResampled_.col(t) - std::log(nParticles_) + std::log(arma::accu(arma::exp(logUnnormalisedWeightsFull_.col(t))));
      logUnnormalisedReweightedWeightsResampled_.col(t) = logUnnormalisedReweightedWeightsResampled_.col(t) - std::log(nParticles_) + logPartialEvidenceEstimates_[t-1];
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // Computing the self-normalised weights for targeting the posterior
  ///////////////////////////////////////////////////////////////////////////
  
  selfNormalisedReweightedWeights_.set_size(nParticles_, nSteps_);
  selfNormalisedReweightedWeightsResampled_.set_size(nParticles_, nSteps_);
  
  for (unsigned int t=0; t<nSteps_; t++)
  {
    selfNormalisedReweightedWeights_.col(t)          = normaliseWeights(logUnnormalisedReweightedWeights_.col(t));
    selfNormalisedReweightedWeightsResampled_.col(t) = normaliseWeights(logUnnormalisedReweightedWeightsResampled_.col(t));
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // Computing the effective sample sizes
  ///////////////////////////////////////////////////////////////////////////
  
  std::cout << "Computing the effective sample sizes" << std::endl;
  
  ess_.set_size(nSteps_);
  essResampled_.set_size(nSteps_);
  cess_.set_size(nSteps_);

  for (unsigned int t=0; t<nSteps_; t++)
  {
    ess_(t)          = 1.0 /  arma::dot(selfNormalisedReweightedWeights_.col(t), selfNormalisedReweightedWeights_.col(t));  
    essResampled_(t) = 1.0 /  arma::dot(selfNormalisedReweightedWeightsResampled_.col(t), selfNormalisedReweightedWeightsResampled_.col(t));
    cess_(t)         = computeCess(1.0, alpha_[t], selfNormalisedWeightsFull_.col(t), logLikelihoodsFull_.col(t));
    
    
  }

      
  ///////////////////////////////////////////////////////////////////////////
  // Computing self-normalised importance-tempering type weights used for
  // approximating integrals with respect to the normalised(!) posterior 
  // distribution.
  ///////////////////////////////////////////////////////////////////////////
  
  std::cout << "Computing self-normalised importance-tempering type weights" << std::endl;
  
  selfNormalisedWeightsEss_.set_size(nParticles_, nSteps_);
  selfNormalisedWeightsEssResampled_.set_size(nParticles_, nSteps_);
  selfNormalisedWeightsCess_.set_size(nParticles_, nSteps_);
  
  double sumOfEss          = arma::accu(ess_);
  double sumOfEssResampled = arma::accu(essResampled_);
  double sumOfCess         = arma::accu(cess_);
  
  for (unsigned int t=0; t<nSteps_; t++)
  {
    selfNormalisedWeightsEss_.col(t)          = selfNormalisedReweightedWeights_.col(t)          * ess_(t)          / sumOfEss;
    selfNormalisedWeightsEssResampled_.col(t) = selfNormalisedReweightedWeightsResampled_.col(t) * essResampled_(t) / sumOfEssResampled;
    selfNormalisedWeightsCess_.col(t)         = selfNormalisedReweightedWeights_.col(t)          * cess_(t)         / sumOfCess;
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // Computing importance-tempering estimators for the model evidence
  ///////////////////////////////////////////////////////////////////////////
  
  std::cout << "Computing importance-tempering estimators for the model evidence" << std::endl;
  
  // The following estimators are based around a version of the ESS
  // suitable for unnormalised weights.
  
  double logC          = logUnnormalisedReweightedWeights_.max();
  double logCResampled = logUnnormalisedReweightedWeightsResampled_.max();
  
  double auxEss1          = 0.0;
  double auxEss2          = 0.0;
  double auxEssResampled1 = 0.0;
  double auxEssResampled2 = 0.0;
  double auxCess1         = 0.0;
  double auxCess2         = 0.0;
  
  for (unsigned int t=0; t<nSteps_; t++) // TODO
  {
    auxEss1          += ess_(t) / (arma::accu(arma::exp(logUnnormalisedReweightedWeights_.col(t) - logC)));
    auxEss2          += 1.0 / (arma::accu(arma::exp(2.0 * (logUnnormalisedReweightedWeights_.col(t) - logC))));
    
    auxEssResampled1 += essResampled_(t) / (arma::accu(arma::exp(logUnnormalisedReweightedWeightsResampled_.col(t) - logCResampled)));
    auxEssResampled2 += 1.0 / (arma::accu(arma::exp(2.0 * (logUnnormalisedReweightedWeightsResampled_.col(t) - logCResampled))));
    
    auxCess1         += cess_(t) / (arma::accu(arma::exp(logUnnormalisedReweightedWeights_.col(t) - logC)));
    auxCess2         += 1.0 / (arma::accu(arma::exp(2.0 * (logUnnormalisedReweightedWeights_.col(t) - logC))));
  }

  logEvidenceEstimateEss_          = logC          + std::log(auxEss1)          - std::log(auxEss2);   
  logEvidenceEstimateEssResampled_ = logCResampled + std::log(auxEssResampled1) - std::log(auxEssResampled2);
  logEvidenceEstimateCess_         = logC          + std::log(auxCess1)         - std::log(auxCess2);  
  
  // The following estimators simple give weights proportional to the ESS or CESS to
  // the unbiased estimates of the normalising constant obtained from each 
  // particle generation:
  
  double evidenceEstimateEssAlternate          = 0.0;
  double evidenceEstimateEssResampledAlternate = 0.0;
  double evidenceEstimateCessAlternate         = 0.0; // TODO: check the validity of this approach!
  
  for (unsigned int t=0; t<nSteps_; t++) // TODO
  {
    evidenceEstimateEssAlternate          += ess_(t)          * arma::accu(arma::exp(logUnnormalisedReweightedWeights_.col(t)          - logC));
    evidenceEstimateEssResampledAlternate += essResampled_(t) * arma::accu(arma::exp(logUnnormalisedReweightedWeightsResampled_.col(t) - logCResampled));
    evidenceEstimateCessAlternate         += cess_(t)         * arma::accu(arma::exp(logUnnormalisedReweightedWeights_.col(t)          - logC));
  }
  
  logEvidenceEstimateEssAlternate_          = logC          + std::log(evidenceEstimateEssAlternate)          - std::log(sumOfEss);
  logEvidenceEstimateEssResampledAlternate_ = logCResampled + std::log(evidenceEstimateEssResampledAlternate) - std::log(sumOfEssResampled);
  logEvidenceEstimateCessAlternate_         = logC          + std::log(evidenceEstimateCessAlternate)         - std::log(sumOfCess);
}


#endif
