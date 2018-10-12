/// \file
/// \brief Generating data and performing inference in a multivariate linear Gaussian state-space model. 
///
/// This file contains the functions for implementing the abstract model
/// class for the in a multivariate linear Gaussian state-space model. 

#ifndef __LINEAR_H
#define __LINEAR_H

#include "base/templates/dynamic/stateSpace/multivariate/linearGaussian/linearGaussian.h"
#include "base/mcmc/Mcmc.h"
// #include "projects/optim/default/default.h" // TODO: are we really using this?
#include "base/kalman.h"
// #include "base/highDim.h" // TODO: are we really using this?

// [[Rcpp::depends("RcppArmadillo")]]

// TODO: obtain speed-ups by sampling all particles in parallel as one arma::mat
// and then converting into an std::vector<arma::colvec>

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the model
////////////////////////////////////////////////////////////////////////////////

/// Holds all the static model parameters.
class ModelParameters
{
public:
  
  /// Returns the number of non-zero diagonals 
  /// on each side of the main diagonal.
  unsigned int getNOffDiagonals() const {return nOffDiagonals_;}
  /// Returns the length of a single latent-variable vector.
  unsigned int getDimLatentVariable() const {return dimLatentVariable_;}
  /// Returns the length of a single observation vector.
  unsigned int getDimObservation() const {return dimObservation_;}
  /// Returns the transition matrix in the state-transition equation.
  const arma::mat& getA() const {return A_;}
  /// Returns the root of the covariance matrix in the transition equation.
  const arma::mat& getB() const  {return B_;}
  /// Returns the transition matrix in the observation equation.
  const arma::mat& getC() const {return C_;}
  /// Returns the root of the covariance matrix in the observation equation.
  const arma::mat& getD() const  {return D_;}
  /// Returns the prior mean of the inital state.
  const arma::colvec& getM0() const {return m0_;}
  /// Returns the prior covariance matrix of the initial state.
  const arma::mat& getC0() const {return C0_;}
  /// Returns one element of the root of the covariance matrix in the transition equation.
  double getB(const unsigned int i, const unsigned int j) const  {return B_(i,j);}
  /// Returns one element of the root of the covariance matrix in the observation equation.
  double getD(const unsigned int i, const unsigned int j) const  {return D_(i,j);}
  /// Returns the one element in the prior covariance matrix of the initial state.
  double getC0(const unsigned int i, const unsigned int j) const  {return C0_(i,j);}
  /// Returns the shape parameter of gamma the prior on b.
  double getShapeHyperB() const {return shapeHyperB_;}
  /// Returns the shape parameter of gamma the prior on d.
  double getShapeHyperD() const {return shapeHyperD_;}
  /// Returns the scale parameter of gamma the prior on b.
  double getScaleHyperB() const {return scaleHyperB_;}
  /// Returns the scale parameter of gamma the prior on d.
  double getScaleHyperD() const {return scaleHyperD_;}
  /// Returns the parametrisation of the static parameters.
  const highDim::ParametrisationType& getParam() const {return param_;}
  
  /// Return some auxiliary quantities needed to avoid duplicate computations
  const arma::mat& getBBT() const {return BBT_;}
  const arma::mat& getDDT() const {return DDT_;}
  const arma::mat& getCA() const  {return CA_;}
  const arma::mat& getInvBBTA() const {return invBBTA_;}
  const arma::mat& getCTinvDDT() const {return CTinvDDT_;}
  
  const arma::mat& getOptPropVar() const {return optPropVar_;}
  const arma::mat& getOptWeightVar() const {return optWeightVar_;}
  const arma::mat& getOptPropInitialVar() const {return optPropInitialVar_;}
  const arma::mat& getOptWeightInitialVar() const {return optWeightInitialVar_;}
  const arma::colvec& getOptWeightInitialMean() const {return optWeightInitialMean_;}
    
  /// Determines the model parameters from arma::colvec theta.
  void setUnknownParameters(const arma::colvec& theta)
  {
    highDim::computeModelParameters(A_, B_, C_, D_, nOffDiagonals_, dimLatentVariable_, dimObservation_, theta, param_);
    
//     std::cout << "Started setting unknown parameters" << std::endl;
    
    BBT_ = B_ * B_.t();
    DDT_ = D_ * D_.t();
    CTinvDDT_ = C_.t() * arma::inv(DDT_);
    
    CA_ = C_ * A_;
    invBBTA_ = arma::inv(BBT_) * A_;
    
    optPropVar_ = arma::inv(arma::inv(BBT_) + CTinvDDT_ * C_);
    
    optWeightVar_        = DDT_ + C_ * BBT_ * C_.t();
    optWeightInitialVar_ = DDT_ + C_ * C0_ * C_.t();
    
    optPropInitialVar_ = arma::inv(arma::inv(C0_) + CTinvDDT_ * C_);
    optWeightInitialMean_ = C_ * m0_;
    
//      std::cout << "Finished setting unknown parameters" << std::endl;
    
  }
  /// Determines the hyperparameters.
  void setKnownParameters(const arma::colvec& hyp)
  {
    dimLatentVariable_ = hyp(4);
    dimObservation_ = hyp(5);
    nOffDiagonals_ = hyp(6);
    m0_.zeros(dimLatentVariable_);
    C0_.eye(dimLatentVariable_, dimLatentVariable_);
    shapeHyperB_ = hyp(0);
    scaleHyperB_ = hyp(1);
    shapeHyperD_ = hyp(2);
    scaleHyperD_ = hyp(3);
  }
  
private:
  
  unsigned int dimLatentVariable_; // (known) dimension of the states
  unsigned int dimObservation_; // (known) dimension of the observations
  unsigned int nOffDiagonals_; // number of non-zero diagonals on each side of the main diagonal
  arma::mat A_, B_, C_, D_; // model parameters as a transformation of theta
  arma::colvec m0_; // initial mean vector
  arma::mat C0_; // initial covariance matrix
  
  /// Some auxiliary parameters:
  arma::mat optWeightInitialVar_;  // the covariance matrix of p(y_1)
  arma::mat optPropVar_; // the covariance matrix of p(x_t|x_{t-1}, y_{t-1})
  arma::mat optWeightVar_; // the covariance matrix of p(x_t|y_{t-1})
  arma::colvec optWeightInitialMean_; // the mean of p(y_1)
  arma::mat optPropInitialVar_; // the covariance matrix of p(x_1|y_1)
  
  arma::mat CA_, invBBTA_, BBT_, DDT_; // C * A; arma::inv(B*B.t()) * A; B*B.t(); D*D.t()
  arma::mat CTinvDDT_; // C.t() * arma::inv(D*D.t())
  
  /// Hyperparameters:
  double shapeHyperB_, scaleHyperB_, shapeHyperD_, scaleHyperD_;
  highDim::ParametrisationType param_ = highDim::HIGHDIM_PARAMETRISATION_NATURAL;

};

////////////////////////////////////////////////////////////////////////////////
// Containers associated with the algorithm
////////////////////////////////////////////////////////////////////////////////

/// Holds some additional auxiliary parameters for the SMC algorithm.
class SmcParameters
{
  
public:
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& algorithmParameters)
  {
    randomWalkProposalSd_ = algorithmParameters; // NOTE: at the moment the only SMC parameters are the scale parameters for the random-walk CSMC algorithm
  }
  /// Returns the standard deviation for the general Gaussian random-walk proposals.
  double getRandomWalkProposalSd(const unsigned int t) const {return randomWalkProposalSd_(t);}
  /// Specifies the standard deviation for the general Gaussian random-walk proposals.
  void setRandomWalkProposalSd(const arma::colvec& randomWalkProposalSd) {randomWalkProposalSd_ = randomWalkProposalSd;}
  
private:
  arma::colvec randomWalkProposalSd_; // the scales (in terms of standard deviation) parameters at each time step used by the general Gaussian random-walk proposal from Tjelmeland (2004). 
  
};
/// Holds some additional auxiliary parameters for the SMC algorithm.
class McmcParameters
{
  
public:
  
  /// Determines the parameters.
  void setParameters(const arma::colvec& algorithmParameters)
  {
    // Empty;
  }
  
private:
};


///////////////////////////////////////////////////////////////////////////////
/// Member functions of the Model class
///////////////////////////////////////////////////////////////////////////////


/// Evaluates the log-prior density of the parameters.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogPriorDensity()
{
  double logDensity = 0;
  logDensity += R::dgamma(1.0/modelParameters_.getB(0,0), modelParameters_.getShapeHyperB()+2.0, 1.0 / modelParameters_.getScaleHyperB(), true);
  logDensity += R::dgamma(1.0/modelParameters_.getD(0,0), modelParameters_.getShapeHyperD()+2.0, 1.0 / modelParameters_.getScaleHyperD(), true);
  return logDensity;
}
/// Samples the set of parameters from the prior.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::sampleFromPrior(arma::colvec& theta)
{
  // TODO: needs to depend on the parametrisation
  theta.set_size(modelParameters_.getNOffDiagonals()+3);

  // NOTE: the prior on each of the diagonal elements of the matrix
  // A, is uniform on the intersection of the stated support 
  // and the space on which the resulting matrix A 
  // induces a stationary latent Markov chain. Sampling from this
  // uniform prior is currently implemented via a simple rejection-sampling
  // step but this could probably be optimised.
  bool isStationary = false;
  
  while (!isStationary)
  {
    for (unsigned int k=0; k<modelParameters_.getNOffDiagonals()+1; k++)
    {
      theta(k) = getSupportMin(k) + (getSupportMax(k) - getSupportMin(k)) * arma::randu();
    }
    isStationary = highDim::checkStationarity(modelParameters_.getDimLatentVariable(), theta, modelParameters_.getNOffDiagonals());
  }
  theta(modelParameters_.getNOffDiagonals()+1) = 1.0 / (rng_.randomGamma(modelParameters_.getShapeHyperB(), 1.0 / modelParameters_.getScaleHyperB()));
  theta(modelParameters_.getNOffDiagonals()+2) = 1.0 / (rng_.randomGamma(modelParameters_.getShapeHyperD(), 1.0 / modelParameters_.getScaleHyperD()));
}
/// Increases the gradient by the gradient of the log-prior density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogPriorDensity(arma::colvec& gradient)
{
//   switch (modelParameters_.getParam())
//   {
//     case HIGHDIM_PARAMETRISATION_UNBOUNDED:
// 
//     case HIGHDIM_PARAMETRISATION_NATURAL:  
//       
//   }
//   gradient(modelParameters_.getNOffDiagonals()+1) += (modelParameters_.getScaleHyperB() / modelParameters_.getB(0,0) - modelParameters_.getShapeHyperB() - 1.0) / modelParameters_.getB(0,0);
//   gradient(modelParameters_.getNOffDiagonals()+2) += (modelParameters_.getScaleHyperD() / modelParameters_.getD(0,0) - modelParameters_.getShapeHyperD() - 1.0) / modelParameters_.getD(0,0);
}
/// Increases the gradient by the gradient of the log-initial density of
/// the latent states.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogInitialDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient)
{
  // Do nothing here as the initial distribution 
  // for the states does not depend on 
  // any of the unknown parameters.
}
/// Increases the gradient by the gradient of the log-transition density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogTransitionDensity(const unsigned int t, const LatentVariable& latentVariableNew, const LatentVariable& latentVariableOld, arma::colvec& gradient)
{
//   // TODO: make this more efficient
// 
//   switch (modelParameters_.getParam())
//   {
//     case HIGHDIM_PARAMETRISATION_UNBOUNDED:
// 
//     case HIGHDIM_PARAMETRISATION_NATURAL:  
//       
//   }
//   
//   for (unsigned int k=0; k<modelParameters_.getNOffDiagonals()+1; k++)
//   {
//     gradient(k) += // TODO
//   }
//   gradient(nOffDiagonals+1) += std::pow(arma::accu(observations_.col(t)) - arma::accu(latentVariable), 2.0)  / std::pow(modelParameters_.getB(0,0), 2.0) - modelParameters_.getDimLatentVariable();
}
/// Increases the gradient by the gradient of the log-observation density.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations>
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::addGradLogObservationDensity(const unsigned int t, const LatentVariable& latentVariable, arma::colvec& gradient)
{
//   // TODO: make this more efficient
//   
//   switch (modelParameters_.getParam())
//   {
//     case HIGHDIM_PARAMETRISATION_UNBOUNDED:
// 
//     case HIGHDIM_PARAMETRISATION_NATURAL:  
//       
//   }
//   gradient(nOffDiagonals+2) += arma::dot(observation_.col(t) - latentVariable, observation_.col(t) - latentVariable)   / std::pow(modelParameters_.getD(0,0), 2.0) - modelParameters_.getDimLatentVariable();
}
/// Evaluates the marginal likelihood of the parameters (with the latent 
/// variables integrated out). Note that analytical expressions for this 
/// will not be available for most models.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
double Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateLogMarginalLikelihood()
{
  return kalman::evaluateLogMarginalLikelihood(modelParameters_.getA(), modelParameters_.getB(), modelParameters_.getC(), modelParameters_.getD(), modelParameters_.getM0(), modelParameters_.getC0(), getObservations()); 
}
/// Evaluates the score.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::evaluateScore(arma::colvec& score)
{
  if (modelParameters_.getDimObservation() != modelParameters_.getLatentVariable()) 
  {
    std::cout << "WARNING: computing the score is currently only implemented for dimX == dimY!" << std::endl;
  }
  else 
  {
    score.zeros(dimTheta_);
    arma::colvec theta(dimTheta_);
    theta(arma::span(0,modelParameters_.getNOffDiagonals())) = modelParameters_.getA()(arma::span(0,modelParameters_.getNOffDiagonals()), arma::span(0,0));
    theta(dimTheta_-2) = modelParameters_.getB(0,0);
    theta(dimTheta_-1) = modelParameters_.getD(0,0);
    highDim::computeScore(score, theta, modelParameters_.getParam(), modelParameters_.getM0(), modelParameters_.getC0(), getObservations());
  }
}
/// Generates latent variables from their full conditional distribution.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations> 
void Model<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations>::runGibbs(LatentPath& latentPath)
{
  latentPath.set_size(modelParameters_.getDimLatentVariable(), observations_.n_cols);
  kalman::runForwardFilteringBackwardSampling(latentPath, modelParameters_.getA(), modelParameters_.getB(), modelParameters_.getC(), modelParameters_.getD(), modelParameters_.getM0(), modelParameters_.getC0(), getObservations());
}


///////////////////////////////////////////////////////////////////////////////
/// Member functions of Smc class
///////////////////////////////////////////////////////////////////////////////

/// Samples particles at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleParticles
(
  const unsigned int t, 
  std::vector<Particle>& particlesNew,
  const std::vector<Particle>& particlesOld
)
{
  double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//   double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
  unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
//   unsigned int dimObservation = model_.getModelParameters().getDimObservation();
  arma::colvec mu(dimLatentVariable);
  
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR) 
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = model_.getModelParameters().getA() * particlesOld[n];
      particlesNew[n] = gaussian::sampleMultivariate(1, mu, b2, false);
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || smcProposalType_ == SMC_PROPOSAL_FA_APF)
  { 
    /*
    double sigma = 1.0/(1.0/d2 + 1.0/b2);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = sigma*(1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesOld[n]);                   
      particlesNew[n] = gaussian::sampleMultivariate(1, mu, sigma, false);
    }
    */
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = model_.getModelParameters().getOptPropVar()*(model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(t) + model_.getModelParameters().getInvBBTA() * particlesOld[n]);                   
      particlesNew[n] = gaussian::sampleMultivariate(1, mu, model_.getModelParameters().getOptPropVar(), false);
    }
  }

  // TODO: check if this is correct
  if (smcProposalType_ == SMC_PROPOSAL_GAUSSIAN_RANDOM_WALK && isConditional_)
  {
//     unsigned int nDimensions = model_.getModelParameters().getDimLatentVariable(); // model dimension
    double proposalSd = smcParameters_.getRandomWalkProposalSd(t) / std::sqrt(2.0);
//     double proposalSd = 1.0 / std::sqrt(dimLatentVariable * 2.0);
    
    // the centre around which the proposed particles are going to be sampled
    arma::colvec newCentre =  particlePath_[t] + proposalSd * arma::randn<arma::colvec>(dimLatentVariable);
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      particlesNew[n] = newCentre + proposalSd * arma::randn<arma::colvec>(dimLatentVariable);
    }
  }
 
  if (isConditional_) {particlesNew[particleIndicesIn_(t)] = particlePath_[t];}
  
}
/// Computes a particle weight at Step 0.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogParticleWeights
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew,
  const std::vector<Particle>& particlesOld,
  arma::colvec& logWeights
)
{
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t), model_.getModelParameters().getC() * particlesNew[n], model_.getModelParameters().getD(0,0), true, true));
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL)
  {     
//     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t), model_.getModelParameters().getCA() * particlesOld[n], model_.getModelParameters().getOptWeightVar(), false, true));
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_FA_APF)
  {     
    if (t+1 < nSteps_)
    {
//       double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//       double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
      for (unsigned int n=0; n<getNParticles(); n++)
      {
        logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t+1), model_.getModelParameters().getCA()* particlesNew[n], model_.getModelParameters().getOptWeightVar(), false, true));
      }
    }
    else
    {
      // Do nothing here because the final-time incremental weights are unity.
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_GAUSSIAN_RANDOM_WALK && isConditional_)
  {
    // TODO: check this!
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += 
        arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t), model_.getModelParameters().getC() * particlesNew[n], model_.getModelParameters().getD(0,0), true, true)) + model_.evaluateLogTransitionDensity(t, particlesNew[n], particlesOld[n]);
//         model_.evaluateLogObservationDensity(t, particlesNew[n] +
//         model_.evaluateLogTransitionDensity(t, particlesNew[n], particlesOld[n]);
        
    }
//     std::cout << arma::sum(logWeights) << "          ";
//     if (!std::isfinite(logWeights(n)))
//     {
 //     std::cout << logWeights(n) << std::endl; ///////////////////
//     }
  }
}
/// Reparametrises particles at Step 0 to obtain the values of Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromParticles
(
  const unsigned int t,
  const std::vector<Particle>& particlesNew, 
  const std::vector<Particle>& particlesOld, 
  std::vector<Aux>& aux1
)
{
  if (model_.getModelParameters().getDimLatentVariable() != model_.getModelParameters().getDimObservation())
  {
    std::cout << "WARNING: determineGaussiansFromParticles() is not implemented for dimX != dimY!" << std::endl;
  }
  
  arma::colvec mu(model_.getModelParameters().getDimLatentVariable());
  double sigma;
  
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  {
    sigma = model_.getModelParameters().getB(0,0);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = model_.getModelParameters().getA() * particlesOld[n];
      aux1[n] = (particlesNew[n] - mu) / sigma;
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || smcProposalType_ == SMC_PROPOSAL_FA_APF)
  {     
    double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
    double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    sigma = std::sqrt(1.0/(1.0/d2 + 1.0/b2));
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = sigma * (1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesOld[n]);
      aux1[n] = (particlesNew[n] - mu) / sigma;
    }
  }
}
/// Reparametrises Gaussians at Step 0 to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineParticlesFromGaussians
(
  const unsigned int t,
  std::vector<Particle>& particlesNew, 
  const std::vector<Particle>& particlesOld,
  const std::vector<Aux>& aux1
)
{
  
  if (model_.getModelParameters().getDimLatentVariable() != model_.getModelParameters().getDimObservation())
  {
    std::cout << "WARNING: determineParticlesFromGaussians() is not implemented for dimX != dimY!" << std::endl;
  }
  
  arma::colvec mu(model_.getModelParameters().getDimLatentVariable());
  double sigma;
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  {
    sigma = model_.getModelParameters().getB(0,0);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = model_.getModelParameters().getA() * particlesOld[n];
      particlesNew[n] = mu + sigma * aux1[n];
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || smcProposalType_ == SMC_PROPOSAL_FA_APF)
  {     
    double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
    double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    sigma = std::sqrt(1.0/(1.0/d2 + 1.0/b2));
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = sigma*(1.0/d2 * model_.getObservations().col(t) + 1.0/b2 * model_.getModelParameters().getA() * particlesOld[n]);
      particlesNew[n] = mu + sigma * aux1[n];
    }
  }
}
/// Samples particles at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::sampleInitialParticles
(
  std::vector<Particle>& particlesNew
)
{
//   double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
  unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
  arma::colvec mu(dimLatentVariable);
  
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR) 
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      particlesNew[n] = gaussian::sampleMultivariate(1, model_.getModelParameters().getM0(), model_.getModelParameters().getC0(), true);
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || smcProposalType_ == SMC_PROPOSAL_FA_APF)
  {      
//     arma::mat sigma = arma::inv(1.0/d2 * arma::eye(dimLatentVariable, dimLatentVariable) + arma::inv(model_.getModelParameters().getC0()));                     
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = model_.getModelParameters().getOptPropInitialVar()*(model_.getModelParameters().getCTinvDDT() * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());        
      particlesNew[n] = gaussian::sampleMultivariate(1, mu, model_.getModelParameters().getOptPropInitialVar(), false);
    }
  }

  
  // TODO: check if this is correct
  if (smcProposalType_ == SMC_PROPOSAL_GAUSSIAN_RANDOM_WALK && isConditional_)
  {
    double proposalSd = smcParameters_.getRandomWalkProposalSd(0) / std::sqrt(2.0);
    
    // the centre around which the proposed particles are going to be sampled
    arma::colvec newCentre = particlePath_[0] + proposalSd * arma::randn<arma::colvec>(dimLatentVariable);
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      particlesNew[n] = newCentre + proposalSd * arma::randn<arma::colvec>(dimLatentVariable);
    }
  }
  
  if (isConditional_) {particlesNew[particleIndicesIn_(0)] = particlePath_[0];}
}
/// Computes the incremental particle weights at Step t.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::computeLogInitialParticleWeights
(
  const std::vector<Particle>& particlesNew,
  arma::colvec& logWeights
)
{
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getC() * particlesNew[n], model_.getModelParameters().getD(0,0), true, true));
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL)
  {   
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getOptWeightInitialMean(), model_.getModelParameters().getOptWeightInitialVar(), false, true));
    }   
  }
  else if (smcProposalType_ == SMC_PROPOSAL_FA_APF)
  {
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    if (nSteps_ > 0)
    {
//       double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//       double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
      for (unsigned int n=0; n<getNParticles(); n++)
      {
        logWeights(n) += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(1), model_.getModelParameters().getCA() * particlesNew[n], model_.getModelParameters().getOptWeightVar(), false, true));
      }
    }
    logWeights += arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getOptWeightInitialMean(), model_.getModelParameters().getOptWeightInitialVar(), false, true));
  }
  else if (smcProposalType_ == SMC_PROPOSAL_GAUSSIAN_RANDOM_WALK && isConditional_)
  {
    // TODO
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      /*
      for (unsigned int d=0; d<particlesNew[n].size(); d++)
      {
        logWeights(n) += 
          R::dnorm(model_.getObservations()(d,0), particlesNew[n](d), 1.0, true) + 
          R::dnorm(particlesNew[n](d), 0.0, 1.0, true);
      }
      */

      
      logWeights(n) += 
        arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(0), model_.getModelParameters().getC() * particlesNew[n], model_.getModelParameters().getD(0,0), true, true)) + model_.evaluateLogInitialDensity(particlesNew[n]);
//         model_.evaluateLogObservationDensity(t, particlesNew[n] +
//         model_.evaluateLogInitialDensity(particlesNew[n]);

      
    }
  }
}
/// Reparametrises the particles at Step t to obtain the value of 
/// Gaussian random variables.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineGaussiansFromInitialParticles
(
  const std::vector<Particle>& particlesNew, 
  std::vector<Aux>& aux1
)
{ 
  if (model_.getModelParameters().getDimLatentVariable() != model_.getModelParameters().getDimObservation())
  {
    std::cout << "WARNING: determineGaussiansFromInitialParticles() is not implemented for dimX != dimY!" << std::endl;
  }
  
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      aux1[n] = (particlesNew[n] - model_.getModelParameters().getM0()) / std::sqrt(model_.getModelParameters().getC0(0,0));
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || smcProposalType_ == SMC_PROPOSAL_FA_APF)
  {     
    unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
    double d2         = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    double sigma      = d2 + 1.0 / (model_.getModelParameters().getC0(0,0));    
    arma::colvec mu(dimLatentVariable);
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = sigma*(1.0/d2 * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());   
      aux1[n] = (particlesNew[n] - mu) / std::sqrt(sigma);
    }
  }
}
/// Reparametrises Gaussians at Step 0 to obtain the particles.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters> 
void Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::determineInitialParticlesFromGaussians
(
  std::vector<Particle>& particlesNew,  
  const std::vector<Aux>& aux1
)
{
  if (model_.getModelParameters().getDimLatentVariable() != model_.getModelParameters().getDimObservation())
  {
    std::cout << "WARNING: determineInitialParticlesFromGaussians() is not implemented for dimX != dimY!" << std::endl;
  }
  
  if (smcProposalType_ == SMC_PROPOSAL_PRIOR)
  {
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      particlesNew[n] = model_.getModelParameters().getM0() + std::sqrt(model_.getModelParameters().getC0(0,0))*aux1[n];
    }
  }
  else if (smcProposalType_ == SMC_PROPOSAL_CONDITIONALLY_LOCALLY_OPTIMAL || smcProposalType_ == SMC_PROPOSAL_FA_APF)
  {     
    unsigned int dimLatentVariable = model_.getModelParameters().getDimLatentVariable();
    double d2         = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    double sigma      = d2 + 1.0 / (model_.getModelParameters().getC0(0,0));    
    arma::colvec mu(dimLatentVariable);
    
    for (unsigned int n=0; n<getNParticles(); n++)
    {
      mu = sigma*(1.0/d2 * model_.getObservations().col(0) + arma::inv(model_.getModelParameters().getC0()) * model_.getModelParameters().getM0());   
      particlesNew[n] = mu + std::sqrt(sigma)*aux1[n];
    }
  }
}
/// Computes (part of the) unnormalised "future" target density needed for 
/// backward or ancestor sampling.
template <class ModelParameters, class LatentVariable, class LatentPath, class LatentPathRepar, class Observations, class Particle, class Aux, class SmcParameters>
double Smc<ModelParameters, LatentVariable, LatentPath, LatentPathRepar, Observations, Particle, Aux, SmcParameters>::logDensityUnnormalisedTarget
(
  const unsigned int t,
  const Particle& particle
)
{
  // For conditionally IID models, we actually only need to return 0;
  // for state-space models, we need to evaluate the log-transition density;
  // for kth-order Markov models, we need to evaluate the log-unnormalised
  // target density k steps in the future.
  
  if (smcProposalType_ == SMC_PROPOSAL_FA_APF) 
  {
//     double b2 = std::pow(model_.getModelParameters().getB(0,0), 2.0);
//     double d2 = std::pow(model_.getModelParameters().getD(0,0), 2.0);
    
    return model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particle) - arma::as_scalar(gaussian::evaluateDensityMultivariate(model_.getObservations().col(t+1), model_.getModelParameters().getCA() * particle, model_.getModelParameters().getOptWeightVar(), false, true));
  }
  else 
  {
    return model_.evaluateLogTransitionDensity(t+1, particlePath_[t+1], particle);
  }
}

#endif
