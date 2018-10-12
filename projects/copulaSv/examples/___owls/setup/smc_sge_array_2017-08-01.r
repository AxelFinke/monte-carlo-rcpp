MODELS_AUX               <- 0:15
MODELS                   <- rep(MODELS_AUX, times=1)

#  0: time-dependent capture probabilities; time-dependent productivity rates
#  1: ... with $\delta_1 = 0$
#  2: time-independent capture probabilities; time-dependent productivity rates
#  3: ... with $\delta_1 = 0$
#  4: time-dependent capture probabilities; time-independent productivity rates
#  5: ... with $\delta_1 = 0$
#  6: ... with $\alpha_3 = 0$
#  7: ... with $\delta_1 = \alpha_3 = 0$
#  8: ... with $\alpha_1 = \alpha_3 = 0$
#  9  ... with $\alpha_1 = \alpha_3 = \delta_1 = 0$
# 10: time-independent capture probabilities; time-independent productivity rates
# 11: ... with $\delta_1 = 0$
# 12: ... with $\alpha_3 = 0$
# 13: ... with $\delta_1 = \alpha_3 = 0$
# 14: ... with $\alpha_1 = \alpha_3 = 0$
# 15  ... with $\alpha_1 = \alpha_3 = \delta_1 = 0$

N_CONFIGS                <- length(MODELS) # number of different model/algorithm configurations

CESS_TARGET              <- rep(0.9993, times=N_CONFIGS) # target conditional effective sample size used for adaptive tempering
CESS_TARGET_FIRST        <- rep(0.999, times=N_CONFIGS) # target conditional effective sample size used for adaptive tempering at Stage 1 of the algorithm (if double tempering is used)
N_PARTICLES_UPPER        <- rep(1000, times=N_CONFIGS) # number of particles used by the SMC sampler
N_PARTICLES_LOWER        <- rep(1000, times=N_CONFIGS) # number of lower-level particles (i.e. number of particles used to approximate the marginal likelihood)

USE_IMPORTANCE_TEMPERING <- FALSE # should we also compute the weights and log-evidence estimates associated with an importance-tempering approach?
N_STEPS                  <- 100 # number of SMC steps (if adaptive tempering is not used)
SIMULATE_DATA            <- FALSE # should we use simulated data? (NOTE: this is currently not implemented for the owls example)

ALPHA                    <- seq(from=0, to=1, length=N_STEPS)
USE_DELAYED_ACCEPTANCE   <- 1 # should delayed acceptance be used?
USE_ADAPTIVE_TEMPERING   <- 1 # should we use adaptive tempering?
USE_ADAPTIVE_CESS_TARGET <- rep(0, each=N_CONFIGS)

USE_ADAPTIVE_PROPOSAL               <- rep(1, each=N_CONFIGS) # should we adapt the proposal covariance matrix along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
USE_ADAPTIVE_PROPOSAL_SCALE_FACTOR1 <- rep(1, each=N_CONFIGS) # should we also adapt the constant by which the sample covariance matrix is multiplied?

USE_DOUBLE_TEMPERING     <- rep(0, times=N_CONFIGS) # should we temper the two likelihood components separately?
LOWER                    <- rep(0, times=N_CONFIGS) # type of algorithm for updating the count-data likelihood (0: pseudo-marginal; 2: MCWM)

SMC_PARAMETERS           <- numeric(0) # additional parameters to be passed to the particle filter 
MCMC_PARAMETERS          <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates

MODEL_NAMES <- c()
for (ii in 1:length(MODELS)) {
  MODEL_NAMES <- c(MODEL_NAMES, paste("M", ii, sep=''))
}
