MODELS_AUX               <- 2
MODELS                   <- rep(MODELS_AUX, times=2)
# Model to be estimated (needed for model comparison):
# 0: time-dependent capture probabilities; time-dependent productivity rates
# 1: time-independent capture probabilities; time-dependent productivity rates
# 2: time-dependent capture probabilities; time-independent productivity rates
# 3: time-independent capture probabilities; time-independent productivity rates
# 4: same as Model 3 but with alpha_3 = 0
# 5: same as Model 3 but with delta_1 = alpha_3 = 0
# 6: same as Model 3 but with alpha_1 = alpha_3 = 0
# 7: same as Model 3 but with alpha_1 = alpha_3 = delta_1 = 0

N_CONFIGS                <- length(MODELS) # number of different model/algorithm configurations

N_PARTICLES              <- rep(2000, times=N_CONFIGS) # number of lower-level particles (i.e. number of particles used to approximate the marginal likelihood)

N_ITERATIONS             <- rep(1000000, times=N_CONFIGS) # number of MCMC iterations
SIMULATE_DATA            <- FALSE # should we use simulated data? (NOTE: this is currently not implemented for the owls example)

USE_DELAYED_ACCEPTANCE   <- c(0,1) # should delayed acceptance be used?

USE_ADAPTIVE_PROPOSAL               <- rep(1, each=N_CONFIGS) # should we adapt the proposal covariance matrix along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
USE_ADAPTIVE_PROPOSAL_SCALE_FACTOR1 <- rep(1, each=N_CONFIGS) # should we also adapt the constant by which the sample covariance matrix is multiplied?

LOWER                    <- rep(0, times=N_CONFIGS) # type of algorithm for updating the count-data likelihood (0: pseudo-marginal; 2: MCWM)
SAMPLE_PATH              <- rep(1, times=N_CONFIGS) # should we sample and store one particle path at each iteration?

LAG_MAX                  <- rep(10000, times=N_CONFIGS) # maximum number of lags to use when storing the autocorrelations
THINNING_INTERVAL        <- rep(100, times=N_CONFIGS) # number of iterations after which the value of the chain is stored
SMC_PARAMETERS           <- numeric(0) # additional parameters to be passed to the particle filter 
MCMC_PARAMETERS          <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates

MODEL_NAMES <- c()
for (ii in 1:length(MODELS)) {
  MODEL_NAMES <- c(MODEL_NAMES, paste("M", ii, sep=''))
}
