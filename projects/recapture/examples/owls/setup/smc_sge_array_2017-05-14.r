MODELS_AUX               <- 0:7
MODELS                   <- rep(MODELS_AUX, times=1)
# Model to be estimated (needed for model comparison):
# 0: time-dependent capture probabilities; time-dependent productivity rates
# 1: time-independent capture probabilities; time-dependent productivity rates
# 2: time-dependent capture probabilities; time-independent productivity rates
# 3: time-independent capture probabilities; time-independent productivity rates
# 4: same as Model 3 but with alpha_3 = 0
# 5: same as Model 3 but with delta_1 = alpha_3 = 0
# 6: same as Model 3 but with alpha_1 = alpha_3 = 0
# 7: same as Model 3 but with alpha_1 = alpha_3 = delta_1 = 0

N_CONFIGS  <- length(MODELS) # number of different model/algorithm configurations

N_PARTICLES_UPPER        <- 10000 # number of particles used by the SMC sampler
N_PARTICLES_LOWER        <- 1000 # number of lower-level particles (i.e. number of particles used to approximate the marginal likelihood)

USE_IMPORTANCE_TEMPERING <- TRUE # should we also compute the weights and log-evidence estimates associated with an importance-tempering approach?
N_STEPS                  <- 100 # number of SMC steps (if adaptive tempering is not used)
SIMULATE_DATA            <- FALSE # should we use simulated data? (NOTE: this is currently not implemented for the owls example)

ALPHA                    <- seq(from=0, to=1, length=N_STEPS)
USE_DELAYED_ACCEPTANCE   <- 1 # should delayed acceptance be used?
USE_ADAPTIVE_TEMPERING   <- 1 # should we use adaptive tempering?
USE_ADAPTIVE_CESS_TARGET <- rep(0, times=N_CONFIGS)
ADAPT_PROPOSAL           <- 1 # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
USE_DOUBLE_TEMPERING     <- rep(0, times=N_CONFIGS) # should we temper the two likelihood components separately?
LOWER                    <- rep(0, times=N_CONFIGS) # type of algorithm for updating the count-data likelihood (0: pseudo-marginal; 2: MCWM)

SMC_PARAMETERS           <- numeric(0) # additional parameters to be passed to the particle filter 
MCMC_PARAMETERS          <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates
 
