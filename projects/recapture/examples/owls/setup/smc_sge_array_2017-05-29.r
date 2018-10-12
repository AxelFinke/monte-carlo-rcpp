MODELS_AUX               <- 0:7
MODELS                   <- rep(MODELS_AUX, times=1)

N_CONFIGS  <- length(MODELS) # number of different model/algorithm configurations

CESS_TARGET              <- c(0.999, 0.999, 0.999, rep(0.999, times=N_CONFIGS-3)) # target conditional effective sample size used for adaptive tempering
CESS_TARGET_FIRST        <- rep(0.9999, times=N_CONFIGS) # target conditional effective sample size used for adaptive tempering at Stage 1 of the algorithm (if double tempering is used)
N_PARTICLES_UPPER        <- c(10000, 10000, 10000, rep(10000, times=N_CONFIGS-3)) # number of particles used by the SMC sampler
N_PARTICLES_LOWER        <- rep(2000, times=N_CONFIGS) # number of lower-level particles (i.e. number of particles used to approximate the marginal likelihood)

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

MODEL_NAMES <- c()
for (ii in 1:length(MODELS)) {
  MODEL_NAMES <- c(MODEL_NAMES, paste("M", ii, sep=''))
}
