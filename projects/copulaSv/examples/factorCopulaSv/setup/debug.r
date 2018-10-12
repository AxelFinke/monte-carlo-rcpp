MODELS_AUX <- c(0,1)
MODELS     <- rep(MODELS_AUX, times=1)
# Model to be used:
# 0: exactly measured log-volatilities
# 1: noisily measured log-volatilities

COPULA_TYPE_H  <- c(0,0)
COPULA_TYPE_Z  <- c(0,0)
COPULA_TYPE_HZ <- c(0,0)
# Type of copula used to model the dependence between the 
# latent factors and the noise variables

N_LOG_EXCHANGE_RATES <- rep(3, length=length(MODELS_AUX)) 
# number of modelled/observed exchange rates

N_PARTICLES_UPPER        <- 100 # number of particles used by the SMC sampler
N_PARTICLES_LOWER        <- 500 # number of lower-level particles (i.e. number of particles used to approximate the marginal likelihood)

USE_IMPORTANCE_TEMPERING <- TRUE # should we also compute the weights and log-evidence estimates associated with an importance-tempering approach?
N_STEPS                  <- 100 # number of SMC steps (if adaptive tempering is not used)
SIMULATE_DATA            <- FALSE # should we use simulated data?

ALPHA                    <- seq(from=0, to=1, length=N_STEPS)
USE_DELAYED_ACCEPTANCE   <- 0 # should delayed acceptance be used?
USE_ADAPTIVE_TEMPERING   <- 1 # should we use adaptive tempering?
ADAPT_PROPOSAL           <- 1 # should we adapt the proposal scale along the lines of Peters et al. (2010)? (only adapt after the burnin phase)

SMC_PARAMETERS           <- numeric(3) # additional parameters to be passed to the particle filter
MCMC_PARAMETERS          <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates



