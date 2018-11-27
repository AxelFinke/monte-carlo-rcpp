MODELS_AUX <- c(12,12,12,12,12)
# c(1,2,3,4,5,10,15,25,50,75,100,250,500,750,1000,2500,5000,7500,10000) # observation sequences which are available
# different "models" only differ by the different observation sequence (and length thereof) used!

MODELS     <- rep(MODELS_AUX, each=2)

N_CONFIGS  <- length(MODELS) # number of different model/algorithm configurations

CESS_TARGET              <- rep(seq(from=0.2, to=0.9, length=length(MODELS_AUX)), times=2) # target conditional effective sample size used for adaptive tempering
N_PARTICLES_UPPER        <- rep(50, times=N_CONFIGS)  # number of particles used by the SMC sampler
N_PARTICLES_LOWER        <- rep(5, times=N_CONFIGS) # number of lower-level particles (i.e. number of particles used to approximate the marginal likelihood)

USE_IMPORTANCE_TEMPERING <- TRUE # should we also compute the weights and log-evidence estimates associated with an importance-tempering approach?
N_STEPS                  <- 100 # number of SMC steps (if adaptive tempering is not used)
SIMULATE_DATA            <- FALSE # should we use simulated data?

ALPHA                    <- seq(from=0, to=1, length=N_STEPS)
USE_DELAYED_ACCEPTANCE   <- 0 # should delayed acceptance be used?
USE_ADAPTIVE_TEMPERING   <- 1 # should we use adaptive tempering?
USE_ADAPTIVE_CESS_TARGET <- rep(c(0,0), each=length(MODELS_AUX)) ##rep(0, times=N_CONFIGS)

USE_ADAPTIVE_PROPOSAL               <- rep(1, each=N_CONFIGS) # should we adapt the proposal covariance matrix along the lines of Peters et al. (2010)? (only adapt after the burnin phase)
USE_ADAPTIVE_PROPOSAL_SCALE_FACTOR1 <-  rep(c(0,1), each=length(MODELS_AUX)) # should we also adapt the constant by which the sample covariance matrix is multiplied?

# USE_DOUBLE_TEMPERING     <- rep(c(0,1), each=length(MODELS_AUX)) # should we temper the two likelihood components separately?
LOWER                    <- rep(3, times=N_CONFIGS) # type of algorithm for updating the intractable marginal ikelihood (0: pseudo-marginal; 2: MCWM; 3: idealised marginal)
N_METROPOLIS_HASTINGS_UPDATES <- rep(1, times=N_CONFIGS) ###########

SMC_PARAMETERS           <- numeric(0) # additional parameters to be passed to the particle filter
MCMC_PARAMETERS          <- numeric(0) # additional parameters to be passed to the MCMC (e.g. PMMH) updates
