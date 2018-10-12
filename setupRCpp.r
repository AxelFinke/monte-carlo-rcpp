nCores <- 1

## ========================================================================= ##
## SETUP PATHS
## ========================================================================= ##

# Directories of input files
pathToInput       <- file.path(pathToInputBase, projectName, "examples", exampleName)
pathToData        <- file.path(pathToInput, "data")
pathToCovar       <- file.path(pathToInput, "covariates")
pathToSetup       <- file.path(pathToInput, "setup")

# Directories of output files
pathToOutput      <- file.path(pathToOutputBase, projectName, exampleName, jobName)
pathToFigures     <- file.path(pathToOutput, "figures")
pathToResults     <- file.path(pathToOutput, "results")
pathToExtra       <- file.path(pathToOutput, "extra")
pathToProcessed   <- file.path(pathToOutput, "processed")

# ensure that the above-mentioned directory exist:
dir.create(file.path(pathToOutputBase, projectName), showWarnings = FALSE) 
dir.create(file.path(pathToOutputBase, projectName, exampleName), showWarnings = FALSE) 
dir.create(pathToOutput,    showWarnings = FALSE)
dir.create(pathToFigures,   showWarnings = FALSE)
dir.create(pathToResults,   showWarnings = FALSE)
dir.create(pathToExtra,     showWarnings = FALSE)
dir.create(pathToProcessed, showWarnings = FALSE)

setwd(pathToInput)

## ========================================================================= ##
## SETUP JUST-IN-TIME COMPILER
## ========================================================================= ##

R_COMPILE_PKGS=TRUE
R_ENABLE_JIT=3

## ========================================================================= ##
## SETUP RCPP
## ========================================================================= ##

library(Rcpp)

Sys.setenv("PKG_CXXFLAGS"=paste("-Wall -std=c++11 -I\"", pathToInputBase, "\" -I/usr/include -fopenmp -O3 -ffast-math -fno-finite-math-only -march=native", sep=''))
Sys.setenv("PKG_LIBS"="-fopenmp")
# note that the option -ffast-math causes the compiler to ignore NaNs or Infs; the option -fno-finite-math-only should circumvent this

## Only needed when we want to use the profiler:
# Sys.setenv("PKG_CXXFLAGS"=paste("-Wall -std=c++11 -I\"", pathToInputBase, "\" -I/usr/include -fopenmp -O3 -ffast-math -march=native", sep=''))
# Sys.setenv("PKG_LIBS"="-fopenmp -lprofiler")
#### Sys.setenv("PKG_CXXFLAGS"=paste("-Wall -std=c++11 -I\"", pathToInputBase, "\" -I/usr/include -I/usr/include/gperftools -fopenmp -O3 -ffast-math -march=native", sep=''))

## Only needed when we use CGAL:
# Sys.setenv("PKG_CXXFLAGS"=paste("-Wall -std=c++11 -I\"", pathToInputBase, "\" -I/usr/include -lCGAL -fopenmp -O3 -ffast-math -march=native", sep=''))
# Sys.setenv("PKG_LIBS"="-fopenmp -lCGAL")

# sourceCpp(paste(pathToInputBase, "/", projectName, "/examples/", exampleName, "/", exampleName, ".cpp", sep=''), rebuild=TRUE, verbose=TRUE)

sourceCpp(file.path(pathToInput, paste(exampleName, ".cpp", sep='')), rebuild=TRUE, verbose=FALSE)

## ========================================================================= ##
## SETUP OTHER R FUNCTIONS AND PARAMETERS
## ========================================================================= ##

# Loads R functions related to the current example.
if (file.exists(file.path(pathToInput, paste(exampleName, "Functions.r", sep='')))) {
  source(file=file.path(pathToInput, paste(exampleName, "Functions.r", sep='')))
}
# Loads R functions related to the current project.
if (file.exists(file.path(pathToInputBase, projectName, paste(projectName, "Functions.r", sep='')))) {
  source(file=file.path(pathToInputBase, projectName, paste(projectName, "Functions.r", sep='')))
}
# Loads other generic R functions.
source(file=file.path(pathToInputBase, "rFunctions/generic_functions.r"))

# Loads parameters for the simulation study.
if (file.exists(file.path(pathToSetup, paste(jobName, ".r", sep='')))) {
  source(file=file.path(pathToSetup, paste(jobName, ".r", sep='')))
}


