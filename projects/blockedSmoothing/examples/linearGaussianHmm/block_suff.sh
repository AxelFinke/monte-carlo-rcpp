#!/bin/bash -l

# Force bash
#$ -S /bin/bash

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=72:00:00

# Requesting RAM (e.g. mem=512M or mem=1G)
#$ -l mem=6G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=4G

# Set the name of the job.
#$ -N block_suff

# Set up the job array (i.e. specifying the task IDs)
#$ -t 1-1000

# Set the working directory to somewhere in your scratch space. This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME. NOTE: this directory must exist!
#$ -wd /home/ucakafi/Scratch/output

# Your work *must* be done in $TMPDIR 
cd $TMPDIR

# Merge stderr and stdout and set the name of the file 
#$ -o $HOME/Scratch/output/blockedSmoothing/linearGaussianHmm/log_suff.$JOB_ID.txt -j y

# Run the application
module unload compilers
module unload mpi
module load r/recommended
R --no-save < $HOME/blockedSmoothing/examples/linearGaussianHmm/block_suff.r > $HOME/Scratch/output/blockedSmoothing/linearGaussianHmm/r_output_suff.$JOB_ID.$SGE_TASK_ID

# Preferably, tar-up (archive) all output files onto the shared scratch area
# this will include the R_output file above.
tar zcvf $HOME/Scratch/output/blockedSmoothing/linearGaussianHmm/files_from_job_suff.$JOB_ID.$SGE_TASK_ID.tgz $TMPDIR

