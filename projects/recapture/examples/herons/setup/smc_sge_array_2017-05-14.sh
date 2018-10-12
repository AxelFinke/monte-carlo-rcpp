#!/bin/bash -l

# Force bash
#$ -S /bin/bash

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=72:00:00

# Requesting RAM (e.g. mem=512M or mem=1G)
#$ -l mem=5G
# [should be 8G]

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=5G
# [should be 8G]

# Set the name of the job.
#$ -N herons

# Set up the job array (i.e. specifying the task IDs)
#$ -t 1-360
# [should be 360]

# Set the working directory to somewhere in your scratch space. This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME. NOTE: this directory must exist!
#$ -wd /home/ucakafi/Scratch/output

# Your work *must* be done in $TMPDIR 
cd $TMPDIR

# Merge stderr and stdout and set the name of the file 
#$ -o $HOME/Scratch/output/cpp/mc/recapture/herons/log_$JOB_ID.txt -j y

# Run the application
module unload compilers
module unload mpi
module load r/recommended
R --no-save < $HOME/code/cpp/mc/recapture/examples/herons/smc_sge_array.r > $HOME/Scratch/output/cpp/mc/recapture/herons/log.$JOB_ID.$SGE_TASK_ID

# Preferably, tar-up (archive) all output files onto the shared scratch area
# this will include the R_output file above.
tar zcvf $HOME/Scratch//utput/cpp/mc/recapture/herons/files_from_job_$JOB_ID.$SGE_TASK_ID.tgz $TMPDIR

