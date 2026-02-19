#!/bin/bash

# test SBATCH --nodelist=None
#SBATCH --ntasks=144
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=8
#SBATCH --time=10-00:00:00
#SBATCH --job-name 6x2pt_TATT_Dzi_nlb
#SBATCH --output /share/nas_mberc2/wongj/mywork/sbatch/out-slurm_%j.out
#SBATCH --error /share/nas_mberc2/wongj/mywork/sbatch/out-slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonathan.wong@manchester.ac.uk

cd /share/nas_mberc2/wongj/mywork/6x2pt/6x2pt_sim/mywork/run_bias/

# How many threads to use per process (same as cpus-per-task). Should be some multiple of 2
export OMP_NUM_THREADS=4

# Here, the number after n is the total number of tasks. Total nodes/compute used will then be:
# ntasks // (ntasks-per-node * cpus-per-task=4)
mpiexec -n 144 python -m mpi4py.futures run_likelihood_6bin_3x2pt_TATT_Photz_nlb.py
