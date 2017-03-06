#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=myMatlabTest
#SBATCH --mail-type=END
#SBATCH --mail-user=sanketh@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load pytorch/intel/20170226  
module load torchvision/0.1.7 

cd /scratch/$USER/DeepLearningNYU

python ladder_impl.py
# cat thtest.m | matlab -nodisplay

