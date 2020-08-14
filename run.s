#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:p40:2
#SBATCH --time=60:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=myTest
#SBATCH --output=slurm_%j.out

cd $SCRATCH/neural_decoding/code/cnn/ALAE/

module purge
module load cudnn/10.0v7.6.2.24
module load cuda/10.0.130
source $HOME/python3.7/bin/activate
python train_alae.py
