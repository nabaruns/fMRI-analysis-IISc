#!/bin/sh
#SBATCH --job-name=labwork # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=72:00:00 # Time limit hrs:min:sec
#SBATCH -o log/%x-%A-%a.out
#SBATCH --gres=gpu:2
#SBATCH --partition=cl1_all_4G
pwd; hostname; date | tee result

jupyter-lab --no-browser --port=7000