#!/bin/bash
#SBATCH -p big_compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --mem 128000M
#SBATCH -t 3-23:59 # time (D-HH:MM)
#SBATCH --job-name="aud_ext"
#SBATCH -o ./Slurm/out%j.txt
#SBATCH -e ./Slurm/err%j.txt
#SBATCH --gres=gpu:0
nvidia-smi
conda env list
source activate cave
spack load cuda/gypzm3r
spack load cudnn
spack load ffmpeg/3fnpeci
python splitter.py
