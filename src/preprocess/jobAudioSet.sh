#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 64000M
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
python audioSetAudioExtract.py  -video_dir /scratch/abhijitdas/Audio-Set -output_dir /scratch/abhijitdas/AudioSetCAV/audioExtract
