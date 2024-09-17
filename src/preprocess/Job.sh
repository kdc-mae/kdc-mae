#!/bin/bash
#SBATCH -p big_compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --mem 150000M
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="vid_ext"
#SBATCH -o ./Slurm/out%j.txt
#SBATCH -e ./Slurm/err%j.txt
#SBATCH --gres=gpu:0
nvidia-smi
conda env list
source activate cave
spack load cuda/gypzm3r
spack load cudnn
spack load ffmpeg/3fnpeci
python video_extract_audioset.py  -video_dir /scratch/abhijitdas/Audio-Set/14000022 -target_fold /scratch/abhijitdas/AudioSetCAV/video_frames
