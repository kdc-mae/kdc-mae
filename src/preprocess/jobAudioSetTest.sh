#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 128000M
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --job-name="aud_ext"
#SBATCH -o ./Slurm/out%j.txt
#SBATCH -e ./Slurm/err%j.txt
#SBATCH --gres=gpu:1
nvidia-smi
conda env list
source activate cave
spack load cuda/gypzm3r
spack load cudnn
spack load ffmpeg/3fnpeci
python audioSetAudioExtract.py  -video_dir /scratch/abhijitdas/Audio-Set/37000011 -output_dir /scratch/abhijitdas/AudioSetCAV/TestaudioExtract -success_log ./test_success.csv -failure_log ./test_failure.csv
