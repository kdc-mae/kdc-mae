#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 32000M
#SBATCH -t 3-23:59 # time (D-HH:MM)
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
#python extract_video_audio_set.py  -video_dir /scratch/abhijitdas/Audio-Set -target_fold /scratch/abhijitdas/AudioSetCAV/video_frames
python extract_video_audio_set2.py  -video_csv /home/abhijitdas/Workbenches/delta/cav-mae/src/preprocess/unbalanced_splits_audioset/split_0.csv -target_fold /scratch/abhijitdas/AudioSetCAV/video_frames -success_log split0_success.csv
