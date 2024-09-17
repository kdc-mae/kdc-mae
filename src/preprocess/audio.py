import os
import numpy as np
import argparse
from pydub import AudioSegment

parser = argparse.ArgumentParser(description='Easy video feature extractor')
parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="Should be a csv file of a single columns, each row is the input video path.")
parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="The place to store the video frames.")
args = parser.parse_args()

input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)
if os.path.exists(args.target_fold) == False:
    os.makedirs(args.target_fold)

# then extract the first channel
for i in range(input_filelist.shape[0]):
    input_f = input_filelist[i]
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    video_id = input_f.split('/')[-1][:-ext_len-1]
    output_f_2 = args.target_fold + '/' + video_id + '.flac'
    if os.path.exists(input_f):  # Check if file exists
        audio = AudioSegment.from_file(input_f)
        audio = audio.set_channels(1) # This will take the first channel only
        audio.export(output_f_2, format='flac')
    else:
        print(f"File {input_f} not found. Skipping.")
