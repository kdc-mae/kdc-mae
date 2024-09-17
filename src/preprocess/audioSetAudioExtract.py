import os
import argparse
import subprocess
from pydub import AudioSegment
import csv

def extract_audio_from_videos(video_dir, output_dir, success_log, failure_log):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load successfully processed video files from the success log
    successfully_processed_videos = set()
    if os.path.exists(success_log):
        with open(success_log, 'r', newline='') as successfile:
            reader = csv.reader(successfile)
            next(reader)  # skip the header
            for row in reader:
                successfully_processed_videos.add(row[0])

    with open(success_log, 'a', newline='') as successfile, open(failure_log, 'a', newline='') as failurefile:
        success_writer = csv.writer(successfile)
        failure_writer = csv.writer(failurefile)

        # Write headers only if the files are empty or don't exist
        if os.path.getsize(success_log) == 0:
            success_writer.writerow(["Video File", "Message"])
        if os.path.getsize(failure_log) == 0:
            failure_writer.writerow(["Video File", "Error Message"])

        # Recursively walk through all subdirectories
        for root, dirs, files in os.walk(video_dir):
            for video_file in files:
                if video_file.endswith(".mp4") and video_file not in successfully_processed_videos:
                    # Extract the unique file ID
                    file_id = video_file[3:-4]  # Remove 'id_' prefix and '.mp4' suffix

                    # File paths for intermediate and final outputs
                    output_wav = os.path.join(output_dir, f"{file_id}_000000_intermediate.wav")
                    output_flac = os.path.join(output_dir, f"{file_id}_000000.flac")

                    # Construct the input video file path
                    video_path = os.path.join(root, video_file)

                    # Use ffmpeg to extract and resample the audio, capturing stderr
                    process = subprocess.Popen(['ffmpeg', '-i', video_path, '-vn', '-ar', '16000', output_wav],
                                               stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    _, stderr = process.communicate()

                    if process.returncode == 0 and os.path.exists(output_wav):
                        audio = AudioSegment.from_wav(output_wav)
                        audio = audio.set_channels(1)  # This will take the first channel only
                        audio.export(output_flac, format='flac')
                        os.remove(output_wav)  # Remove the intermediate .wav file
                        success_writer.writerow([video_file, "Audio extracted successfully"])
                    else:
                        error_message = stderr.decode('utf-8').strip() if stderr else f"Intermediate file {output_wav} not found."
                        failure_writer.writerow([video_file, error_message])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio feature extractor from video')
    parser.add_argument("-video_dir", type=str, required=True, help="Root directory containing video files and subfolders.")
    parser.add_argument("-output_dir", type=str, required=True, help="Directory to save extracted audio files.")
    parser.add_argument("-success_log", type=str, default="audio_success_log.csv", help="Path to save the success log CSV file.")
    parser.add_argument("-failure_log", type=str, default="audio_failure_log.csv", help="Path to save the failure log CSV file.")
    args = parser.parse_args()

    extract_audio_from_videos(args.video_dir, args.output_dir, args.success_log, args.failure_log)

