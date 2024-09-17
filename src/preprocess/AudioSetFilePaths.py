import os
import csv
from argparse import ArgumentParser

def log_filepaths_to_csv(directory, output_csv):
    """
    Log all .mp4 file paths from the specified directory to a CSV file.
    :param directory: Path to the root directory to search.
    :param output_csv: Path to the output CSV file.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Video File Path"])  # Write the header
        
        for dirpath, _, filenames in os.walk(directory):
            for video_file in filenames:
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(dirpath, video_file)
                    writer.writerow([video_path])

if __name__ == "__main__":
    parser = ArgumentParser(description="Python script to log paths of all .mp4 files in a directory to a CSV.")
    parser.add_argument("-dir", type=str, required=True, help="Directory to search for .mp4 files.")
    parser.add_argument("-output", type=str, default='video_paths.csv', help="Output CSV file to store video paths.")
    args = parser.parse_args()

    log_filepaths_to_csv(args.dir, args.output)

