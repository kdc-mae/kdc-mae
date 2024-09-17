import os
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import csv
from argparse import ArgumentParser
preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])

def log_to_csv(filepath, data):
    """
    Append a row of data to a CSV.
    :param filepath: Path to the CSV file.
    :param data: List containing data to be written.
    """
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
def initialize_log(filepath, headers):
     """
    Initialize the CSV log. If the file does not exist, create it and write the headers.
    :param filepath: Path to the CSV file.
    :param headers: Headers to be written in the CSV file.
    """
     if not os.path.exists(filepath):
         with open(filepath, 'w', newline='') as csvfile:
             writer = csv.writer(csvfile)
             writer.writerow(headers)

def extract_frame(input_video_path, target_fold, extract_frame_num=10):
    # Extract the unique file ID
    file_id = input_video_path.split('/')[-1][3:-4]  # Remove 'id_' prefix and '.mp4' suffix

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num/extract_frame_num))
        print(f'Extract frame {i} from original frame {frame_idx}, total video frame {total_frame_num} at frame rate {int(fps)}.')
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()

        if frame is not None:
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            image_tensor = preprocess(pil_im)

            # Create directory for this frame if not exists
            frame_dir = os.path.join(target_fold, f'frame_{i}')
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)

            # Save the image with the naming convention: fileid_000000.jpg
            output_filename = os.path.join(frame_dir, f"{file_id}_000000.jpg")
            save_image(image_tensor, output_filename)
        else:
            print(f"Failed to extract frame {i} from video {input_video_path}")

def get_processed_videos(filepath):
    """
    Get a set of already processed videos from the success log.
    :param filepath: Path to the success log CSV file.
    :return: Set of processed video paths.
    """
    processed_videos = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip the headers
            for row in reader:
                processed_videos.add(row[0])  # the video path is the first column
    return processed_videos



if __name__ == "__main__":
    parser = ArgumentParser(description="Python script to extract frames from videos and save as jpgs.")
    
    # Change the argument to accept a CSV file containing video paths
    parser.add_argument("-video_csv", type=str, required=True, help="CSV file containing video paths.")
    parser.add_argument("-target_fold", type=str, default='./sample_frames/', help="The place to store the video frames.")
    parser.add_argument("-success_log", type=str, default='video_success_audioset.csv', help="CSV file to log successfully processed videos.")
    parser.add_argument("-failure_log", type=str, default='video_failure_audioset.csv', help="CSV file to log failed videos.")
    args = parser.parse_args()

    # Check and create target directory
    if not os.path.exists(args.target_fold):
        os.makedirs(args.target_fold)

    initialize_log(args.success_log, ["Video File", "Status"])
    initialize_log(args.failure_log, ["Video File", "Error Message"])

    # Get the set of already processed videos
    processed_videos = get_processed_videos(args.success_log)

    # Read video paths from the provided CSV
    with open(args.video_csv, 'r') as csvfile:
        video_paths = [row[0] for row in csv.reader(csvfile)]

    # Iterate over the video paths
    for video_path in video_paths:
        # Skip the video if it's already processed
        if video_path in processed_videos:
            print(f"Skipping already processed video: {video_path}")
            continue

        try:
            extract_frame(video_path, args.target_fold)
            log_to_csv(args.success_log, [video_path, "Processed successfully"])
        except Exception as e:
            print(f"Error processing video {video_path}. Error:")
            log_to_csv(args.failure_log, [video_path, str(e)])
