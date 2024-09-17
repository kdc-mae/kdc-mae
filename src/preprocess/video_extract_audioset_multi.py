import os
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import csv
from argparse import ArgumentParser
import multiprocessing

preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()
])

def log_to_csv(filepath, data):
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def extract_frame(input_video_path, target_fold, extract_frame_num=10):
    file_id = input_video_path.split('/')[-1][3:-4]
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num/extract_frame_num))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()

        if frame is not None:
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            image_tensor = preprocess(pil_im)
            frame_dir = os.path.join(target_fold, f'frame_{i}')
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            output_filename = os.path.join(frame_dir, f"{file_id}_000000.jpg")
            save_image(image_tensor, output_filename)
        else:
            raise Exception(f"Failed to extract frame {i} from video {input_video_path}")

def process_video_batch(video_batch, args):
    for video_path in video_batch:
        try:
            extract_frame(video_path, args.target_fold)
            log_to_csv(args.success_log, [video_path, "Processed successfully"])
        except Exception as e:
            log_to_csv(args.failure_log, [video_path, str(e)])

def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    parser = ArgumentParser(description="Python script to extract frames from videos and save as jpgs.")
    parser.add_argument("-video_dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument("-target_fold", type=str, default='./sample_frames/')
    parser.add_argument("-success_log", type=str, default='video_success_audioset.csv')
    parser.add_argument("-failure_log", type=str, default='video_failure_audioset.csv')
    args = parser.parse_args()

    if not os.path.exists(args.target_fold):
        os.makedirs(args.target_fold)

    with open(args.success_log, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Video File", "Status"])

    with open(args.failure_log, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Video File", "Error Message"])

    video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
    
    batch_size = 5  # Number of videos to process in a single batch. Adjust this based on your requirements.
    video_batches = list(batchify(video_files, batch_size))
    
    pool = multiprocessing.Pool(processes=32)  # Adjust the number of processes based on your CPU and experimentation.
    pool.starmap(process_video_batch, [(batch, args) for batch in video_batches])
    pool.close()
    pool.join()

