import cv2
import time
import os
from tqdm import tqdm
import pandas as pd
import face_alignment
from skimage import io


def preprocess(dataset_dir, csv_path):
    ids = os.listdir(dataset_dir)
    videos = []
    for idx in ids:
        temp = os.listdir(os.path.join(dataset_dir, idx))
        temp = [os.path.join(dataset_dir, idx, vid) for vid in temp]
        videos += temp

    for video_path in tqdm(videos):
        video_to_frames(video_path, video_path.split('.')[0])

    extract_landmarks(dataset_dir, csv_path)


def extract_landmarks(dataset_dir, csv_path):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    preds = fa.get_landmarks_from_directory(dataset_dir)

    landmarks = [preds[key] for key in preds.keys()]
    images = ["/".join(key.split("/")[1:]) for key in preds.keys()]
    idx = [img_path.split("/")[0] for img_path in images]
    video = [img_path.split("/")[1] for img_path in images]
    image = [img_path.split("/")[2] for img_path in images]
    df = pd.DataFrame({'idx': idx, 'video': video, 'image': image, 'path': images, 'landmarks': landmarks})
    df.to_csv(csv_path)


def video_to_frames(input_loc, output_loc, skip_frames=15):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
        skip_frames: samples a frame every skip_frames frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        return
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video ", input_loc, "...\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count + 1), frame)
        if count % skip_frames != 0:
            os.remove(output_loc + "/%#05d.jpg" % (count + 1))
        count = count + 1

        # If there are no more frames left
        if (count > (video_length - 1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds forconversion." % (time_end - time_start))
            break


if __name__ == '__main__':
    preprocess('/home/starc/SBU/Sem-1/CV/Project/sample/', 'landmarks.csv')
