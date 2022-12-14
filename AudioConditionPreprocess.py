import os
from tqdm import tqdm
import pandas as pd
import argparse

def extract_audio(video_path):
    video_name = video_path.split('/')[-1]
    id_path = '/'.join(video_path.split('/')[:-1])
    wav_file = os.path.join(id_path, video_name[:-3] + 'wav')
    extract_wav_cmd = 'ffmpeg -i ' + video_path + ' -f wav -ar 16000 ' + wav_file
    os.system(extract_wav_cmd)


def extract_audio_features(id_path):
    extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + id_path
    os.system(extract_ds_cmd)


def video_to_audio_features(csv_path, frac=(1/6), frac_id=0):
    df = pd.read_csv(csv_path, index_col=0)
    idxs = df['idx'].unique()
    list_idxs = [idxs[start:start+int(len(idxs)*frac)] for start in range(0, len(idxs), int(len(idxs)*frac))]
    new_idxs = list_idxs[frac_id]
    for idx in tqdm(new_idxs):
        idx_df = df[df['idx'] == idx]
        idx_path = '/'.join(idx_df.iloc[0]['path'].split('/')[:-2])
        for index, row in idx_df.iterrows():
            video_path = '/'.join(row['path'].split('/')[:-1]) + '.mp4'
            audio_path = video_path[:-3] + 'wav'
            if not os.path.isfile(audio_path):
                extract_audio(video_path)
        videos_in_id = [ele for ele in os.listdir(idx_path) if os.path.isfile(os.path.join(idx_path, ele)) and ele[-3:] == "mp4"]
        if not os.path.isfile(os.path.join(idx_path, videos_in_id[0][:-3] + 'npy')):
            extract_audio_features(idx_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', type=float,
                        default=(1/3), help='fraction to break the df into for multi-processing')
    parser.add_argument('--frac_id', type=int,
                        default=0, help='frac_id to know which fraction to run')
    args = parser.parse_args()
    video_to_audio_features('/scratch/tan/train_landmarks.csv', frac=args.frac, frac_id=args.frac_id)
    video_to_audio_features('/scratch/tan/val_landmarks.csv',  frac=args.frac, frac_id=args.frac_id)

