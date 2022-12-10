import os
from tqdm import tqdm
import pandas as pd


def extract_audio(video_path):
    print('--- Step0: extract deepspeech feature ---')
    video_name = video_path.split('/')[-1]
    id_path = '/'.join(video_path.split('/')[:-1])
    wav_file = os.path.join(id_path, video_name[:-3]+'wav')
    extract_wav_cmd = 'ffmpeg -i ' + video_path + ' -f wav -ar 16000 ' + wav_file
    os.system(extract_wav_cmd)


def extract_audio_features(id_path):
    extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + id_path
    os.system(extract_ds_cmd)


def video_to_audio_features(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    idxs = df['idx'].unique()
    for idx in tqdm(idxs[:10]):
        idx_df = df[df['idx'] == idx]
        idx_path = '/'.join(idx_df.iloc[0]['path'].split('/')[:-2])
        for row in idx_df.iterrows():
            video_path = '/'.join(row['path'].split('/')[:-1])+'.mp4'
            extract_audio(video_path)
        extract_audio_features(idx_path)


if __name__ == '__main__':
    video_to_audio_features('/scratch/tan/train_landmarks.csv')
    video_to_audio_features('/scratch/tan/val_landmarks.csv')
