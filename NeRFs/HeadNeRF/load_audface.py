import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def load_audface_data(basedir, testskip=1, test_file=None, test_rof_file=None, aud_file=None, test_size=-1):
    if test_file is not None:
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        lms = []
        image_size = np.array(imageio.imread(os.path.join(basedir,
                                                          'ori_imgs',
                                                          str(meta['frames'][0]['img_id']) + '.jpg'))).shape[0]
        aud_features = np.load(os.path.join(basedir, aud_file))
        cur=0
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            auds.append(aud_features[frame['aud_id']])
            lmname = os.path.join(basedir, 'ori_imgs',
                                  str(frame['img_id']) + '.lms')
            lms.append(lmname)
            cur+=1
            if cur == aud_features.shape[0] or cur == test_size:
                break
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        lms = np.array(lms)
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(
            meta['cx']), float(meta['cy'])
        rof_emb = torch.load(os.path.join(basedir, test_rof_file))
        return poses, auds, lms, image_size, bc_img, [H, W, focal, cx, cy], rof_emb

    splits = ['train', 'val']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    all_auds = []
    all_sample_rects = []
    all_lms = []
    aud_features = np.load(os.path.join(basedir, 'aud.npy'))
    counts = [0]
    image_size = np.array(imageio.imread(os.path.join(basedir,
                                                      'ori_imgs',
                                                      str(meta['frames'][0]['img_id']) + '.jpg'))).shape[0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        auds = []
        lms = []
        sample_rects = []
        mouth_rects = []
        #exps = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, 'head_imgs',
                                 str(frame['img_id']) + '.jpg')
            lmname = os.path.join(basedir, 'ori_imgs',
                                 str(frame['img_id']) + '.lms')
            lms.append(lmname)
            imgs.append(fname)
            poses.append(np.array(frame['transform_matrix']))
            auds.append(
                aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
            sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))
        imgs = np.array(imgs)
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_auds.append(auds)
        all_lms.append(lms)
        all_sample_rects.append(sample_rects)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    auds = np.concatenate(all_auds, 0)
    lms = np.concatenate(all_lms, 0)
    sample_rects = np.concatenate(all_sample_rects, 0)

    bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))

    H, W = bc_img.shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(
        meta['cx']), float(meta['cy'])

    return imgs, poses, auds, lms, image_size, bc_img, [H, W, focal, cx, cy], sample_rects, sample_rects, i_split
