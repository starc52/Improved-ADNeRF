# Improved-ADNeRF
Improving ADNeRF to make it audio identity agnostic by disentangling audio-mouth features. Built on top of https://github.com/YudongGuo/AD-NeRF

## Facial Feature Disentangling
* All code in Landmark*.py
* Run ```LandmarkMain.py``` to train model.

## Audio-Lip Synchronisation
* All code in AudioCondition*.py
* Run AudioConditionMain.py

## Improved AD-NeRF
* All code added to NeRFs/HeadNeRF/run_nerf.py, NeRFs/HeadNeRF/run_nerf_helpers.py, NeRFs/HeadNeRF/load_audface.py
* Run training using: ```python NeRFs/HeadNeRF/run_nerf.py --load_aud_cond ./best_audcond.pt --i_testset 30000 --N_iters 300000 --N_rand 768 --chunk 384 --netchunk 24756 --config dataset/$id/HeadNeRF_config.txt```
* Run inference code on original audio: ```python NeRFs/HeadNeRF/run_nerf.py --render_only --use_train_lms --test_file=transforms_train.json  --config=./dataset/$id/HeadNeRFTest_config.txt --aud_file=aud.npy --test_size=-1```
* Run inference code on new audio placed in ```./dataset/$id```: ```python NeRFs/HeadNeRF/run_nerf.py --render_only --use_train_lms --test_file=transforms_train.json  --config=./dataset/$id/HeadNeRFTest_config.txt --aud_file=<your_audio>.npy --test_size=-1```
