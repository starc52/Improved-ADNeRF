# Improved-ADNeRF
Improving ADNeRF to make it audio identity agnostic by disentangling audio-mouth features. Built on top of https://github.com/YudongGuo/AD-NeRF

Note that you must set up your environment before running the following code. Please try setting up with ```run_ada.yml``` first. If it doesn't work, please set up the environment following the original AD-NeRF code instructions.

## Facial Feature Disentangling
* All code in Landmark*.py
* Run ```LandmarkMain.py``` to train model.

## Audio-Lip Synchronisation
* All code in AudioCondition*.py
* Run ```AudioConditionMain.py``` to train the model

## Improved AD-NeRF
* All code added to NeRFs/HeadNeRF/run_nerf.py, NeRFs/HeadNeRF/run_nerf_helpers.py, NeRFs/HeadNeRF/load_audface.py
* Run training using: ```python NeRFs/HeadNeRF/run_nerf.py --load_aud_cond ./best_audcond.pt --i_testset 30000 --N_iters 300000 --N_rand 768 --chunk 384 --netchunk 24756 --config dataset/$id/HeadNeRF_config.txt```
* Run inference code on original audio: ```python NeRFs/HeadNeRF/run_nerf.py --render_only --use_train_lms --test_file=transforms_train.json  --config=./dataset/$id/HeadNeRFTest_config.txt --aud_file=aud.npy --test_size=-1```
* Run inference code on new audio placed in ```./dataset/$id```: ```python NeRFs/HeadNeRF/run_nerf.py --render_only --use_train_lms --test_file=transforms_train.json  --config=./dataset/$id/HeadNeRFTest_config.txt --aud_file=<your_audio>.npy --test_size=-1```

Here ```$id``` is the name of the folder with the data, for eg. Obama. 

Note that in order to run the above functions on novel data, you must preprocess your video, using ```process_data.sh``` first. Guidance for that is available in the original AD-NeRF paper.