#!/usr/bin/env bash

wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar -xvf "./images.tar" -C "./"

#### Stanford Dog dataset split for Fold validation ###
python3 0_split_fold.py --data_path ./Image/

#### Train transfer learning using EfficientNet  ###
python3 1_train_model.py --mode B0 --weights imagenet  ##### imagenet noisystudent advprob autoaugment

#### Train transfer learning using EfficientNet with Image augmentation ###
#python3 2_train_model_augment.py --mode B0 --weights imagenet

#### Best model result check using history file ####
python3 3_eval_history.py --mode B0

#### Model evlauation ####
python3 4_eval_model.py --mode B0 --valid_fold 1 --model_path ./model/imagenet_FOLD1_B0.hdf5

exit 0

