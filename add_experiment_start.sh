#!/usr/bin/env bash

wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar

tar -xvf "./images.tar" -C "./"
tar -xvf "./annotation.tar" -C "./"

#### Image extract using annotation file ####
python3 0_annotation_extract.py --data_path ./Image/ --annotation_path ./Annotation/

#### Stanford Dog dataset split for Fold validation ####
python3 0_split_fold.py --data_path ./Image/

#### Train transfer learning using EfficientNet  ####
python3 1_train_model.py --data_path ./extract_dataset/ --affix extract  --mode B0 --weights imagenet    ##### imagenet noisystudent advprob autoaugment

#### Train transfer learning using EfficientNet with Image augmentation ###
#python3 2_train_model_augment.py --affix extract --data_path ./extract_dataset/ --mode B0 --weights imagenet

#### Best model result check using history file ####
python3 3_eval_history.py --affix extract --mode B0

#### Model evlauation ####
python3 4_eval_model.py --data_path ./extract_dataset/ --mode B0 --valid_fold 1 --model_path ./model/extract_imagenet_B0_FOLD1.hdf5

exit 0

