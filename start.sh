#!/usr/bin/ bash
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints
# weight download https://www.kaggle.com/datasets/ipythonx/efficientnet-keras-noisystudent-weights-b0b7

#### Stanford Dog dataset split for Fold validation ###
python3 0_split_fold.py --data_path ./stanford_dog/image/Image/

#### Train transfer learning using EfficientNet  ###
python3 1_train_model.py --mode B0 --weights imagenet  ##### imagenet noisystudent advprob autoaugment

#### Train transfer learning using EfficientNet with Image augmentation ###
#python3 2_train_model_augment.py --mode B0 --weights imagenet

#### Best model result check using history file ####
python3 3_eval_history.py --mode B0

exit 0

