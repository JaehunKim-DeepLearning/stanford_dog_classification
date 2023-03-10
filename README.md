# Stanford Dog Image Classification

We proposed stanford dog image classification baseline model using Pretrained EfficientNet

## Getting Started
### Quick Start - Main experiment
```  
./main_experiment_start.sh
```
* Download stanford dog image dataset
```
http://vision.stanford.edu/aditya86/ImageNetDogs/main.html
```
* Download efficientnet model weights
```
https://www.kaggle.com/datasets/ipythonx/efficientnet-keras-noisystudent-weights-b0b7

weights data path

./weights/
  └imagenet
    └imagenet.notop-b0.h5
    └imagenet.notop-b1.h5
  └nosystudent
  └advprob
  └autoaugment
```
* Dataset split for fold-validation
```
python3 0_split_fold.py --data_path ./Image/

original image dataset path for split

./Image/
  └n02085620-Chihuahua
    └n02085620_7.jpg
    └n02085620_199.jpg
    └n02085620_242.jpg
  └n02085782-Japanese_spaniel
  └n02085936-Maltese_dog
  ... 
```
* Train start
```
python3 1_train_model.py --mode B0 --weights imagenet

--mode : B0 ~ B7
--weights : imagenet, nosystudent, advprob, autoaugment
```
* Train start with image augmentation
```
python3 2_train_model_augment.py --mode B0 --weights imagenet
```
* Best model result check using history
```
python3 3_eval_history.py --mode B0
```
* Model evaluation
```
python3 4_eval_model.py --mode B0 --valid_fold 1 --model_path ./model/imagenet_B0_FOLD1.hdf5
```
### Quick start - Additional experiment 
```
./add_experiment_start.sh
```
* Image extract using annotation file
```
python3 0_annotation_extract.py --data_path ./Image/ --annotation_path ./Annotation/
```
## Result (5-fold-validation)
|                                     | B0    | B1      | B2    | B3    | B4        | B5    |
|-------------------------------------|-------|---------|-------|-------|-----------|-------|
| Model with Imagenet                 | 80.3% | 83.1% | 84.9% | 87.2% | 89.5%     | 88.7% |
| Model with NoisyStudent          | 81.1% | 84.2% | 85.7% | 87.6% | 89.1%     | 88.6% |
| Model with Imagenet, Augment     | 79.8% | 82.5% | 84.1% | 87.1% | **89.8%** | 89.1% |
| Model with NoisyStudent, Augment | 80.9% | 83.9% | 85.3% | 87.3% | 89.5%     | 89.0% |
| Model with Imagenet, Extract     | 81.8% | 84.1% | 85.6% | 86.7% | 88.0%     | 88.2% |
| Model with NoisyStudent, Extract | 83.0% | 85.1% | 86.2% | 87.7% | 88.6%     | 88.7% |

[model download link](http://naver.me/G0JEYARU)


