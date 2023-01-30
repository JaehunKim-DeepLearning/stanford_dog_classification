# Stanford Dog Image Classification

We proposed stanford dog image classification base line model using Pretrained EfficientNet

## Getting Started
* Download stanford dog image dataset
```
http://vision.stanford.edu/aditya86/ImageNetDogs/main.html
```
* Download efficientnet model weights
```
https://www.kaggle.com/datasets/ipythonx/efficientnet-keras-noisystudent-weights-b0b7

weights data path

./weights/
  --imagenet
    --imagenet.notop-b0.h5
    --imagenet.notop-b1.h5
  --nosystudent
  --advprob
  --autoaugment
```
* Dataset split for fold-validation
```
python3 0_split_fold.py --data_path ./Image/

original image dataset path for split

./Image/
  --n02085620-Chihuahua
    --n02085620_7.jpg
    --n02085620_199.jpg
    --n02085620_242.jpg
  --n02085782-Japanese_spaniel
  --n02085936-Maltese_dog
  ... 
```
* Train start
```
python3 1_train_model.py --mode B0 --weights imagenet

mode : B0~B7
weights : imagenet, nosystudent, advprob, autoaugment
```
* Train start with image augmentation
```
python3 2_train_model_augment.py --mode B0 --weights imagenet

mode : B0~B7
weights : imagenet, nosystudent, advprob, autoaugment
```
* Best model result check using history
```
python3 3_eval_history.py --mode B0
```

## Result (5-fold-validation)
||B0|B1|B2|B3|B4|B5|
|------|---|---|---|---|---|---|
|Baseline with Imagenet|80.3%|83.1%|84.9%|87.2%|**89.5%**|88.7%|
|Baseline with NoisyStudent|81.1%|84.2%|85.7%|87.6%|89.1%|88.6%|

[model download link](http://naver.me/G0JEYARU)

**Best model is using Imagenet weight and B4 architecture**


