import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--mode", type=str, default='B0')
parser.add_argument("--valid_fold", type=int, default=1)
parser.add_argument("--model_path", type=str, default='./model/imagenet_B0_FOLD1.hdf5')
args = parser.parse_args()

gpu = args.gpu
mode = args.mode
model_path = args.model_path
valid_fold = args.valid_fold

#### GPU select ####
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

#### pre-trained model select & model create ####
def model_info(mode):
    if mode == 'B0':
        IMAGE_SIZE = 224
        BATCH_SIZE = 32
    elif mode == 'B1':
        IMAGE_SIZE = 240
        BATCH_SIZE = 32
    elif mode == 'B2':
        IMAGE_SIZE = 260
        BATCH_SIZE = 32
    elif mode == 'B3':
        IMAGE_SIZE = 300
        BATCH_SIZE = 32
    elif mode == 'B4':
        IMAGE_SIZE = 380
        BATCH_SIZE = 30
    elif mode == 'B5':
        IMAGE_SIZE = 456
        BATCH_SIZE = 14

    return IMAGE_SIZE, BATCH_SIZE

#### final accuracy report ####
def valid_report(test, y_pred):
    print("\n###############################")
    print("###### PREDICTION RESULT ######")
    print("###############################\n")

    print('Confusion Matrix')
    cm = confusion_matrix(test, y_pred)
    print(cm)
    print('\nClassification Report')
    print(classification_report(test, y_pred))
    np.set_printoptions(precision=2)

IMAGE_SIZE, BATCH_SIZE = model_info(mode)

valid_ds = image_dataset_from_directory(
    directory='./dataset/' + str(valid_fold),
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE))

model = tf.keras.models.load_model(model_path)

predictions = np.array([])
labels = np.array([])

for x, y in valid_ds:
  predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

valid_report(labels, predictions)
