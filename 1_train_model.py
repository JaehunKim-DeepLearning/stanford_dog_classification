# weight download https://www.kaggle.com/datasets/ipythonx/efficientnet-keras-noisystudent-weights-b0b7
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints

import os
import gc
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--mode", type=str, default='B0')
parser.add_argument("--data_path", type=str, default='./dataset/')
parser.add_argument("--weights", type=str, default='imagenet') # noisystudent advprob autoaugment imagenet
parser.add_argument("--fold", type=int, default=5)
args = parser.parse_args()

gpu = args.gpu
mode = args.mode
data_path = args.data_path
weights = args.weights
fold = args.fold

epochs = 100
patience = 5

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

os.makedirs('./history', exist_ok=True)
os.makedirs('./model', exist_ok=True)
os.makedirs('./weights', exist_ok=True)

if weights == 'noisystudent':
    weights_path = './weights/noisystudent/noisy.student.notop-' + mode.lower() + '.h5'
    use_weights = None
elif weights == 'advprob':
    weights_path = './weights/advprob/adv.prop.notop-' + mode.lower() + '.h5'
    use_weights = None
elif weights == 'autoaugment':
    weights_path = './weights/autoaugment/auto.augment.notop-' + mode.lower() + '.h5'
    use_weights = None
else:
    use_weights = 'imagenet'

result = []

image_path = os.listdir(data_path)
image_path.sort()

def model_create(mode):
    if mode == 'B0':
        base_model = tf.keras.applications.EfficientNetB0(weights=use_weights, include_top=False, pooling='avg')
        IMAGE_SIZE = 224
        BATCH_SIZE = 32
        DROP_OUT = 0.2
    elif mode == 'B1':
        base_model = tf.keras.applications.EfficientNetB1(weights=use_weights, include_top=False, pooling='avg')
        IMAGE_SIZE = 240
        BATCH_SIZE = 32
        DROP_OUT = 0.2
    elif mode == 'B2':
        base_model = tf.keras.applications.EfficientNetB2(weights=use_weights, include_top=False, pooling='avg')
        IMAGE_SIZE = 260
        BATCH_SIZE = 32
        DROP_OUT = 0.3
    elif mode == 'B3':
        base_model = tf.keras.applications.EfficientNetB3(weights=use_weights, include_top=False, pooling='avg')
        IMAGE_SIZE = 300
        BATCH_SIZE = 32
        DROP_OUT = 0.3
    elif mode == 'B4':
        base_model = tf.keras.applications.EfficientNetB4(weights=use_weights, include_top=False, pooling='avg')
        IMAGE_SIZE = 380
        BATCH_SIZE = 30
        DROP_OUT = 0.4
    elif mode == 'B5':
        base_model = tf.keras.applications.EfficientNetB5(weights=use_weights, include_top=False, pooling='avg')
        IMAGE_SIZE = 456
        BATCH_SIZE = 14
        DROP_OUT = 0.4

    if weights != 'imagenet':
        base_model.load_weights(weights_path, by_name=True)

    x = base_model.output
    x = tf.keras.layers.Dropout(DROP_OUT)(x)
    x = tf.keras.layers.Dense(units=120)(x) 
    output = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, IMAGE_SIZE, BATCH_SIZE, DROP_OUT

for valid_fold in range(fold):
    valid_path = image_path[valid_fold]
    train_path = image_path.copy()
    train_path.pop(valid_fold)

    model, IMAGE_SIZE, BATCH_SIZE, DROP_OUT = model_create(mode)

    for idx, i in enumerate(train_path):
        if idx == 0 :
            train_ds = image_dataset_from_directory(
                directory=data_path + '/' + str(i),
                labels='inferred',
                label_mode='categorical',
                batch_size=BATCH_SIZE,
                image_size=(IMAGE_SIZE, IMAGE_SIZE))
        else:
            data = image_dataset_from_directory(
                directory=data_path + '/' + str(i),
                labels='inferred',
                label_mode='categorical',
                batch_size=BATCH_SIZE,
                image_size=(IMAGE_SIZE, IMAGE_SIZE))
            train_ds = train_ds.concatenate(data)

    valid_ds = image_dataset_from_directory(
        directory=data_path + '/' + str(valid_path),
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE, IMAGE_SIZE))

    checkponiter = tf.keras.callbacks.ModelCheckpoint(filepath='./model/' + weights + '_' + mode + '_FOLD' + str(valid_fold+1)  + '.hdf5' , monitor='val_accuracy', verbose=1, mode='max', save_best_only=True)
    earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max', restore_best_weights=True)

    print("TRAIN START")
    print("TRAIN FOLD -> ", train_path)
    print("VALID FOLD -> ", [str(valid_path)])
    history = model.fit(train_ds, 
        epochs=epochs, 
        validation_data=valid_ds,
        verbose=1,
        shuffle=True,
        callbacks=[checkponiter, earlystopper])
    print("TRAIN COMPLETE")

    hist_df = pd.DataFrame(history.history) 
    result.append(hist_df['val_accuracy'].max())
    hist_csv_file = './history/' + weights + '_' + mode + '_FOLD' + str(valid_fold+1) + '_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    print("HISTORY SAVE COMPLETE")

    tf.keras.backend.clear_session()
    del model
    gc.collect()


print("\nFINAL RESULTS")
for idx, i in enumerate(result):
    print('VALID FOLD', idx+1, 'ACCURACY : %0.3f' %i)
avg_acc = sum(result) / len(result)
print('\nTOTAL AVG FOLD ACCURACY : %0.3f' %avg_acc)