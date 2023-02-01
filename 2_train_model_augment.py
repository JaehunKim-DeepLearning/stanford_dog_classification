import gc
import os
import argparse
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--mode", type=str, default='B1')
parser.add_argument("--data_path", type=str, default='./dataset/')
parser.add_argument("--weights", type=str, default='imagenet') # noisystudent advprob autoaugment imagenet
parser.add_argument("--affix", type=str, default='')
parser.add_argument("--fold", type=int, default=5)
args = parser.parse_args()

gpu = args.gpu
mode = args.mode
data_path = args.data_path
weights = args.weights
fold = args.fold
affix = args.affix

epochs = 100
steps_epochs = 512 * 4
patience = 5

#### GPU select ####
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

os.makedirs('./history', exist_ok=True)
os.makedirs('./model', exist_ok=True)

#### model weights select ####
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

#### pre-trained model select & model create ####
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

#### data generator concatenate funtion ####
def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)

#### data generator with image augmentation ####
train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        shear_range=30,
        rotation_range=30,
        brightness_range=[0.7,1.3],
        horizontal_flip=True,
        )
#Creates our batch of one image



def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)#rescale=1./255)

#### N-fold traning start ####
for valid_fold in range(fold):
    valid_path = image_path[valid_fold]
    train_path = image_path.copy()
    train_path.pop(valid_fold)

    model, IMAGE_SIZE, BATCH_SIZE, DROP_OUT = model_create(mode)

    train_list = []

    for idx, i in enumerate(train_path):
        tmp_generator = train_datagen.flow_from_directory(
                directory = data_path + '/' + str(i),
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE,
                class_mode='categorical')
        train_list.append(tmp_generator)

        #augmented_images = [tmp_generator[0][0][0] for i in range(5)]
        #plotImages(augmented_images)

    train_generator = combine_gen(train_list[0],train_list[1],train_list[2],train_list[3])

    valid_generator = test_datagen.flow_from_directory(
        directory = data_path + '/' + str(valid_path),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    if affix != '':
        checkponiter = tf.keras.callbacks.ModelCheckpoint(filepath='./model/AUG_' + affix + '_' + weights + '_' + mode + '_FOLD' + str(valid_fold + 1) + '.hdf5', monitor='val_accuracy', verbose=1, mode='max', save_best_only=True)
    else:
        checkponiter = tf.keras.callbacks.ModelCheckpoint(filepath='./model/AUG_' + weights + '_' + mode + '_FOLD' + str(valid_fold + 1) + '.hdf5',monitor='val_accuracy', verbose=1, mode='max', save_best_only=True)

    earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max', restore_best_weights=True)

    print("TRAIN START")
    print("TRAIN FOLD -> ", train_path)
    print("VALID FOLD -> ", [str(valid_path)])
    history = model.fit_generator(
        train_generator, 
        epochs=epochs, 
        steps_per_epoch=steps_epochs,
        validation_data=valid_generator,
        verbose=1,
        shuffle=True,
        callbacks=[earlystopper, checkponiter])
    print("TRAIN COMPLETE")

    hist_df = pd.DataFrame(history.history) 
    result.append(hist_df['val_accuracy'].max())

    if affix != '':
        hist_csv_file = './history/AUG_' + affix + '_' + weights + '_' + mode + '_FOLD' + str(valid_fold+1) + '_history.csv'
    else:
        hist_csv_file = './history/AUG_' + weights + '_' + mode + '_FOLD' + str(valid_fold+1) + '_history.csv'

    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    print("HISTORY SAVE COMPLETE")

    tf.keras.backend.clear_session()
    del model
    gc.collect()

#### Final result print ####
print("\nFINAL RESULTS")
for idx, i in enumerate(result):
    print('VALID FOLD', idx+1, 'ACCURACY : %0.3f' %i)
avg_acc = sum(result) / len(result)
print('\nTOTAL AVG FOLD ACCURACY : %0.3f' %avg_acc)