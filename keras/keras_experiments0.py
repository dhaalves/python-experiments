import os

import numpy as np

from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from sklearn.metrics import classification_report, confusion_matrix

from keras.applications import nasnet, resnet50, inception_v3
from keras.preprocessing import image
from keras import losses, metrics, callbacks, optimizers, activations, models, layers
from keras import backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

K.set_image_dim_ordering('tf')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def _inception(num_classes, pretrained=True, freezed=True):
    weights = 'imagenet' if pretrained else None
    base_model = inception_v3.InceptionV3(weights=weights, include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense({{choice([256, 512, 1024])}}, activation=activations.relu)(x)
    predictions = layers.Dense(
        num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


dataset_path = '/mnt/sdb1/dataset/mammoset/exp5-2_B'


def model(train_gen, val_gen):

    i = 0
    img_size = 299
    generator = image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    #                                     shear_range=0.2,
    #                                     zoom_range=0.2,
    #                                     horizontal_flip=True)

    train_data_gen = generator.flow_from_directory(dataset_path,
                                                   # color_mode='grayscale',
                                                   target_size=(
                                                       img_size, img_size),
                                                   subset='training',
                                                   batch_size={{choice([16, 32, 64, 128])}})

    test_data_gen = generator.flow_from_directory(dataset_path,
                                                  # color_mode='grayscale',
                                                  target_size=(img_size, img_size),
                                                  subset='validation',
                                                  batch_size=1,
                                                  shuffle=False)

    model = None
    model_ckp = os.path.basename(dataset_path) + '_inception_ckp_' + str(i) + '.hdf5'

    checkpointer = callbacks.ModelCheckpoint(
        filepath=model_ckp, verbose=1, save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    tensorboard = callbacks.TensorBoard()
    # progbar = callbacks.ProgbarLogger()
    # earlystop = callbacks.EarlyStopping(patience=5)
    csv_logger = callbacks.CSVLogger(os.path.basename(
        dataset_path) + '_inception' + str(i) + '.csv')

    # if os.path.exists(model_ckp):
    #     model = models.load_model(model_ckp)
    # else:
    model = _inception(train_data_gen.num_classes, freezed=False)
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])

    #     model.summary()

    hist = model.fit_generator(train_data_gen, epochs=5, validation_data=test_data_gen, workers=8, verbose=2,
                               callbacks=[checkpointer, csv_logger, tensorboard])

    prob = model.predict_generator(test_data_gen, verbose=1)
    y_pred = np.argmax(prob, axis=1)
    y_true = test_data_gen.classes

    #     print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).astype(int)
    print(cm)


# model_features = models.Model(inputs=model.input, outputs=model.get_layer('#layer-name#').output)
#
# features = model_features.predict_generator(test_data_gen)
