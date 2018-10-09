import os

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from keras.applications import nasnet, resnet50, inception_v3
from keras.preprocessing import image
from keras import losses, metrics, callbacks, optimizers, activations, models, layers
from keras import backend as K
K.set_image_dim_ordering('tf')


def _nasnet(num_classes, pretrained=True, freezed=True):
    weights = 'imagenet' if pretrained else None
    base_model = nasnet.NASNetMobile(weights=weights, include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(
        num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


def _inception(num_classes, pretrained=True, freezed=True):
    weights = 'imagenet' if pretrained else None
    base_model = inception_v3.InceptionV3(weights=weights, include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation=activations.relu)(x)
    predictions = layers.Dense(
        num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


def _resnet(num_classes, pretrained=True, freezed=True):
    weights = 'imagenet' if pretrained else None
    base_model = resnet50.ResNet50(input_shape=(
        224, 224, 3), weights=weights, include_top=False)
    x = base_model.output
    x = layers.Flatten()(x)
    predictions = layers.Dense(
        num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


dataset_path = '/mnt/sdb1/dataset/mammoset/exp5-2_aug'

# for i in range(5):
i = 0
img_size = 299
batch_size = 16
generator = image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
# samplewise_center=True, featurewise_center=True, zca_whitening=True)
# ,
#                                     shear_range=0.2,
#                                     zoom_range=0.2,
#                                     horizontal_flip=True)

train_data_gen = generator.flow_from_directory(dataset_path,
                                               # color_mode='grayscale',
                                               target_size=(
                                                   img_size, img_size),
                                               subset='training',
                                               batch_size=batch_size)

test_data_gen = generator.flow_from_directory(dataset_path,
                                              # color_mode='grayscale',
                                              target_size=(img_size, img_size),
                                              subset='validation',
                                              batch_size=1,
                                              shuffle=False)

model = None
model_ckp = os.path.basename(dataset_path) + '_resnet_ckp_' + str(i) + '.hdf5'

checkpointer = callbacks.ModelCheckpoint(
    filepath=model_ckp, verbose=1, save_best_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
tensorboard = callbacks.TensorBoard()
progbar = callbacks.ProgbarLogger()
earlystop = callbacks.EarlyStopping(patience=5)
csv_logger = callbacks.CSVLogger(os.path.basename(
    dataset_path) + '_resnet_' + str(i) + '.csv')

# if os.path.exists(model_ckp):
#     model = models.load_model(model_ckp)
# else:
model = _inception(train_data_gen.num_classes, freezed=False)
model.compile(optimizer=optimizers.Adam(lr=0.01),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

#     model.summary()

hist = model.fit_generator(train_data_gen, epochs=3, validation_data=test_data_gen, workers=8, verbose=2,
                           callbacks=[checkpointer, earlystop, csv_logger, tensorboard])
print(hist.history)

prob = model.predict_generator(test_data_gen, verbose=1)
y_pred = np.argmax(prob, axis=1)
y_true = test_data_gen.classes

#     print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred).astype(int)
print(cm)
np.savetxt(os.path.basename(dataset_path) + '_resnet_confusion_matrix_' + str(i) + '.csv',
           cm, delimiter=',')


# model_features = models.Model(inputs=model.input, outputs=model.get_layer('#layer-name#').output)
#
# features = model_features.predict_generator(test_data_gen)
