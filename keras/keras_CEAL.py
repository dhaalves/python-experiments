import tensorflow as tf
from keras.datasets import cifar10
from keras.applications import mobilenet, nasnet
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, ProgbarLogger

from keras.layers import Input, Dense, GlobalAveragePooling2D, Reshape, Conv2D

from keras.models import Model
import numpy as np
# from keras.utils import plot_model
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot


num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

batch_size = 100
initial_train_perc = 0.2
initial_train_size = int(x_train.shape[0] * initial_train_perc)

datagen = ImageDataGenerator()

x_train_initial, y_train_initial = iter(datagen.flow(
    x_train, y_train, batch_size=initial_train_size, shuffle=True)).next()

print(x_train_initial.shape)
print(y_train_initial.shape)

print(x_train.shape)
print(y_train.shape)

input_shape = x_train[-1, ].shape
# input_tensor = Input(shape=input_shape)


def _nasnet(num_classes, input_shape=(32, 32, 3), pretrained=True, freezed=True):
    input_tensor = Input(shape=input_shape)

    weights = 'imagenet' if pretrained else None
    base_model = nasnet.NASNetMobile(
        input_tensor=input_tensor, weights=weights, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax',
                        name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def _mobilenet(num_classes, input_shape=(32, 32, 3), pretrained=True, freezed=True):
    input_tensor = Input(shape=input_shape)

    weights = 'imagenet' if pretrained else None
    base_model = mobilenet.MobileNet(
        input_tensor=input_tensor, weights=weights, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    predictions = Reshape((num_classes,), name='reshape_2')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# model = _mobilenet(num_classes)
# model = _nasnet(num_classes, input_shape)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy', metrics=['acc'])

# SVG(model_to_dot(model).create(prog='dot', format='svg'))
model = load_model('ceal_initial_nasnet.hdf5')
model.summary()


# checkpointer = ModelCheckpoint(
#     filepath='ceal_initial_nasnet.hdf5', verbose=1, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
# tensorboard = TensorBoard()
# progbar = ProgbarLogger()
# earlystop = EarlyStopping(patience=5)
#
# hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20,
#                  callbacks=[checkpointer, reduce_lr, tensorboard, progbar, earlystop])



from sklearn.metrics import classification_report, confusion_matrix
import scipy as sc
import heapq

##### CEAL parameters #####

# maximum iteration number
T = 10
# fine-tuning interval
t = 2
# threshold decay rate
dr = 0.0033
# high confidence samples selection threshold
sigma = 0.005
# uncertain samples selection size
K = 2

# unlabeled samples
DU = None
# initially labeled samples
DL = None
# high confidence samples
DH = None

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def least_confidence(y_pred_prob, y_true):

    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    max_prob_index = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index,
                           max_prob,
                           max_prob_index,
                           y_true))

    lci = lci[lci[:, 1].argsort()]

    return lci, lci[:, 0].astype(int)


def margin_sampling(y_pred_prob):
    for row in y_pred_prob:
        #         a = heapq.nlargest(2, range(len(row)), row.take)
        a = row.argsort()[-2:][::-1]
        b = np.take(row, a)
#
    return np.sort(np.amax(y_pred_prob, axis=1))


def entropy(y_pred_prob):
    #     entropy = sc.stats.entropy(y_pred_prob, base=2, axis=1)
    #     entropy = np.nan_to_num(entropy)
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    return entropy


def high_confidence(y_pred_prob, threshold=0.05):
    print(y_pred_prob)
    en = entropy(y_pred_prob)
    print(en)
    j = np.argmax(en)
#     print(np.all(en < threshold))k
    return j


dataset_size = T * 2
num_classes = 10


# np.random.seed(1)

x = x_test[0:dataset_size]
y = y_test[0:dataset_size]

# y_pred_prob = model.predict(x, verbose=1)

y_pred_prob = np.random.rand(y.shape[0], num_classes)
y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)

y_pred_prob = np.around(y_pred_prob, 3)
# print(y_pred_prob)
# print(np.sum(y_pred_prob))
hc = high_confidence(y_pred_prob, sigma)
_, idx = least_confidence(y_pred_prob, y)


for i in range(T):
    step = i * K
    x_uncertain = np.take(x, idx[step:step + K], axis=0)
    y_uncertain = np.take(y, idx[step:step + K], axis=0)

    if DL is None:
        DL = x_uncertain, y_uncertain
    else:
        x_l, y_l = DL
        x_l = np.append(x_l, x_uncertain, axis=0)
        y_l = np.append(y_l, y_uncertain, axis=0)
        DL = x_l, y_l
    print(DL[1].shape)
#     predicted = model.predict(x, verbose=1)

print(y)
print(DL[1])


# print(classification_report(np.argmax(y, axis=1), y_pred_idx))
# print(confusion_matrix(np.argmax(y, axis=1), y_pred_idx))
