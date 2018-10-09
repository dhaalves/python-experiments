import numpy as np
from keras import losses, metrics, callbacks, optimizers, activations, models, layers
from keras.applications import resnet50, inception_v3
from keras.preprocessing.image import ImageDataGenerator


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

def _inception(num_classes, pretrained=True, freezed=True):
    weights = 'imagenet' if pretrained else None
    base_model = inception_v3.InceptionV3(weights=weights, include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation=activations.relu)(x)
    predictions = layers.Dense(num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)

dataset_dir = '/mnt/sdb1/datasets/leaves/leaves1'

img_target_size = (224, 224)
batch_size = 20

data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2);

train_data = data_gen.flow_from_directory(dataset_dir, batch_size=batch_size, target_size=img_target_size, subset='training')
test_data = data_gen.flow_from_directory(dataset_dir, batch_size=batch_size, target_size=img_target_size, subset='validation', shuffle=False)


checkpointer = callbacks.ModelCheckpoint(
    filepath='leaves_resnet50_ckpt.hdf5', verbose=1, save_best_only=True)

model = _resnet(train_data.num_classes)
model.compile(loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy], optimizer=optimizers.Adagrad())
model.fit_generator(train_data, validation_data=test_data, epochs=50, callbacks=[checkpointer])

model = models.load_model('leaves_resnet50_ckpt.hdf5')

model.summary()


def least_confidence(y_pred_prob, y_true):

    origin_index = np.arange(0,len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    max_prob_index = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index,
                           max_prob,
                           max_prob_index,
                           y_true))

    lci = lci[lci[:,1].argsort()]
    return lci, lci[:,0].astype(int)

def margin_sampling(y_pred_prob, y_true):

    origin_index = np.arange(0,len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    max_prob_index = np.argmax(y_pred_prob, axis=1)

    for row in y_pred_prob:
        #         a = heapq.nlargest(2, range(len(row)), row.take)
        a = row.argsort()[-2:][::-1]
        b = np.take(row, a)
    #
    return np.sort(np.amax(y_pred_prob, axis=1))

def entropy(y_pred_prob, y_true):
    #     entropy = sc.stats.entropy(y_pred_prob, base=2, axis=1)
    #     entropy = np.nan_to_num(entropy)
    origin_index = np.arange(0,len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    max_prob_index = np.argmax(y_pred_prob, axis=1)
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    eni = np.column_stack((origin_index,
                           entropy,
                           max_prob,
                           max_prob_index,
                           y_true))

    eni = eni[(-eni[:,1]).argsort()]
    return eni, eni[:,0].astype(int)

def high_confidence(y_pred_prob, y_true, delta):
    eni, eni_idx = entropy(y_pred_prob, y_true)
    hcs = eni[eni[:,1] < delta]
    return hcs, hcs[:,0].astype(int)


##### CEAL parameters #####

# #maximum iteration numbers
# T=50
# #fine-tuning interval
# t=1
# #threshold decay rate
# dr= 0.00033
# #high confidence samples selection threshold
# delta=0.005
# #uncertain samples selection size
# K=1000
#
# #unlabeled samples
# DU = None
# #initially labeled samples
# DL = None
# #high confidence samples
# DH = None
#
# dataset_size = 1000
# num_classes = 10
#
# np.random.seed(1)
#
# x = x_train#[0:dataset_size]
# y = y_train#[0:dataset_size]
#
# # y_pred_prob = np.random.rand((y.shape[0]), num_classes)
# # y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)
# # y_pred_prob = np.around(y_pred_prob, 3)
# # print(y_pred_prob)
# # print(np.sum(y_pred_prob))
#
# for i in range(T):
#     y_pred_prob = model.predict(x, verbose=0)
#     #     y_pred_prob = np.around(y_pred_prob, 3)
#
#     isa, isa_idx = least_confidence(y_pred_prob, y)
#     # isa, isa_idx = margin_sampling(y_pred_prod, y)
#     # isa, isa_idx = entropy(y_pred_prob, y)
#     hcs, hcs_idx = high_confidence(y_pred_prob, y, delta)
#
#     idx_concat = np.concatenate((isa_idx, hcs_idx),0)
#     idx = np.unique(idx_concat, return_index=True)[1]
#     idx = np.array([idx_concat[i] for i in sorted(idx)])
#
#     step = i*K
#     DH = np.take(x, idx[step+K:], axis=0), np.take(y, idx[step+K:], axis=0)
#     DL = np.take(x, idx[step:step+K], axis=0), np.take(y, idx[step:step+K], axis=0)
#     x = np.delete(x, idx[step:step+K], axis=0)
#     y = np.delete(y, idx[step:step+K], axis=0)
#
#     al_x, al_y = np.append(DL[0], DH[0], axis=0), np.append(DL[1], DH[1], axis=0)
#     if i % t == 0:
#         model.fit(al_x, al_y, epochs=5, verbose=0)
#         delta -= (dr * (i//t))
#     evaluate = model.evaluate(x_test, y_test, verbose=1)
#     print(evaluate)
#
#
# # print(classification_report(np.argmax(y, axis=1), y_pred_idx))
# # print(confusion_matrix(np.argmax(y, axis=1), y_pred_idx))

