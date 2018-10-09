import argparse

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K, callbacks, optimizers, losses
from keras.preprocessing.image import ImageDataGenerator


def split_train_test(path, perc_split, recreate=False):
    import os
    import shutil
    import numpy as np
    train_root_path = path + '_train'
    test_root_path = path + '_test'
    new_split = not os.path.exists(test_root_path) or not os.path.exists(train_root_path) or recreate
    if new_split:
        if os.path.exists(train_root_path):
            shutil.rmtree(train_root_path)
        if os.path.exists(test_root_path):
            shutil.rmtree(test_root_path)
        shutil.copytree(path, train_root_path)
        classes = os.listdir(train_root_path)
        for folder in classes:
            train_path = os.path.join(train_root_path, folder)
            test_path = os.path.join(test_root_path, folder)
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            files = os.listdir(os.path.join(train_root_path, folder))
            for f in files:
                if np.random.rand(1) < perc_split:
                    shutil.move(train_path + '/' + f, test_path + '/' + f)
                    # plot_log('result/log.csv')
    return train_root_path, test_root_path


def load_dataset(dataset, batch_size, target_size):


    train_path, test_path = split_train_test(datasets[dataset], 0.2)

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        # color_mode='grayscale',
        class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=target_size,
        batch_size=batch_size,
        # color_mode='grayscale',
        class_mode='categorical')

    return (train_generator, test_generator)


def g(generator):
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])


datasets = {'LEAVES1': '/home/daniel/Desktop/datasets/leaves/leaves1',
            'SOYBEAN1': '/home/daniel/Desktop/datasets/soybean/soybean1',
            'SOYBEAN2': '/home/daniel/Desktop/datasets/soybean/soybean2',
            'SOYBEAN3': '/home/daniel/Desktop/datasets/soybean/soybean3',
            'PARASITES1': '/home/daniel/Desktop/datasets/parasites/parasites1'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.1, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=1.0, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--dataset', default='LEAVES1')
    args = parser.parse_args()
    print(args)

    (train_generator, test_generator) = load_dataset(args.dataset, args.batch_size, (299, 299))

    input_shape = train_generator.image_shape

    n_class = train_generator.num_classes

    print(n_class)
    print(input_shape)


    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=True)

    base_model.layers.pop()




    print(base_model.summary())
    # add a global spatial average pooling layer
    x = base_model.output
    # let's add a fully-connected layer
    # x = Dense(2048)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(n_class, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # print(model.summary())
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss='categorical_crossentropy', metrics=['mae', 'acc'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers

    log = callbacks.CSVLogger(args.save_dir + '/inception/log_' + args.dataset + '.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/inception/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/inception/weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))


    steps_per_epoch = int(train_generator.samples / args.batch_size)
    validation_steps = int(test_generator.samples / args.batch_size)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_generator,
                        validation_steps=1,
                        epochs=args.epochs,
                        callbacks=[log, tb, checkpoint, lr_decay])