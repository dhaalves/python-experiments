
import os
import tensorflow as tf


from keras.applications.xception import *
from keras.preprocessing import image
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = Xception(weights='imagenet', include_top=False)
# model.summary()

for layer in model.layers[:5]:
    layer.trainable=False


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print('Predited:', decode_predictions(preds, top=5)[0])
