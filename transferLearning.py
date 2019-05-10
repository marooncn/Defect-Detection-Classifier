import numpy as np
np.random.seed(0)

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import h5py
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'   # build TensorFlow from source, it can be faster on your machine.


train_data_dir = './data/train'
validation_data_dir = './data/test'
weights_path = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# dimensions of our images.
img_width, img_height = 256, 256

top_model_weights_path = './models/bottleneck_fc_model.h5'
bottleneck_features_train = './bottleneck_features_train.npy'
bottleneck_features_validation = './bottleneck_features_validation.npy'
nb_train_samples = 320
nb_validation_samples = 80
epochs = 10
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights=None, input_shape=(256,256,3))  # channel last, if not, change ~/.keras/keras.json
    model.load_weights(weights_path)
    print('Model loaded.')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)  
    print("Saved model %s" % top_model_weights_path)


if __name__ == '__main__':
    if not os.path.exists(bottleneck_features_train and bottleneck_features_validation):
        save_bottlebeck_features()
   
    train_top_model()
