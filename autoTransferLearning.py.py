import autokeras as ak
import transferLearning
import h5py
import numpy as np
import os

bottleneck_features_train = './bottleneck_features_train.npy'
bottleneck_features_validation = './bottleneck_features_validation.npy'
nb_train_samples = 320
nb_validation_samples = 80

def train_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    nsamples, nx, ny, nz = train_data.shape
    train_data = train_data.reshape((nsamples,nx*ny*nz))  # scikit-learn expects 2d num arrays for the training dataset for a fit function
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    clf = ak.ImageClassifier()
    clf.fit(train_data, train_labels)
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    nsamples, nx, ny, nz = validation_data.shape
    validation_data = validation_data.reshape((nsamples,nx*ny*nz))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    clf.final_fit(train_data, train_labels, validation_data, validation_labels, retrain=True)

    y = clf.evaluate(validation_data, validation_labels)

    print("auto Transfer Learning accuracy: %f" % y)


if __name__ == '__main__':
    if not os.path.exists(bottleneck_features_train and bottleneck_features_validation):
        transferLearning.save_bottlebeck_features()
    train_model()

