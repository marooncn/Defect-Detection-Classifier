from sklearn.svm import SVC  # libSVM
import transferLearning
import h5py
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

bottleneck_features_train = './bottleneck_features_train.npy'
bottleneck_features_validation = './bottleneck_features_validation.npy'
nb_train_samples = 320
nb_validation_samples = 80


def train_svm():
    train_data = np.load(open('bottleneck_features_train.npy'))
    nsamples, nx, ny, nz = train_data.shape
    train_data = train_data.reshape((nsamples,nx*ny*nz))  # scikit-learn expects 2d num arrays for the training dataset for a fit function
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    clf = SVC(gamma='auto')
    clf.fit(train_data, train_labels)
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    nsamples, nx, ny, nz = validation_data.shape
    validation_data = validation_data.reshape((nsamples,nx*ny*nz))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    accuracy = clf.score(validation_data, validation_labels)
    print("SVM accuracy: %f" % accuracy) 


if __name__ == '__main__':
    if not os.path.exists(bottleneck_features_train and bottleneck_features_validation):
        transferLearning.save_bottlebeck_features()
    train_svm()
