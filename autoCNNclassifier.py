import autokeras as ak
from autokeras.image.image_supervised import load_image_dataset
from keras.models import load_model
from keras.utils import plot_model


train_data_dir = "./data/train"
validation_data_dir = "./data/test"


def train_model():

    clf = ak.ImageClassifier(verbose=True, augment=False)
    train_data, train_labels = load_image_dataset(csv_file_path=train_data_dir+"/label.csv",
                                      images_path=train_data_dir)
    validation_data, validation_labels = load_image_dataset(csv_file_path=validation_data_dir+"/label.csv",
                                      images_path=validation_data_dir)
    clf.fit(train_data, train_labels)
    clf.final_fit(train_data, train_labels, validation_data, validation_labels, retrain=True)
    y = clf.evaluate(validation_data, validation_labels)
    print("auto CNN classifier accuracy: %f" % y)
    clf.load_searcher().load_best_model().produce_keras_model().save('shallowCNN_model.h5')


def visualize_model():
    model = load_model('shallowCNN_model.h5') #See 'How to export keras models?' to generate this file before loading it.
    # plot_model(model, to_file='shallowCNN_model.png')
    model.summary()

    
if __name__ == '__main__':
    train_model()
    visualize_model()
