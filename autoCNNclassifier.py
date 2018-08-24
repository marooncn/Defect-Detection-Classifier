import autokeras as ak


train_data_dir = './data/train'
validation_data_dir = './data/test'


def train_model():

    clf = ak.ImageClassifier()
    train_data, train_labels = clf.load_image_dataset(train_data_dir)
    validation_data, validation_labels = clf.load_image_dataset(validation_data_dir)
    clf.fit(train_data, train_labels)
    clf.final_fit(train_data, train_labels, validation_data, validation_labels, retrain=True)
    y = clf.evaluate(validation_data, validation_labels)

    print("auto CNN classifier accuracy: %f" % y)


if __name__ == '__main__':
    train_model()

