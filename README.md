## Defect-Detection-Classifier
    Defect detection of water-cooled wall. The sample is hard to collect, so we only have a little dataset 
    which includes 320 training images(160 normal+ 160 defect) and 80 testing images(40 normal+ 40 defect). 
    The image size is 256*256. The dataset is collected by Dong Jin. Thanks the advice from Yu Fang about 
    the using of gcForest.
## dataset 
    the above three images are normal examples and the below are defect.
![normal1](https://github.com/marooncn/Defect-Detection-Classifier/blob/master/data/train/normal/2.jpg)
![normal2](https://github.com/marooncn/Defect-Detection-Classifier/blob/master/data/train/normal/3.jpg)
![normal3](https://github.com/marooncn/Defect-Detection-Classifier/blob/master/data/train/normal/4.jpg)
![defect1](https://github.com/marooncn/Defect-Detection-Classifier/blob/master/data/train/defect/2.jpg)
![defect2](https://github.com/marooncn/Defect-Detection-Classifier/blob/master/data/train/defect/3.jpg)
![defect3](https://github.com/marooncn/Defect-Detection-Classifier/blob/master/data/train/defect/4.jpg)
## Classifier
    We use Support Vector Machine(SVM) with different feature extractors, deep forest and Convolutional Neural 
    Network to train the classifier.
* Gauss filter+LBP+SVM(rbf kernel)

      Use Gaussian filter and laplacian operator to denoise and extracts edges, then LBP(Local Binary Patt-
      ern) extract features of preprocessed images as the input of SVM.
* CNN+SVM(rbf kernel)
      
      Use VGG16 to extract features as the input of SVM., the weight of VGG16 is trained on ImageNet.
* simple CNN(3 Conv+1 FC)

      Build a simple neural network to train. The network consists of three convolutional layers and a fully
      connected layer.
* transfer Learning(VGG16)
    
      Use VGG16 to extract features as input of a simple network that consists of a fully-connected layer.
* Neural Network Search
    
      Use NNS to search a best network.
* gcForest
    
      Use deep forest(Only cascade forest structure/With multi-grained forests) to train the ensemble classifier. 
### Result
 
|                classifier                  |   accuracy   | 
|--------------------------------------------|--------------|
|      Gauss filter+LBP+SVM(rbf kernel)      |    97.25%    | 
|            CNN+SVM(rbf kernel)             |    73.75%    | 
|          simple CNN(3 Conv+1 FC)           |    70.00%    | 
|         transfer Learning(VGG16)           |    82.50%    |
|          Neural Network Search             |    82.28%    |
|  gcForest (without multi-grained forests)  |    77.50%    |
| gcForest (with multi-grained forests, i=8) |    88.75%    |

## run
### Dependencies ###
* Keras `sudo pip install keras`
* NumPy `sudo pip install numpy`
* h5py `sudo pip install h5py`
* scikit-learn `sudo pip install scikit-learn`
* [gcForest](https://github.com/kingfengji/gcForest)
* [AutoKeras](https://github.com/jhfjhfj1/autokeras) <br>
compile from source code and revise according to [this issue](https://github.com/jhfjhfj1/autokeras/issues/144)
### run ###
~~~
# read README.md in models folder and download weight file of pre-trained VGG on the ImageNet dataset.
# dataset
cp -rf normal_add/* ./normal
rm -rf normal_add/
cp -rf defect_add/* ./defect
rm -rf defect_add
# CNN+SVM(rbf kernel)
python cnnSVM.py
# simple CNN(3 Conv+1 FC)
python CNNclassifier.py
# transfer Learning(VGG16)
python transferLearning.py
# gcForest (without multi-grained forests) 
python ./data/train/write_label.py
python ./data/test/write_label.py
cd 
python ./gcForest/demo_Defect-Detection-Classifier.py --model ./gcForest/demo_Defect-Detection-Classifier-ca.json
# gcForest (with multi-grained forests, i=8) 
python ./gcForest/demo_Defect-Detection-Classifier.py --model ./gcForest/demo_Defect-Detection-Classifier-gc8.json
# Neural Network Search
python ./data/train/write_label2.py
python ./data/test/write_label2.py
python autoCNNclassifier.py
~~~
## reference
[scikit-learn tutorial](http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html) </br>
[Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
