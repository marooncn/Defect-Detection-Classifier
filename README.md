## Defect-Detection-Classifier
    Defect detection of water-cooled wall. The sample is hard to collect, so we only have a little dataset 
    which includes 320 training images(160 normal+ 160 defect) and 80 testing images(40 normal+ 40 defect). 
    The image size is 256*256. The dataset is collected by Dong Jin.
## Classifier 
    We use Support Vector Machine(SVM) with different feature extractors and Convolutional Neural Network 
    to train the classifier.
* Gauss filter+LBP+SVM(rbf kernel)
        
      Use Gaussian filter and laplacian operator to denoise and extracts edges, then LBP(Local Binary Patt-
      ern) extract features of preprocessed images as the input of SVM.
* CNN+SVM(rbf kernel)
        
      Use VGG19 to extract features as the input of SVM., the weight of VGG19 is trained by imagenet.
* simple CNN(3 Conv+1 FC)

      Build a simple neural network to train. The network consists of three convolutional layers and a fully
      connected layer.
* transfer Learning(VGG19)
    
      Use VGG19 to extract features as input of a simple network that consists of a fully-connected layer.
### Result
 
|           classifier                   |   accuracy   | 
|----------------------------------------|--------------|
|    Gauss filter+LBP+SVM(rbf kernel)    |    97.25%    | 
|         CNN+SVM(rbf kernel)            |    73.75%    | 
|       simple CNN(3 Conv+1 FC)          |    68.75%    | 
|       transfer Learning(VGG19)         |    82.50%    | 

## run
### Dependencies ###
* Keras `sudo pip install keras`
* NumPy `sudo pip install numpy`
* h5py `sudo pip install h5py`
* scikit-learn `sudo pip install scikit-learn`
### run ###
~~~
# dataset
cp -rf normal_add/* ./normal
rm -rf normal_add/
cp -rf defect_add/* ./defect
rm -rf defect_add
# CNN+SVM(rbf kernel)
python cnnSVM.py
# simple CNN(3 Conv+1 FC)
python CNNclassifier.py
# transfer Learning(VGG19)
python transferLearning.py
~~~
## reference
![scikit-learn tutorial](http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html) </br>
![Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)


  

    
    
