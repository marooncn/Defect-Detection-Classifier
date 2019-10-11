import cv2
import os
import numpy as np
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm


class LocalBinaryPatterns:
  def __init__(self,num_points,radius):
    # 存储数据点数目和半径
    self.numPoints = num_points
    self.radius = radius

  def describe(self,image,eps=1e-7):
    # 计算图像的LBP表征，然后使用LBP表征来构建起直方图模式
    lbp = feature.local_binary_pattern(image,self.numPoints,self.radius,method='uniform')
    # 直接返回的lbp特征是无法直接使用的，它与输入图像等宽高，取值范围为0到numPoitns+2
    # 构建直方图可以用来计算 每个LBP模式出现的次数
    (hist,_) = np.histogram(lbp.ravel(),bins = np.arange(0,self.numPoints+3), range=(0,self.numPoints +2))
    # 归一化直方图
    hist = hist.astype("float")
    hist /= (hist.sum()+eps)
    return hist


def prepare_lbp_data(image_path):
  desc = LocalBinaryPatterns(24,8)
  data = []
  pbar = tqdm(image_path)
  for image in pbar:
    pbar.set_description("处理图像 %s" % image)
    im = cv2.imread(image)
    #print("图像尺寸是\t",im.shape)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    hist = desc.describe(gray)
    data.append(hist)
  return data

def train_model(data,label):
  model =LinearSVC(C=100.0, random_state=42)
  model.fit(data,label)
  return model

def model_predict(model,test_img):
  im = cv2.imread(test_img)
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  desc = LocalBinaryPatterns(24, 8)
  hist = desc.describe(gray)
  prediction = model.predict(hist.reshape(1,-1))
  cv2.putText(im,prediction[0],(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
  cv2.imshow("SVC predict ",im)
  cv2.waitKey(0)

def get_train_test_data():
  train_imgs = []
  train_labels = []
  test_imgs = []
  test_labels = []
  base_dir = "./data/"
  for subset in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subset)
    if subset.startswith("defect") or subset.startswith("normal"):
      for img in os.listdir(subpath):
        img_file = os.path.join(subpath,img)
        train_imgs.append(img_file)
        if subset.startswith("defect"):
          train_labels.append(1)
        else:
          train_labels.append(0)
    elif subset.startswith("test"):
      test_paths = os.listdir(subpath)
      for path in test_paths:
        new_path = os.path.join(subpath,path)
        if os.path.isdir(new_path):
          for img in os.listdir(new_path):
            img_file = os.path.join(new_path,img)
            test_imgs.append(img_file)
            if path =="defect":
              test_labels.append(1)
            else:
              test_labels.append(0)
  return train_imgs,np.array(train_labels),test_imgs,np.array(test_labels)

if __name__ =="__main__":
  train_imgs, train_labels, test_imgs, test_labels = get_train_test_data()
  indexs = [x for x in range(0,len(train_imgs))]
  np.random.shuffle(indexs)
  print("获取的图片概览..")
  print("训练集前5 ",train_imgs[:5])
  print("训练集标签前5 ",train_labels[:5])
  # 数据混排
  real_train_imgs = []
  real_train_labels =[]
  for i in indexs:
      real_train_imgs.append(train_imgs[i])
      real_train_labels.append(train_labels[i])
  print("训练集 图片总数 : %d，标签总数: %d ,测试集 图片总数 : %d，标签总数 %d  " % (
  len(train_imgs), len(train_labels), len(test_imgs), len(test_labels)))

  train_data = prepare_lbp_data(real_train_imgs)
  model = train_model(train_data,real_train_labels)
  save_file = "defect_normal_svm_875.bin"
  out = open(save_file,"wb")
  model_save = pickle.dumps(model)
  out.write(model_save)
  out.close()
  input = open(save_file, 'rb')
  real_model = input.read()
  model = pickle.loads(real_model)
  test_data = prepare_lbp_data(test_imgs)
  y_pred = model.predict(test_data)
  print("模型测试准确率\t",accuracy_score(test_labels,y_pred))