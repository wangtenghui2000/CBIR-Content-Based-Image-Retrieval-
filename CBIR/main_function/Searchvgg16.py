import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.preprocessing import image
from numpy import linalg as LA
import h5py
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
import math
import matplotlib.image as mpimg

class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        # include_top：是否保留顶层的3个全连接网络
        # weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
        # input_tensor：可填入Keras tensor作为模型的图像输出tensor
        # input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，图片的宽高必须大于48，如(200,200,3)
        # pooling：当include_top = False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。‘avg’代表全局平均池化，‘max’代表全局最大值池化。
        # classes：可选，图片分类的类别数，仅当include_top = True并且不加载预训练权重时可用。
        self.model_vgg = VGG16(weights=self.weight,
                               input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                               pooling=self.pooling, include_top=False)
        self.model_vgg.predict(np.zeros((1, 224, 224, 3)))

    '''
    Use vgg16/Resnet model to extract features
    Output normalized feature vector
    '''

    # 提取vgg16最后一层卷积特征
    def vgg_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = self.model_vgg.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

app = Tk()  # 初始化GUI程序
app.withdraw()  # 仅显示对话框，隐藏主窗口


showinfo(title="提示",
             message="请选择您要检索的图像")

open_file_path = askopenfilename(title="请选择一个要打开的图像文件",
                                     )
showinfo(title="提示",
             message="载入图片成功，请等待图像检索")

app.destroy()

index = '../models/vgg_featureCNN.h5'
query =open_file_path

result='../database'
h5f = h5py.File(index, 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute simlarity score and sort
queryVec = model.vgg_extract_feat(query)  # 修改此处改变提取特征的网络

scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
# number of top retrieved images to show
maxres = 16  # 检索出十六张相似度最高的图片
imlist = []
sumacc=0

for i, index in enumerate(rank_ID[0:maxres]):
    imlist.append(imgNames[index])
    sumacc += rank_score[i]
    #print("image names: " + str(imgNames[index]) + " scores: %f" % rank_score[i])
#print("top %d images in order are: " % maxres, imlist)

aveacc=sumacc/16.0

plt.figure(num='result',figsize=(16,8))  #创建一个名为result的窗口,并设置大小
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+100+100")

x=1
for i, im in enumerate(imlist):
    image = mpimg.imread(result + "/" + str(im, 'utf-8'))
    #plt.title("search output %d" % (i + 1))
    plt.subplot(4, 4, x)  # 第i个子窗口
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标尺寸
    x+=1
plt.show()
print("检索评分为")
print(aveacc)