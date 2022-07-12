import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from numpy import linalg as LA
from tqdm import tqdm
import glob
import h5py
import os


class ResNet:
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

        self.model_resnet = ResNet50(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        self.model_resnet.predict(np.zeros((1, 224, 224, 3)))

    #提取resnet50最后一层卷积特征
    def resnet_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_resnet(img)
        feat = self.model_resnet.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat


if __name__ == "__main__":
    imgset = glob.glob("../database/*.jpg")
    index = '../models/resnet_featureCNN.h5'
    #img_list = get_imlist(database)

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = ResNet()
    for img_path in tqdm(imgset):
        norm_feat = model.resnet_extract_feat(img_path)  # 修改此处改变提取特征的网络
        img_name = os.path.split(img_path)[1]

        feats.append(norm_feat)
        names.append(img_name)


    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    # output = args["index"]
    output = index
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
