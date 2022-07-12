import cv2 as cv
from skimage import feature as skif
import numpy as np
import glob
from time import sleep
from tqdm import tqdm

#获取图像的lbp特征
def get_lbp_data(image_path, lbp_radius=1, lbp_point=8):
    # img = utils.change_image_rgb(image_path)
    img = cv.imread(image_path)
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 使用LBP方法提取图像的纹理特征.
    #lbp_point：选取中心像素周围的像素点的个数；lbp_radius：选取的区域的半径
    #以下为5种不同的方法提取的lbp特征，相应的提取到的特征维度也不一样
    #'default': original local binary pattern which is gray scale but notrotation invariant
    #'ror': extension of default implementation which is gray scale androtation invariant
    #'uniform': improved rotation invariance with uniform patterns andfiner quantization of the angular space which is gray scale and rotation invariant.
    #'nri_uniform': non rotation-invariant uniform patterns variantwhich is only gray scale invariant
    #'var': rotation invariant variance measures of the contrast of localimage texture which is rotation but not gray scale invariant
    lbp = skif.local_binary_pattern(image, lbp_point, lbp_radius, 'default')
    # 统计图像的直方图
    max_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return hist



file = open("../Repository/LBP.txt", 'w+')
imgset = glob.glob("../dataset/*/*.jpg")
for i in tqdm(imgset):
    feature = get_lbp_data(i)
    f = feature.tolist()
    out = "{" + str(i) + "}\n" + str(f) + "\n"
    file.write(out)
    sleep(0.01)
file.close()

