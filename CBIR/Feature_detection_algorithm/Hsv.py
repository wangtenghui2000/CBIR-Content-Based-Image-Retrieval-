import cv2
import numpy as np
import glob
from time import sleep
from tqdm import tqdm

def color_moments(img):
    if img is None:
        return
    # 将BGR转换为HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 分割通道 - h,s,v
    h, s, v = cv2.split(hsv)
    # 初始化颜色特征
    color_feature = []
    # 第一个中心矩--平均
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # 第二中心矩--标准差
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # 第三中心矩--偏斜度的第三根
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature

file=open('../Repository/colorjuData.txt','w+')
imgset = glob.glob("../dataset/*/*.jpg")
for i in tqdm(imgset):
    img = cv2.imread(i)
    out = color_moments(img)
    file.write(str(i) + "\n" + str(out) + "\n")
    sleep(0.01)
file.close()
