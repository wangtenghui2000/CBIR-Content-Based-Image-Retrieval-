import cv2
import math
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from PIL import Image
from skimage import feature as skif
import numpy as np

app = Tk()  # 初始化GUI程序
app.withdraw()  # 仅显示对话框，隐藏主窗口


showinfo(title="提示",
             message="请选择您要检索的图像")

open_file_path = askopenfilename(title="请选择一个要打开的图像文件",
                                     )
showinfo(title="提示",
             message="载入图片成功，请等待图像检索")

app.destroy()
inputpath = open_file_path
inputimg = cv2.imread(inputpath)
crop_size = (224, 224)
img_new = cv2.resize(inputimg, crop_size, interpolation = cv2.INTER_CUBIC)
file = open("../Repository/LBP.txt")

#获取图像的lbp特征
def get_lbp_data(image_path, lbp_radius=1, lbp_point=8):
    # img = utils.change_image_rgb(image_path)
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    #print(max_bins)
    # hist size:256
    hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return hist

inlist=get_lbp_data(inputpath)
inlist=inlist.tolist()

def calc(input, data):
    res = 0
    for i in range (0,256):
        if input[i] == 0:
            continue
        res += abs(float(input[i]-data[i]))/abs(float(input[i]))
    return res



i = 1
outputlist = []
outacc = []
flag = 0

while 1:
    line = file.readline()
    if i % 2 == 1:
        name = line
    if i % 2 == 0:
        if not line:
            break
        lstr = str(line)
        length = len(lstr)
        lstr = lstr[1:length-2]
        tlist = [float(x) for x in lstr.split(',')]
        num = calc(inlist, tlist)
        if flag == 1 and num <= 80:
            outputlist.append(name[1:-2])
            outacc.append(num)

        if num <= 0.5 and flag == 0:

            outputlist.append(name[1:-2])
            outacc.append(num)

            flag = 1

    i+=1

plt.figure(num='result',figsize=(16,8))  #创建一个名为result的窗口,并设置大小
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+100+100")
for i in range(1,17):
    plt.subplot(4, 4, i)  # 第i个子窗口
    #plt.title(str(outputlist[i])[10:-4] ) # 第i幅图片标题
    plt.imshow(Image.open(outputlist[i-1]))
    plt.axis('off')  # 不显示坐标尺寸

plt.show()   #显示窗口

#定义转换函数：
def change(list):
    x=max(list)
    for i in range(0,len(list)):
        list[i]=x-list[i]
    return list
#定义标准化函数：
def standardize(list):
    sum=0
    for i in range(0,len(list)):
        sum=sum+list[i]**2
    sum=math.sqrt(sum)
    for i in range(0,len(list)):
        list[i]=list[i]/sum
    return list

#定义打分函数：

def calscore(list):
    maxlist=max(list)
    minlist=min(list)
    sum1=0
    sum2=0
    for i in range(0,len(list)):
        sum1=sum1+(list[i]-maxlist)**2
        sum2=sum2+(list[i]-minlist)**2
    d1=math.sqrt(sum1)
    d2=math.sqrt(sum2)
    s=d2/(d1+d2)
    print(s)
outacc=change(outacc)
outacc=standardize(outacc)
print("检索评分为：")

calscore(outacc)