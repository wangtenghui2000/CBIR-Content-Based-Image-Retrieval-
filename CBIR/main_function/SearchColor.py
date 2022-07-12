import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from PIL import Image

app = Tk()  # 初始化GUI程序
app.withdraw()  # 仅显示对话框，隐藏主窗口


showinfo(title="提示",
             message="请选择您要检索的图像")

open_file_path = askopenfilename(title="请选择一个要打开的图像文件",
                                     )
showinfo(title="提示",
             message="载入图片成功，请等待图像检索")

app.destroy()
def color_moments(filename):
    img = cv2.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature

def cal(input,data):
    ans = 0
    for i in range(0,9):
        if input[i] == 0:
            continue
        ans += abs(input[i] - data[i])/(abs(input[i]))
    return ans

file = open(r"../Repository/colorjuData.txt")
inputpath = open_file_path
inputimg = cv2.imread(inputpath)
inputft = color_moments(inputpath)
crop_size = (224, 224)
img_new = cv2.resize(inputimg, crop_size, interpolation = cv2.INTER_CUBIC)

i = 0
outputlist = []
outacc = []
flag = 0

while 1:
    line = file.readline()
    if not line:
        break
    if i % 2 == 0:
        name = line
    if i % 2 == 1:
        lstr = str(line)
        try:
            tlist = [float(x) for x in lstr.split(',')]
        except ValueError as e:
            # print("error", e, "on line", i)
            continue
        num = cal(inputft,tlist)
        if flag == 1 and num < 1:
            outputlist.append(name[0:-1])
            outacc.append(num)
        if num <= 0.5 and flag == 0:
            outputlist.append(name[0:-1])
            outacc.append(num)
            flag = 1
    i+=1

plt.figure(num='result',figsize=(16,8))  #创建一个名为result的窗口,并设置大小
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+100+100")
for i in range(1,17):
    plt.subplot(4, 4, i)  # 第i个子窗口
    #a=index_out_pic[i-1]
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