import cv2
import math
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from PIL import Image
import numpy as np

####载入图像窗口
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

file_texture = open("../Repository/GreyMatrixData.txt")
file_color = open(r"../Repository/colorjuData2.txt")
file_shape = open("../Repository/ShapeNchangeData.txt")

#####   texture部分
gray_level = 16

def maxGrayLevel(img):
    max_gray_level=0
    (height,width)=img.shape
    # print height,width
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level+1

def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape

    max_gray_level=maxGrayLevel(input)

    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height-d_y):
        for i in range(width-d_x):
             rows = srcdata[j][i]
             cols = srcdata[j + d_y][i+d_x]
             ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)

    return ret

def feature_computer(p):
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Asm+=p[i][j]*p[i][j]
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
    return Asm,Con,-Eng,Idm

def test(img):

    try:
        img_shape=img.shape
    except:
        print ('imread error')
        return -1

    img=cv2.resize(img,(int(img_shape[1]/2),int(img_shape[0]/2)),interpolation=cv2.INTER_CUBIC)

    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0=getGlcm(img_gray, 1,0)
    #glcm_1=getGlcm(src_gray, 0,1)
    #glcm_2=getGlcm(src_gray, 1,1)
    #glcm_3=getGlcm(src_gray, -1,1)

    asm,con,eng,idm=feature_computer(glcm_0)

    return asm,con,eng,idm

def calc_texture(input, data):
    res = 0
    for i in range (0,4):
        if input[i] == 0:
            continue
        res += abs(float(input[i]-data[i]))/abs(float(input[i]))
    return res
inputlist_texture = (test(inputimg))

#####   color部分
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

def cal_color(input,data):
    ans = 0
    for i in range(0,9):
        if input[i] == 0:
            continue
        ans += abs(input[i] - data[i])/(abs(input[i]))
    return ans

#####   shape部分

def ft(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img_gray)
    humoments = cv2.HuMoments(moments)
    return humoments

def cal_shape(input, data):
    ans = 0
    for i in range(0,6):
        if input[i] == 0:
            continue
        ans += abs(input[i] - data[i])/abs(input[i])
    return ans

inputft = color_moments(inputpath)

inlist = ft(inputimg)

linelist = []
outputlist = []
outacc = []
flag_tex=0
flag_color=0
flag_shape=0
i=1
flag=0
final_flag=0
while 1:
    line_tex=file_texture.readline()
    line_color=file_color.readline()
    line_shape=file_shape.readline()
    if i % 2 ==1:
        name=line_tex
    if i % 2 == 0:
        if not line_tex:
            break
        #求tex_num
        lstr_tex = str(line_tex)
        length_tex = len(lstr_tex)
        lstr_tex = lstr_tex[1:length_tex-2]
        tlist_tex = [float(x) for x in lstr_tex.split(',')]
        num_tex = calc_texture(inputlist_texture, tlist_tex)
        #求col_num
        lstr_color = str(line_color)
        try:
            tlist_color = [float(x) for x in lstr_color.split(',')]
        except ValueError as e:
            continue
        num_color = cal_color(inputft, tlist_color)

        if flag == 1 and num_tex <= 0.5:
            #outputlist.append(name[1:-2])
            #outacc.append(num_tex)
            flag_tex=1


        if num_tex <= 0.001 and flag == 0:

            #outputlist.append(name[1:-2])
            #outacc.append(num_tex)
            flag_tex=1
            final_flag=1
            #flag = 1
        if flag == 1 and num_color < 1:
            #outputlist.append(name[0:-1])
            # outacc.append(num)
            flag_color=1
        if num_color <= 0.1 and flag == 0:
            outputlist.append(name[0:-1])
            # outacc.append(num)
            #flag = 1
            flag_color=1
            final_flag=1

    i+=1
    case = (i-1) % 8

    if case != 1 and case != 0:
        linelist.append(float(line_shape[2:-2]))
    elif case == 0:
        linelist.append(float(line_shape[2:-3]))

    if case == 0 and len(linelist) == 7:
        num_shape = cal_shape(inlist, linelist)
        if flag == 1 and num_shape < 5:
            #outputlist.append(name)
            # outacc.append(num)
            flag_shape=1

        if num_shape < 0.1 and flag == 0:
            #outputlist.append(name)
            # outacc.append(num)
            # flag = 1
            flag_shape=1
            final_flag=1
    if case == 0 and len(linelist) == 7:
        linelist = []

    if flag_shape or flag_tex or flag_color:
        flag_tex=flag_color=flag_shape=0
        outputlist.append(name[1:-2])
        outacc.append(num_tex)
    if final_flag:
        flag=1


plt.figure(num='result',figsize=(16,8))  #创建一个名为result的窗口,并设置大小
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+100+100")


for i in range(1,17):
    plt.subplot(4, 4, i)  # 第i个子窗口
    #plt.title(str(outputlist[i])[10:-4] ) # 第i幅图片标题
    plt.imshow(Image.open(outputlist[i]))
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