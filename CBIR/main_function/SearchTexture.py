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

file = open("../Repository/GreyMatrixData.txt")

inputpath = open_file_path
inputimg = cv2.imread(inputpath)
crop_size = (224, 224)
img_new = cv2.resize(inputimg, crop_size, interpolation = cv2.INTER_CUBIC)

#定义最大灰度级数
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

def calc(input, data):
    res = 0
    for i in range (0,4):
        if input[i] == 0:
            continue
        res += abs(float(input[i]-data[i]))/abs(float(input[i]))
    return res
inputlist = (test(inputimg))


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
        num = calc(inputlist, tlist)

        if flag == 1 and num <= 0.5:
            outputlist.append(name[1:-2])
            outacc.append(num)

        if num <= 0.001 and flag == 0:

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