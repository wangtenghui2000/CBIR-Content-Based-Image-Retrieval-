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
inputpath = open_file_path
inputimg = cv2.imread(inputpath)
crop_size = (224, 224)
img_new = cv2.resize(inputimg, crop_size, interpolation = cv2.INTER_CUBIC)

file = open("../Repository/ShapeNchangeData.txt")
inputimg = cv2.imread(inputpath)



def ft(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img_gray)
    humoments = cv2.HuMoments(moments)
    return humoments

def cal(input, data):
    ans = 0
    for i in range(0,6):
        if input[i] == 0:
            continue
        ans += abs(input[i] - data[i])/abs(input[i])
    return ans


inlist = ft(inputimg)
i = 0
linelist = []
outputlist = []
outacc = []
flag = 0
loop = 0
while 1:
    line = file.readline()
    if not line:
        break
    i += 1
    case = i % 8
    if case == 1:
        name = line[1:-2]
    elif case != 1 and case != 0:
        linelist.append(float(line[2:-2]))
    elif case == 0:
        linelist.append(float(line[2:-3]))

    if case == 0 and len(linelist) == 7:
        num = cal(inlist, linelist)
        n=float(num)
        if flag == 1 and num < 5:
            outputlist.append(name)
            outacc.append(n)
        if num < 0.1 and flag == 0:
            outputlist.append(name)
            outacc.append(n)
            flag = 1
    if case == 0 and len(linelist) == 7:
        linelist = []


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
