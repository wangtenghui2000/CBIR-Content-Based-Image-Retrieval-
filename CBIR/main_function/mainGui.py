import os
import tkinter
top=tkinter.Tk()
def color():
    os.system('python SearchColor.py')

def texture():
    os.system('python SearchTexture.py')

def shape():
    os.system('python SearchShape.py')

def mix():
     os.system('python SearchMix.py')

def LBP():
    os.system('python SearchLBP.py')

def VGG16():
    os.system('python Searchvgg16.py')

def Resnet50():
    os.system('python SearchResNet.py')

def DenseNet121():
    os.system('python SearchDensenet.py')

screenwidth = top.winfo_screenwidth()
screenheight = top.winfo_screenheight()
width=500
heigh=500
top.geometry('%dx%d+%d+%d'%(width, heigh, (screenwidth-width)/2, (screenheight-heigh)/2))
top.title('CBIR')
B1=tkinter.Button(top,text="基于颜色特征检索：HSV 中心距法",command= color)
B2=tkinter.Button(top,text="基于纹理特征检索：灰度矩阵法",command= texture)
B3=tkinter.Button(top,text="基于形状特征检索：形状Hu不变矩法",command= shape)
B4=tkinter.Button(top,text="---混合检索---",command= mix)
B5=tkinter.Button(top,text="---LBP检索---",command= LBP)
B6=tkinter.Button(top,text="---VGG16检索---",command= VGG16)
B7=tkinter.Button(top,text="---Resnet50检索---",command= Resnet50)
B8=tkinter.Button(top,text="---DenseNet121检索---",command= DenseNet121)


B1.pack(fill=tkinter.X, padx=5,pady=6)
B2.pack(fill=tkinter.X, padx=5,pady=6)
B3.pack(fill=tkinter.X, padx=5,pady=6)
B4.pack(fill=tkinter.X, padx=5,pady=6)
B5.pack(fill=tkinter.X, padx=5,pady=6)
B6.pack(fill=tkinter.X, padx=5,pady=6)
B7.pack(fill=tkinter.X, padx=5,pady=6)
B8.pack(fill=tkinter.X, padx=5,pady=6)

top.mainloop()