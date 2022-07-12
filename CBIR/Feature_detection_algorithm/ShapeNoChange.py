
import glob
import cv2
from time import sleep
from tqdm import tqdm

def test(img):
    moments = cv2.moments(img)
    humoments = cv2.HuMoments(moments)
    return humoments

if __name__=='__main__':
    f = open("../Repository/ShapeNchangeData.txt", 'w+')
    imgset = glob.glob("../dataset/*/*.jpg")
    for i in tqdm(imgset):
        img = cv2.imread(i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = "{" + str(i) + "}\n" + str(test(img_gray)) + "\n"
        #print(out)
        f.write(out)
        sleep(0.01)
    f.close()