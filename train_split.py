import cv2
import numpy as np 
import PIL
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import os
import glob

def get_exif(image_path):
    t = os.path.getmtime(image_path)
    d = datetime.datetime.fromtimestamp(t)
    captured_datetime = d.strftime("%Y%m%d_%H%M%S")
    return captured_datetime

files = glob.glob("./train/*")

for file in files:
    fname = get_exif(file)
    imgL = cv2.imread(file)
    imgL = imgL[256:(256+700*4),(984):(984+700*4)]
    count=0
    for i in range(4):
        for j in range(4):
            count = count+1
            img2 = imgL[(700*i):(700*(i+1)),(700*j):(700*(j+1))]
            cv2.imwrite(f'./train_split/{fname}_{count}.jpg',img2)
    

