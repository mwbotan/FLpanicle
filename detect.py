import cv2
import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import os
import glob
import datetime
import onnxruntime
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

def get_exif(image_path):
    t = os.path.getmtime(image_path)
    d = datetime.datetime.fromtimestamp(t)
    captured_datetime = d.strftime("%Y%m%d_%H%M%S")
    return captured_datetime

model = 'YOLOX/yolox_s.onnx'

output_dir ='onnx_out_woBox'
csv_dir ='csv_out'


target_dir = ['./timelapse/220801','./timelapse/220813','./timelapse/220823','./timelapse/220904','./timelapse/220914','./timelapse/220922']
x_margin = [810, 928, 1064, 888, 992,748]

yL=500

dfArea=pd.DataFrame(columns = ["date","left","right"])
index=0
mkdir(output_dir)
mkdir(csv_dir)
dfALL = pd.DataFrame(columns = ['class-id','class','score','x-min','y-min','x-max','y-max','L','a','b',"blue","green","red","h","s","v",'area','fname'])

for n in range(len(target_dir)):
    target = target_dir[n]
    files = glob.glob(f'{target}/**/*.JPG',recursive=True)
    xL = x_margin[n]
    for file in files:
        fname = get_exif(file)
        imgL = cv2.imread(file)
        img_blue, img_green, img_red = cv2.split(imgL)
        if (img_blue==img_green).all() & (img_green==img_red ).all():
            dfArea.loc[index,'date'] = fname
            index = index+1
        else:
            imgL = imgL[yL:(yL+700*4),(xL):(xL+700*4)]
            imgL2 = cv2.imread(file)[yL:(yL+700*4),(xL):(xL+700*4)]


            dfL = pd.DataFrame(columns = ['class-id','class','score','x-min','y-min','x-max','y-max','L','a','b',"blue","green","red","h","s","v","area"])
            count=0
            areaL=0
            areaR=0
            numL=0
            numR=0
            for i in range(4):
                for j in range(4):
                    count=count + 1
                    imgS = imgL[(700*i):(700*(i+1)),(700*j):(700*(j+1))]
                    imgS2 = imgL2[(700*i):(700*(i+1)),(700*j):(700*(j+1))]
                    input_shape = (640,640)
                    img, ratio = preprocess(imgS, input_shape)
                    session = onnxruntime.InferenceSession(model)
                    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
                    output = session.run(None, ort_inputs)
                    predictions = demo_postprocess(output[0], input_shape)[0]
                    boxes = predictions[:, :4]
                    scores = predictions[:, 4:5] * predictions[:, 5:]
                    boxes_xyxy = np.ones_like(boxes)
                    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
                    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
                    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
                    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
                    boxes_xyxy /= ratio
                    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.5)
                    result = []
                    if dets is not None:
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                        imgS = vis(imgS, final_boxes, final_scores, final_cls_inds,
                                          0.3, class_names=COCO_CLASSES)
                        [result.extend((final_cls_inds[x],COCO_CLASSES[int(final_cls_inds[x])],final_scores[x],final_boxes[x][0],final_boxes[x][1],final_boxes[x][2],final_boxes[x][3]) for x in range(len(final_scores)))]


                    df = pd.DataFrame(result, columns = ['class-id','class','score','x-min','y-min','x-max','y-max'])
                    df["L"],df["a"],df["b"],df["blue"],df["green"],df["red"],df["h"],df["s"],df["v"] = [0,0,0,0,0,0,0,0,0]
                    if len(df)!=0:
                        for n in range(len(df)):
                            imgT = imgS2[max(int(df.iloc[n,4]),0):min(int(df.iloc[n,6]),700),max(int(df.iloc[n,3]),0):min(int(df.iloc[n,5]),700)]
                            lab = cv2.cvtColor(imgT, cv2.COLOR_BGR2LAB)
                            hsv = cv2.cvtColor(imgT, cv2.COLOR_BGR2HSV)
                            df.loc[n,"L"]=lab.T[0].flatten().mean()
                            df.loc[n,"a"]=lab.T[1].flatten().mean()
                            df.loc[n,"b"]=lab.T[2].flatten().mean()
                            df.loc[n,"blue"]=imgT.T[0].flatten().mean()
                            df.loc[n,"green"]=imgT.T[1].flatten().mean()
                            df.loc[n,"red"]=imgT.T[2].flatten().mean()
                            df.loc[n,"h"]=hsv.T[0].flatten().mean()
                            df.loc[n,"s"]=hsv.T[1].flatten().mean()
                            df.loc[n,"v"]=hsv.T[2].flatten().mean()
                            df.loc[n,"area"]=(df.loc[n,"x-max"]-df.loc[n,"x-min"])*(df.loc[n,"y-max"]-df.loc[n,"y-min"])
                        dfc = df[df['b'] > 140]
                        dfc = dfc[dfc['area'] < 6000]
                        dfc = dfc[dfc['score'] > 0.7]
                        for n in range(len(dfc)):
                            a=1+1
                            #cv2.rectangle(imgS,(int(dfc.iloc[n,3]),int(dfc.iloc[n,4])),(int(dfc.iloc[n,5]),int(dfc.iloc[n,6])),color=(255, 0, 255),thickness=5)
                        if j < 2:
                            #areaL=areaL+sum((df["x-max"]-df["x-min"])*(df["y-max"]-df["y-min"]))
                            areaL=areaL+sum(dfc["y-max"]-dfc["y-min"])
                        else:
                            #areaR=areaR+sum((df["x-max"]-df["x-min"])*(df["y-max"]-df["y-min"]))
                            areaR=areaR+sum(dfc["y-max"]-dfc["y-min"])

                        df["x-min"]=df["x-min"]+700*j
                        df["x-max"]=df["x-max"]+700*j
                        df["y-min"]=df["y-min"]+700*i
                        df["y-max"]=df["y-max"]+700*i

                        dfL=pd.concat([dfL, df])
            dfL["fname"] = fname
            dfALL=pd.concat([dfALL, dfL])

            dfArea.loc[index,'date'] = fname
            dfArea.loc[index,'left'] = areaL
            dfArea.loc[index,'right'] = areaR
            index = index+1
            dfArea.to_csv(f'./{csv_dir}/FLarea.csv')
            dfALL.to_csv(f'./{csv_dir}/dfALL.csv')
            if len(dfL)!=0:
                cv2.imwrite(f'{output_dir}/{fname}.jpg',imgL)


    