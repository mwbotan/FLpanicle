Scripts for detecting rice flower openings using YOLOX

**train.split.py**

 Generate 700x700 images for annotation from timelapse images.

**yolox_train.py**

 Train YOLOX_S with COCO format training datasets and convert best_ckpt.pth to yolox_s.onnx.

**detect.py**

 Detect flower openings in timelapse images and calculate the feature of each detected region.

**Data sets**

COCO format training data
 
 https://drive.google.com/file/d/1-UIa2l6U8zI3UsJUrYiGwcXr950USKSB

Trained yolox_x model coverted to ONNX format
 
 https://drive.google.com/file/d/1-anoV9C5lyDZ-BdNDGVxQy5ZKESzECkQ
