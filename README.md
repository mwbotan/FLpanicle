Scripts to use YOLOX to detect rice flower openings.

**train.split.py**

 Generate 700x700 images for annotation from timelapse images.

**yolox_train.py**

 Train YOLOX_S with COCO format training datasets and convert best_ckpt.pth to yolox_s.onnx.

**detect.py**

 Detect flower openings in timelapse images and calculate the feature of each detected region.
