import os

!python YOLOX/tools/train.py -f yolox_s.py -d 1 -b 4 --fp16 -o -c YOLOX/weights/yolox_s.pth

!python YOLOX/tools/export_onnx.py --output-name YOLOX/yolox_s.onnx -f YOLOX/yolox_s.py -c YOLOX_outputs/yolox_s/best_ckpt.pth

