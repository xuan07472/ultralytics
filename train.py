import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/ultralytics/cfg/models/v8/yolov8_C2f_SCConv_b2_Triplet_v1_c.yaml')
    # model = YOLO('/home/ubuntu/guidang/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml')
    model.train(data='ultralytics/datasets/luderick_base/luderick_base.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='b150n_C2f_SCConv_b2_Triplet_v1_c',
                )
    