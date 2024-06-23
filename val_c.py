import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 
    model = YOLO('/home/ubuntu/guidang/ultralytics/runs/train/b150n-C2f_SCConv_b2_Triplet_v1_c2/weights/best.pt')
    # model = YOLO('/home/ubuntu/ultralytics/runs/prune/yolov8_b150n_C2f_SCConv_b2_Triplet_v1_lamp200_s2ng-prune/weights/prune.pt')
    model.val(data='/home/ubuntu/guidang/ultralytics/datasets/luderick_base/luderick_base.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              # project='runs/val/2080ti',
              project='runs/val/',
              name='b150n-C2f_SCConv_b2_Triplet_v1_c',
              )

