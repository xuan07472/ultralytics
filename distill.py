import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'ultralytics/cfg/models/v8/yolov8n-fasternet.yaml',
        'data':'dataset/data.yaml',
        'imgsz': 640,
        'epochs': 250,
        'batch': 8,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 10,
        'project':'runs/distill',
        'name':'test',
        
        # distill
        'prune_model': False,
        'teacher_weights': 'ultralytics/cfg/models/v8/yolov8n-fasternet.yaml',
        'teacher_cfg': 'runs/train/yolov8n-fasternet/weights/best.pt',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 0.4,
        
        'teacher_kd_layers': '0-1,0-2,0-3,0-4,4',
        'student_kd_layers': '0-1,0-2,0-3,0-4,4',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.2
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()