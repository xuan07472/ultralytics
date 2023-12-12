# Compress Experiment (For BiliBili魔鬼面具)
### Model:yolov8n.yaml Dataset:Visdrone only using 30% Training Data

```
------------------ train base model ------------------
model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=300,
            batch=32,
            close_mosaic=30,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n-visdrone',
            )

nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log
```

```
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-fps.log
------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp2-fps.log
------------------ lamp exp3 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp3',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-lamp-exp3.log 2>&1 & tail -f logs/yolov8n-lamp-exp3.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp3-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp3-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp3-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp3-fps.log
```

```
------------------ group-taylor exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp1',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-grouptaylor-exp1.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouptaylor-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-fps.log
------------------ group-taylor exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp2',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-grouptaylor-exp2.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouptaylor-exp2-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp2-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-fps.log
```

```
------------------ group-hessian exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 24,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouphessian-exp1',
    
    # prune
    'prune_method':'group_hessian',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-grouphessian-exp1.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouphessian-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouphessian-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouphessian-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-fps.log
```

```
------------------ slim exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-slim-exp1',
    
    # prune
    'prune_method':'slim',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.04,
    'reg_decay': 0.05,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-slim-exp1.log 2>&1 & tail -f logs/yolov8n-slim-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-slim-exp1-test.log 2>&1 & tail -f logs/yolov8n-slim-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-slim-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-slim-exp1-fps.log 2>&1 & tail -f logs/yolov8n-slim-exp1-fps.log
```

```
------------------ group_sl exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-groupsl-exp1',
    
    # prune
    'prune_method':'group_sl',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.015,
    'reg_decay': 0.05,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-groupsl-exp1.log 2>&1 & tail -f logs/yolov8n-groupsl-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-groupsl-exp1-test.log 2>&1 & tail -f logs/yolov8n-groupsl-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-groupsl-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-groupsl-exp1-fps.log 2>&1 & tail -f logs/yolov8n-groupsl-exp1-fps.log
```

```
------------------ group_slim exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-groupslim-exp1',
    
    # prune
    'prune_method':'group_slim',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.02,
    'reg_decay': 0.05,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-groupslim-exp1.log 2>&1 & tail -f logs/yolov8n-groupslim-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-groupslim-exp1-test.log 2>&1 & tail -f logs/yolov8n-groupslim-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-groupslim-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-groupslim-exp1-fps.log 2>&1 & tail -f logs/yolov8n-groupslim-exp1-fps.log
```

```
python plot_channel_image.py --base-weights runs/prune/yolov8n-visdrone-lamp-exp3-prune/weights/model_c2f_v2.pt --prune-weights runs/prune/yolov8n-visdrone-lamp-exp3-prune/weights/prune.pt
```

### Model:yolov8n-Faster-GFPN-P2-EfficientHead.yaml Dataset:Visdrone

```
------------------ train base model ------------------
model = YOLO('yolov8n-Faster-GFPN-P2-EfficientHead.yaml')
model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_visdrone/data_exp.yaml',
            cache=True,
            imgsz=640,
            epochs=300,
            batch=12,
            close_mosaic=30,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n-visdrone',
            )

nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log
```

```
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}
CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-lamp-exp1-fps.log
```

```
------------------ group-taylor exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 12,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp1',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': True,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-grouptaylor-exp1.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-grouptaylor-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp1-fps.log

------------------ group-taylor exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 12,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '1',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouptaylor-exp2',
    
    # prune
    'prune_method':'group_taylor',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-grouptaylor-exp2.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-grouptaylor-exp2-test.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouptaylor-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouptaylor-exp2-fps.log 2>&1 & tail -f logs/yolov8n-grouptaylor-exp2-fps.log
```

```
------------------ group-hessian exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-visdrone/weights/best.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 12,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-visdrone-grouphessian-exp1',
    
    # prune
    'prune_method':'group_hessian',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=1 nohup python compress.py > logs/yolov8n-grouphessian-exp1.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-grouphessian-exp1-test.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-visdrone-grouphessian-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 100 --testtime 200 > logs/yolov8n-grouphessian-exp1-fps.log 2>&1 & tail -f logs/yolov8n-grouphessian-exp1-fps.log
```

### Model:yolov8-BIFPN-EfficientRepHead.yaml Dataset:Seaship
```
------------------ train base model ------------------
model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_seaship/data.yaml',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=32,
            close_mosaic=20,
            workers=8,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n',
            )

CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log
nohup python get_FPS.py --weights runs/train/yolov8n/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-fps.log 2>&1 & tail -f logs/yolov8n-fps.log

------------------ train base-light model ------------------
model = YOLO('yolov8n-BIFPN-EfficientRepHead.yaml')
# model.load('yolov8n.pt') # loading pretrain weights
model.train(data='/root/data_ssd/dataset_seaship/data.yaml',
            cache=True,
            imgsz=640,
            epochs=200,
            batch=32,
            close_mosaic=20,
            workers=8,
            device='1',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False # close amp
            # fraction=0.2,
            project='runs/train',
            name='yolov8n-light',
            )

nohup python train.py > logs/yolov8n-light.log 2>&1 & tail -f logs/yolov8n-light.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-light-test.log 2>&1 & tail -f logs/yolov8n-light-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-light-lamp-exp1-prune/weights/model_c2f_v2.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-light-fps.log 2>&1 & tail -f logs/yolov8n-light-fps.log
```

```
------------------ lamp exp1 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-light/weights/best.pt',
    'data':'/root/data_ssd/dataset_seaship/data.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-light-lamp-exp1',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.0,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-light-lamp-exp1.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-light-lamp-exp1-test.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp1-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-light-lamp-exp1-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-light-lamp-exp1-fps.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp1-fps.log

------------------ lamp exp2 ------------------
param_dict = {
    # origin
    'model': 'runs/train/yolov8n-light/weights/best.pt',
    'data':'/root/data_ssd/dataset_seaship/data.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/prune',
    'name':'yolov8n-light-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': False,
    'speed_up': 2.5,
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None
}

CUDA_VISIBLE_DEVICES=0 nohup python compress.py > logs/yolov8n-light-lamp-exp2.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-light-lamp-exp2-test.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp2-test.log
nohup python get_FPS.py --weights runs/prune/yolov8n-light-lamp-exp2-prune/weights/prune.pt --batch 32 --device 0 --warmup 200 --testtime 400 > logs/yolov8n-light-lamp-exp2-fps.log 2>&1 & tail -f logs/yolov8n-light-lamp-exp2-fps.log
```

# 使用教程
    剪枝操作问题，报错问题统一群里问，我群里回复谢谢~

## 视频

整体的流程说明和讲解(第一个必须要看的视频):
链接：https://pan.baidu.com/s/147wIPfk4-4_WXKOc6LZtLw?pwd=pman 
提取码：pman # BiliBili 魔鬼面具

yolov8-Faster-GFPN-P2-EfficientHead教程:
链接：https://pan.baidu.com/s/103ljiewi9nG3bDJupH8jdw?pwd=b939 
提取码：b939 # BiliBili 魔鬼面具

yolov8-BIFPN-EfficientRepHead教程:
链接：https://pan.baidu.com/s/1qJqvXN__5Ow0RakqYeKv4A?pwd=wedo 
提取码：wedo # BiliBili 魔鬼面具

EfficientHead中的PConv跳层教程:
链接：https://pan.baidu.com/s/1MLVEqLicH6keIp6GhYDECw?pwd=ejqn 
提取码：ejqn # BiliBili 魔鬼面具

yolov5v7的示例讲解:
1. yolov5n+C3-Faster+RepConv
链接：https://pan.baidu.com/s/11UVcQINQUlzUQzWjTpp6fw?pwd=sa15 
提取码：sa15 # BiliBili 魔鬼面具

2. yolov5n+RepViT+C2f
链接：https://pan.baidu.com/s/1TtcLwwer3ANcc4aFZDoUxw?pwd=bodk 
提取码：bodk # BiliBili 魔鬼面具

3. 原yolov7-tiny, yolov7-tiny+mobilenetv3+LSKBlock+TSCODE+RepConv, yolov7-tiny+Yolov7_Tiny_E_ELAN_DCN+AFPN
链接：https://pan.baidu.com/s/1n7Y7Ec93jeznJM6XGb20Yg?pwd=vjpd 
提取码：vjpd # BiliBili 魔鬼面具

4. yolov7-tiny+FasterNet+DBB
链接：https://pan.baidu.com/s/19HycIie3sa2lEj0HMF5jeA?pwd=s066 
提取码：s066 # BiliBili 魔鬼面具

5. yolov7-tiny+ReXNet+VoVGSCSP+DyHead+DecoupledHead
链接：https://pan.baidu.com/s/1ycaI3COdqS6eTvEYR3xNpQ?pwd=ctxf 
提取码：ctxf # BiliBili 魔鬼面具

## 我自己跑的实验数据
1. yolov8n.yaml
链接：https://pan.baidu.com/s/1T4XrW28Tj1O88TC00y5cRw?pwd=2ar0 
提取码：2ar0 # BiliBili 魔鬼面具
2. yolov8-Faster-GFPN-P2-EfficientHead.yaml
链接：https://pan.baidu.com/s/15V67npN4V6DX-ugFF9Ip2A?pwd=4r3k 
提取码：4r3k # BiliBili 魔鬼面具
3. yolov8-BIFPN-EfficientRepHead.yaml
链接：https://pan.baidu.com/s/18gRkJV9ZAC-gHJr0AzIFcg?pwd=79nc 
提取码：79nc # BiliBili 魔鬼面具
链接：https://pan.baidu.com/s/16FoDC2yiIOlwlTN5o87PpQ?pwd=5c05 
提取码：5c05 # BiliBili 魔鬼面具 Seaship数据集