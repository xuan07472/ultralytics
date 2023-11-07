import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='data.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default='', help='data yaml path')
    
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    cocoEval = COCOeval(anno, pred, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    precisions = cocoEval.eval['precision'] # TP/(TP+FP) right/detection
    recalls = cocoEval.eval['recall'] # iou*class_num*Areas*Max_det TP/(TP+FN) right/gt
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[0],np.mean(precisions[0, :, :, 0, -1]),np.mean(recalls[0, :, 0, -1])))
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[1],np.mean(precisions[1, :, :, 0, -1]),np.mean(recalls[1, :, 0, -1])))
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[2],np.mean(precisions[2, :, :, 0, -1]),np.mean(recalls[2, :, 0, -1])))
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[3],np.mean(precisions[3, :, :, 0, -1]),np.mean(recalls[3, :, 0, -1])))
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[4],np.mean(precisions[4, :, :, 0, -1]),np.mean(recalls[4, :, 0, -1])))
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[5],np.mean(precisions[5, :, :, 0, -1]),np.mean(recalls[5, :, 0, -1])))
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[6],np.mean(precisions[6, :, :, 0, -1]),np.mean(recalls[6, :, 0, -1])))
    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[7],np.mean(precisions[7, :, :, 0, -1]),np.mean(recalls[7, :, 0, -1])))