# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python3 test.py --data coco.yaml --fp16
"""
import os
import sys
import yaml
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5Optimized/ to path

from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, colorstr
from utils.metrics import ap_per_class
from utils.torch_utils import time_synchronized

MIN_SHAPE = (192, 192)
MAX_SHAPE = (800, 800)
DEFAULT_WORKSPACE_SIZE = 3 << 20

def load_model(arch='yolo5s', dev=torch.device('cuda'), reload=False, pretrained=True):
    from yolo.detector import yolo5s, yolo5l, yolo5m, yolo5x
    model = eval(arch)(pretrained=pretrained, force_reload=reload).to(dev)
    return model

def load_engine(model, arch='yolo5s', reload=False, fp16=True, int8=False, strict=False, bs=2):
    # dynamic/static input
    minH, minW = MIN_SHAPE
    maxH, maxW = MAX_SHAPE
    min_shapes = [(3, minH, minW)]
    max_shapes = [(3, maxH, maxW)]
    spec = [[3, minH, minW]]
    dynamic_axes={'input_0': {0: 'batch_size'}}
    if maxH != minH:
        spec[0][1] = -1
        dynamic_axes['input_0'][2] = 'height'
    if maxW != minW:
        spec[0][2] = -1
        dynamic_axes['input_0'][3] = 'width'
    name = f"{arch}-bs{bs}_{spec[0][-2]}x{spec[0][-1]}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}{strict and '_strict' or ''}"
    model.deploy(
        name=name, 
        batch_size=bs, 
        dynamic_axes=dynamic_axes,
        min_shapes=min_shapes,
        max_shapes=max_shapes, 
        spec=spec, 
        fp16=fp16, 
        backend='trt', 
        reload=reload,
        workspace_size=DEFAULT_WORKSPACE_SIZE
    )
    return model.engine, model.head

@torch.no_grad()
def run(data,
        arch='yolo5s',  # model arch
        batch_size=2,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        verbose=False,  # verbose output
        fp16=True,  # use FP16 half-precision inference
        dataloader=None,
        save_dir=Path(''),
        plots=True
        ):

    training = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check

    # Load model
    model = load_model(arch=arch, dev=device)  # load FP32 model
    gs = max(int(model.module.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check image size
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    names = {k: v for k, v in enumerate(model.module.names if hasattr(model.module, 'names') else model.module.names)}

    engine, head = load_engine(model, arch=arch, bs=batch_size, fp16=fp16)
    del model
    
    # Configure
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    task = 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task], imgsz, 2, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '), workers=0)[0]

    seen = 0
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t_ = time_synchronized()
        img = img.to(device)
        img = img.float()  # uint8 to 32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_synchronized()
        t0 += t - t_

        # Run engine
        outputs = engine.predict(img)
        x = outputs
        
        z = []
        # logging.info(f"x={[tuple(xi.shape) for xi in x]}, feats={[tuple(feats.shape) for feats in features]}")
        for i in range(head.nl):
            bs, na, ny, nx, no = x[i].shape
            if head.grid[i].shape[2:4] != x[i].shape[2:4]:
                head.grid[i] = head._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            # logging.info(f"y={tuple(y.shape)}, head.grid[{i}]={head.grid[i].shape}, head.stride[{i}]={head.stride[i].shape}")
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + head.grid[i].to(x[i].device)) * head.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * head.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, head.no))
        out = torch.cat(z, 1)

        t1 += time_synchronized() - t

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t2 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Return results
    if not training:
        s = ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

def parse_opt():
    """
    Usage:
        python3 test.py --data coco.yaml --fp16 
    """
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--arch', type=str, default='yolo5s', help='model arch')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='run task')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--fp16', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    return opt

def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    if opt.task in ('val'):
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
