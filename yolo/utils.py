import logging

import cv2 as cv
import numpy as np

import torch
from torchvision.ops import clip_boxes_to_image, nms, box_iou

def xcycwh2xyxy(x, inplace=False):
    """Convert Nx4 boxes from center, width and height to top-left and bottom-right coordinates.
    Args:
        x(Tensor[N, 4]): N boxes in the format [xc, yc, w, h]
        inplace(bool): whether to modify the input
    Returns:
        boxes(Tensor[N, 4]): N boxes in the format of [x1, y1, x2, y2]
    """
    # Boolean mask causes a new tensor unless assignment inplace
    y = torch.zeros_like(x)
    even = (x[:, 2] % 2) == 0
    odd =  (x[:, 2] % 2) == 1
    y[:, 0][even] = x[:, 0][even] - x[:, 2][even].div(2, rounding_mode='floor') + 1
    y[:, 0][odd] = x[:, 0][odd] - x[:, 2][odd].div(2, rounding_mode='floor')
    even = (x[:, 3] % 2) == 0
    odd =  (x[:, 3] % 2) == 1
    y[:, 1][even] = x[:, 1][even] - x[:, 3][even].div(2, rounding_mode='floor') + 1
    y[:, 1][odd] = x[:, 1][odd] - x[:, 3][odd].div(2, rounding_mode='floor')
    y[:, 2] = x[:, 0] + x[:, 2].div(2, rounding_mode='floor')
    y[:, 3] = x[:, 1] + x[:, 3].div(2, rounding_mode='floor')
    if inplace:
        x.copy_(y)
        return x
    return y

def resize(img, size, constraint='shorter', interpolation=cv.INTER_LINEAR, **kwargs):
    '''Resize input image of PIL/accimage, OpenCV BGR or torch tensor.
    Args:
        size(Tuple[int], int): tuple of height and width or length on both sides following torchvision resize semantics
        constraint(str): resize by the shorter (ImageNet) or longer edge (YOLO)
    '''

    W, H = img.shape[-3:-1][::-1]
    if isinstance(size, int):
        # with aspect ratio preserved
        if constraint == 'shorter':
            # by the shorter edge
            if H < W:
                h, w = size, int(W / H * size)
            else:
                h, w = int(H / W * size), size
        else:
            # by the longer edge
            if H < W:
                h, w = int(H / W * size), size
            else:
                h, w = size, int(W / H * size)
    else:
        # exact as is
        h, w = size
    return cv.resize(img, (w, h), interpolation=interpolation)

def letterbox(img, size=640, color=114, auto=True, stretch=False, upscaling=True):
    """Resize and pad to the new shape.
    Args:
        img(BGR): CV2 BGR image
        size[416 | 512 | 608 | 32*]: target long side to resize to in multiples of 32
        color(tuple): Padding color
        auto(bool): Padding up to the short side or not
        stretch(bool): Scale the short side without keeping the aspect ratio
        upscaling(bool): Allows to scale up or not
    """
    # Resize image to a multiple of 32 pixels on both sides 
    # https://github.com/ultralytics/yolov3/issues/232
    color = isinstance(color, int) and (color,) * img.shape[-1] or color
    shape = img.shape[:2]
    if isinstance(size, int):
        size = (size, size)

    r = min(size[0] / shape[0], size[1] / shape[1])
    if not upscaling: 
        # Only scale down but no scaling up for better test mAP
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    pw = int(round(shape[1] * r))
    ph = int(round(shape[0] * r))
    new_unpad = pw, ph  # actual size to scale to (w, h)
    dw, dh = size[1] - new_unpad[0], size[0] - new_unpad[1]         # padding on sides

    if auto: 
        dw, dh = dw % 32, dh % 32
    elif stretch:  
        # Stretch the short side to the exact target size
        dw, dh = 0.0, 0.0
        new_unpad = size
        ratio = size[0] / shape[0], size[1] / shape[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = resize(img, (new_unpad[::-1]))

    # Fractional to integral padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    resized = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return resized, dict(
        shape=shape,        # HxW
        offset=(top, left), # H, W
        ratio=ratio,        # H, W
    )

def preprocess(image, size=640):
    """Sequential preprocessing of input images for YOLO
    Args:
        image(str | list[str] | ndarray | list[ndarray] | ): 
            image filename(s) | CV BGR image(s)
    Returns:
        images(Tensor[BCHW]):
    """
    if isinstance(image, (str, np.ndarray)):
        images = [image]
    else:
        images = image

    if isinstance(images[0], str):
        images = [cv.imread(image) for image in images]

    # minimal only when all shapes are the same as in a batch
    shapes = [img.shape for img in images]
    minimal = all(map(lambda s: s == images[0].shape, shapes))

    resized = []
    metas = []

    # resize w/ optional padding to a mulitple of 32
    for img in images:
        img, meta = letterbox(img, size=size, auto=minimal)
        resized.append(torch.from_numpy(img).flip(-1).permute(2, 0, 1))
        metas.append(meta)
    
    resized = torch.stack(resized).to(dtype=torch.get_default_dtype()).div(255)
    return resized, metas

def batched_nms(predictions, 
                conf_thres=0.3, iou_thres=0.6, 
                agnostic=False, merge=True, 
                multi_label=False, classes=None):
    """Perform NMS on inference results
    Args:
        prediction(B, AG, 4+1+80): center, width and height refinement, plus anchor and class scores 
                                   per anchor and grid combination
        conf_thres(float): anchor confidence threshold
        iou_thres(float): NMS IoU threshold
        agnostic(bool): class agnostic NMS or per class NMS
        merge(bool): weighted merge by IoU for best mAP
        multi_label(bool): whether to select mulitple class labels above the threshold or just the max
        classes(list | tuple): class ids of interest to retain
    Returns:
        output(List[Tensor[B, N, 6]]): list of detections per image in (x1, y1, x2, y2, conf, cls)
    """
    min_wh, max_wh = 2, 4096                                        # minimum and maximum box width and height
    B, _, nc = predictions.shape
    nc -= 5
    multi_label &= nc > 1                                           # multiple labels per box if nc > 1 too
    output = [None] * B
    
    for b, x in enumerate(predictions):                             # image index and inference
        x = x[x[:, 4] > conf_thres]                                 # Threshold anchors by confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)] # width/height constraints
        if x.numel() == 0:
            continue

        # Compute resulting class scores = anchor * class
        x[..., 5:] *= x[..., 4:5]

        # xcycwh to xyxy
        boxes = xcycwh2xyxy(x[:, :4].round())

        # Single or multi-label boxes
        if multi_label:
            # Combinations of repeated boxes with different classes: [x1, y1, x2, y2, conf, class]
            keep, cls = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((boxes[keep], x[keep, 5 + cls].unsqueeze(1), cls.float().unsqueeze(1)), 1)
        else:  
            # Best class only: [x1, y1, x2, y2, conf, class]
            conf, cls = x[:, 5:].max(1)
            x = torch.cat((boxes, conf.unsqueeze(1), cls.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter out boxes not in any specified classses
        if classes:
            x = x[(cls.view(-1, 1) == torch.tensor(classes, device=cls.device)).any(1)]

        if x.numel() == 0:
            continue

        # Batched NMS
        scores = x[:, 4]
        boxes = x[:, :4]
        cls = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes = boxes + cls.view(-1, 1) * boxes.max()  # boxes (offset by class), scores
        keep = nms(boxes, scores, iou_thres)
        if merge and (1 < x.shape[0] < 3e3):
            # Weighted NMS box merge by IoU * scoress
            try:
                iou = box_iou(boxes[keep], boxes) > iou_thres   # Filtered IoU
                weights = iou * scores[None]                    # weighted IoU by class scores
                x[keep, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            except Exception as e:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                logging.error(f"Failed to merge NMS boxes by weighted IoU: {e}")
                print(x, x.shape, keep, keep.shape)
        output[b] = x[keep].to(predictions.dtype)

    return output

def postprocess(predictions, metas, 
                conf_thres=0.3, iou_thres=0.6, 
                agnostic=False, merge=True, 
                multi_label=False, classes=None):
    """Post-process to restore predictions on pre-processed images back.
    Args:
        predictions(Tensor[B,K,4+1+80]): batch output predictions from YOLO
    Returns:
        dets(xyxysc):
    """
    dets = [None] * len(predictions)
    predictions = batched_nms(predictions, conf_thres, iou_thres, agnostic, merge, multi_label, classes)
    for b, (pred, meta) in enumerate(zip(predictions, metas)):
        if pred is None or len(pred) == 0:
            continue
        # Shift back
        top, left = meta['offset']
        pred[:, [0, 2]] -= left
        pred[:, [1, 3]] -= top
        # Scale back
        rH, rW = meta['ratio']
        pred[:, [0, 2]] /= rW
        pred[:, [1, 3]] /= rH
        # Clip boxes
        pred[:, :4] = clip_boxes_to_image(pred[:, :4].round(), meta['shape'])
        dets[b] = pred
    return dets