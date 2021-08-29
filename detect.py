import os
import random
import argparse
from pathlib import Path

# third party
import cv2
import tqdm
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_jpeg, read_image

# custom
from yolo.coco import COCO80_CLASSES
from yolo.detector import yolo5s, yolo5l, yolo5m, yolo5x

MIN_SHAPE = (192, 192)
MAX_SHAPE = (800, 800)
DEFAULT_WORKSPACE_SIZE = 3 << 20
SUPPORTED_FORMATS = ['jpeg', 'jpg', 'png']

COLORS = lambda x: [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(x)]

def get_model(arch='yolo5s', dev=torch.device('cuda'), reload=False, pretrained=True, fp16=True, int8=False, strict=False, bs=2):
    model = eval(arch)(pretrained=pretrained, force_reload=reload).to(dev)
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
    return model

def detect(inputs, outputs, model, fp16=True):
    inputs = Path(inputs)
    assert inputs.exists(), f'Provided input path: {inputs} does not exist'
    outputs = Path(outputs)
    outputs.mkdir(parents=True, exist_ok=True)

    input_images = []
    for fmt in SUPPORTED_FORMATS:
        input_images += inputs.glob(f'*.{fmt}')

    for image_path in tqdm.tqdm(input_images):
        img = cv2.imread(str(image_path))
        with torch.cuda.amp.autocast(enabled=fp16):
            dets = model.detect([img])
        dets = [det.float() for det in dets]

        image = read_image(str(image_path))
        boxes = dets[0][:, :4]
        labels = ['{}[{:.2f}]'.format(COCO80_CLASSES[c.int().item()], s.item()) for s, c in dets[0][:, -2:]]
        colors = COLORS(len(labels))
        image = draw_bounding_boxes(image, boxes=boxes, labels=labels, colors=colors, font_size=15, width=2)
        filename = os.path.basename(str(image_path))
        out_path = os.path.join(str(outputs), filename)
        write_jpeg(image, out_path)

def parse_args():
    """
    The program will read all the [.jpeg, .jpg, .png] files from input path 
    and store the rendered images in output paths
    NOTE: Make sure to enable fp16 if the memory on the device is limited 
          (especially Jetson Nanos) or increase the DEFAULT_WORKSPACE_SIZE
    Run:
        python3 detect.py --fp16 --inputs assets --outputs outputs 
    """
    parser = argparse.ArgumentParser('Run detector on images')
    parser.add_argument('--arch', type=str, default='yolo5s', 
                        choices=['yolo5s', 'yolo5l', 'yolo5m', 'yolo5x'], help='model arch')
    parser.add_argument('--inputs', type=str, default='./assets', help='input dir')
    parser.add_argument('--outputs', type=str, default='./outputs', help='output dir')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--reload', action='store_true', help='reload model and engine build')
    parser.add_argument('--fp16', action='store_true', help='use FP16 inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.arch, dev, args.reload, fp16=args.fp16, bs=args.batch_size)
    detect(args.inputs, args.outputs, model, fp16=args.fp16)

if __name__ == '__main__':
    main()
