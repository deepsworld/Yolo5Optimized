import os
import logging
from pathlib import Path

import torch as th
from torch import nn
from torch import hub

from .coco import download, COCO80_CLASSES
from .utils import preprocess, postprocess

def yolo5s(pretrained=False, force_reload=False, **kwargs):
    from .model import yolo5s
    m = yolo5s(pretrained, force_reload=force_reload, **kwargs)
    return YOLODetector(m)

def yolo5l(pretrained=False, force_reload=False, **kwargs):
    from .model import yolo5l
    m = yolo5l(pretrained, force_reload=force_reload, **kwargs)
    return YOLODetector(m)

def yolo5m(pretrained=False, force_reload=False, **kwargs):
    from .model import yolo5m
    m = yolo5m(pretrained, force_reload=force_reload, **kwargs)
    return YOLODetector(m)

def yolo5x(pretrained=False, force_reload=False, **kwargs):
    from .model import yolo5x
    m = yolo5x(pretrained, force_reload=force_reload, **kwargs)
    return YOLODetector(m)

class YOLODetector(nn.Module):
    def __init__(self, model, classes=COCO80_CLASSES):
        super().__init__()
        self.module = model
        self.classes = classes
        self.engine = None

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        return outputs

    def deploy(self, name='yolo5s', batch_size=2, spec=[(3, 384, 640)], fp16=True, backend='trt', reload=False, **kwargs):
        r"""Deploy optimized runtime backend.
        Args:
            batch_size(int): max batch size
            spec(Tuple[int]): preprocessed frame shape which must be fixed through the batch
            amp(bool): mixed precision with FP16
            kwargs:
                dynamix_axes: dynamic axes for each input ==> {'input_0': {0: 'batch_size', 2: 'height'}}
                min_shapes: min input shapes ==> [(3, 320, 674)]
                max_shapes: max input shapes ==> [(3, 674, 674)]
        """
        from .deploy.utils import build
        module = self.module
        module.model[-1].export = True
        self.head = module.model[-1]
        int8 = kwargs.get('int8', False)
        strict = kwargs.get('strict', False)
        if int8:
            def preprocessor(size=(384, 640)):
                from PIL import Image
                from torchvision import transforms
                trans = transforms.Compose([transforms.Resize(size),
                                            transforms.ToTensor()])

                H, W = size
                def preprocess(image_path, *shape):
                    r'''Preprocessing for TensorRT calibration
                    Args:
                        image_path(str): path to image
                        channels(int):
                    '''
                    image = Image.open(image_path)
                    logging.debug(f"image.size={image.size}, mode={image.mode}")
                    image = image.convert('RGB')
                    C = len(image.mode)
                    im = trans(image)
                    assert im.shape == (C, H, W)
                    return im

                return preprocess

            int8_calib_max = kwargs.get('int8_calib_max', 5000)
            int8_calib_batch_size = kwargs.get('int8_calib_batch_size', max(batch_size, 64)) 
            cache = f'{name}-COCO2017-val-{int8_calib_max}-{int8_calib_batch_size}.cache'
            cache_path = Path(os.path.join(hub.get_dir(), cache))
            kwargs['int8_calib_cache'] = str(cache_path)
            kwargs['int8_calib_data'] = download(split='val2017', reload=False)
            kwargs['int8_calib_preprocess_func'] = preprocessor()
            kwargs['int8_calib_max'] = int8_calib_max
            kwargs['int8_calib_batch_size'] = int8_calib_batch_size
          
        self.engine = build(f"{name}-bs{batch_size}_{spec[0][-2]}x{spec[0][-1]}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}{strict and '_strict' or ''}",
                                   self,
                                   spec,
                                   backend=backend, 
                                   reload=reload,
                                   batch_size=batch_size,
                                   fp16=fp16,
                                   strict_type_constraints=strict,
                                   **kwargs)
        del self.module

    def detect(self, images, **kwargs):
        """Perform object detection.
        Args:
            images(str | List[str] | ndarray[HWC] | List[ndarray[HWC]]): filename, list of filenames or an image batch
        Returns:
            detection(List[Tensor[N, 6]]): list of object detection tensors in [x1, y1, x2, y2, score, class] per image
        """
        self.eval()
        param = next(self.parameters())

        size = kwargs.get('size', 640)
        cfg = dict(
            conf_thres = kwargs.get('cls_thres', 0.4),
            iou_thres = kwargs.get('nms_thres', 0.5),
            agnostic = kwargs.get('agnostic', False),
            merge = kwargs.get('merge', True),
        )
        batch, metas = preprocess(images, size=size)
        with th.no_grad():
            if self.engine is None:
                outputs = self(batch.to(param.device))
                predictions = outputs[0]
            else:
                # XXX Remaining inference skipped in Detect for ONNX export
                outputs = self.engine.predict(batch.to(param.device))
                x = outputs
                z = []
                head = self.head
                for i in range(head.nl):
                    bs, na, ny, nx, no = x[i].shape
                    if head.grid[i].shape[2:4] != x[i].shape[2:4]:
                        head.grid[i] = head._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + head.grid[i].to(x[i].device)) * head.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * head.anchor_grid[i]  # wh
                    z.append(y.view(bs, -1, head.no))
                predictions = th.cat(z, 1)
        dets = postprocess(predictions, metas, **cfg)
        dtype = th.float32
        for dets_f in dets:
            if dets_f is not None:
                dtype = dets_f.dtype
                break

        dets = list(map(lambda det: th.empty(0, 6, dtype=dtype, device=param.device) if det is None else det, dets))

        return dets