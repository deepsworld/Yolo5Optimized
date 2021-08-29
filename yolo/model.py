import torch
from torch import nn
from torch.nn import functional as F

# FIXME: workaround for torch 1.9 bug: https://github.com/pytorch/vision/issues/4156
#torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX

def yolo5(chkpt, pretrained=False, force_reload=False, tag='v5.0', autoshape=False, channels=3, **kwargs):
    m = torch.hub.load(f'ultralytics/yolov5:{tag}', chkpt, pretrained=pretrained, force_reload=force_reload, autoshape=autoshape, channels=channels, **kwargs)
    for module in m.modules():
        # XXX export friendly for ONNX/TRT
        if isinstance(getattr(module, 'act', None), nn.Hardswish):
            module.act = Hardswish()
    return m

def yolo5s(pretrained=False, force_reload=False, **kwargs):
    return yolo5('yolov5s', pretrained, force_reload, **kwargs)

def yolo5m(pretrained=False, force_reload=False, **kwargs):
    return yolo5('yolov5m', pretrained, force_reload, **kwargs)

def yolo5l(pretrained=False, force_reload=False, **kwargs):
    return yolo5('yolov5l', pretrained, force_reload, **kwargs)

def yolo5x(pretrained=False, force_reload=False, **kwargs):
    return yolo5('yolov5x', pretrained, force_reload, **kwargs)