import os
from uuid import uuid4

from PIL import Image
import torch
import tensorrt as trt

from ml import logging

logging.Logger(name='Calibrator').setLevel('INFO')

def preprocessor(size=256, crop=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    r"""ImageNet preprocessing through resize and center crop followed by normalization.
    """
    from collections.abc import Iterable
    from torchvision import transforms
    normalize = transforms.Normalize(mean, std)
    trans = transforms.Compose([transforms.Resize(size),
                                transforms.CenterCrop(crop),
                                transforms.ToTensor(),
                                normalize])

    if isinstance(crop, int):
        W, H = (crop, crop)
    elif isinstance(crop, Iterable):
        W, H = crop * 2 if len(crop) == 1 else crop

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

# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html
class Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator

    Calibration data is randomly generated based on given input shapes and batch size

    Parameters
    ----------
    batch_size: int
        Number of images to pass through in one batch during calibration
    input_shape: Tuple[int]
        Tuple of integers defining the shape of input to the model (Default: (3, 224, 224))
    cache: str
        Name of file to read/write calibration cache from/to.
    device: str or int 
        Device for calibration data (Default: 0 ==> cuda:0)
    max_calid_data: int
        Maximum caliration dataset size (Default: 512)
    """
    def __init__(self,
                 batch_size=32,
                 inputs=[],
                 cache=None,
                 calibration_files=[],
                 max_calib_data=512,
                 preprocess_func=None,
                 algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2,
                 device=torch.cuda.default_stream().device):
        super().__init__()

        self.inputs = inputs
        if not inputs:
            raise ValueError('Input shapes is required to generate calibration dataset')

        # unique cache file name in case mutliple engines are built in parallel
        self.cache = cache or f'{uuid4().hex}.cache'
        self.batch_size = batch_size
        self.max_calib_data = max_calib_data
        self.algorithm = algorithm
        self.files = calibration_files
        self.buffers = [torch.empty((batch_size, *input_shape), dtype=torch.float32, device=device)
                        for input_shape in inputs]

        # Pad the list so it is a multiple of batch_size
        if self.files and len(self.files) % self.batch_size != 0:
            logging.info("Padding number of calibration files to be a multiple of batch_size {:}".format(self.batch_size))
            self.files += calibration_files[(len(calibration_files) % self.batch_size):self.batch_size]

        if not preprocess_func:
            logging.warning('default preprocessing applied to convert input to RGB tensor followed by ImageNet resize, crop and normaliztion.')
        self.preprocess_func = preprocess_func or preprocessor()
        self.batches = self.load_batches()

    def get_algorithm(self):
        return self.algorithm

    def load_batches(self):
        if self.files:
            # Populates a persistent buffer with images.
            for index in range(0, len(self.files), self.batch_size):
                for offset in range(self.batch_size):
                    image_path = self.files[index + offset]
                    for i, input_shape in enumerate(self.inputs):
                        self.buffers[i][offset] = self.preprocess_func(image_path).contiguous()
                logging.info(f"preprocessed calibration images: {index + self.batch_size}/{len(self.files)}")
                yield
        else:
            for index in range(0, self.max_calib_data, self.batch_size):
                for offset in range(self.batch_size):
                    for i, input_shape in enumerate(self.inputs):
                        rand_batch = torch.rand((self.batch_size, *input_shape), dtype=torch.float32).contiguous()
                        self.buffers[i].copy_(rand_batch)
                logging.info(f"generated random calibration data batch: {index + self.batch_size}/{self.max_calib_data}")
                yield

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            next(self.batches)
            # Pass buffer pointer to tensorrt calibration algorithm
            return [int(buffer.data_ptr()) for buffer in self.buffers]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache):
            with open(self.cache, "rb") as f:
                logging.info("Using calibration cache to save time: {:}".format(self.cache))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache, "wb") as f:
            logging.info("Caching calibration data for future use: {:}".format(self.cache))
            f.write(cache)
