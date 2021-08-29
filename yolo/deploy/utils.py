import os
import glob
import errno
import random
from pathlib import Path

import torch as th
import logging

def get_calibration_files(calibration_data, max_calibration_size=None, allowed_extensions=(".jpeg", ".jpg", ".png")):
    """Returns a list of all filenames ending with `allowed_extensions` found in the `calibration_data` directory.

    Parameters
    ----------
    calibration_data: str
        Path to directory containing desired files.
    max_calibration_size: int
        Max number of files to use for calibration. If calibration_data contains more than this number,
        a random sample of size max_calibration_size will be returned instead. If None, all samples will be used.

    Returns
    -------
    calibration_files: List[str]
         List of filenames contained in the `calibration_data` directory ending with `allowed_extensions`.
    """

    logging.info("Collecting calibration files from: {:}".format(calibration_data))
    calibration_files = [path for path in glob.iglob(os.path.join(calibration_data, "**"), recursive=True)
                         if os.path.isfile(path) and path.lower().endswith(allowed_extensions)]
    logging.info("Number of Calibration Files found: {:}".format(len(calibration_files)))

    if len(calibration_files) == 0:
        raise Exception("ERROR: Calibration data path [{:}] contains no files!".format(calibration_data))

    if max_calibration_size:
        if len(calibration_files) > max_calibration_size:
            logging.warning("Capping number of calibration images to max_calibration_size: {:}".format(max_calibration_size))
            random.seed(42)  # Set seed for reproducibility
            calibration_files = random.sample(calibration_files, max_calibration_size)

    return calibration_files

def get_input_output_names(engine):
    from six import string_types
    input_names = []
    output_names = []
    for idx_or_name in range(engine.num_bindings):
        # name or index
        if isinstance(idx_or_name, string_types):
            name = idx_or_name
            index  = engine.get_binding_index(name)
            if index == -1:
                raise IndexError("Binding name not found: %s" % name)
        else:
            index = idx_or_name
            name  = engine.get_binding_name(index)
            if name is None:
                raise IndexError("Binding index out of range: %i" % index)

        if engine.binding_is_input(index):
            input_names.append(name)
        else:
            output_names.append(name)

    return input_names, output_names

def export(model, spec, path, **kwargs):
    """
    Args:
        spec(List[Tuple[int,...]]): list of input shapes to export the model in ONNX
    Kwargs:
        export_params=True
        training=None
        input_names=None
        output_names=None
        dynamic_axes=None
        fixed_batch_size=False
        onnx_shape_inference=False
        verbose=False
    """
    path = Path(path)
    if path.suffix == '.onnx':
        from torch import onnx
        logging.info(f"Exporting pytorch model to {path}")
        device = next(model.parameters()).device
        args = tuple([th.rand(1, *shape, device=device) for shape in spec])
        input_names = [f'input_{i}' for i in range(len(spec))]
        dynamic_axes = {input_name: {0: 'batch_size'} for input_name in input_names}
        onnx.export(model, args, str(path), 
                    input_names=input_names, 
                    dynamic_axes=dynamic_axes,
                    opset_version=kwargs.pop('opset_version', 11),
                    **kwargs)
    else:
        raise ValueError(f"Unknown suffix `{path.suffix}` to export")

def build(name, model, spec, model_dir=None, backend='trt', reload=False, **kwargs):
    r"""
    Args:
        name(str): checkpoint name to save and load
        model(nn.Module): pytorch model on CPU
        spec(Tuple[B, C, H, W]): input shape including dynamic axies as -1
    Kwargs:
        model_dir(str, Path): path to save and load model checkpoint
        backend(str): deployment backend
        reload(bool): whether to force deploying the model with backend

        input_names(List[str]): list of names of input tensor args
        output_names([List[str]): list of names of output tensors
        batch_size(int): max batch size as the dynamic axis 0
        workspace_size(int):
        fp16(bool):
        int8(bool):
        strict_type_constraints(bool): type mode strictly forced or not
        int8_calib_batch_size(int):
        int8_calib_preprocess_func(Callable):
        min_shapes(Tuple):
        opt_shapes(Tuple):
        max_shapes(Tuple)
    """
    from torch import hub
    if model_dir is None:
        hub_dir = hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise e

    from time import time
    t = time()
    if backend in ['trt', 'tensorrt']:
        # XXX No intermmediate ONNX archive
        from . import trt as backend
        chkpt_path = Path(f"{model_dir}/{name}.pth")
        if chkpt_path.exists() and not reload:
            logging.info(f"Loading torch2trt checkpoint from {chkpt_path}")
            chkpt = th.load(chkpt_path)
            predictor = backend.TRTPredictor()
            predictor.load_state_dict(chkpt)
            return predictor

        trt = Path(f"{model_dir}/{name}.trt")
        input_names = kwargs.pop('input_names', None)
        output_names = kwargs.pop('output_names', None)
        
        if trt.exists() and not reload:
            # Load from previous saved deployment engine
            logging.info(f"Building TensorRT inference engine from {trt}")
            engine = backend.build(trt)
            if not (input_names and output_names):
                input_names, output_names = get_input_output_names(engine)
            predictor = backend.TRTPredictor(engine=engine, 
                                             input_names=input_names,
                                             output_names=output_names)
        else:
            batch_size = kwargs.pop('batch_size', 1)
            workspace_size = kwargs.pop('workspace_size', 3 << 20)
            fp16 = kwargs.pop('amp', False)
            fp16 = fp16 or kwargs.pop('fp16', False)
            int8 = kwargs.pop('int8', False)
            strict_type_constraints = kwargs.pop('strict_type_constraints', False)  # amp implied
            int8_calib_batch_size = kwargs.pop('int8_calib_batch_size', 4)
            device = next(model.parameters()).device
            min_shapes = kwargs.get('min_shapes')
            if min_shapes is not None:
                inputs = tuple([th.rand(1, *shape, device=device) for shape in min_shapes])
            else:
                inputs = tuple([th.rand(1, *shape, device=device) for shape in spec])
            predictor = backend.torch2trt(model,
                                          inputs,
                                          max_batch_size=batch_size,
                                          max_workspace_size=workspace_size,
                                          input_names=input_names,
                                          output_names=output_names,
                                          fp16_mode=fp16,
                                          int8_mode=int8,
                                          int8_calib_batch_size=int8_calib_batch_size,
                                          strict_type_constraints=strict_type_constraints,
                                          use_onnx=True,
                                          **kwargs)
        logging.info(f"Saving TensorRT checkpoint to {chkpt_path}")
        th.save(predictor.state_dict(), chkpt_path)
        logging.info(f"Built TensorRT inference engine for {time() - t:.3f}s")
        return predictor                
    else:
        raise ValueError(f"Unsupported backend: {backend}")
