import io
import logging

import torch as th
from .utils import get_calibration_files

try:
    import torch2trt as t2t
    import tensorrt as trt
except ImportError as e:
    raise ImportError(e, f"torch2trt required: `pip install --install-option='--plugins' git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@b0cc8e77a0fbd61e96b971a66bbc11326f77c6b5`")

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

class TRTPredictor(t2t.TRTModule):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTPredictor, self).__init__(engine, input_names, output_names)

    def predict(self, *inputs, out=None, sync=False):
        with th.no_grad():
            output = self(*inputs, out=out)
            if sync:
                th.cuda.synchronize()
            return output

    def forward(self, *inputs, out=None):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            # XXX Conclude dynamic input shape
            idx = self.engine.get_binding_index(input_name)
            binding_shape = tuple(self.context.get_binding_shape(idx))
            arg_shape = tuple(inputs[i].shape)
            if binding_shape != arg_shape:
                # logging.info(f"Reallocate {input_name}.shape{binding_shape} -> {arg_shape}")
                self.context.set_binding_shape(idx, trt.Dims(arg_shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        if out is None:
            for i, output_name in enumerate(self.output_names):
                idx = self.engine.get_binding_index(output_name)
                dtype = t2t.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                shape = tuple(self.context.get_binding_shape(idx))
                #assert shape[0] == batch_size
                device = t2t.torch_device_from_trt(self.engine.get_location(idx))
                output = th.empty(size=shape, dtype=dtype, device=device)
                outputs[i] = output
                bindings[idx] = output.data_ptr()
        else:
            for i, output_name in enumerate(self.output_names):
                idx = self.engine.get_binding_index(output_name)
                outputs[i] = out[i]
                bindings[idx] = out[i].data_ptr()

        self.context.execute_async_v2(
            bindings=bindings, 
            stream_handle=th.cuda.current_stream().cuda_stream
        )

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

def torch2trt(module, 
              inputs, 
              input_names=None, 
              output_names=None, 
              max_batch_size=1,
              max_workspace_size=1<<20, 
              strict_type_constraints=False, 
              fp16_mode=False, 
              int8_mode=False, 
              int8_calib_batch_size=16,
              int8_calib_algorithm=t2t.DEFAULT_CALIBRATION_ALGORITHM,
              keep_network=True, 
              log_level=trt.Logger.INFO, 
              use_onnx=True,
              **kwargs):
    """Revise to support dynamic batch size through ONNX by default
    Args:
        module(nn.Module):
        inputs(List[Tensor]): list of tensors
    Kwargs:
        input_names:
        output_names:
        dynamic_axes(Dict[str, Dict[int, str]]): {input_key: {index: name}}
        min_shapes(Dict[int, Tuple(int, int, int)]): minimum shape for each dynamic input
        opt_shapes(Dict[int, Tuple(int, int, int)]): optimal shape for each dynamic input
        max_shapes(Dict[int, Tuple(int, int, int)]): maximum shape for each dynamic input

        max_batch_size(int): max batch size as the dynamic axis 0
        int8_calib_cache_file(str):
        int8_calib_data_path(str):
        int8_calib_max_data(int):
        int8_calib_batch_size(int): batch size to load for calibration
        int8_calib_preprocess_func(input):
        opset_version(int):
    """

    # copy inputs to avoid modifications to source data
    inputs_in = inputs
    inputs = [tensor.clone()[0:1] for tensor in inputs]  # only run single entry

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # run once to get num outputs
    outputs = module(*inputs)
    if not isinstance(outputs, tuple) and not isinstance(outputs, list):
        outputs = (outputs,)

    def reduce(value, outputs):
        nonlocal count
        if th.is_tensor(outputs):
            value += 1
        else:
            for output in outputs:
                value = reduce(value, output)
        return value
    if input_names is None:
        # list of tensors expected
        input_names = t2t.default_input_names(len(inputs))
    if output_names is None:
        # in case of nested tensors
        count = reduce(0, outputs)
        output_names = t2t.default_output_names(count)
        # logging.info(f"len(outputs)={len(outputs)}, count={count}")
    logging.info(f"input_names={input_names}")
    logging.info(f"output_names={output_names}")

    dynamic_axes = kwargs.pop('dynamic_axes', None)
    if dynamic_axes is None and max_batch_size > 1:
        dynamic_axes = {input_name: {0: 'batch_size'} for input_name in input_names}
    if use_onnx:
        f = io.BytesIO()
        th.onnx.export(module, inputs, f,
                       input_names=input_names,
                       output_names=output_names,
                       dynamic_axes=dynamic_axes,
                       opset_version=kwargs.pop('opset_version', 11))
        f.seek(0)
        onnx_bytes = f.read()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        parser.parse(onnx_bytes)
    else:
        # FIXME No dynamic batch size by default
        network = builder.create_network()
        with t2t.ConversionContext(network) as ctx:
            ctx.add_inputs(inputs, input_names)
            outputs = module(*inputs)
            if not isinstance(outputs, tuple) and not isinstance(outputs, list):
                outputs = (outputs,)
            ctx.mark_outputs(outputs, output_names)

    builder.max_batch_size = max_batch_size
    """ Removed in tensorrt > 8.0 """
    builder.max_workspace_size = max_workspace_size
    builder.fp16_mode = fp16_mode
    builder.int8_mode = int8_mode
    builder.strict_type_constraints = strict_type_constraints
    """"""""""""""""""""""""""""""""
    if dynamic_axes is None:
        for i in range(network.num_inputs):
            logging.info(f"network.get_input({i}).shape={network.get_input(i).shape}")
        for i in range(network.num_outputs):
            logging.info(f"network.get_output({i}).shape={network.get_output(i).shape}")
        engine = builder.build_cuda_engine(network)
    else:
        cfg = builder.create_builder_config()
        if fp16_mode:
            cfg.flags |= 1 << int(trt.BuilderFlag.FP16)
            if strict_type_constraints:
                cfg.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)

        if int8_mode:
            from .calibrator import Calibrator
            cfg.set_flag(trt.BuilderFlag.INT8)
            calib_cache = kwargs.pop('int8_calib_cache', None)            # cache of calibration dataset
            calib_data = kwargs.pop('int8_calib_data', None)              # path to calibration data
            calib_max = kwargs.pop('int8_calib_max', 128)                 # max amount of calibration data to use
            calib_preprocess_func = kwargs.pop('int8_calib_preprocess_func', None)  # preprocessing of calibration data
            calib_files = calib_data and get_calibration_files(calib_data, calib_max) or []

            # TODO: test calibrator with dynamic shapes other than batch size dimension
            cfg.int8_calibrator = Calibrator(
                batch_size=int8_calib_batch_size,
                inputs=[tuple(tensor.shape[1:]) for tensor in inputs],
                cache=calib_cache,
                calibration_files=calib_files,
                max_calib_data=calib_max,
                preprocess_func=calib_preprocess_func,
                algorithm=int8_calib_algorithm
            )

        # XXX: set max_workspace in config for dynamic input
        cfg.max_workspace_size = max_workspace_size
        min_shapes = kwargs.pop('min_shapes', None)
        max_shapes = kwargs.pop('max_shapes', None)
        opt_shapes = kwargs.pop('opt_shapes', None)
        logging.info(f"dynamic_axes={dynamic_axes}")

        profiles = {}
        for i in range(network.num_inputs):
            shape = network.get_input(i).shape
            dynamic = any([s < 1 for s in shape])
            if dynamic:
                logging.info(f"dynamic network.get_input({i}).shape={shape}")
                profile = builder.create_optimization_profile()
                min = min_shapes and (1, *min_shapes[i]) or (1, *shape[1:])
                max = max_shapes and (max_batch_size, *max_shapes[i]) or (max_batch_size, *shape[1:])
                opt = opt_shapes and (max_batch_size, *opt_shapes[i]) or max
                profile.set_shape(input_names[i], min=trt.Dims(min), opt=trt.Dims(opt), max=trt.Dims(max))
                idx = cfg.add_optimization_profile(profile)
                profiles[idx] = profile
                logging.info(f"set dynamic {input_names[i]}.shape to min={min}, opt={opt}, max={max} in profile[{idx}]")
            else: 
                logging.info(f"network.get_input({i}).shape={shape}")
        for i in range(network.num_outputs):
            shape = network.get_output(i).shape
            logging.info(f"network.get_output({i}).shape={shape}")
        logging.info(f"building TensorRT engine with fp16={fp16_mode}, int8={int8_mode}, strict={strict_type_constraints}")
        if int8_mode:
            assert network.num_inputs == 1, "Only one dynamic tensor input is supported for int8 calibration"
            cfg.set_calibration_profile(profiles[0])
        engine = builder.build_engine(network, cfg)
    module_trt = TRTPredictor(engine, input_names, output_names)
    if keep_network:
        module_trt.network = network
    return module_trt

def build(path):
    """Build an inference engine from a serialized model.
    Args:
        path: model or path to a saved onnx/trt checkpoint
    """
    logging.info("Deserializing the TensorRT engine from {}".format(path))
    with open(path, "rb") as f, trt.Logger() as logger, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
