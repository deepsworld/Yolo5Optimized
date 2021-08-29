# Yolo5 for Jetson Devices

This repo contains optimized yolo5 for jetson devices. 

- All the dependencies including ones to train and validate in a compact docker container.  
- On the fly TensorRT engine build with `FP32, FP16, INT8 with calibration`.
- `No need` for any custom plugins.
- Upto `4X inference` speed up...could be more with different devices and configs. 

## Usage

### Clone the repo from `github.com`:

```sh
git clone https://github.com/deepsworld/Yolo5Optimized
cd Yolo5Optimized
```

### Pull and run pre-built docker image

- The output rendered images will be stored in outputs dir. It will take ~5 mins to build fp16 engine on Jetson Nano

```sh
make run
cd yolo5
# run inference on images in assets 
python detect.py --fp16 --inputs assets --outputs outputs
```

### Build docker image

```sh
make build 
```