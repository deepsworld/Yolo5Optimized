# Yolo5 for Jetson Devices

<a href="https://hub.docker.com/r/deep1362/yolo5"><img src="https://img.shields.io/docker/pulls/deep1362/yolo5?logo=docker" alt="Docker Pulls"></a>

This repo contains optimized yolo5 for jetson devices. 

- All the dependencies including ones to train and validate in a compact docker container.  
- On the fly TensorRT engine build with `FP32, FP16, INT8 with calibration`.
- `No need` for any custom plugins.
- At least `4X inference` speed up...could be more with different devices and configs. 
- `Dynamic shape` input is also supported

## Usage

### Clone the repo from `github.com`:

```sh
git clone https://github.com/deepsworld/Yolo5Optimized
cd Yolo5Optimized
```

### Pull and run pre-built docker image

- `DETECT`: run sample detection on images in a directory. The output rendered images will be stored in outputs dir.
```sh
make run
cd yolo5
# run inference on images in assets 
python3 detect.py --fp16 --inputs assets --outputs outputs
```

- `EVALUATE`: run evaluation on COCO validation set (2017)
```sh
make run
cd yolo5
# start evaluation script
python3 test.py --data coco.yaml --fp16
```

### Build docker image

```sh
make build 
```

## Evaluation on COCO
- Device: Jetson Nano 4GB
- Arch: yolo5s
- Batch_Size: 2
- Scale: 640
- FP16: True

Class   |    Images   |   Labels   |   P     |   R     |   mAP@.5   |   mAP@.5:.95 
|-------|-------|-------|-------|-------|-------|-------|
all     |    5000     |  36335     |   66.6 |   49.8 |   54.9    |   35.3

`Speed:` `1.3/52.4/53.7` ms inference/NMS/total per `640x640` image at `batch-size 2`

## Notes
- It will take ~5-6 mins to build fp16 engine on Jetson Nano
- Consider disabling GUI to free up some extra memory
- Run in 10W power mode with `sudo nvpmodel -m 0` or `make max_power`
- The validation script can be used to validate over some other test set with minor changes if needed.
- If using a Jetson Nano with 2GB memory, set `DEFAULT_WORKSPACE_SIZE = 2 << 20 or 1 << 20`. 

## Common Issues and Resolutions
- Docker build fails with `make build`: This could happen if the default docker runtime is not set to `nvidia`. To fix this do as follow:
```sh
# edit docker daemon json
sudo vi /etc/docker/daemon.json
```
```py
# Make sure it looks like this
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
```sh
# restart docker daemon
sudo systemctl restart docker
```
- Disable GUI to free up some extra memory
```sh
sudo systemctl set-default multi-user
sudo reboot now
```

## References
- https://github.com/necla-ml/ML-Vision
- https://github.com/necla-ml/ML

## Contributing
- Please raise an issue if you find any bugs or problems in running the code. 
