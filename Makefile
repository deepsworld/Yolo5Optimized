NS ?= deep1362
IMAGE_NAME ?= yolo5
VERSION ?= jetson
CONTAINER_NAME ?= yolo5

DOCKERFILE=docker/.
CURRENT_DIR=$(shell pwd)
DEST_DIR=/home/yolo5
TENSORRT_VOLUME=/usr/lib/python3.6/dist-packages/tensorrt:/usr/lib/python3.6/dist-packages/tensorrt

PHONY: max_power build push run

max_power:
	sudo nvpmodel -m 0  # 10 W

build: max_power
	docker build -t "$(NS)/$(IMAGE_NAME):$(VERSION)" $(DOCKERFILE)

push:
	docker push $(NS)/$(IMAGE_NAME):$(VERSION)

run: max_power
	docker run -it --rm --net=host \
	--ipc host --runtime nvidia \
	-v $(CURRENT_DIR):$(DEST_DIR) \
	-v $(TENSORRT_VOLUME) \
	$(NS)/$(IMAGE_NAME):$(VERSION)