import os
import logging
from pathlib import Path

COCO80_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

BASE_DOWNLOAD_URL = lambda split: f'http://images.cocodataset.org/zips/{split}.zip'

def download(split='val2017', reload=False):
    """
    Downloads COCO split
    Returns:
        Path to downloaded split directory
    """
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile
    from torch import hub

    download_url = BASE_DOWNLOAD_URL(split)
    download_dir = Path(os.path.join(hub.get_dir(), 'COCO'))
    download_dir.mkdir(exist_ok=True, parents=True)
    split_dir = Path(os.path.join(download_dir, f'{split}'))
 
    if split_dir.exists() and any(split_dir.iterdir()) and not reload:
        # already exists
        logging.info(f'Skipping download of COCO: {split} as it already exists')
    else:
        # download 
        with urlopen(download_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(download_dir)
        logging.info(f'Downloaded COCO: {split}')
        
    return str(split_dir)