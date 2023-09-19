# YOLOv8 🚀 by Ultralytics, AGPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov8

Usage:
    import torch
    model = torch.hub.load('sblmh/ultralytics', 'yolov8s')  # official model
    model = torch.hub.load('sblmh/ultralytics', 'custom', 'yolov8s.pt')  # custom/local model
"""

import torch

def _create(name, pretrained=True, channels=3, classes=80, autoshace=True, verbose=True, device=None):
    from ultralytics import YOLO
    model = YOLO(name)
    return model

def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='model name')
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [
        'data/images/zidane.jpg',  # filename
        Path('data/images/zidane.jpg'),  # Path
        'https://ultralytics.com/images/zidane.jpg',  # URI
        cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
        Image.open('data/images/bus.jpg'),  # PIL
        np.zeros((320, 640, 3))]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()
