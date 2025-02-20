import cv2
import numpy as np

import torch
from torchvision import transforms as T


def roi_preprocess(h,w):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def det_preprocess(h,w):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform



def roi_postprocess(output, ori_shape, sigmoid_included=False, thresh=None, dilate=False, k_size=3, iter=1):
    output = torch.squeeze(output)
    if not sigmoid_included:
        output = output.sigmoid()
    if thresh:
        output = torch.where(output > thresh, 1.0, 0.0)
    output = 255 * output.detach().cpu().numpy().astype(np.uint8) 
    if dilate:
        kernel = np.ones((k_size, k_size), np.uint8)
        output = cv2.dilate(output, kernel, iterations = iter)

    output_fullres = cv2.resize(output, (ori_shape[1], ori_shape[0]))
    
    return output_fullres, output


def det_postprocess(output):
    return output[0]
    

unet_DC = dict(
    weights = "weights/DroneCrowd-uR18.pt",
    in_size = (192,320),
    postprocess_args = dict(
        thresh = 0.5,
        dilate = False,
        k_size = 3,
        iter = 1,
        sigmoid_included = True,
    ),
    transform = roi_preprocess,
    postprocess = roi_postprocess,
)

unet_SDS = dict(
    weights = "weights/SeaDronesSee-uR18.pt",
    in_size = (448,768),
    postprocess_args = dict(
        thresh = 0.5,
        dilate = True,
        k_size = 7,
        iter = 1,
        sigmoid_included = True,
    ),
    transform = roi_preprocess,
    postprocess = roi_postprocess,
)


yolov7t_SDS = dict(
    weights = "weights/SeaDronesSee-yolov7t.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_preprocess, 
    postprocess = det_postprocess,
    resize = True,
)

yolov7t_DC = dict(
    weights = "weights/DroneCrowd-yolov7t.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    transform = det_preprocess, 
    postprocess = det_postprocess,
    resize = False,
)

tracker = dict(
    module_name = 'SORT', 
    class_name = 'Sort',
    args = dict(
        max_age = 10,
        min_hits = 1,
        iou_threshold = 0.3,
        min_confidence = 0.3
    ),
)