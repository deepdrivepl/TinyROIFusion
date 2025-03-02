import os

import cv2
import numpy as np

import torch
from torchvision import transforms as T
from utils.bboxes import letterbox


class SingleDetectionDataset(torch.utils.data.Dataset):
    
    def __init__(self, paths, dataset, det_inf_size, det_transform):
        self.paths = [x for x in paths if os.path.isfile(x)]
        self.dataset = dataset
        self.size = det_inf_size
        self.det_transform = det_transform(self.size[0], self.size[1])
        
        
    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = np.ascontiguousarray(img[:, :, ::-1])
        h,w = img.shape[:2]
        
        img = self.det_transform(img)
        metadata = {
            "image_path": os.path.abspath(self.paths[idx]), 
            "image_idx": idx,
            "coco": self.dataset.get_image_metadata(self.paths[idx])
        }
        return img, metadata

        
        

class ROIDataset(torch.utils.data.Dataset):
    def __init__(self, paths, dataset, roi_inf_size, roi_transform):
        self.paths = [x for x in paths if os.path.isfile(x)]
        self.roi_transform = roi_transform(roi_inf_size[0], roi_inf_size[1])
        self.dataset = dataset



    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W = image.shape[:2]
        
        image = self.roi_transform(image)

        metadata = {
            'image_path': os.path.abspath(self.paths[idx]),
            'coco': self.dataset.get_image_metadata(self.paths[idx])
        }            
    
        return image, metadata
    

    
class WindowDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset, bboxes, size):
        self.path = os.path.abspath(path)
        self.image = cv2.imread(path)[:,:,::-1]
        self.image = np.ascontiguousarray(self.image)
        self.bboxes = torch.tensor(bboxes)
        self.size = size
        self.transform = T.ToTensor()
        self.h, self.w = self.image.shape[:2]
        self.dataset = dataset


    def __len__(self):
        return len(self.bboxes)


    def __getitem__(self, idx):

        # handle cropping
        xmin, ymin, xmax, ymax = map(int, self.bboxes[idx])
        roi_h, roi_w = ymax-ymin, xmax-xmin # vertical or horizontal
        crop_image = self.image[ymin:ymax,xmin:xmax,:]

        # rotate if needed
        if (roi_h > roi_w and self.size[0] > self.size[1]) or (roi_h <= roi_w and self.size[0] <= self.size[1]):
            rotate = False
        else:
            rotate = True

        if rotate: # same orientation as the detection window
            crop_image = cv2.rotate(crop_image, cv2.ROTATE_90_CLOCKWISE)

        # resize if needed
        det_h, det_w = self.size
        crop_h, crop_w = crop_image.shape[:2]
        resize = False
        unpadded = [0.0,0.0]
        if crop_h > det_h or crop_w > det_w:
            resize = True
            crop_image, (unpadded) = letterbox(crop_image, (det_h, det_w), auto=False)
        paste_h, paste_w = crop_image.shape[:2]

        img_in = np.full((*self.size, 3), (114, 114, 114)).astype(np.uint8)
        img_in[:paste_h,:paste_w,...] = crop_image

        
        metadata = {
            'bbox': self.bboxes[idx], 
            'translate': torch.tensor([xmin, ymin, xmin, ymin]).float() ,
            'image_path': self.path, 
            'rotation': rotate, 
            'resize': resize,
            'det_shape': torch.tensor([det_h, det_w]), 
            'crop_shape': torch.tensor([crop_h, crop_w]), 
            'unpadded_shape': torch.tensor([unpadded[0], unpadded[1]]).long() ,
            'coco': self.dataset.get_image_metadata(self.path),
            'roi_shape': torch.tensor([roi_h, roi_w]),
        }

        return self.transform(img_in), metadata


