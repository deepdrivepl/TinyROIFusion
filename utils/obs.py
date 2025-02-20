import torch
from .bboxes import box_iou

def OverlappingBoxSuppresion(windows, bboxes, th=0.6):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # detections from window = 0
    for i, unique_window in enumerate(unique_windows):
        win_ind = torch.unique(torch.where(windows==unique_window)[0])
        intersections[win_ind, i, :] = 0
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    confs, areas = [], []
    for i in det_ind:
        xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
        confs.append(conf.item())
        areas.append((xmax-xmin)*(ymax-ymin))
        
    confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    ious = normalize(ious.unsqueeze(-1))
    mean = torch.mean(torch.stack([ious,confs, areas]), 0)

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            continue
        if to_del[i, 2].item() in to_del_ids:
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    return windows_filtered, bboxes_filtered