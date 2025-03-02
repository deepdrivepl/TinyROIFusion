import argparse
import os
import importlib
import json

from glob import glob
from tqdm import tqdm
from statistics import mean

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from configs import CONFIG
from data_loader import SingleDetectionDataset, ROIDataset, WindowDetectionDataset
from utils.bboxes import getDetectionBboxesSorted, NMS, non_max_suppression, scale_coords, xyxy2xywh, findBboxes, rot90points, getSlidingWindowBBoxes
from utils.general import save_args, load_model
from utils.drawing import make_vis
from utils.obs import OverlappingBoxSuppresion
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--ds', type=str, default="SeaDronesSee", choices=CONFIG.keys())
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--flist', type=str, help='If provided, infer images listed in flist.txt; if not, infer split images.')
    parser.add_argument('--name', type=str, help='Name for img list provided in flist.txt')
    # general
    parser.add_argument('--mode', type=str, default='roi_track', choices=['roi', 'track', 'roi_track'])
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--out_dir', type=str, default='detections')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--vis_conf_th', type=float, default=0.3)
    parser.add_argument('--show_label', default=False, action='store_true')
    # NMS
    parser.add_argument('--second_nms', default=False, action='store_true')
    parser.add_argument('--merge', default=False, action='store_true')
    parser.add_argument('--redundant', default=False, action='store_true')
    parser.add_argument('--max_det', type=int, default=500)
    parser.add_argument('--agnostic', default=False, action='store_true')
    # tracker
    parser.add_argument('--frame_delay', type=int, default=3)
    # obs
    parser.add_argument('--obs_iou_th', type=float, default=0.7)
    
    args = parser.parse_args()
    
    # save args
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = args.out_dir
    save_args(out_dir, args)

    if args.debug:
        windows_dir = f'{args.out_dir}/vis-windows'
        detections_dir = f'{args.out_dir}/vis-detections'
        os.makedirs(windows_dir, exist_ok=True); os.makedirs(detections_dir, exist_ok=True)
    
    # load models
    config = CONFIG[args.ds]
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 and not args.cpu else 'cpu'
    cfg_det = config["det_model"]
    net_det = load_model(cfg_det, device)

    cfg_roi = config["roi_model"]
    net_roi = load_model(cfg_roi, device)
    
    cfg_trk = config["tracker"]
    trk_class = getattr(importlib.import_module(cfg_trk['module_name']), cfg_trk['class_name'])
    
    # get dataset
    flist = args.flist if args.flist is None else [x.rstrip() for x in open(args.flist)]
    ds = (config["dataset"])(split=args.split, flist=flist, name=args.name)
    seq2images = ds.get_seq2imgs()
    
    # inference
    annotations = []
    for seq_name, seq_flist in tqdm(seq2images.items()):
        seq_flist = sorted(seq_flist)
        if 'roi' in args.mode:
            dataset = ROIDataset(seq_flist, ds, cfg_roi["in_size"], cfg_roi["transform"])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        else:
            dataset = SingleDetectionDataset(seq_flist, ds, cfg_det["in_size"], cfg_det["transform"])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        tracker = (trk_class)(**cfg_trk['args'])
        
        seg_mask, mot_mask = None, None
        with torch.no_grad():
            for i, (img, metadata) in tqdm(enumerate(dataloader)):

                H_orig, W_orig = metadata['coco']['height'].item(), metadata['coco']['width'].item()
                original_shape = (H_orig, W_orig)


                seg_bboxes, mot_bboxes, det_bboxes = np.empty((0,4)), np.empty((0,4)), np.empty((0,4))
                seg_mask_fullres = None
                
                    
                if args.mode == 'track' and i < args.frame_delay: # detect (for frame_delay frames) to initialize the tracker

                    # sliding-window initialization method
                    det_bboxes = getSlidingWindowBBoxes([0,0,W_orig,H_orig], cfg_det['in_size'])[0]
                    det_bboxes = np.array(det_bboxes).astype(np.int32)
                    det_dataset =  WindowDetectionDataset(metadata['image_path'][0], ds, det_bboxes, cfg_det['in_size'])
                    det_dataloader = DataLoader(det_dataset, batch_size=len(det_dataset) if len(det_dataset)>0 else 1, shuffle=False, num_workers=4) # all windows in a single batch

                    img_out = torch.empty((0, 6))
                    win_out = torch.empty((0, 4))

                    for j, (img_det, det_metadata) in enumerate(det_dataloader):
                        out = net_det(img_det.to(device))
                        out = cfg_det["postprocess"](out)
                        out = non_max_suppression(
                            out, 
                            conf_thres = cfg_det['conf_thresh'], 
                            iou_thres = cfg_det['iou_thresh'],
                            multi_label = True,
                            labels = [],
                            merge = args.merge,
                            agnostic = args.agnostic
                        )
                        
                        for si, pred in enumerate(out):
                            if len(pred) == 0:
                                continue
                                
                            win_out = torch.cat((
                                win_out,
                                det_metadata['bbox'][si].repeat(len(pred),1)
                            ))

                            if det_metadata['resize'][si].item():

                                pred[:,:4] = scale_coords(det_metadata['unpadded_shape'][si], pred[:, :4], det_metadata['crop_shape'][si])

                            if det_metadata['rotation'][si].item():
                                h_window, w_window = det_metadata['roi_shape'][si]
                                xmin_, ymax_ = rot90points(pred[:,0], pred[:,1], [w_window.item(),h_window.item()])
                                xmax_, ymin_ = rot90points(pred[:,2], pred[:,3], [w_window.item(),h_window.item()])
                                pred[:,0] = xmin_
                                pred[:,1] = ymin_
                                pred[:,2] = xmax_
                                pred[:,3] = ymax_

                            pred[:,:4] = pred[:,:4] + det_metadata['translate'][si].to(device) # to the coordinates of the original image
              
      
                        if not img_out.numel():
                            img_out = torch.cat(out)
                        else:
                            img_out = torch.cat((img_out, out), 0)

                    win_out = win_out.to(device)
                    img_out = img_out.to(device)

                    if args.second_nms and img_out is not None:
                        img_out, win_out = NMS(
                            img_out, 
                            win_out, 
                            iou_thres = cfg_det["iou_thresh"], 
                            redundant = args.redundant, 
                            merge = args.merge, 
                            max_det = args.max_det, 
                            agnostic = args.agnostic
                        )
                        
                    win_out = win_out[~torch.any(img_out.isnan(),dim=1)]
                    img_out = img_out[~torch.any(img_out.isnan(),dim=1)]

                    # OBS
                    win_out, img_out = OverlappingBoxSuppresion(win_out, img_out, th=args.obs_iou_th)

                    trks = tracker.get_pred_locations()
                    tracker.update(img_out.detach().cpu().numpy()[:, :-1], trks)
                    if args.debug:
                        frame = cv2.imread(metadata['image_path'][0])
                        
                        frame, frame_dets = make_vis(
                            frame, 
                            seg_mask_fullres, 
                            mot_mask, 
                            det_bboxes, 
                            img_out, 
                            ds.classes, 
                            ds.colors, 
                            args.vis_conf_th, 
                            show_label=args.show_label
                        )
                        out_fname_wins = f"{windows_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                        out_fname_dets = f"{detections_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                        os.makedirs(os.path.dirname(out_fname_wins), exist_ok=True); os.makedirs(os.path.dirname(out_fname_dets), exist_ok=True)
                        cv2.imwrite(out_fname_wins, frame); cv2.imwrite(out_fname_dets, frame_dets)

                    img_out[:,:4] = xyxy2xywh(img_out[:,:4])
                    for p in img_out.tolist():
                        annotations.append(
                            {
                                "id": len(annotations), 
                                "image_id": int(metadata['coco']['id'].item()),
                                "category_id": int(p[-1]),
                                "bbox": [round(x, 3) for x in p[:4]],
                                "area": p[2] * p[3],
                                "score": round(p[4], 5),
                                "iscrowd": 0,
                            }
                        )
                    continue # do not run the window detection, just track for frame_delay frames 

                elif 'roi' in args.mode: # predict ROIs
                    seg_mask = net_roi(img.to(device))
                    seg_mask_fullres, seg_mask = cfg_roi["postprocess"](
                        seg_mask, 
                        original_shape, 
                        **cfg_roi["postprocess_args"],
                    )

                    seg_bboxes = findBboxes(seg_mask, original_shape, seg_mask.shape)
                    merged_bboxes = seg_bboxes

                if 'track' in args.mode:
                    trks = tracker.get_pred_locations()
                    if i >= args.frame_delay:

                        mot_bboxes = trks[:,:-1]

                        mot_bboxes[:,0] = np.where(mot_bboxes[:,0] < 0, 0, mot_bboxes[:,0])
                        mot_bboxes[:,1] = np.where(mot_bboxes[:,1] < 0, 0, mot_bboxes[:,1])
                        mot_bboxes[:,2] = np.where(mot_bboxes[:,2] >= W_orig, W_orig-1, mot_bboxes[:,2])
                        mot_bboxes[:,3] = np.where(mot_bboxes[:,3] >= H_orig, H_orig-1, mot_bboxes[:,3])

                        indices = np.nonzero(((mot_bboxes[:,2]-mot_bboxes[:,0]) > 0) & ((mot_bboxes[:,3]-mot_bboxes[:,1]) > 0))
                        mot_bboxes = mot_bboxes[indices[0], :]

                    mot_mask = np.zeros((H_orig, W_orig), dtype=np.uint8)
                    for mot_bbox in mot_bboxes:
                        xmin,ymin,xmax,ymax = map(int, mot_bbox[:4])
                        mot_mask[ymin:ymax+1, xmin:xmax+1] = 255

                    if seg_mask is not None:
                        mask_h,mask_w = seg_mask.shape[:2]
                        mot_mask_low = cv2.resize(mot_mask, (mask_w, mask_h))
                        seg_mask = np.logical_or(seg_mask, mot_mask_low).astype(np.uint8)*255 

                        merged_bboxes = findBboxes(seg_mask, original_shape, seg_mask.shape)
                    else:
                        merged_bboxes = findBboxes(mot_mask, original_shape, mot_mask.shape)

                det_bboxes = getDetectionBboxesSorted(
                    merged_bboxes, 
                    H_orig, W_orig, 
                    cfg_det['in_size'], 
                    cfg_det['resize'],
                )

                det_bboxes = np.array(det_bboxes).astype(np.int32)
                
                if len(det_bboxes) > 0:
                    indices = np.nonzero(((det_bboxes[:,2]-det_bboxes[:,0]) > 0) & ((det_bboxes[:,3]-det_bboxes[:,1]) > 0))
                    indices = indices[0]
                    det_bboxes = det_bboxes[indices, :]

                det_dataset =  WindowDetectionDataset(metadata['image_path'][0], ds, det_bboxes, cfg_det['in_size'])
                det_dataloader = DataLoader(det_dataset, batch_size=len(det_dataset) if len(det_dataset)>0 else 1, shuffle=False, num_workers=4) # all windows in a single batch

                img_out = torch.empty((0, 6))
                win_out = torch.empty((0, 4))

                for j, (img_det, det_metadata) in enumerate(det_dataloader):
                    out = net_det(img_det.to(device))
                    out = cfg_det["postprocess"](out)
                    out = non_max_suppression(
                        out, 
                        conf_thres = cfg_det['conf_thresh'], 
                        iou_thres = cfg_det['iou_thresh'],
                        multi_label = True,
                        labels = [],
                        merge = args.merge,
                        agnostic = args.agnostic
                    )
                    
                    for si, pred in enumerate(out):
                        if len(pred) == 0:
                            continue
                            
                        win_out = torch.cat((
                            win_out,
                            det_metadata['bbox'][si].repeat(len(pred),1)
                        ))

                        if det_metadata['resize'][si].item():

                            pred[:,:4] = scale_coords(det_metadata['unpadded_shape'][si], pred[:, :4], det_metadata['crop_shape'][si])

                        if det_metadata['rotation'][si].item():
                            h_window, w_window = det_metadata['roi_shape'][si]
                            xmin_, ymax_ = rot90points(pred[:,0], pred[:,1], [w_window.item(),h_window.item()])
                            xmax_, ymin_ = rot90points(pred[:,2], pred[:,3], [w_window.item(),h_window.item()])
                            pred[:,0] = xmin_
                            pred[:,1] = ymin_
                            pred[:,2] = xmax_
                            pred[:,3] = ymax_

                        pred[:,:4] = pred[:,:4] + det_metadata['translate'][si].to(device) # to the coordinates of the original image
          
  
                    if not img_out.numel():
                        img_out = torch.cat(out)
                    else:
                        img_out = torch.cat((img_out, out), 0)

                win_out = win_out.to(device)
                img_out = img_out.to(device)

                if args.second_nms and img_out is not None:
                    img_out, win_out = NMS(img_out, win_out, iou_thres=cfg_det["iou_thresh"], redundant=args.redundant, merge=args.merge, max_det=args.max_det, agnostic=args.agnostic)
                    
                win_out = win_out[~torch.any(img_out.isnan(),dim=1)]
                img_out = img_out[~torch.any(img_out.isnan(),dim=1)]

                # OBS
                win_out, img_out = OverlappingBoxSuppresion(win_out, img_out, th=args.obs_iou_th)
                    
                if 'track' in args.mode:
                    tracker.update(img_out.detach().cpu().numpy()[:, :-1], trks)

                
                if args.debug:
                    frame = cv2.imread(metadata['image_path'][0])
                    frame, frame_dets = make_vis(
                        frame, 
                        seg_mask_fullres, 
                        mot_mask, 
                        det_bboxes, 
                        img_out, 
                        ds.classes, 
                        ds.colors, 
                        args.vis_conf_th, 
                        show_label=args.show_label
                    )
                    out_fname_wins = f"{windows_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                    out_fname_dets = f"{detections_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                    os.makedirs(os.path.dirname(out_fname_wins), exist_ok=True); os.makedirs(os.path.dirname(out_fname_dets), exist_ok=True)
                    cv2.imwrite(out_fname_wins, frame); cv2.imwrite(out_fname_dets, frame_dets)
                
                img_out[:,:4] = xyxy2xywh(img_out[:,:4])
                for p in img_out.tolist():
                    annotations.append(
                        {
                            "id": len(annotations), 
                            "image_id": int(metadata['coco']['id'].item()),
                            "category_id": int(p[-1]),
                            "bbox": [round(x, 3) for x in p[:4]],
                            "area": p[2] * p[3],
                            "score": round(p[4], 5),
                            "iscrowd": 0,
                        }
                    )

        
    with open(os.path.join(out_dir, f'results-{args.split if args.flist is None else args.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
