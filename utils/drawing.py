import cv2
import numpy as np


def plot_mask(mask, image, color = [255, 144, 30], alpha=0.4):
    mask = cv2.merge((mask, mask, mask))
    color = np.full(mask.shape, np.array(color))
    
    mask = np.where(mask>0, color, image).astype(np.uint8)
    masked_image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0) 
    return masked_image


def plot_one_box(bbox, img, color, label=None, lw=2, draw_label=True, font_scale=1.1):
    
    xmin,ymin,xmax,ymax = list(map(int, bbox))
    img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, lw)

    if draw_label:
        ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        img = cv2.rectangle(img, (xmin, ymin - int(1.3 * text_height)), (xmin + text_width, ymin), color, -1)
        
        img = cv2.putText(
            img,
            text=label,
            org=(xmin, ymin - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
    )
    return img


def make_vis(frame, seg_mask, mot_mask, det_bboxes, detections, classes, colors, vis_conf_th=0.1, show_label=True):

    for det_bbox in det_bboxes:
        frame = plot_one_box(list(map(int, det_bbox)), frame, color=(0,0,0), label='WIN', draw_label=show_label) 

    frame_wins = frame.copy()  
    if seg_mask is not None:
        frame_wins = plot_mask(seg_mask, frame_wins, alpha=0.6)

    if mot_mask is not None:
        frame_wins = plot_mask(mot_mask, frame_wins, color=[0, 128, 255], alpha=0.6)
    
    detections = detections[detections[:, -2] >= vis_conf_th]
    for det in detections.tolist():
        xmin,ymin,xmax,ymax,conf,cls = det
        frame = plot_one_box(
            list(map(int, [xmin,ymin,xmax,ymax])), 
            frame, 
            color=colors[int(cls)], 
            label=f'{classes[int(cls)]} {int(conf*100)}%' if show_label else '',
            draw_label=show_label
        )
    
    return frame_wins, frame