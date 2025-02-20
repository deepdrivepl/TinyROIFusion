from .config import unet_DC, unet_SDS, yolov7t_SDS, yolov7t_DC, tracker
from .datasets.DroneCrowd import DroneCrowdDataset
from .datasets.SeaDronesSee import SeaDronesSeeDataset



CONFIG = dict(
    SeaDronesSee = dict(
        dataset = SeaDronesSeeDataset,
        roi_model = unet_SDS,
        det_model = yolov7t_SDS,
        tracker = tracker,
    ),
    DroneCrowd = dict(
        dataset = DroneCrowdDataset,
        roi_model = unet_DC,
        det_model = yolov7t_DC,
        tracker = tracker,
    )
)