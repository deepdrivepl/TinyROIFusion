python inference_sequence.py --out_dir results/SeaDronesSee \
--ds SeaDronesSee --split val \
--mode 'roi_track' \
--merge  \
--obs_iou_th 0.1 \
--debug --vis_conf_th 0.1 --show_label

python inference_sequence.py --out_dir results/DroneCrowd \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--mode 'roi_track' \
--merge \
--obs_iou_th 0.7 --second_nms \
--debug --vis_conf_th 0.1