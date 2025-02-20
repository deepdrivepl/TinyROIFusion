FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx wget libglib2.0-0 g++
RUN pip install seaborn thop tensorboard opencv-python gdown Pillow scikit-image filterpy lap

WORKDIR /TinyROITrack
COPY ./scripts/download_DroneCrowd_test.sh /download_DroneCrowd_test.sh 
RUN chmod +x /download_DroneCrowd_test.sh 

COPY ./scripts/download_SeaDronesSee.sh /download_SeaDronesSee.sh
RUN chmod +x /download_SeaDronesSee.sh 

CMD ["/bin/sh", "-c", "/download_DroneCrowd_test.sh && /download_SeaDronesSee.sh"]