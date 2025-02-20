OUT_DIR="weights"

mkdir -p $OUT_DIR

echo "Downloading models"
wget "https://github.com/deepdrivepl/TinyROITrack/releases/download/v0.1/DroneCrowd-uR18.pt" -P $OUT_DIR
wget "https://github.com/deepdrivepl/TinyROITrack/releases/download/v0.1/DroneCrowd-yolov7t.pt" -P $OUT_DIR
wget "https://github.com/deepdrivepl/TinyROITrack/releases/download/v0.1/SeaDronesSee-uR18.pt" -P $OUT_DIR
wget "https://github.com/deepdrivepl/TinyROITrack/releases/download/v0.1/SeaDronesSee-yolov7t.pt" -P $OUT_DIR