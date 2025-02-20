./scripts/download_models.sh
./scripts/download_DroneCrowd_test.sh
./scripts/download_SeaDronesSee.sh

docker build -t tinyroitrack .
docker run --ipc=host --gpus all -it -v $(pwd):/TinyROITrack tinyroitrack 