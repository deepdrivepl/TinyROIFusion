./scripts/download_models.sh
docker build -t tinyroitrack .
docker run --ipc=host --gpus all -it -v $(pwd):/TinyROITrack tinyroitrack 