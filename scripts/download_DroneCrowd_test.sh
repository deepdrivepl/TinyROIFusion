OUT_DIR="/TinyROITrack/data/DroneCrowd"

mkdir -p $OUT_DIR

echo "Downloading DroneCrowd dataset"
gdown https://drive.google.com/uc?id=1NeUK0AqgACG1iPiu4rjz3J68Pnsaj5LN -O $OUT_DIR/annotations.zip
gdown https://drive.google.com/uc?id=1C9PGUZ9GPB_NnUBVGbkJQFXhRxD3bfT9 -O $OUT_DIR/test_data.zip

unzip $OUT_DIR/annotations.zip -d $OUT_DIR
unzip $OUT_DIR/test_data.zip -d $OUT_DIR

rm $OUT_DIR/annotations.zip $OUT_DIR/test_data.zip 