#!/bin/bash
BASE_NAME="live-triton"
BASE_PORT=8081
IMAGE_NAME="live-portrait:triton"
CHECKPOINTS_DIR="/avatar/checkpoints"
IMAGES_DIR="/avatar/images"

cd /avatar/LivePortraitOnTriton

git pull

docker build -t live-portrait:triton .

# docker tag live-portrait:triton hypelaunchdev/live-portrait:triton

# docker push hypelaunchdev/live-portrait:triton

docker ps -a --filter "name=live-triton" --format "{{.Names}}" | xargs -r -n 1 docker stop
docker ps -a --filter "name=live-triton" --format "{{.Names}}" | xargs -r -n 1 docker rm

cd nginx

docker build -t hypelaunchdev/analytics-nginx .

docker push hypelaunchdev/analytics-nginx

docker stop nginx-avatar

docker rm nginx-avatar

docker run --restart unless-stopped --network host -d --name nginx-avatar hypelaunchdev/analytics-nginx

for i in {0..0}; do
    CONTAINER_NAME="${BASE_NAME}${i}"
    PORT=$((BASE_PORT + i))
    docker run --restart unless-stopped --gpus=all -d --name $CONTAINER_NAME \
        -v $CHECKPOINTS_DIR:/LivePortraitOnTriton/checkpoints \
        -v $IMAGES_DIR:/LivePortraitOnTriton/images \
        --network host -e PORT=$PORT $IMAGE_NAME
done