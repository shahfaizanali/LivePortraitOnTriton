#!/bin/bash

cd /avatar/LivePortraitOnTriton

git pull

docker build -t live-portrait:triton .

docker stop live-triton

docker rm live-triton

docker stop live-triton-1

docker rm live-triton-1

docker stop live-triton-2

docker rm live-triton-2



cd nginx

docker build -t hypelaunchdev/analytics-nginx .

# docker push hypelaunchdev/analytics-nginx

docker stop nginx-avatar

docker rm nginx-avatar

docker run --network host -d --name nginx-avatar hypelaunchdev/analytics-nginx

docker run --gpus=all -d --name live-triton -v /avatar/checkpoints:/LivePortraitOnTriton/checkpoints -v /avatar/images:/LivePortraitOnTriton/images --network host -e PORT=8081 live-portrait:triton
docker run --gpus=all -d --name live-triton-1 -v /avatar/checkpoints:/LivePortraitOnTriton/checkpoints -v /avatar/images:/LivePortraitOnTriton/images --network host -e PORT=8082 live-portrait:triton
docker run --gpus=all -d --name live-triton-2 -v /avatar/checkpoints:/LivePortraitOnTriton/checkpoints -v /avatar/images:/LivePortraitOnTriton/images --network host -e PORT=8083 live-portrait:triton