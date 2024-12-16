#!/bin/bash

cd /avatar/LivePortraitOnTriton

git pull

docker build -t live-portrait:triton .

docker stop live-triton

docker rm live-triton

cd nginx

docker build -t hypelaunchdev/analytics-nginx .

docker push hypelaunchdev/analytics-nginx

docker stop nginx-avatar

docker rm nginx-avatar

docker run --network host -d --name nginx-avatar hypelaunchdev/analytics-nginx

docker run --gpus=all -d --name live-triton -v /avatar/checkpoints:/LivePortraitOnTriton/checkpoints --network host -e PORT=8082 live-portrait:triton