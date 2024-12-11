#!/bin/bash

cd /avatar/LivePortraitOnTriton

git pull

docker build -t live-portrait .

docker stop live

docker rm live

docker run -d --name live -v /avatar/checkpoints:/LivePortraitOnTriton/checkpoints --network host -e PORT=8081 live-portrait