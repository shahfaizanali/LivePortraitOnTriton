#!/bin/bash

cd /avatar/LivePortraitOnTriton

git pull

docker build -t live-portrait .

docker rm live

docker run -d --name live --network host -e PORT=8081 live-portrait