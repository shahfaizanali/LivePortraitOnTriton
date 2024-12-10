FROM nvcr.io/nvidia/tensorrt:24.11-py3

COPY requirements.txt /opt/requirements.txt

RUN pip install --no-cache-dir -r /opt/requirements.txt

RUN pip install --no-cache-dir onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

RUN mkdir -p /opt/ffmpeg
RUN cd /opt/ffmpeg \
    && git clone -q -b sdk/11.0 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
    && cd nv-codec-headers && \
    make install
RUN apt update && \
    apt install build-essential \
        pkg-config \
        yasm \
        cmake \
        libtool \
        libc6 \
        libc6-dev \
        unzip \
        wget \
        libnuma1 \
        libnuma-dev \
        libx264-dev \
        libwebp-dev \
        libmp3lame-dev \
        libffmpeg-nvenc-dev -y \
    && rm -rf /var/lib/apt/lists/*
RUN cd /opt/ffmpeg \
    && git clone -q -b release/6.1 https://git.ffmpeg.org/ffmpeg.git ffmpeg/ && \
    cd ffmpeg && \
    ./configure --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-nvenc \
    --enable-libnpp \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --disable-static \
    --enable-shared \
    --enable-gpl \
    --enable-libwebp \
    --enable-libmp3lame \
    --enable-libx264 && \
    make -j 8 && make install && rm -rf /opt/ffmpeg

RUN pip install --no-cache-dir torch torchvision cupy-cuda12x
RUN apt update && apt install libgl1 -y && rm -rf /var/lib/apt/lists/*

WORKDIR /LivePortraitOnTriton

COPY . .

CMD ["python3", "server.py"]
#RUN cd /root/FasterLivePortrait/src/models/XPose/models/UniPose/ops && python setup.py build install