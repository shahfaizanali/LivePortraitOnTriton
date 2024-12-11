import argparse
import asyncio
import datetime
import json
import logging
import os
import platform
import pdb
import subprocess
import time
import uuid
from typing import List
import cv2
import ffmpeg
import numpy as np
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCRtpSender,
)
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# STUN server configuration
ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]

# Optimized RTCConfiguration
rtc_configuration = RTCConfiguration(
    iceServers=ICE_SERVERS,
    #iceTransportPolicy="all",
    #bundlePolicy="max-bundle",
    #rtcpMuxPolicy="require",
    #iceCandidatePoolSize=0,
)


# Load a source image for animation (you may want to make this configurable)
default_source_image = cv2.imread("deepfake_cleveland.png")
default_source_image = cv2.cvtColor(default_source_image, cv2.COLOR_BGR2RGB)

image_map = {"default": "deepfake_cleveland.png"}


# Assign default values to variables
default_src_image = "deepfake_cleveland.png"
default_dri_video = "assets/examples/driving/d14.mp4"
default_cfg = "configs/trt_infer.yaml"
default_paste_back = False


infer_cfg = OmegaConf.load(default_cfg)
infer_cfg.infer_params.flag_pasteback = default_paste_back
pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)
ret = pipe.prepare_source(default_src_image, realtime=True)
if not ret:
    logger.info(f"no face in {default_src_image}! exit!")
    exit(1)

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.source_image = self.load_source_image(image_map["default"])
        self.last_animated_face = None
        self.initialized = False
        self.uid = str(uuid.uuid4())
        self.infer_times = []
        self.frame_ind = 0

    def load_source_image(self, image_path):
        image = None
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
        else:
            image = self.load_default_image()
        
        # resize the image to the next size that is divisible by two
        height, width = image.shape[:2]
        new_height = (height + 1) // 2 * 2
        new_width = (width + 1) // 2 * 2
        image = cv2.resize(image, (new_width, new_height))

        return image


    def update_source_image(self, file_key):
        self.source_image = self.load_source_image(image_map[file_key])
        self.initialized = False

    def start_ffmpeg_process(self):
        return subprocess.Popen([
            '/bin/ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', '556x556',  # Adjust resolution as needed
            '-r', '30',  # Adjust framerate as needed
            '-i', '-',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            '-b:v', '2M',  # Adjust bitrate as needed
            '-maxrate', '2M',
            '-bufsize', '4M',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', '60',  # Keyframe interval
            '-f', 'flv',
            'rtmp://localhost:1935/live/stream'
        ], stdin=subprocess.PIPE)

    async def recv(self):
        logger.debug(f"VideoTransformTrack recv called #{self.uid}")
        frame = None
        animated_face = None
        try:
            logger.debug("Received frame from original track")
            frame = await self.track.recv()
            # Convert frame to numpy array
            img = frame.to_ndarray(format="rgb24")

            if not self.initialized:
                self.initialized = True
            
            t0 = time.time()
            first_frame = self.frame_ind == 0
            dri_crop, out_crop, out_org = pipe.run(img, pipe.src_imgs[0], pipe.src_infos[0], first_frame=first_frame)
            self.frame_ind += 1
            if out_crop is None:
                logger.info(f"no face in driving frame:{self.frame_ind}")
                return frame
            self.infer_times.append(time.time() - t0)
            logger.info(time.time() - t0)
            # dri_crop = cv2.resize(dri_crop, (512, 512))
            # out_crop = np.concatenate([dri_crop, out_crop], axis=1)
            # out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
            
            # #self.ffmpeg_process.stdin.write(animated_face.tobytes())
            
            # # # Convert back to VideoFrame
            new_frame = VideoFrame.from_ndarray(out_crop, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            # logger.debug("Returning processed frame")
            return new_frame
        
        except Exception as e:
            logger.error(f"Error in VideoTransformTrack.recv: {e}")
            return frame

    def handle_message(self, message):
        if message['type'] == 'strength':
            self.power = float(message['value'])
            global POWER
            POWER = self.power
        elif message['type'] == 'reset':
            logger.info()
            # lia_model.force_set_source_motion()

    def stop(self):
        logger.info("Stopping VideoTransformTrack and closing RTMP stream")
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None

relay = MediaRelay()

async def health(request):
    return web.Response(status=200)

async def index(request):
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def on_ice_candidate(request):
    params = await request.json()
    pc = request.app['peer_connections'].get(params['sessionId'])
    if pc:
        candidate = RTCIceCandidate(
            sdpMid=params['sdpMid'],
            sdpMLineIndex=params['sdpMLineIndex'],
            candidate=params['candidate']
        )
        await pc.addIceCandidate(candidate)
    return web.Response(status=201)


async def offer(request):
    logger.info("Received offer request")
    params = await request.json()
    logger.debug(f"Offer parameters: {params}")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(rtc_configuration)
    pcs.add(pc)

    local_video = None  # Store the VideoTransformTrack instance

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state changed to {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            if local_video:
                local_video.stop()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state changed to: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            logger.error("Connection failed or closed")
            await pc.close()
            pcs.discard(pc)
            if local_video:
                local_video.stop()

    @pc.on("track")
    def on_track(track):
        nonlocal local_video
        logger.info(f"Received track: {track.kind}")
        if track.kind == "video":
            logger.info("Creating VideoTransformTrack")
            local_video = VideoTransformTrack(relay.subscribe(track, buffered=False))
            pc.addTrack(local_video)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            logger.info(f"Message: {message}")
            if isinstance(message, str):
                data = json.loads(message)
                logger.info(f"Message: {message}")
                for sender in pc.getTransceivers():
                    if sender.sender.track and sender.sender.track.kind == "video":
                        if isinstance(sender.sender.track, VideoTransformTrack):
                            sender.sender.track.handle_message(data)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()

    # Set codec preferences
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            codecs = RTCRtpSender.getCapabilities("video").codecs
            preferred_codecs = [codec for codec in codecs if codec.mimeType == "video/H264"]
            transceiver.setCodecPreferences(preferred_codecs)

    await pc.setLocalDescription(answer)

    logger.info("Sending answer back to client")
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

pcs = set()

async def on_shutdown(app):
    logger.info("Shutting down")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def upload_image(request):
    reader = await request.multipart()
    field = await reader.next()
    if field.name == 'file':
        filename = field.filename
        extension = filename.split(".")[-1]
        file_path = os.path.join('/images', f"{uuid.uuid4()}.{extension}")
        with open(file_path, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        image_map[filename] = file_path
        return web.Response(text=json.dumps(list(image_map.keys())), content_type='application/json')
    return web.Response(status=400, text="No file uploaded")

async def update_source_image(request):
    params = await request.json()
    image_key = params.get("image")
    
    if image_key not in image_map:
        return web.Response(status=400, text="No source image path provided")

    
    if not os.path.exists(image_map[image_key]):
        return web.Response(status=404, text="Source image not found")

    for pc in pcs:
        for sender in pc.getSenders():
            if sender.track and sender.track.kind == "video":
                if isinstance(sender.track, VideoTransformTrack):
                    sender.track.update_source_image(image_key)
    
    return web.Response(status=200)

async def get_available_files(request):
    return web.Response(text=json.dumps(list(image_map.keys())), content_type='application/json')


@web.middleware
async def logging_middleware(request, handler):
    logger.info(f"Received request: {request.method} {request.path}")
    response = await handler(request)
    logger.info(f"Sending response: {response.status}")
    return response

async def perf(request):
    content = open("perf-test.html", "r").read()
    return web.Response(content_type="text/html", text=content)

if __name__ == "__main__":
    app = web.Application(middlewares=[logging_middleware])
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/health", health)
    app.router.add_get("/", index)
    app.router.add_get("/perf", perf)
    app.router.add_post("/offer", offer)
    app.router.add_post("/upload", upload_image)
    app.router.add_post("/update-source-image", update_source_image)
    app.router.add_get("/get-available", get_available_files)
    #app.router.add_post("/ice-candidate", on_ice_candidate)
    port = int(os.getenv("PORT", 8081))
    logger.info(f"Starting server on {port}")
    web.run_app(app, host="0.0.0.0", port=port)