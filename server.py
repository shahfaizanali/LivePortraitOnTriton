import asyncio
import json
import logging
import os
import subprocess
import time
import uuid

import cv2
import ffmpeg
import numpy as np
import aiohttp_cors
from aiohttp import web
import aiohttp
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
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
from middleware import is_authenticated_middleware
from file_downloader import download_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# STUN server configuration
ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]

rtc_configuration = RTCConfiguration(
    iceServers=ICE_SERVERS,
)

def create_image_map(images_dir='./images'):
    images_path = os.path.abspath(images_dir)

    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"The directory {images_dir} does not exist.")

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}

    image_map = {}

    for entry in os.listdir(images_path):
        entry_path = os.path.join(images_path, entry)
        
        if os.path.isfile(entry_path):
            filename, ext = os.path.splitext(entry)
            if ext.lower() in allowed_extensions:
                relative_path = os.path.relpath(entry_path, os.getcwd())
                relative_path = relative_path.replace(os.sep, '/')
                image_map[filename] = f"./{relative_path}"

    return image_map

image_map = create_image_map()

# Default values
default_src_image = "deepfake_cleveland.png"
default_cfg = "configs/trt_infer.yaml"
default_paste_back = False

infer_cfg = OmegaConf.load(default_cfg)
infer_cfg.infer_params.flag_pasteback = default_paste_back

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, user_id, source_image, cfg):
        super().__init__()
        self.track = track
        self.user_id = user_id
        self.source_image = source_image
        self.last_animated_face = None
        self.initialized = False
        self.infer_times = []
        self.frame_ind = 0
        self.ffmpeg_process = None
        self.cfg = cfg
        # self.ffmpeg_process = self.start_ffmpeg_process()

    def start_ffmpeg_process(self):
        # Create a personalized RTMP URL
        rtmp_url = f"rtmp://localhost:1935/live/{self.user_id}"

        # Adjust these parameters as needed (frame size, framerate, bitrate, etc.)
        return subprocess.Popen([
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', '512x512',  # Adjust resolution as needed
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
            rtmp_url
        ], stdin=subprocess.PIPE)

    def update_source_image(self, file_key):
        self.source_image = image_map[file_key]
        self.initialized = False

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")

        if not self.initialized:
            self.pipe = FasterLivePortraitPipeline(cfg=self.cfg, is_animal=False)
            await self.pipe.initialize()
            await self.pipe.prepare_source(self.source_image, realtime=True)
            self.initialized = True

        t0 = time.time()
        first_frame = (self.frame_ind == 0)
        dri_crop, out_crop, out_org = await self.pipe.run(img, self.pipe.src_imgs[0], self.pipe.src_infos[0], first_frame=first_frame)
        self.frame_ind += 1
        if out_crop is None:
            logger.info(f"No face in driving frame: {self.frame_ind}")
            # No output, just return the original frame
            return frame
        # self.infer_times.append(time.time() - t0)
        # logger.info(time.time() - t0)

        # # Ensure out_crop is 556x556
        # out_crop = cv2.resize(out_crop, (556, 556))

        # # Write the processed frame to FFmpeg
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            self.ffmpeg_process.stdin.write(out_crop.tobytes())

        # # Return the processed frame to the WebRTC client as well (optional)
        new_frame = VideoFrame.from_ndarray(out_crop, format="rgb24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def handle_message(self, message):
        logger.info(f"handling message: {message['type']}")
        if message['type'] == 'reset':
          self.pipe.src_lmk_pre = None
          self.frame_ind = 0

    def stop(self):
        logger.info("Stopping VideoTransformTrack and closing RTMP stream")
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
        super().stop()

relay = MediaRelay()

# pcs set is used for cleanup on shutdown
broadcasters = set()

async def health(request):
    return web.Response(status=200)

async def index(request):
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def stream(request):
    content = open("viewer.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def create_whip_client(broadcaster_pc):
    whip_url = "http://localhost:8080/api/whip"

    pc = broadcaster_pc.whip_pc

    # Create an SDP offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    authorization = f"Bearer {broadcaster_pc.user_id}"

    # Send the offer to the WHIP server
    async with aiohttp.ClientSession() as session:
        headers = {'Content-Type': 'application/sdp', 'authorization': authorization }
        async with session.post(whip_url, data=pc.localDescription.sdp, headers=headers) as response:
            answer_sdp = await response.text()
            response.close()
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type='answer'))

async def offer(request):
    logger.info("Received offer request")
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    avatar_url = params["avatar_url"]
    config = OmegaConf.create(params["config"])
    merged_cfg = OmegaConf.merge(infer_cfg, config)
    user_id = request["user_id"]
    source_image = await download_file(avatar_url)
    pc = RTCPeerConnection(rtc_configuration)
    pc.whip_pc = RTCPeerConnection()
    pc.user_id = request["user_id"]
    broadcasters.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state (broadcaster): {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            broadcasters.discard(pc)
            # if local_video:
            #     local_video.stop()
            

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state (broadcaster): {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            logger.error("Connection failed or closed (broadcaster)")
            await pc.whip_pc.close()
            await pc.close()
            broadcasters.discard(pc)
            if pc.video_track:
                pc.video_track.stop()

    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track: {track.kind}")
        if track.kind == "video":
            local_video = VideoTransformTrack(relay.subscribe(track, buffered=False), user_id, source_image, merged_cfg)
            relayed = relay.subscribe(local_video, buffered=False)
            pc.video_track = local_video
            pc.addTrack(relayed)
            pc.whip_pc.addTrack(relayed)
              
        if track.kind == "audio":
            relayed = relay.subscribe(track, buffered=True)
            pc.audio_track = track
            pc.whip_pc.addTrack(relayed)
        if hasattr(pc, "audio_track") and hasattr(pc, "video_track"):
          if pc.audio_track and pc.video_track:
              asyncio.ensure_future(create_whip_client(pc))

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            logger.info(f"Datachannel message: {message}")
            if isinstance(message, str):
                data = json.loads(message)
                if(pc.video_track):
                    pc.video_track.handle_message(data)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()

    # Prefer H264 if available
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            codecs = RTCRtpSender.getCapabilities("video").codecs
            preferred_codecs = [codec for codec in codecs if codec.mimeType == "video/H264"]
            transceiver.setCodecPreferences(preferred_codecs)

    await pc.setLocalDescription(answer)

    # Return the RTMP URL to the client so they know where to connect via OBS
    rtmp_url = f"rtmp://{request.host.split(':')[0]}:1935/live/{user_id}"
    viewer_url = f"https://{request.host.split(':')[0]}/stream?user_id={user_id}"
    logger.info(f"User {user_id} RTMP URL: {rtmp_url}")

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "user_id": user_id,
            # "stream_url": rtmp_url,
            "stream_url": viewer_url,
        })
    )

@web.middleware
async def logging_middleware(request, handler):
    logger.info(f"Received request: {request.method} {request.path}")
    response = await handler(request)
    logger.info(f"Sending response: {response.status}")
    return response

async def perf(request):
    content = open("perf-test.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def on_shutdown(app):
    logger.info("Shutting down")
    coros = [pc.close() for pc in broadcasters]
    await asyncio.gather(*coros)
    broadcasters.clear()

if __name__ == "__main__":
    app = web.Application(middlewares=[logging_middleware, is_authenticated_middleware])
    cors = aiohttp_cors.setup(app, defaults={
    "https://ps-dev-ce1b0.ravai.hypelaunch.io": {  # Replace with your frontend's origin
        "allow_headers": "*",
        "allow_credentials": True,  # Allow cookies
        }
    })
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/health", health)
    app.router.add_get("/", index)
    app.router.add_get("/stream", stream)
    app.router.add_get("/perf", perf)
    offer_route = app.router.add_post("/offer", offer)
    cors.add(offer_route)

    port = int(os.getenv("PORT", 8081))
    logger.info(f"Starting server on {port}")
    web.run_app(app, host="0.0.0.0", port=port)
