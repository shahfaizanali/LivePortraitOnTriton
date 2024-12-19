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
from aiohttp import web
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

    def __init__(self, track, user_id):
        super().__init__()
        self.track = track
        self.user_id = user_id
        self.source_image = image_map["default"]
        self.last_animated_face = None
        self.initialized = False
        self.infer_times = []
        self.frame_ind = 0
        self.pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)
        self.ffmpeg_process = self.start_ffmpeg_process()

    def start_ffmpeg_process(self):
        # Create a personalized RTMP URL
        rtmp_url = f"rtmp://localhost:1935/live/{self.user_id}"

        # Adjust these parameters as needed (frame size, framerate, bitrate, etc.)
        return subprocess.Popen([
            '/bin/ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', '556x556',  # Must match the output frame size you're processing
            '-r', '30',       # Framerate
            '-i', '-',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            '-b:v', '2M',
            '-maxrate', '2M',
            '-bufsize', '4M',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', '60',
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
            self.pipe.prepare_source(self.source_image, realtime=True)
            self.initialized = True

        t0 = time.time()
        first_frame = self.frame_ind == 0
        dri_crop, out_crop, out_org = self.pipe.run(img, self.pipe.src_imgs[0], self.pipe.src_infos[0], first_frame=first_frame)
        self.frame_ind += 1
        if out_crop is None:
            logger.info(f"No face in driving frame: {self.frame_ind}")
            # In case of no output, just return the original frame
            return frame

        self.infer_times.append(time.time() - t0)
        logger.info(time.time() - t0)

        # Ensure out_crop is 556x556 to match ffmpeg input
        out_crop = cv2.resize(out_crop, (556, 556))

        # Write the processed frame to FFmpeg
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            self.ffmpeg_process.stdin.write(out_crop.tobytes())

        # Return the processed frame to the WebRTC client as well (optional)
        new_frame = VideoFrame.from_ndarray(out_crop, format="rgb24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def handle_message(self, message):
        # Handle any datachannel messages if needed
        pass

    def stop(self):
        logger.info("Stopping VideoTransformTrack and closing RTMP stream")
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
        super().stop()

relay = MediaRelay()
pcs = set()

async def health(request):
    return web.Response(status=200)

async def index(request):
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    logger.info("Received offer request")
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Generate a user_id for this session
    user_id = params.get("userId", str(uuid.uuid4()))

    pc = RTCPeerConnection(rtc_configuration)
    pcs.add(pc)

    local_video = None

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            if local_video:
                local_video.stop()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
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
            local_video = VideoTransformTrack(relay.subscribe(track, buffered=False), user_id)
            pc.addTrack(local_video)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            logger.info(f"Datachannel message: {message}")
            if isinstance(message, str):
                data = json.loads(message)
                for sender in pc.getSenders():
                    if sender.track and sender.track.kind == "video" and isinstance(sender.track, VideoTransformTrack):
                        sender.track.handle_message(data)

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
    logger.info(f"User {user_id} RTMP URL: {rtmp_url}")

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "user_id": user_id,
            "stream_url": rtmp_url
        })
    )

async def upload_image(request):
    reader = await request.multipart()
    field = await reader.next()
    if field.name == 'file':
        filename = field.filename
        extension = filename.split(".")[-1]
        file_path = os.path.join('./images', f"{uuid.uuid4()}.{extension}")
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
        return web.Response(status=400, text="Image key not found")

    if not os.path.exists(image_map[image_key]):
        return web.Response(status=404, text="Source image not found")

    for pc in pcs:
        for sender in pc.getSenders():
            if sender.track and sender.track.kind == "video" and isinstance(sender.track, VideoTransformTrack):
                sender.track.update_source_image(image_key)
    
    return web.Response(status=200)

async def get_available_files(request):
    global image_map
    image_map = create_image_map()
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

async def on_shutdown(app):
    logger.info("Shutting down")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

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

    port = int(os.getenv("PORT", 8081))
    logger.info(f"Starting server on {port}")
    web.run_app(app, host="0.0.0.0", port=port)