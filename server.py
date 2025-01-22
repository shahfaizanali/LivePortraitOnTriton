import asyncio
import json
import logging
import os
import subprocess
import time
import traceback
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
from aiortc.contrib.media import MediaRecorder
from av import VideoFrame
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio
from middleware import is_authenticated_middleware
from file_downloader import download_file

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# STUN server configuration
ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]

rtc_configuration = RTCConfiguration(
    iceServers=ICE_SERVERS,
)

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
        self.reset_pose = False
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

    async def recv(self):
      frame = None
      try:  
        frame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")

        if not self.initialized:
            self.pipe = FasterLivePortraitPipeline(cfg=self.cfg, is_animal=False)
            await self.pipe.initialize()
            await self.pipe.prepare_source(self.source_image, realtime=True)
            self.initialized = True

        t0 = time.time()
        first_frame = (self.frame_ind == 0)
        if self.reset_pose:
            self.pipe.src_lmk_pre = None
            first_frame = True
            self.reset_pose = False
        out_crop = await self.pipe.run(img, self.pipe.src_imgs[0], self.pipe.src_infos[0], first_frame=first_frame)
        self.frame_ind += 1
        if out_crop is None:
            logger.info(f"No face in driving frame: {self.frame_ind}")
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
      except Exception as e:
            traceback.print_exc()
            return frame
      

    def handle_message(self, message):
        logger.info(f"handling message: {message['type']}")
        if message['type'] == 'reset':
            self.reset_pose = True

    def stop(self):
        logger.info("Stopping VideoTransformTrack and closing RTMP stream")
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
        super().stop()

# pcs set is used for cleanup on shutdown
broadcasters = set()

async def cleanup_peer_connection(pc):
    """
    Cleanly stop all tracks, close all associated RTCPeerConnections,
    and remove from the broadcasters set.
    """
    logger.info(f"Cleaning up PeerConnection for user {getattr(pc, 'user_id', 'unknown')}")

    if hasattr(pc, "recorder") and pc.recorder is not None:
        logger.info("Closing recorder")
        await pc.recorder.stop()
        pc.recorder = None

    # 1. Stop any custom or local tracks you have saved on the PC
    #    For example, if you store them on pc.video_track / pc.audio_track
    if hasattr(pc, "video_track") and pc.video_track is not None:
        logger.info("Stopping video track")
        pc.video_track.stop()
        pc.video_track = None
    if hasattr(pc, "audio_track") and pc.audio_track is not None:
        logger.info("Stopping audio track")
        pc.audio_track.stop()
        pc.audio_track = None
        
    if hasattr(pc, "video_relayed_track") and pc.video_relayed_track is not None:
        logger.info("Stopping video relayed track")
        pc.video_relayed_track.stop()
        pc.video_relayed_track = None

    if hasattr(pc, "audio_relayed_track") and pc.audio_relayed_track is not None:
        logger.info("Stopping audio relayed track")
        pc.audio_relayed_track.stop()
        pc.audio_relayed_track = None

    # 2. Close the 'whip_pc' if it exists
    if hasattr(pc, "whip_pc") and pc.whip_pc is not None:
        logger.info("Closing whip_pc")
        await pc.whip_pc.close()
        pc.whip_pc = None

    # 3. Close the main PeerConnection
    logger.info("Closing main PeerConnection")
    await pc.close()

    # 4. Remove from broadcasters set if still present
    if pc in broadcasters:
        broadcasters.discard(pc)
    logger.info("Cleanup complete")


async def health(request):
    return web.Response(status=200)

async def index(request):
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def stream(request):
    content = open("viewer.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def handle_recording(broadcaster_pc):
    # logger.info("Starting Recording")
    # recording_path = f"/recordings/{broadcaster_pc.user_id}/{uuid.uuid4()}.mp4"
    # os.makedirs(os.path.dirname(recording_path), exist_ok=True)
    # recorder = broadcaster_pc.recorder = MediaRecorder(recording_path)
    # recorder.addTrack(broadcaster_pc.realyed_audio_track)
    # recorder.addTrack(broadcaster_pc.realyed_video_track)
    await broadcaster_pc.recorder.start()

async def handle_live_streaming(broadcaster_pc):
    whip_url = "http://localhost:8080/api/whip"

    pc = broadcaster_pc.whip_pc = RTCPeerConnection()
    pc.addTrack(broadcaster_pc.realyed_video_track)
    pc.addTrack(broadcaster_pc.realyed_audio_track)

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
    user_id = "ravaiavatartestuser"
    source_image = await download_file(avatar_url)
    recording = params["recording"]
    pc = RTCPeerConnection(rtc_configuration)
    pc.user_id = user_id
    broadcasters.add(pc)
    relay = MediaRelay()


    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state (broadcaster): {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
          await cleanup_peer_connection(pc)
            

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state (broadcaster): {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            logger.error("Connection failed or closed (broadcaster)")
            await cleanup_peer_connection(pc)

    @pc.on("track")
    def on_track(track):
        @track.on("ended")
        async def on_ended():
            if hasattr(pc, "recorder") and pc.recorder is not None:
              logger.info("Track ended Closing recorder")
              await pc.recorder.stop()
        logger.info(f"Received track: {track.kind}")
        if track.kind == "video":
            local_video = VideoTransformTrack(relay.subscribe(track, buffered=True), user_id, source_image, merged_cfg)
            relayed = relay.subscribe(local_video, buffered=True)
            pc.video_track = local_video
            pc.realyed_video_track = relayed
            pc.addTrack(relayed)
              
        if track.kind == "audio":
            relayed = relay.subscribe(track, buffered=True)
            pc.audio_track = track
            pc.realyed_audio_track = relayed
        if hasattr(pc, "audio_track") and hasattr(pc, "video_track"):
          if pc.audio_track and pc.video_track:
              if recording:
                  logger.info("Starting Recording")
                  recording_path = f"/recordings/{pc.user_id}/{uuid.uuid4()}.mp4"
                  os.makedirs(os.path.dirname(recording_path), exist_ok=True)
                  recorder = pc.recorder = MediaRecorder(recording_path)
                  recorder.addTrack(pc.audio_track)
                  recorder.addTrack(pc.video_track)
                  asyncio.ensure_future(handle_recording(pc))
              else:    
                  asyncio.ensure_future(handle_live_streaming(pc))

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
        },
    "https://ps-dev-0ca77.ravai.hypelaunch.io": {  # Replace with your frontend's origin
        "allow_headers": "*",
        "allow_credentials": True,  # Allow cookies
        },
    "http://localhost:5173": {  # Replace with your frontend's origin
        "allow_headers": "*",
        # "allow_credentials": True,  # Allow cookies
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
