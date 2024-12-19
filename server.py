import asyncio
import json
import logging
import os
import subprocess
import time
import uuid

import cv2
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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# STUN server configuration
ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
]

rtc_configuration = RTCConfiguration(iceServers=ICE_SERVERS)


def create_image_map(images_dir="./images"):
    images_path = os.path.abspath(images_dir)

    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"The directory {images_dir} does not exist.")

    allowed_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

    image_map = {}

    for entry in os.listdir(images_path):
        entry_path = os.path.join(images_path, entry)

        if os.path.isfile(entry_path):
            filename, ext = os.path.splitext(entry)
            if ext.lower() in allowed_extensions:
                relative_path = os.path.relpath(entry_path, os.getcwd())
                relative_path = relative_path.replace(os.sep, "/")
                image_map[filename] = f"./{relative_path}"

    return image_map


image_map = create_image_map()

# Default values
default_src_image = "deepfake_cleveland.png"
default_cfg = "configs/trt_infer.yaml"
default_paste_back = False

infer_cfg = OmegaConf.load(default_cfg)
infer_cfg.infer_params.flag_pasteback = default_paste_back


class StreamHandler:
    """
    Manages the FFmpeg process and named pipes for a single user's stream.
    """

    def __init__(
        self,
        user_id,
        video_resolution=(512, 512),  # Updated resolution
        framerate=15,                  # Updated framerate
        sample_rate=48000,
        channels=2,
    ):
        self.user_id = user_id
        self.video_resolution = video_resolution
        self.framerate = framerate
        self.sample_rate = sample_rate
        self.channels = channels
        self.video_pipe = f"/tmp/aiortc_video_{self.user_id}.raw"
        self.audio_pipe = f"/tmp/aiortc_audio_{self.user_id}.raw"
        self.ffmpeg_process = None
        self.video_fd = None
        self.audio_fd = None

        self.setup_pipes()
        self.start_ffmpeg()
        self.open_pipes(retries=10, delay=0.5)

    def setup_pipes(self):
        """
        Creates named pipes for video and audio.
        """
        for pipe in [self.video_pipe, self.audio_pipe]:
            if os.path.exists(pipe):
                os.remove(pipe)
            os.mkfifo(pipe)
        logging.info(f"Named pipes created for user {self.user_id}")

    def start_ffmpeg(self):
        """
        Starts the FFmpeg process to read from both audio and video pipes and stream to RTMP using NVENC.
        """
        rtmp_url = f"rtmp://localhost:1935/live/{self.user_id}"
        cmd = [
            "ffmpeg",
            "-y",
            # Video input
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.video_resolution[0]}x{self.video_resolution[1]}",  # 512x512
            "-r", str(self.framerate),  # 15 fps
            "-i", self.video_pipe,
            # Audio input
            "-f", "s16le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-i", self.audio_pipe,
            # Video encoding with NVENC
            "-c:v", "h264_nvenc",      # Use NVIDIA's H.264 encoder
            "-preset", "p2",            # Preset for lower latency; adjust as needed
            "-b:v", "2M",                # Video bitrate
            "-maxrate", "2M",
            "-bufsize", "4M",
            "-g", str(int(self.framerate) * 2),  # GOP size (e.g., 15 fps * 2 = 30)
            "-tune", "zerolatency",      # For low latency
            # Audio encoding
            "-c:a", "aac",
            "-b:a", "128k",
            # Sync options
            "-shortest",
            "-f", "flv",
            rtmp_url,
        ]
        logging.info(f"Starting FFmpeg for user {self.user_id}")
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Start asynchronous logging of FFmpeg's stderr
        asyncio.ensure_future(self.log_ffmpeg_output())

    async def log_ffmpeg_output(self):
        """
        Asynchronously logs FFmpeg's stderr output.
        """
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.stderr.readline)
            if not line:
                break
            logging.info(f"FFmpeg ({self.user_id}): {line.decode().strip()}")

    def open_pipes(self, retries=5, delay=0.5):
        """
        Attempts to open the named pipes for writing with retries.
        """
        for attempt in range(retries):
            try:
                self.video_fd = os.open(self.video_pipe, os.O_WRONLY | os.O_NONBLOCK)
                self.audio_fd = os.open(self.audio_pipe, os.O_WRONLY | os.O_NONBLOCK)
                logging.info(f"Opened pipes for writing for user {self.user_id}")
                return
            except OSError as e:
                logging.warning(f"Attempt {attempt + 1}/{retries}: Error opening pipes for user {self.user_id}: {e}")
                time.sleep(delay)
        logging.error(f"Failed to open pipes for user {self.user_id} after {retries} attempts.")
        self.stop_ffmpeg()
        raise e

    def write_video_frame(self, frame: np.ndarray):
        """
        Writes a processed video frame to the video pipe.
        """
        if self.video_fd:
            try:
                os.write(self.video_fd, frame.tobytes())
            except BlockingIOError:
                logging.warning(f"FFmpeg not ready to read video for user {self.user_id}")
            except Exception as e:
                logging.error(f"Error writing video frame for user {self.user_id}: {e}")

    def write_audio_frame(self, pcm_data: np.ndarray):
        """
        Writes raw PCM audio data to the audio pipe.
        """
        if self.audio_fd:
            try:
                os.write(self.audio_fd, pcm_data.tobytes())
            except BlockingIOError:
                logging.warning(f"FFmpeg not ready to read audio for user {self.user_id}")
            except Exception as e:
                logging.error(f"Error writing audio frame for user {self.user_id}: {e}")

    def stop_ffmpeg(self):
        """
        Terminates the FFmpeg process and cleans up named pipes.
        """
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
            logging.info(f"FFmpeg process for user {self.user_id} terminated.")
        # Close file descriptors
        if self.video_fd:
            os.close(self.video_fd)
            self.video_fd = None
        if self.audio_fd:
            os.close(self.audio_fd)
            self.audio_fd = None
        # Remove named pipes
        for pipe in [self.video_pipe, self.audio_pipe]:
            try:
                os.remove(pipe)
                logging.info(f"Removed pipe {pipe}")
            except Exception as e:
                logging.error(f"Error removing pipe {pipe}: {e}")
                
class VideoTransformTrack(MediaStreamTrack):
    """
    Processes video frames and writes them to the StreamHandler's video pipe.
    """

    kind = "video"

    def __init__(self, track, stream_handler: StreamHandler):
        super().__init__()
        self.track = track
        self.stream_handler = stream_handler
        self.source_image = image_map.get("default", default_src_image)
        self.initialized = False
        self.infer_times = []
        self.frame_ind = 0
        self.pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)

    def update_source_image(self, file_key):
        """
        Updates the source image for video processing.
        """
        self.source_image = image_map[file_key]
        self.initialized = False
        logger.info(f"Updated source image for user {self.stream_handler.user_id} to {file_key}")

    async def recv(self):
        """
        Receives a video frame, processes it, and writes it to the video pipe.
        """
        frame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")

        if not self.initialized:
            self.pipe.prepare_source(self.source_image, realtime=True)
            self.initialized = True

        t0 = time.time()
        first_frame = self.frame_ind == 0
        dri_crop, out_crop, out_org = self.pipe.run(
            img, self.pipe.src_imgs[0], self.pipe.src_infos[0], first_frame=first_frame
        )
        self.frame_ind += 1
        if out_crop is None:
            logger.info(f"No face detected in frame {self.frame_ind} for user {self.stream_handler.user_id}")
            # In case of no output, write the original frame resized
            out_crop = cv2.resize(img, self.stream_handler.video_resolution)
            self.stream_handler.write_video_frame(out_crop)
            return frame

        inference_time = time.time() - t0
        self.infer_times.append(inference_time)
        logger.debug(f"Inference time for user {self.stream_handler.user_id}: {inference_time:.4f} seconds")

        # Ensure out_crop matches the video resolution
        out_crop = cv2.resize(out_crop, self.stream_handler.video_resolution)

        # Write the processed video frame to FFmpeg's video pipe
        self.stream_handler.write_video_frame(out_crop)

        # Optionally, return the processed frame to the WebRTC client
        new_frame = VideoFrame.from_ndarray(out_crop, format="rgb24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def handle_message(self, message):
        """
        Handles messages received via the data channel.
        """
        if message["type"] == "strength":
            # Example: Handle strength adjustments
            pass
        elif message["type"] == "reset":
            # Example: Handle reset commands
            pass

    def stop(self):
        """
        Stops the VideoTransformTrack.
        """
        logger.info(f"Stopping VideoTransformTrack for user {self.stream_handler.user_id}")
        super().stop()


class AudioTransformTrack(MediaStreamTrack):
    """
    Pass-through audio track that writes raw audio data to the StreamHandler's audio pipe.
    """

    kind = "audio"

    def __init__(self, track, stream_handler: StreamHandler):
        super().__init__()
        self.track = track
        self.stream_handler = stream_handler

    async def recv(self):
        """
        Receives an audio frame and writes it to the audio pipe.
        """
        frame = await self.track.recv()
        pcm_data = frame.to_ndarray()

        # Write PCM audio data to the StreamHandler's audio pipe
        self.stream_handler.write_audio_frame(pcm_data)

        # Return the original frame (pass-through)
        return frame

    def handle_message(self, message):
        """
        Handles messages received via the data channel.
        """
        # Currently, no audio processing, so no handling needed
        pass

    def stop(self):
        """
        Stops the AudioTransformTrack.
        """
        logger.info(f"Stopping AudioTransformTrack for user {self.stream_handler.user_id}")
        super().stop()


relay = MediaRelay()
pcs = set()


async def health(request):
    return web.Response(status=200)


async def index(request):
    try:
        content = open("index.html", "r").read()
        return web.Response(content_type="text/html", text=content)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return web.Response(status=500, text="Internal Server Error")


async def offer(request):
    logger.info("Received offer request")
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Generate a unique user_id for this session
    user_id = params.get("userId", str(uuid.uuid4()))
    logger.info(f"Generated user_id: {user_id}")

    # Create a StreamHandler for this user
    stream_handler = StreamHandler(user_id)
    logger.info(f"Created StreamHandler for user {user_id}")

    # Create PeerConnection
    pc = RTCPeerConnection(rtc_configuration)
    pcs.add(pc)
    logger.info(f"Created PeerConnection for user {user_id}")

    local_video = None
    local_audio = None

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state for user {user_id}: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            if local_video:
                local_video.stop()
            if local_audio:
                local_audio.stop()
            stream_handler.stop_ffmpeg()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state for user {user_id}: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            logger.error(f"Connection failed or closed for user {user_id}")
            await pc.close()
            pcs.discard(pc)
            if local_video:
                local_video.stop()
            if local_audio:
                local_audio.stop()
            stream_handler.stop_ffmpeg()

    @pc.on("track")
    def on_track(track):
        nonlocal local_video, local_audio
        logger.info(f"Received track for user {user_id}: {track.kind}")
        if track.kind == "video":
            local_video = VideoTransformTrack(relay.subscribe(track, buffered=False), stream_handler)
            pc.addTrack(local_video)
            logger.info(f"Added VideoTransformTrack for user {user_id}")
        elif track.kind == "audio":
            local_audio = AudioTransformTrack(relay.subscribe(track, buffered=False), stream_handler)
            pc.addTrack(local_audio)
            logger.info(f"Added AudioTransformTrack for user {user_id}")

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            logger.info(f"Datachannel message from user {user_id}: {message}")
            if isinstance(message, str):
                data = json.loads(message)
                if local_video:
                    local_video.handle_message(data)
                if local_audio:
                    local_audio.handle_message(data)

    # Set remote description
    await pc.setRemoteDescription(offer)
    logger.info(f"Set remote description for user {user_id}")

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info(f"Created answer for user {user_id}")

    # Prefer H264 codec if available
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            codecs = RTCRtpSender.getCapabilities("video").codecs
            preferred_codecs = [codec for codec in codecs if codec.mimeType == "video/H264"]
            if preferred_codecs:
                transceiver.setCodecPreferences(preferred_codecs)
                logger.info(f"Set codec preferences to H264 for user {user_id}")

    # Prepare RTMP URL
    host = request.host.split(":")[0]  # e.g., 'localhost' or 'avatar.prod.hypelaunch.io'
    rtmp_url = f"rtmp://{host}:1935/live/{user_id}"
    logger.info(f"User {user_id} RTMP URL: {rtmp_url}")

    # Return the answer and RTMP URL to the client
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "user_id": user_id,
                "stream_url": rtmp_url,
            }
        ),
    )


async def upload_image(request):
    reader = await request.multipart()
    field = await reader.next()
    if field.name == "file":
        filename = field.filename
        extension = filename.split(".")[-1]
        file_path = os.path.join("./images", f"{uuid.uuid4()}.{extension}")
        with open(file_path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        image_map[filename] = file_path
        logger.info(f"Uploaded image: {filename} -> {file_path}")
        return web.Response(
            text=json.dumps(list(image_map.keys())), content_type="application/json"
        )
    logger.warning("No file uploaded in /upload request")
    return web.Response(status=400, text="No file uploaded")


async def update_source_image(request):
    params = await request.json()
    image_key = params.get("image")

    if image_key not in image_map:
        logger.warning(f"Image key not found: {image_key}")
        return web.Response(status=400, text="Image key not found")

    if not os.path.exists(image_map[image_key]):
        logger.warning(f"Source image not found: {image_map[image_key]}")
        return web.Response(status=404, text="Source image not found")

    for pc in pcs:
        for sender in pc.getSenders():
            track = sender.track
            if track and track.kind == "video" and isinstance(track, VideoTransformTrack):
                track.update_source_image(image_key)
                logger.info(
                    f"Updated source image for user {track.stream_handler.user_id} to {image_key}"
                )

    return web.Response(status=200, text="Source image updated")


async def get_available_files(request):
    global image_map
    image_map = create_image_map()
    return web.Response(
        text=json.dumps(list(image_map.keys())), content_type="application/json"
    )


@web.middleware
async def logging_middleware(request, handler):
    logger.info(f"Received request: {request.method} {request.path}")
    response = await handler(request)
    logger.info(f"Sending response: {response.status}")
    return response


async def perf(request):
    try:
        content = open("perf-test.html", "r").read()
        return web.Response(content_type="text/html", text=content)
    except Exception as e:
        logger.error(f"Error serving perf-test.html: {e}")
        return web.Response(status=500, text="Internal Server Error")


async def on_shutdown(app):
    logger.info("Shutting down server")
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
    logger.info(f"Starting server on port {port}")
    web.run_app(app, host="0.0.0.0", port=port)