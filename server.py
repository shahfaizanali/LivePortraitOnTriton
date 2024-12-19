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
