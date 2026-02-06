#!/usr/bin/env python3
import cv2
import subprocess
import os
import sys
import time
import threading
import queue
import numpy as np
from collections import OrderedDict
from tkinter import *
from tkinter.ttk import Separator, Style, Progressbar
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.ttk import Separator, Style, Progressbar
from PIL import Image, ImageTk
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union, Set
from enum import Enum, auto
import traceback
import json
from pathlib import Path
import re
from fractions import Fraction
import math
import signal
from datetime import datetime
import argparse
import shutil
import tempfile
import atexit
import zipfile
import requests
import platform
import glob

DEFAULT_THEME = "dark"
DEFAULT_QUEUE_FILE = "favencoder_queue.json"
ENABLE_LOGGING = False
DEFAULT_VIDEO_CODEC = "H.264 (x264)"
DEFAULT_AUDIO_CODEC = "AAC"
DEFAULT_OUTPUT_FORMAT = "MP4"
DEFAULT_VIDEO_ENCODER_ARGS = ""
DEFAULT_AUDIO_ENCODER_ARGS = ""
PRESETS_FILE = "favencoder_presets.json"

if ENABLE_LOGGING:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('favencoder.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
logger = logging.getLogger(__name__) if ENABLE_LOGGING else None

DISPLAY_MAX_W = 900
DISPLAY_MAX_H = 500

class AppState(Enum):

    NO_VIDEO = auto()
    VIDEO_LOADED = auto()
    PLAYING = auto()
    CONVERTING = auto()
    CONVERSION_PAUSED = auto()
    CROP_MODE = auto()

class VideoCodec(Enum):

    NO_VIDEO = "No video (Audio Only)"
    FFV1 = "FFV1 (Lossless)"
    H264_X264 = "H.264 (x264)"
    H265_X265 = "H.265 (x265)"
    AV1_SVT = "AV1 (SVT-AV1)"
    VP9 = "VP9"
    PRORES = "ProRes"
    DNXHD = "DNxHD"
    MPEG2 = "MPEG-2"
    MJPEG = "MJPEG"
    H264_NVENC = "H.264 (NVENC)"
    H265_NVENC = "H.265 (NVENC)"
    H264_QSV = "H.264 (QSV)"
    H265_QSV = "H.265 (QSV)"
    H264_AMF = "H.264 (AMF)"
    H265_AMF = "H.265 (AMF)"
    AV1_AOM = "AV1 (AOM)"
    VP8 = "VP8"
    MPEG4 = "MPEG-4"
    DV = "DV"
    RAW = "RAW Video"
    CUSTOM = "Custom (Advanced)"

class AudioCodec(Enum):

    NO_AUDIO = "No audio"
    FLAC = "FLAC (Lossless)"
    PCM_16 = "PCM 16-bit"
    PCM_24 = "PCM 24-bit"
    PCM_32 = "PCM 32-bit"
    AAC = "AAC"
    OPUS = "Opus"
    MP3 = "MP3"
    AC3 = "AC3"
    DTS = "DTS"
    VORBIS = "Vorbis"
    CUSTOM = "Custom (Advanced)"

class QualityMode(Enum):

    CQ = "CQ"
    BITRATE = "Bitrate"

class BitrateType(Enum):

    VBR = "VBR"
    CBR = "CBR"

class VideoFormat(Enum):

    MKV = "MKV"
    MP4 = "MP4"
    MOV = "MOV"
    AVI = "AVI"
    WEBM = "WebM"
    FLV = "FLV"
    TS = "TS"
    CUSTOM = "Custom"

class ResolutionMode(Enum):

    ORIGINAL = "Original"
    AUTO_240 = "Auto 240p"
    AUTO_360 = "Auto 360p"
    AUTO_480 = "Auto 480p"
    AUTO_720 = "Auto 720p (HD)"
    AUTO_1080 = "Auto 1080p (Full HD)"
    AUTO_1440 = "Auto 1440p (2K)"
    AUTO_2160 = "Auto 2160p (4K)"
    AUTO_4320 = "Auto 4320p (8K)"
    AI_2X_CPU = "AI 2x Enhancement (CPU)"
    AI_3X_CPU = "AI 3x Enhancement (CPU)"
    AI_4X_CPU = "AI 4x Enhancement (CPU)"
    AI_2X_GPU_ANIME = "AI 2x Enhancement (GPU/anime)"
    AI_3X_GPU_ANIME = "AI 3x Enhancement (GPU/anime)"
    AI_4X_GPU_ANIME = "AI 4x Enhancement (GPU/anime)"
    AI_4X_GPU_GENERAL = "AI 4x Enhancement (GPU/general)"
    CUSTOM = "Custom"
    CUSTOM_WIDTH = "Custom Width"
    CUSTOM_HEIGHT = "Custom Height"

@dataclass
class CropRect:

    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @classmethod
    def from_coords(cls, x1: int, y1: int, x2: int, y2: int) -> 'CropRect':
        return cls(
            x=min(x1, x2),
            y=min(y1, y2),
            width=abs(x2 - x1),
            height=abs(y2 - y1)
        )
    
    def ensure_even_dimensions(self) -> 'CropRect':
        width = self.width - (self.width % 2)
        height = self.height - (self.height % 2)
        width = max(2, width)
        height = max(2, height)
        return CropRect(self.x, self.y, width, height)
    
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0
    
    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 1.0
        return self.width / self.height
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CropRect':
        return cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height']
        )

@dataclass
class ConversionJob:

    id: int
    input_path: str
    output_path: str
    video_settings: 'VideoSettings'
    audio_settings: 'AudioSettings'
    output_settings: 'OutputSettings'
    crop_rect: Optional[CropRect] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    status: str = "Pending"
    progress: float = 0.0
    added_time: datetime = field(default_factory=datetime.now)
    ffmpeg_command: Optional[str] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    video_fps: Optional[float] = None
    
    def get_filename(self):
        return os.path.basename(self.input_path)
    
    def get_status_display(self):
        return f"{self.get_filename()} - {self.status}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'input_path': self.input_path,
            'output_path': self.output_path,
            'video_settings': self.video_settings.to_dict(),
            'audio_settings': self.audio_settings.to_dict(),
            'output_settings': self.output_settings.to_dict(),
            'crop_rect': self.crop_rect.to_dict() if self.crop_rect else None,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'status': self.status,
            'progress': self.progress,
            'added_time': self.added_time.isoformat(),
            'ffmpeg_command': self.ffmpeg_command,
            'video_width': self.video_width,
            'video_height': self.video_height,
            'video_fps': self.video_fps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversionJob':
        return cls(
            id=data['id'],
            input_path=data['input_path'],
            output_path=data['output_path'],
            video_settings=VideoSettings.from_dict(data['video_settings']),
            audio_settings=AudioSettings.from_dict(data['audio_settings']),
            output_settings=OutputSettings.from_dict(data['output_settings']),
            crop_rect=CropRect.from_dict(data['crop_rect']) if data['crop_rect'] else None,
            start_frame=data['start_frame'],
            end_frame=data['end_frame'],
            status=data['status'],
            progress=data['progress'],
            added_time=datetime.fromisoformat(data['added_time']),
            ffmpeg_command=data.get('ffmpeg_command'),
            video_width=data.get('video_width'),
            video_height=data.get('video_height'),
            video_fps=data.get('video_fps')
        )

@dataclass
class VideoSettings:

    codec: VideoCodec = VideoCodec(DEFAULT_VIDEO_CODEC)
    quality_mode: QualityMode = QualityMode.CQ
    cq_value: int = 23
    bitrate: int = 8000
    bitrate_type: BitrateType = BitrateType.VBR
    output_fps: Optional[float] = None
    crop: Optional[CropRect] = None
    resolution_mode: ResolutionMode = ResolutionMode.ORIGINAL
    custom_width: Optional[int] = None
    custom_height: Optional[int] = None
    custom_dimension_value: Optional[int] = None
    speed_preset: Optional[str] = "medium"
    custom_encoder: Optional[str] = None
    custom_encoder_args: Optional[str] = DEFAULT_VIDEO_ENCODER_ARGS
    
    def calculate_output_dimensions(self, original_width: int, original_height: int, crop_rect: Optional[CropRect] = None) -> Tuple[int, int]:
        if crop_rect:
            src_width = crop_rect.width
            src_height = crop_rect.height
        else:
            src_width = original_width
            src_height = original_height
        
        if src_width == 0 or src_height == 0:
            return original_width, original_height
        
        aspect_ratio = src_width / src_height
        
        if self.resolution_mode == ResolutionMode.ORIGINAL:
            return src_width, src_height
        
        elif self.resolution_mode == ResolutionMode.CUSTOM:
            if self.custom_width and self.custom_height:
                width = self.custom_width - (self.custom_width % 2)
                height = self.custom_height - (self.custom_height % 2)
                return max(2, width), max(2, height)
            return src_width, src_height
        
        elif self.resolution_mode == ResolutionMode.CUSTOM_WIDTH and self.custom_dimension_value:
            width = self.custom_dimension_value - (self.custom_dimension_value % 2)
            width = max(2, width)
            height = int(width / aspect_ratio)
            height = height - (height % 2)
            height = max(2, height)
            return width, height
        
        elif self.resolution_mode == ResolutionMode.CUSTOM_HEIGHT and self.custom_dimension_value:
            height = self.custom_dimension_value - (self.custom_dimension_value % 2)
            height = max(2, height)
            width = int(height * aspect_ratio)
            width = width - (width % 2)
            width = max(2, width)
            return width, height
        
        elif self.resolution_mode in [ResolutionMode.AI_2X_CPU, ResolutionMode.AI_2X_GPU_ANIME]:
            width = src_width * 2
            height = src_height * 2
            width = width - (width % 2)
            height = height - (height % 2)
            return max(2, width), max(2, height)
        
        elif self.resolution_mode in [ResolutionMode.AI_3X_CPU, ResolutionMode.AI_3X_GPU_ANIME]:
            width = src_width * 3
            height = src_height * 3
            width = width - (width % 2)
            height = height - (height % 2)
            return max(2, width), max(2, height)
        
        elif self.resolution_mode in [ResolutionMode.AI_4X_CPU, ResolutionMode.AI_4X_GPU_ANIME, ResolutionMode.AI_4X_GPU_GENERAL]:
            width = src_width * 4
            height = src_height * 4
            width = width - (width % 2)
            height = height - (height % 2)
            return max(2, width), max(2, height)
        
        else:
            target_height = 0
            if self.resolution_mode == ResolutionMode.AUTO_240:
                target_height = 240
            elif self.resolution_mode == ResolutionMode.AUTO_360:
                target_height = 360
            elif self.resolution_mode == ResolutionMode.AUTO_480:
                target_height = 480
            elif self.resolution_mode == ResolutionMode.AUTO_720:
                target_height = 720
            elif self.resolution_mode == ResolutionMode.AUTO_1080:
                target_height = 1080
            elif self.resolution_mode == ResolutionMode.AUTO_1440:
                target_height = 1440
            elif self.resolution_mode == ResolutionMode.AUTO_2160:
                target_height = 2160
            elif self.resolution_mode == ResolutionMode.AUTO_4320:
                target_height = 4320
            
            if target_height > 0:
                if src_width >= src_height:
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
                
                new_width = new_width - (new_width % 2)
                new_height = new_height - (new_height % 2)
                new_width = max(2, new_width)
                new_height = max(2, new_height)
                
                return new_width, new_height
        
        return src_width, src_height
    
    def is_ai_enhancement(self) -> bool:
        return self.resolution_mode in [
            ResolutionMode.AI_2X_CPU, ResolutionMode.AI_3X_CPU, ResolutionMode.AI_4X_CPU,
            ResolutionMode.AI_2X_GPU_ANIME, ResolutionMode.AI_3X_GPU_ANIME, ResolutionMode.AI_4X_GPU_ANIME,
            ResolutionMode.AI_4X_GPU_GENERAL
        ]
    
    def is_gpu_ai_enhancement(self) -> bool:
        return self.resolution_mode in [
            ResolutionMode.AI_2X_GPU_ANIME, ResolutionMode.AI_3X_GPU_ANIME, ResolutionMode.AI_4X_GPU_ANIME,
            ResolutionMode.AI_4X_GPU_GENERAL
        ]
    
    def get_ai_scale_factor(self) -> int:
        if self.resolution_mode in [ResolutionMode.AI_2X_CPU, ResolutionMode.AI_2X_GPU_ANIME]:
            return 2
        elif self.resolution_mode in [ResolutionMode.AI_3X_CPU, ResolutionMode.AI_3X_GPU_ANIME]:
            return 3
        elif self.resolution_mode in [ResolutionMode.AI_4X_CPU, ResolutionMode.AI_4X_GPU_ANIME, ResolutionMode.AI_4X_GPU_GENERAL]:
            return 4
        return 1
    
    def get_ai_model_type(self) -> str:
        """Get the model type for Real-ESRGAN based on resolution mode"""
        if self.resolution_mode in [ResolutionMode.AI_2X_GPU_ANIME, ResolutionMode.AI_3X_GPU_ANIME, ResolutionMode.AI_4X_GPU_ANIME]:
            return "anime"
        elif self.resolution_mode in [ResolutionMode.AI_4X_GPU_GENERAL]:
            return "general"
        else:
            return "anime"  # Default for CPU modes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'codec': self.codec.value,
            'quality_mode': self.quality_mode.value,
            'cq_value': self.cq_value,
            'bitrate': self.bitrate,
            'bitrate_type': self.bitrate_type.value,
            'output_fps': self.output_fps,
            'crop': self.crop.to_dict() if self.crop else None,
            'resolution_mode': self.resolution_mode.value,
            'custom_width': self.custom_width,
            'custom_height': self.custom_height,
            'custom_dimension_value': self.custom_dimension_value,
            'speed_preset': self.speed_preset,
            'custom_encoder': self.custom_encoder,
            'custom_encoder_args': self.custom_encoder_args
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoSettings':
        # Handle backward compatibility for removed AI GPU options
        resolution_mode_value = data['resolution_mode']
        # Map old AI GPU options to their anime equivalents
        old_to_new_mapping = {
            "AI 2x Enhancement (GPU)": "AI 2x Enhancement (GPU/anime)",
            "AI 3x Enhancement (GPU)": "AI 3x Enhancement (GPU/anime)",
            "AI 4x Enhancement (GPU)": "AI 4x Enhancement (GPU/anime)"
        }
        if resolution_mode_value in old_to_new_mapping:
            resolution_mode_value = old_to_new_mapping[resolution_mode_value]
        
        return cls(
            codec=VideoCodec(data['codec']),
            quality_mode=QualityMode(data['quality_mode']),
            cq_value=data['cq_value'],
            bitrate=data['bitrate'],
            bitrate_type=BitrateType(data['bitrate_type']),
            output_fps=data['output_fps'],
            crop=CropRect.from_dict(data['crop']) if data['crop'] else None,
            resolution_mode=ResolutionMode(resolution_mode_value),
            custom_width=data['custom_width'],
            custom_height=data['custom_height'],
            custom_dimension_value=data['custom_dimension_value'],
            speed_preset=data['speed_preset'],
            custom_encoder=data['custom_encoder'],
            custom_encoder_args=data['custom_encoder_args']
        )

@dataclass
class AudioSettings:

    codec: AudioCodec = AudioCodec(DEFAULT_AUDIO_CODEC)
    bitrate: int = 256
    samplerate: int = 48000
    custom_encoder: Optional[str] = None
    custom_encoder_args: Optional[str] = DEFAULT_AUDIO_ENCODER_ARGS
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'codec': self.codec.value,
            'bitrate': self.bitrate,
            'samplerate': self.samplerate,
            'custom_encoder': self.custom_encoder,
            'custom_encoder_args': self.custom_encoder_args
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioSettings':
        return cls(
            codec=AudioCodec(data['codec']),
            bitrate=data['bitrate'],
            samplerate=data['samplerate'],
            custom_encoder=data['custom_encoder'],
            custom_encoder_args=data['custom_encoder_args']
        )

@dataclass
class OutputSettings:

    format: VideoFormat = VideoFormat(DEFAULT_OUTPUT_FORMAT)
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    custom_format: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'format': self.format.value,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'custom_format': self.custom_format
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputSettings':
        return cls(
            format=VideoFormat(data['format']),
            start_frame=data['start_frame'],
            end_frame=data['end_frame'],
            custom_format=data.get('custom_format')
        )

@dataclass
class ConversionProgress:

    current_time: float = 0.0
    total_time: float = 0.0
    percentage: float = 0.0
    speed: float = 0.0
    fps: float = 0.0
    elapsed_time: float = 0.0
    estimated_total_time: float = 0.0
    estimated_remaining: float = 0.0

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def format_time_compact(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_ffmpeg_progress(line: str) -> Optional[ConversionProgress]:
    progress = ConversionProgress()
    
    time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2}\.\d+)', line)
    if time_match:
        h, m, s = time_match.groups()
        progress.current_time = int(h) * 3600 + int(m) * 60 + float(s)
    
    speed_match = re.search(r'speed=([\d\.]+)x', line)
    if speed_match:
        progress.speed = float(speed_match.group(1))
    
    fps_match = re.search(r'fps=([\d\.]+)', line)
    if fps_match:
        progress.fps = float(fps_match.group(1))
    
    if progress.current_time > 0:
        return progress
    
    return None

def simplify_aspect_ratio(width: int, height: int) -> str:
    if width == 0 or height == 0:
        return "N/A"
    
    try:
        ratio = Fraction(width, height)
        
        common_ratios = {
            (16, 9): "16:9",
            (4, 3): "4:3",
            (1, 1): "1:1",
            (3, 2): "3:2",
            (5, 4): "5:4",
            (2, 1): "2:1",
            (21, 9): "21:9",
            (32, 9): "32:9"
        }
        
        for (w, h), name in common_ratios.items():
            if abs(ratio - Fraction(w, h)) < 0.01:
                return name
        
        return f"{ratio.numerator}:{ratio.denominator}"
    except:
        return f"{width}:{height}"

class LRUCache:

    def __init__(self, maxsize: int = 50):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key: Any) -> Any:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        for img in self.cache.values():
            if hasattr(img, 'close'):
                img.close()
        self.cache.clear()

class VideoPlayer:

    def __init__(self, video_path: str, fps: float, start_frame: int = 0):
        self.video_path = video_path
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.playing = False
        self.current_frame = start_frame
        self.stop_event = threading.Event()
        self.playback_thread = None
        
        self.frame_queue = queue.Queue(maxsize=30)
        self.frame_available = threading.Event()
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = self.width * self.height * 3
        
        self.cap.release()
        self.cap = None
    
    def start_playback(self) -> None:
        if self.playing:
            return
            
        self.playing = True
        self.stop_event.clear()
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self.playback_thread.start()
    
    def stop_playback(self) -> None:
        self.playing = False
        self.stop_event.set()
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_frame(self) -> Optional[Tuple[Image.Image, int]]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _playback_worker(self) -> None:
        process = None
        try:
            start_time = self.current_frame / self.fps
            
            cmd = [
                'ffmpeg',
                '-ss', f"{start_time:.6f}",
                '-i', self.video_path,
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo',
                '-vsync', '0',
                '-'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1024 * 1024,
                stdin=subprocess.DEVNULL
            )
            
            frame_count = self.current_frame
            last_frame_time = time.time()
            
            while self.playing and not self.stop_event.is_set():
                next_frame_time = last_frame_time + self.frame_interval
                
                try:
                    raw_frame = process.stdout.read(self.frame_size)
                except (IOError, ValueError) as e:
                    if ENABLE_LOGGING: logger.error(f"Error reading frame: {e}")
                    break
                    
                if len(raw_frame) != self.frame_size:
                    break
                
                try:
                    frame = np.frombuffer(raw_frame, np.uint8).reshape(
                        (self.height, self.width, 3)
                    )
                    img = Image.fromarray(frame)
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Error converting frame: {e}")
                    break
                
                try:
                    self.frame_queue.put((img, frame_count), timeout=0.1)
                except queue.Full:
                    if ENABLE_LOGGING: logger.warning("Frame queue full, dropping frame")
                    continue
                
                frame_count += 1
                
                current_time = time.time()
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_frame_time = next_frame_time if sleep_time > 0 else current_time
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"Playback error: {e}")
                logger.error(traceback.format_exc())
        finally:
            if process:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Error cleaning up FFmpeg process: {e}")
            
            self.playing = False
    
    def seek(self, frame_num: int) -> None:
        self.current_frame = max(0, min(frame_num, self.total_frames - 1))
        
        if self.playing:
            self.stop_playback()
            time.sleep(0.05)
            self.start_playback()

class VideoCropper:

    DEFAULT_THEME = DEFAULT_THEME
    MIN_CROP_SIZE = 10
    QUEUE_FILE = DEFAULT_QUEUE_FILE
    
    CODEC_PAIRINGS = {
        VideoCodec.AV1_SVT: AudioCodec.OPUS,
        VideoCodec.AV1_AOM: AudioCodec.OPUS,
        
        VideoCodec.H265_X265: AudioCodec.AAC,
        VideoCodec.H265_NVENC: AudioCodec.AAC,
        VideoCodec.H265_QSV: AudioCodec.AAC,
        VideoCodec.H265_AMF: AudioCodec.AAC,
        
        VideoCodec.H264_X264: AudioCodec.AAC,
        VideoCodec.H264_NVENC: AudioCodec.AAC,
        VideoCodec.H264_QSV: AudioCodec.AAC,
        VideoCodec.H264_AMF: AudioCodec.AAC,
        
        VideoCodec.VP9: AudioCodec.OPUS,
        VideoCodec.VP8: AudioCodec.VORBIS,
        
        VideoCodec.FFV1: AudioCodec.FLAC,
        VideoCodec.RAW: AudioCodec.PCM_16,
        
        VideoCodec.PRORES: AudioCodec.PCM_24,
        VideoCodec.DNXHD: AudioCodec.PCM_16,
        
        VideoCodec.MPEG2: AudioCodec.MP3,
        VideoCodec.MJPEG: AudioCodec.PCM_16,
        VideoCodec.MPEG4: AudioCodec.AAC,
        VideoCodec.DV: AudioCodec.PCM_16,
        VideoCodec.CUSTOM: AudioCodec.AAC,
    }
    
    DEFAULT_PAIRING = AudioCodec.AAC
    
    ENCODER_PRESETS = {
        VideoCodec.H264_X264: ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"],
        VideoCodec.H265_X265: ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "veryslow", "placebo"],
        
        VideoCodec.AV1_SVT: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        VideoCodec.AV1_AOM: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        
        VideoCodec.VP9: ["0", "1", "2", "3", "4", "5"],
        
        VideoCodec.H264_NVENC: ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        VideoCodec.H265_NVENC: ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        
        VideoCodec.H264_QSV: ["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        VideoCodec.H265_QSV: ["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        
        VideoCodec.H264_AMF: ["speed", "balanced", "quality"],
        VideoCodec.H265_AMF: ["speed", "balanced", "quality"],
        
        VideoCodec.VP8: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"],
        VideoCodec.MPEG4: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        VideoCodec.MJPEG: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    }
    
    LONGEST_STATUS_MESSAGES = [
        "Crop mode: Drag to select area. Use handles to resize.",
        "Conversion complete!",
        "Conversion failed: FFmpeg returned error code 1",
        "Conversion stopped by user",
        "Ready",
        "Converting...",
        "Conversion Paused"
    ]
    
    def __init__(self, root: Tk = None):
        self.root = root
        self.no_gui_mode = root is None
        # Initialize tkinter variables for GUI mode
        if not self.no_gui_mode:
            self.video_args_var = StringVar()
            self.audio_args_var = StringVar()
            self.video_encoder_var = StringVar()
            self.audio_encoder_var = StringVar()
            self.cq_var = StringVar(value="23")
            self.bitrate_var = StringVar(value="8000")
            self.audio_bitrate_var = StringVar(value="256")
            self.samplerate_var = StringVar(value="48000")
            self.fps_var = StringVar()
            self.speed_var = StringVar(value="medium")
            self.custom_output_var = StringVar()
        
        if not self.no_gui_mode:
            self.root.title("FAVencoder - Frame-accurate Video Encoder")
            self.root.geometry("1280x720")
            self.root.resizable(True, True)
        
        self.state = AppState.NO_VIDEO
        
        self.frame_lock = threading.Lock()
        self.playback_lock = threading.Lock()
        self.conversion_lock = threading.Lock()
        
        self.video_settings = VideoSettings()
        self.audio_settings = AudioSettings()
        self.output_settings = OutputSettings()
        
        if self.audio_settings.codec != AudioCodec.CUSTOM:
                self.audio_codec_manually_changed = False
        
        self.custom_resolutions_history = []
        self.max_custom_history = 10
        
        self.conversion_queue = []
        self.next_job_id = 1
        self.current_job = None
        self.queue_window = None
        
        self.themes = {
            "grey": {
                "bg_color": "#f0f0f0",
                "fg_color": "#000000",
                "entry_bg": "#ffffff",
                "entry_fg": "#000000",
                "button_bg": "#e0e0e0",
                "button_fg": "#000000",
                "highlight_bg": "#d0d0d0",
                "canvas_bg": "#000000",
                "status_bg": "#d0d0d0",
                "status_fg": "#000000",
                "accent_color": "#4CAF50",
                "accent_color2": "#2196F3",
                "border_color": "#c0c0c0",
                "menu_bg": "#ffffff",
                "menu_fg": "#000000",
                "menu_active_bg": "#d0d0d0",
                "menu_active_fg": "#000000",
                "progress_bg": "#c0c0c0",
                "progress_fg": "#4CAF50",
                "progress_text_fg": "#000000",
                "dropdown_bg": "#ffffff",
                "dropdown_fg": "#000000",
                "tree_bg": "#ffffff",
                "tree_fg": "#000000",
                "tree_field_bg": "#f0f0f0",
                "tree_selected_bg": "#d0d0d0",
                "tree_selected_fg": "#000000"
            },
            "dark": {
                "bg_color": "#1a1a1a",
                "fg_color": "#ffffff",
                "entry_bg": "#2d2d2d",
                "entry_fg": "#ffffff",
                "button_bg": "#3a3a3a",
                "button_fg": "#ffffff",
                "highlight_bg": "#4a4a4a",
                "canvas_bg": "#000000",
                "status_bg": "#0f0f0f",
                "status_fg": "#cccccc",
                "accent_color": "#4CAF50",
                "accent_color2": "#2196F3",
                "border_color": "#333333",
                "menu_bg": "#2d2d2d",
                "menu_fg": "#ffffff",
                "menu_active_bg": "#4a4a4a",
                "menu_active_fg": "#ffffff",
                "progress_bg": "#333333",
                "progress_fg": "#4CAF50",
                "progress_text_fg": "#ffffff",
                "dropdown_bg": "#2d2d2d",
                "dropdown_fg": "#ffffff",
                "tree_bg": "#2d2d2d",
                "tree_fg": "#ffffff",
                "tree_field_bg": "#3a3a3a",
                "tree_selected_bg": "#4a4a4a",
                "tree_selected_fg": "#ffffff"
            },
            "white": {
                "bg_color": "#ffffff",
                "fg_color": "#000000",
                "entry_bg": "#f5f5f5",
                "entry_fg": "#000000",
                "button_bg": "#e8e8e8",
                "button_fg": "#000000",
                "highlight_bg": "#d8d8d8",
                "canvas_bg": "#000000",
                "status_bg": "#e8e8e8",
                "status_fg": "#000000",
                "accent_color": "#4CAF50",
                "accent_color2": "#2196F3",
                "border_color": "#d0d0c0",
                "menu_bg": "#f5f5f5",
                "menu_fg": "#000000",
                "menu_active_bg": "#d8d8d8",
                "menu_active_fg": "#000000",
                "progress_bg": "#e0e0e0",
                "progress_fg": "#4CAF50",
                "progress_text_fg": "#000000",
                "dropdown_bg": "#f5f5f5",
                "dropdown_fg": "#000000",
                "tree_bg": "#ffffff",
                "tree_fg": "#000000",
                "tree_field_bg": "#f0f0f0",
                "tree_selected_bg": "#d0d0d0",
                "tree_selected_fg": "#000000"
            }
        }
        
        self.current_theme = self.DEFAULT_THEME
        
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 25.0
        self.video_width = 0
        self.video_height = 0
        self.duration = 0.0
        self.original_aspect_ratio: float = 1.0
        
        self.playing = False
        self.player: Optional[VideoPlayer] = None
        self.playback_update_id: Optional[str] = None
        
        self.crop_mode = False
        self.crop_rect: Optional[CropRect] = None
        self.crop_rect_id: Optional[int] = None
        self.crop_handles: List[int] = []
        self.crop_start: Optional[Tuple[int, int]] = None
        self.crop_move_mode = False
        self.crop_resize_mode = False
        self.crop_resize_corner: Optional[str] = None
        self.crop_original_coords: Optional[Tuple[float, ...]] = None
        
        self.display_img_x = 0
        self.display_img_y = 0
        self.display_img_w = 0
        self.display_img_h = 0
        self.aspect_ratio: Optional[float] = None
        
        self.frame_cache = LRUCache(maxsize=50)
        
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        
        self.conversion_thread: Optional[threading.Thread] = None
        self.conversion_process: Optional[subprocess.Popen] = None
        self.conversion_stop_event = threading.Event()
        self.conversion_pause_event = threading.Event()
        self.conversion_paused = False
        
        self.segment_duration = 0.0
        self.conversion_start_time = 0.0
        self.last_progress_update_time = 0.0
        self.last_percentage = 0.0
        
        self.current_context_menu = None
        
        # AI enhancement variables
        self.realesrgan_available = False
        self.realesrgan_vulkan_available = False
        self.super_image_available = False
        self.ai_initialized = False
        
        # Real-ESRGAN executable paths
        self.realesrgan_dir = os.path.join(os.path.expanduser("~"), ".favencoder", "realesrgan")
        self.realesrgan_exe = None
        self.realesrgan_downloaded = False
        
        # Real-ESRGAN process reference for pausing
        self.realesrgan_process = None
        
        # Clean up temporary folders from previous sessions
        self.cleanup_temp_folders()
        
        
        self.ai_model = None
        self.realesrgan_models = {}
        
        if not self.no_gui_mode:
            self.style = Style()
            self.apply_theme(self.current_theme)
            self.build_ui()
            
            self.root.bind("<Configure>", self.on_window_resize)
            self.root.bind("<Left>", lambda e: self.prev_frame())
            self.root.bind("<Right>", lambda e: self.next_frame())
            self.root.bind("<space>", lambda e: self.toggle_play())
            self.root.bind("s", lambda e: self.set_start())
            self.root.bind("e", lambda e: self.set_end())
            self.root.bind("<Escape>", lambda e: self.cancel_crop_mode())
            
            self.root.bind("<Button-1>", self.close_context_menu)
            
            self.enable_drag_drop()
        
        self.load_queue()
    
    def cleanup_temp_folders(self):
        """Clean up temporary folders from previous AI enhancement sessions"""
        try:
            temp_dir = tempfile.gettempdir()
            
            # Look for Real-ESRGAN temporary folders
            realesrgan_patterns = [
                os.path.join(temp_dir, "realesrgan_*"),
                os.path.join(temp_dir, "favencoder_ai_*"),
                os.path.join(temp_dir, "*realesrgan*"),
            ]
            
            # Look for super-image temporary folders
            super_image_patterns = [
                os.path.join(temp_dir, "super_image_*"),
                os.path.join(temp_dir, "*superimage*"),
            ]
            
            all_patterns = realesrgan_patterns + super_image_patterns
            
            cleaned_folders = []
            for pattern in all_patterns:
                for folder in glob.glob(pattern):
                    if os.path.isdir(folder):
                        try:
                            shutil.rmtree(folder, ignore_errors=True)
                            cleaned_folders.append(os.path.basename(folder))
                            if ENABLE_LOGGING: logger.info(f"Cleaned temporary folder: {folder}")
                        except Exception as e:
                            if ENABLE_LOGGING: logger.warning(f"Failed to clean folder {folder}: {e}")
            
            if cleaned_folders and ENABLE_LOGGING:
                logger.info(f"Cleaned {len(cleaned_folders)} temporary AI enhancement folders")
                
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error cleaning temporary folders: {e}")
    
    def _download_realesrgan(self):
        """Download Real-ESRGAN executable from GitHub releases"""
        try:
            os.makedirs(self.realesrgan_dir, exist_ok=True)
            
            # Determine OS and appropriate download
            system = platform.system()
            if system == "Windows":
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
                exe_name = "realesrgan-ncnn-vulkan.exe"
            elif system == "Linux":
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
                exe_name = "realesrgan-ncnn-vulkan"
            elif system == "Darwin":  # macOS
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
                exe_name = "realesrgan-ncnn-vulkan"
            else:
                if ENABLE_LOGGING: logger.error(f"Unsupported OS: {system}")
                return False
            
            zip_path = os.path.join(self.realesrgan_dir, "realesrgan.zip")
            
            # Download the zip file
            if ENABLE_LOGGING: logger.info(f"Downloading Real-ESRGAN from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            if ENABLE_LOGGING: logger.info(f"Extracting Real-ESRGAN to {self.realesrgan_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.realesrgan_dir)
            
            # Find the executable
            for root, dirs, files in os.walk(self.realesrgan_dir):
                for file in files:
                    if file == exe_name:
                        self.realesrgan_exe = os.path.join(root, file)
                        break
            
            # Clean up zip file
            os.remove(zip_path)
            
            # Make executable on Linux/macOS
            if system in ["Linux", "Darwin"] and self.realesrgan_exe:
                os.chmod(self.realesrgan_exe, 0o755)
            
            self.realesrgan_downloaded = True
            
            # Download models if they don't exist
            models_dir = os.path.join(self.realesrgan_dir, "models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
                
                model_urls = [
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3-x2.bin",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3-x2.param",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3-x3.bin",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3-x3.param",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3-x4.bin",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3-x4.param",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus-anime.bin",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus-anime.param",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.bin",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.param"
                ]
                
                for model_url in model_urls:
                    model_name = os.path.basename(model_url)
                    model_path = os.path.join(models_dir, model_name)
                    if not os.path.exists(model_path):
                        if ENABLE_LOGGING: logger.info(f"Downloading model: {model_name}")
                        response = requests.get(model_url)
                        with open(model_path, 'wb') as f:
                            f.write(response.content)
            
            if self.realesrgan_exe and os.path.exists(self.realesrgan_exe):
                if ENABLE_LOGGING: logger.info(f"Real-ESRGAN executable found at: {self.realesrgan_exe}")
                return True
            else:
                if ENABLE_LOGGING: logger.error(f"Failed to find Real-ESRGAN executable after download")
                return False
                
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to download Real-ESRGAN: {e}")
            return False
    
    def _check_ai_availability(self):
        """Check which AI enhancement backends are available"""
        # Check for super-image (CPU)
        try:
            import super_image
            from super_image import EdsrModel
            self.super_image_available = True
            if ENABLE_LOGGING: logger.info("super-image is available")
        except ImportError:
            if ENABLE_LOGGING: logger.info("super-image is not available")
            self.super_image_available = False
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error checking super-image: {e}")
            self.super_image_available = False
        
        # Check for Real-ESRGAN executable (GPU)
        self.realesrgan_available = False
        self.realesrgan_vulkan_available = False
        
        # Disable GPU AI enhancement on ARM
        import platform
        machine = platform.machine().lower()
        if machine in ['aarch64', 'arm64', 'armv8l', 'armv7l', 'arm']:
            if ENABLE_LOGGING: logger.info(f"ARM architecture detected ({machine}), disabling GPU AI enhancement")
            return
        
        # First check if we already have it downloaded
        if not self.realesrgan_exe or not os.path.exists(self.realesrgan_exe):
            # Try to find it in the standard location
            system = platform.system()
            if system == "Windows":
                exe_name = "realesrgan-ncnn-vulkan.exe"
            else:
                exe_name = "realesrgan-ncnn-vulkan"
            
            # Check in the realesrgan directory
            for root, dirs, files in os.walk(self.realesrgan_dir):
                for file in files:
                    if file == exe_name:
                        self.realesrgan_exe = os.path.join(root, file)
                        break
        
        # If not found, we'll download it when needed
        if self.realesrgan_exe and os.path.exists(self.realesrgan_exe):
            self.realesrgan_available = True
            # Test if Vulkan is available by running the executable with --help
            try:
                result = subprocess.run([self.realesrgan_exe, "--help"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or "Usage:" in result.stdout or "Usage:" in result.stderr:
                    self.realesrgan_vulkan_available = True
                    if ENABLE_LOGGING: logger.info("Real-ESRGAN executable with Vulkan is available")
            except Exception as e:
                if ENABLE_LOGGING: logger.info(f"Real-ESRGAN available but Vulkan may not be: {e}")
                self.realesrgan_vulkan_available = False
    
    def _initialize_ai_model(self, use_gpu: bool = False) -> None:
        """Initialize AI model based on availability and requested backend"""
        self._check_ai_availability()
        
        scale_factor = self.video_settings.get_ai_scale_factor()
        is_gpu_mode = use_gpu or self.video_settings.is_gpu_ai_enhancement()
        
        # Check if we're on ARM64 and trying to use GPU
        import platform
        machine = platform.machine().lower()
        if is_gpu_mode and machine in ['aarch64', 'arm64', 'armv8l', 'armv7l']:
            if ENABLE_LOGGING: logger.warning("GPU AI enhancement requested on ARM64, forcing CPU mode")
            is_gpu_mode = False
            # Update the resolution mode to CPU equivalent
            if self.video_settings.resolution_mode == ResolutionMode.AI_2X_GPU_ANIME:
                self.video_settings.resolution_mode = ResolutionMode.AI_2X_CPU
            elif self.video_settings.resolution_mode == ResolutionMode.AI_3X_GPU_ANIME:
                self.video_settings.resolution_mode = ResolutionMode.AI_3X_CPU
            elif self.video_settings.resolution_mode == ResolutionMode.AI_4X_GPU_ANIME:
                self.video_settings.resolution_mode = ResolutionMode.AI_4X_CPU
            elif self.video_settings.resolution_mode == ResolutionMode.AI_4X_GPU_GENERAL:
                self.video_settings.resolution_mode = ResolutionMode.AI_4X_CPU
        
        # Try to use Real-ESRGAN executable if requested
        if is_gpu_mode:
            if not self.realesrgan_available:
                # Try to download it
                if ENABLE_LOGGING: logger.info("Real-ESRGAN not available, attempting to download...")
                if self._download_realesrgan():
                    self._check_ai_availability()
                else:
                    if ENABLE_LOGGING: logger.error("Failed to download Real-ESRGAN")
            
            if self.realesrgan_available and self.realesrgan_exe:
                self.ai_initialized = True
                self.ai_model = None  # Clear super-image model if it was loaded
                
                backend = "GPU (Vulkan)" if self.realesrgan_vulkan_available else "GPU (CPU fallback)"
                if ENABLE_LOGGING: logger.info(f"AI initialized with Real-ESRGAN executable ({backend}) for scale {scale_factor}")
                return
                
            else:
                if ENABLE_LOGGING: logger.warning("GPU acceleration requested but Real-ESRGAN not available")
                # Fall through to super-image
        
        # Try to use super-image for CPU or as fallback
        if self.super_image_available:
            try:
                from super_image import EdsrModel
                
                # Load appropriate model based on scale factor
                if scale_factor == 2:
                    self.ai_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
                elif scale_factor == 3:
                    # For 3x, we'll use 2x model and resize
                    self.ai_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
                elif scale_factor == 4:
                    # For 4x, we'll apply 2x twice
                    self.ai_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
                
                self.ai_initialized = True
                if ENABLE_LOGGING: logger.info(f"AI initialized with super-image (CPU) for scale {scale_factor}")
                return
                
            except Exception as e:
                if ENABLE_LOGGING: logger.error(f"Failed to initialize super-image model: {e}")
        
        # If neither works, show error
        if not self.ai_initialized:
            error_msg = "Failed to initialize AI enhancement.\n\n"
            if is_gpu_mode:
                error_msg += "GPU acceleration requested but not available.\n"
                error_msg += "The script will attempt to download Real-ESRGAN executable automatically.\n\n"
            error_msg += "For CPU-only enhancement, install super-image:\n"
            error_msg += "pip install super-image\n\n"
            error_msg += "Note: For pipx installation, use:\n"
            error_msg += "pipx install super-image"
            
            messagebox.showerror("AI Enhancement Error", error_msg)
            raise RuntimeError("AI model not initialized")
    
    def _enhance_image_with_ai(self, img: Image.Image, scale_factor: int) -> Image.Image:
        """Enhance image using AI with the best available backend"""
        if not self.ai_initialized:
            raise RuntimeError("AI model not initialized")
        
        is_gpu_mode = self.video_settings.is_gpu_ai_enhancement()
        
        # Try Real-ESRGAN executable first if available and GPU mode requested
        if is_gpu_mode and self.realesrgan_available and self.realesrgan_exe:
            try:
                # Determine video directory for temporary files
                video_dir = os.path.dirname(self.video_path) if hasattr(self, 'video_path') and self.video_path else None
                if video_dir and os.path.exists(video_dir):
                    # Use video directory for temp files
                    temp_dir_context = tempfile.TemporaryDirectory(prefix="favencoder_ai_", dir=video_dir)
                else:
                    # Fallback to system temp directory
                    temp_dir_context = tempfile.TemporaryDirectory(prefix="favencoder_ai_")
                temp_dir = temp_dir_context.name
                
                try:
                    # Save input image
                    input_path = os.path.join(temp_dir, "input.png")
                    output_path = os.path.join(temp_dir, "output.png")
                    img.save(input_path)
                    
                    # Determine model based on scale factor and model type
                    model_type = self.video_settings.get_ai_model_type()
                    
                    if model_type == "anime":
                        if scale_factor == 2:
                            model = "realesr-animevideov3-x2"
                        elif scale_factor == 3:
                            model = "realesr-animevideov3-x3"
                        elif scale_factor == 4:
                            model = "realesr-animevideov3-x4"
                        else:
                            model = "realesrgan-x4plus-anime"
                    else:  # general
                        if scale_factor == 4:
                            model = "realesrgan-x4plus"
                        else:
                            # For 2x and 3x general, use 4x model (will be scaled appropriately)
                            model = "realesrgan-x4plus"
                    
                    # Build command
                    cmd = [
                        self.realesrgan_exe,
                        "-i", input_path,
                        "-o", output_path,
                        "-s", str(scale_factor),
                        "-n", model
                    ]
                    
                    # Run Real-ESRGAN
                    if ENABLE_LOGGING: logger.info(f"Running Real-ESRGAN command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and os.path.exists(output_path):
                        enhanced_img = Image.open(output_path).convert('RGB')
                        return enhanced_img
                    else:
                        if ENABLE_LOGGING: logger.error(f"Real-ESRGAN failed: {result.stderr}")
                        raise Exception(f"Real-ESRGAN failed: {result.stderr}")
                        
                finally:
                    temp_dir_context.cleanup()
            except Exception as e:
                if ENABLE_LOGGING: logger.error(f"Real-ESRGAN enhancement failed: {e}")
                # Fall through to super-image
        
        # Try super-image (CPU)
        if self.ai_model and self.super_image_available:
            try:
                from super_image import ImageLoader
                
                inputs = ImageLoader.load_image(img)
                preds = self.ai_model(inputs)
                enhanced_img = ImageLoader.save_image(preds)
                
                # Handle scale factors not directly supported
                if scale_factor == 3:
                    # For 3x, enhance to 2x then resize to 3x
                    enhanced_img = enhanced_img.resize(
                        (enhanced_img.width * 3 // 2, enhanced_img.height * 3 // 2),
                        Image.Resampling.BILINEAR
                    )
                elif scale_factor == 4:
                    # For 4x, apply 2x enhancement twice
                    inputs2 = ImageLoader.load_image(enhanced_img)
                    preds2 = self.ai_model(inputs2)
                    enhanced_img = ImageLoader.save_image(preds2)
                
                return enhanced_img
                
            except Exception as e:
                if ENABLE_LOGGING: logger.error(f"Super-image enhancement failed: {e}")
                raise
        
        # If neither works, do simple resize with warning
        if ENABLE_LOGGING: logger.warning("AI enhancement failed, using simple resize")
        new_width = img.width * scale_factor
        new_height = img.height * scale_factor
        return img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    def enable_drag_drop(self):
        try:
            try:
                from tkinterdnd2 import DND_FILES, TkinterDnD
                if not isinstance(self.root, TkinterDnD.Tk):
                    self.root.destroy()
                    self.root = TkinterDnD.Tk()
                    self.root.title("FAVencoder - Frame-accurate Video Encoder")
                    self.root.geometry("1280x720")
                    self.root.resizable(True, True)
                    self.apply_theme(self.current_theme)
                    self.build_ui()
                    self.setup_dnd_events()
                else:
                    self.setup_dnd_events()
                if ENABLE_LOGGING: logger.info("Using tkinterdnd2 for drag and drop")
                return
            except ImportError:
                pass
            
            try:
                self.root.drop_target_register('*')
                self.root.dnd_bind('<<Drop>>', self.on_drop)
                self.root.dnd_bind('<<DragEnter>>', self.on_drag_enter)
                self.root.dnd_bind('<<DragLeave>>', self.on_drag_leave)
                
                self.canvas.drop_target_register('*')
                self.canvas.dnd_bind('<<Drop>>', self.on_drop)
                self.canvas.dnd_bind('<<DragEnter>>', self.on_drag_enter)
                self.canvas.dnd_bind('<<DragLeave>>', self.on_drag_leave)
                
                if ENABLE_LOGGING: logger.info("Using native tkinter DND")
            except Exception as e:
                if ENABLE_LOGGING: logger.warning(f"Native tkinter DND failed: {e}")
                
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Drag and drop setup failed: {e}")
    
    def setup_dnd_events(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop_tkinterdnd2)
            self.root.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.root.dnd_bind('<<DragLeave>>', self.on_drag_leave)
            
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind('<<Drop>>', self.on_drop_tkinterdnd2)
            self.canvas.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.canvas.dnd_bind('<<DragLeave>>', self.on_drag_leave)
            
            self.top_frame = Frame(self.root)
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to setup tkinterdnd2 events: {e}")
    
    def on_drop_tkinterdnd2(self, event):
        self.canvas.config(bg=self.canvas_bg)
        
        try:
            files = event.data.strip().split()
            
            video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.m4v', '.wmv', '.mpg', '.mpeg')
            video_files = []
            for file in files:
                file = file.strip('{}')
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    if os.path.exists(file):
                        video_files.append(file)
            
            if video_files:
                if len(video_files) == 1:
                    self._load_single_video_preview(video_files[0])
                else:
                    self.process_multiple_videos(video_files)
            else:
                messagebox.showwarning("No Videos", "No valid video files were dropped.")
                
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error handling drop: {e}")
            messagebox.showerror("Error", f"Failed to process dropped files: {str(e)}")
    
    def on_drag_enter(self, event):
        self.canvas.config(bg="#2a5298")
        return 'copy'
    
    def on_drag_leave(self, event):
        self.canvas.config(bg=self.canvas_bg)
    
    def on_drop(self, event):
        self.canvas.config(bg=self.canvas_bg)
        
        try:
            files = []
            if hasattr(event, 'data'):
                files = [event.data]
            
            video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.m4v', '.wmv', '.mpg', '.mpeg')
            video_files = []
            for file in files:
                file = file.strip('{}')
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    if os.path.exists(file):
                        video_files.append(file)
            
            if video_files:
                if len(video_files) == 1:
                    self._load_single_video_preview(video_files[0])
                else:
                    self.process_multiple_videos(video_files)
            else:
                messagebox.showwarning("No Videos", "No valid video files were dropped.")
                
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error handling drop: {e}")
            messagebox.showerror("Error", f"Failed to process dropped files: {str(e)}")
    
    def run_no_gui_mode(self):
        if not self.conversion_queue:
            print("No jobs in queue. Exiting.")
            return
        
        print(f"Starting no-GUI mode with {len(self.conversion_queue)} job(s) in queue")
        
        try:
            while self.conversion_queue:
                job = self.conversion_queue[0]
                print(f"\nProcessing job: {job.get_filename()}")
                
                if job.video_settings.is_ai_enhancement():
                    self._process_job_with_ai_no_gui(job)
                else:
                    self._process_job_no_gui(job)
                
                if job.status == "Completed":
                    self.conversion_queue.pop(0)
                    self.save_queue()
                else:
                    print(f"\n  ✗ Job failed, stopping queue processing")
                    break
            
            print("\nAll jobs completed!")
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user")
            if self.current_job:
                self.current_job.status = "Interrupted"
                print(f"  Current job '{self.current_job.get_filename()}' was interrupted")
            print("App closed")
            sys.exit(0)
    
    def _process_job_no_gui(self, job: ConversionJob):
        job.status = "Processing"
        
        try:
            cap = cv2.VideoCapture(job.input_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {job.input_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load video for job {job.id}: {e}")
            job.status = "Failed"
            return
        
        total_duration = total_frames / fps
        if job.start_frame is not None or job.end_frame is not None:
            start_time = job.start_frame / fps if job.start_frame else 0
            end_time = job.end_frame / fps if job.end_frame else total_duration
            segment_duration = max(0.001, end_time - start_time)
        else:
            segment_duration = total_duration
            start_time = 0
        
        cmd = self._build_ffmpeg_command_job(job)
        if not cmd:
            job.status = "Failed"
            return
        
        print(f"  Output: {os.path.basename(job.output_path)}")
        print(f"  Duration: {format_time(segment_duration)}")
        
        start_time_actual = time.time()
        last_update_time = start_time_actual
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            while True:
                try:
                    line = process.stderr.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        continue
                    
                    progress = self._parse_ffmpeg_progress_with_eta_no_gui(line, segment_duration, start_time_actual)
                    if progress:
                        self._update_progress_no_gui(progress)
                except KeyboardInterrupt:
                    print("\n\n⚠️  Conversion interrupted by user")
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    job.status = "Interrupted"
                    print(f"  Job '{job.get_filename()}' was interrupted")
                    raise
            
            returncode = process.wait()
            
            if returncode == 0:
                job.status = "Completed"
                job.progress = 100.0
                print(f"\n  ✓ Completed successfully")
            elif returncode == -15 or returncode == 255:
                job.status = "Interrupted"
                print(f"\n  ⚠️  Job interrupted")
            else:
                job.status = "Failed"
                print(f"\n  ✗ Failed with return code {returncode}")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Conversion error for job {job.id}: {e}")
            job.status = "Failed"
            print(f"\n  ✗ Error: {e}")
    
    def _process_job_with_ai_no_gui(self, job: ConversionJob):
        job.status = "Processing"
        
        scale_factor = job.video_settings.get_ai_scale_factor()
        is_gpu = job.video_settings.is_gpu_ai_enhancement()
        
        print(f"  AI {scale_factor}x Enhancement ({'GPU' if is_gpu else 'CPU'})")
        print(f"  Output: {os.path.basename(job.output_path)}")
        
        # Check AI backend availability
        self._check_ai_availability()
        
        backend = "Unknown"
        if is_gpu and self.realesrgan_available:
            backend = "GPU (Real-ESRGAN)"
            if self.realesrgan_vulkan_available:
                backend += " with Vulkan"
            else:
                backend += " (CPU fallback)"
        elif self.super_image_available:
            backend = "CPU (super-image)"
        else:
            backend = "CPU (simple resize)"
        
        print(f"  Backend: {backend}")
        
        if is_gpu and not self.realesrgan_available:
            print(f"  WARNING: GPU acceleration requested but Real-ESRGAN not available!")
            print(f"  Attempting to download Real-ESRGAN...")
        
        if not is_gpu:
            print(f"  WARNING: AI enhancement is CPU-intensive and can take a long time!")
        
        try:
            self._initialize_ai_model(use_gpu=is_gpu)
            
            # Create temporary directory in the same folder as the video file
            video_dir = os.path.dirname(job.input_path) or os.getcwd()
            with tempfile.TemporaryDirectory(prefix="favencoder_ai_", dir=video_dir) as temp_dir:
                frames_dir = os.path.join(temp_dir, "frames")
                enhanced_dir = os.path.join(temp_dir, "enhanced")
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(enhanced_dir, exist_ok=True)
                
                print(f"  Step 1/4: Extracting frames...")
                extract_cmd = ["ffmpeg"]
                
                if job.start_frame is not None:
                    start_time = job.start_frame / job.video_fps
                    extract_cmd.extend(["-ss", f"{start_time:.6f}"])
                
                extract_cmd.extend(["-i", job.input_path])
                
                if job.end_frame is not None and job.start_frame is not None:
                    duration = (job.end_frame - job.start_frame) / job.video_fps
                    extract_cmd.extend(["-t", f"{duration:.6f}"])
                elif job.end_frame is not None:
                    duration = job.end_frame / job.video_fps
                    extract_cmd.extend(["-t", f"{duration:.6f}"])
                
                if job.crop_rect:
                    extract_cmd.extend(["-vf", f"crop={job.crop_rect.width}:{job.crop_rect.height}:{job.crop_rect.x}:{job.crop_rect.y}"])
                
                extract_cmd.extend([
                    "-vsync", "0",
                    os.path.join(frames_dir, "frame_%06d.png")
                ])
                
                extract_process = subprocess.Popen(
                    extract_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                extract_process.wait()
                
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
                total_frames = len(frame_files)
                
                if total_frames == 0:
                    raise ValueError("No frames extracted from video")
                
                print(f"  Step 2/4: Enhancing {total_frames} frames with AI...")
                for i, frame_file in enumerate(frame_files):
                    try:
                        progress_percentage = (i / total_frames) * 50
                        job.progress = progress_percentage
                        
                        bar_length = 30
                        filled_length = int(bar_length * progress_percentage / 100)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        backend_text = f"{backend} | " if i == 0 else ""
                        print(f"\r  [{bar}] {progress_percentage:.1f}% | {backend_text}AI Processing Frames", end='', flush=True)
                        
                        frame_path = os.path.join(frames_dir, frame_file)
                        img = Image.open(frame_path).convert('RGB')
                        
                        try:
                            enhanced_img = self._enhance_image_with_ai(img, scale_factor)
                        except Exception as e:
                            if ENABLE_LOGGING: logger.error(f"Failed to enhance frame {frame_file}: {e}")
                            new_width = img.width * scale_factor
                            new_height = img.height * scale_factor
                            enhanced_img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
                        
                        enhanced_path = os.path.join(enhanced_dir, frame_file)
                        enhanced_img.save(enhanced_path)
                    except KeyboardInterrupt:
                        print("\n\n⚠️  AI enhancement interrupted by user")
                        job.status = "Interrupted"
                        print(f"  AI enhancement for '{job.get_filename()}' was interrupted")
                        raise
                
                print(f"\n  Step 3/4: Creating video from enhanced frames...")
                
                temp_video_path = os.path.join(temp_dir, "enhanced_video.mkv")
                
                video_cmd = [
                    "ffmpeg",
                    "-framerate", str(job.video_fps),
                    "-i", os.path.join(enhanced_dir, "frame_%06d.png"),
                    "-i", job.input_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-y", temp_video_path
                ]
                
                video_process = subprocess.Popen(
                    video_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                start_time = time.time()
                while True:
                    try:
                        line = video_process.stderr.readline()
                        if not line:
                            if video_process.poll() is not None:
                                break
                            continue
                        
                        progress = self._parse_ffmpeg_progress_with_eta_no_gui(line, 1, start_time)
                        if progress:
                            progress['percentage'] = 50 + (progress['percentage'] / 2)
                            self._update_ai_progress_no_gui(progress, "Creating Video")
                    except KeyboardInterrupt:
                        print("\n\n⚠️  Video creation interrupted by user")
                        video_process.terminate()
                        try:
                            video_process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            video_process.kill()
                            video_process.wait()
                        job.status = "Interrupted"
                        print(f"  Video creation for '{job.get_filename()}' was interrupted")
                        raise
                
                returncode = video_process.wait()
                
                if returncode != 0:
                    raise ValueError(f"Video creation failed: {returncode}")
                
                print(f"\n  Step 4/4: Final encoding...")
                
                final_cmd = self._build_ffmpeg_command_for_enhanced_video(temp_video_path, job)
                
                if final_cmd:
                    start_time = time.time()
                    final_process = subprocess.Popen(
                        final_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        bufsize=1
                    )
                    
                    while True:
                        try:
                            line = final_process.stderr.readline()
                            if not line:
                                if final_process.poll() is not None:
                                    break
                                continue
                            
                            progress = self._parse_ffmpeg_progress_with_eta_no_gui(line, 1, start_time)
                            if progress:
                                self._update_progress_no_gui(progress)
                        except KeyboardInterrupt:
                            print("\n\n⚠️  Final encoding interrupted by user")
                            final_process.terminate()
                            try:
                                final_process.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                final_process.kill()
                                final_process.wait()
                            job.status = "Interrupted"
                            print(f"  Final encoding for '{job.get_filename()}' was interrupted")
                            raise
                    
                    final_returncode = final_process.wait()
                    
                    if final_returncode == 0:
                        job.status = "Completed"
                        job.progress = 100.0
                        print(f"\n  ✓ AI enhancement completed successfully")
                    else:
                        job.status = "Failed"
                        print(f"\n  ✗ Final encoding failed: {final_returncode}")
                else:
                    shutil.copy2(temp_video_path, job.output_path)
                    job.status = "Completed"
                    job.progress = 100.0
                    print(f"\n  ✓ AI enhancement completed successfully")
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"AI conversion error for job {job.id}: {e}")
            job.status = "Failed"
            print(f"\n  ✗ AI enhancement failed: {e}")
    
    def _update_ai_progress_no_gui(self, progress: Dict[str, float], status: str):
        percentage = progress['percentage']
        elapsed = format_time_compact(progress['elapsed_time'])
        
        bar_length = 30
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r  [{bar}] {percentage:.1f}% | {status} | Elapsed: {elapsed}", end='', flush=True)
    
    def _parse_ffmpeg_progress_with_eta_no_gui(self, line: str, segment_duration: float, start_time: float) -> Optional[Dict[str, float]]:
        time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2}\.\d+)', line)
        if not time_match:
            return None
        
        try:
            h, m, s = time_match.groups()
            current_time = int(h) * 3600 + int(m) * 60 + float(s)
            
            if segment_duration > 0:
                percentage = min(100.0, (current_time / max(0.001, segment_duration)) * 100.0)
                
                current_actual_time = time.time()
                elapsed_time = current_actual_time - start_time
                
                if percentage > 0:
                    estimated_total_time = elapsed_time / (percentage / 100.0)
                    estimated_remaining = max(0.0, estimated_total_time - elapsed_time)
                else:
                    estimated_total_time = 0
                    estimated_remaining = 0
                
                return {
                    'percentage': percentage,
                    'current_time': current_time,
                    'elapsed_time': elapsed_time,
                    'estimated_remaining': estimated_remaining
                }
        except (ValueError, TypeError) as e:
            if ENABLE_LOGGING: logger.warning(f"Error parsing time from FFmpeg output: {e}")
        
        return None
    
    def _update_progress_no_gui(self, progress: Dict[str, float]):
        percentage = progress['percentage']
        elapsed = format_time_compact(progress['elapsed_time'])
        remaining = format_time_compact(progress['estimated_remaining'])
        
        bar_length = 30
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r  [{bar}] {percentage:.1f}% | Elapsed: {elapsed} | Remaining: {remaining}", end='', flush=True)
    
    def save_queue(self) -> None:
        try:
            queue_data = [job.to_dict() for job in self.conversion_queue]
            with open(self.QUEUE_FILE, 'w') as f:
                json.dump(queue_data, f, indent=2, default=str)
            if ENABLE_LOGGING: logger.info(f"Queue saved to {self.QUEUE_FILE} with {len(self.conversion_queue)} jobs")
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to save queue: {e}")
    
    def load_queue(self) -> None:
        try:
            if os.path.exists(self.QUEUE_FILE):
                with open(self.QUEUE_FILE, 'r') as f:
                    queue_data = json.load(f)
                    for job_data in queue_data:
                        try:
                            job = ConversionJob.from_dict(job_data)
                            self.conversion_queue.append(job)
                            self.next_job_id = max(self.next_job_id, job.id + 1)
                        except Exception as e:
                            if ENABLE_LOGGING: logger.error(f"Failed to load job from queue data: {e}")
                if ENABLE_LOGGING: logger.info(f"Queue loaded from {self.QUEUE_FILE} with {len(self.conversion_queue)} jobs")
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load queue: {e}")
    
    def load_presets(self) -> Dict[str, Any]:
        """Load saved presets from file"""
        try:
            if os.path.exists(PRESETS_FILE):
                with open(PRESETS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load presets: {e}")
        return {}
    
    def save_presets(self, presets: Dict[str, Any]) -> None:
        """Save presets to file"""
        try:
            with open(PRESETS_FILE, 'w') as f:
                json.dump(presets, f, indent=2, default=str)
            if ENABLE_LOGGING: logger.info(f"Presets saved to {PRESETS_FILE}")
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to save presets: {e}")
    
    def save_current_preset(self) -> None:
        """Save current settings as a preset"""
        preset_name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if not preset_name:
            return
        
        self._update_current_settings_from_ui()
        
        preset = {
            'video_settings': self.video_settings.to_dict(),
            'audio_settings': self.audio_settings.to_dict(),
            'output_settings': self.output_settings.to_dict()
        }
        
        presets = self.load_presets()
        presets[preset_name] = preset
        self.save_presets(presets)
        
        messagebox.showinfo("Preset Saved", f"Preset '{preset_name}' saved successfully.")
    
    def load_selected_preset(self) -> None:
        """Load a selected preset"""
        presets = self.load_presets()
        if not presets:
            messagebox.showinfo("No Presets", "No presets saved yet.")
            return
        
        # Create selection dialog
        preset_window = Toplevel(self.root)
        preset_window.title("Load Preset")
        preset_window.geometry("300x400")
        preset_window.configure(bg=self.bg_color)
        preset_window.transient(self.root)
        preset_window.grab_set()
        
        Label(
            preset_window,
            text="Select a preset to load:",
            font=("Arial", 10, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        ).pack(pady=10)
        
        listbox = Listbox(preset_window, selectmode=SINGLE, bg=self.entry_bg, fg=self.entry_fg)
        scrollbar = Scrollbar(preset_window)
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        for preset_name in sorted(presets.keys()):
            listbox.insert(END, preset_name)
        
        listbox.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=RIGHT, fill=Y, pady=10)
        
        def on_select():
            selection = listbox.curselection()
            if not selection:
                return
            
            preset_name = listbox.get(selection[0])
            preset = presets[preset_name]
            self.apply_preset(preset)
            preset_window.destroy()
            messagebox.showinfo("Preset Loaded", f"Preset '{preset_name}' loaded successfully.")
        
        button_frame = Frame(preset_window, bg=self.bg_color)
        button_frame.pack(fill=X, padx=10, pady=(0, 10))
        
        Button(
            button_frame,
            text="Load",
            bg=self.accent_color,
            fg=self.button_fg,
            command=on_select,
            width=10
        ).pack(side=LEFT, padx=(0, 5))
        
        Button(
            button_frame,
            text="Cancel",
            bg=self.button_bg,
            fg=self.button_fg,
            command=preset_window.destroy,
            width=10
        ).pack(side=RIGHT)
    
    def apply_preset(self, preset: Dict[str, Any]) -> None:
        """Apply preset settings to UI"""
        try:
            # Update settings objects
            self.video_settings = VideoSettings.from_dict(preset['video_settings'])
            self.audio_settings = AudioSettings.from_dict(preset['audio_settings'])
            self.output_settings = OutputSettings.from_dict(preset['output_settings'])
            
            # Update UI variables
            self.video_codec_var.set(self.video_settings.codec.value)
            self.audio_codec_var.set(self.audio_settings.codec.value)
            self.output_format_var.set(self.output_settings.format.value)
            self.resolution_var.set(self.video_settings.resolution_mode.value)
            
            # Update custom fields
            self.video_encoder_var.set(self.video_settings.custom_encoder or "")
            self.video_args_var.set(self.video_settings.custom_encoder_args or "")
            self.audio_encoder_var.set(self.audio_settings.custom_encoder or "")
            self.audio_args_var.set(self.audio_settings.custom_encoder_args or "")
            self.custom_output_var.set(self.output_settings.custom_format or "")
            
            # Update quality settings
            self.video_quality_var.set(self.video_settings.quality_mode.value)
            self.cq_var.set(str(self.video_settings.cq_value))
            self.bitrate_var.set(str(self.video_settings.bitrate))
            self.bitrate_type_var.set(self.video_settings.bitrate_type.value)
            self.audio_bitrate_var.set(str(self.audio_settings.bitrate))
            self.samplerate_var.set(str(self.audio_settings.samplerate))
            self.fps_var.set(str(self.video_settings.output_fps) if self.video_settings.output_fps else "")
            
            # FIX: Set the speed preset BEFORE calling update_encoder_speed_options    
            # This ensures it doesn't get overwritten by the default
            if self.video_settings.speed_preset:
                self.speed_var.set(self.video_settings.speed_preset)
        
            # Update custom resolution fields
            if self.video_settings.resolution_mode == ResolutionMode.CUSTOM:
                if self.video_settings.custom_width:
                    self.width_entry.delete(0, END)
                    self.width_entry.insert(0, str(self.video_settings.custom_width))
                if self.video_settings.custom_height:
                    self.height_entry.delete(0, END)
                    self.height_entry.insert(0, str(self.video_settings.custom_height))
            elif self.video_settings.resolution_mode in [ResolutionMode.CUSTOM_WIDTH, ResolutionMode.CUSTOM_HEIGHT]:
                if self.video_settings.custom_dimension_value:
                    self.custom_dim_entry.delete(0, END)
                    self.custom_dim_entry.insert(0, str(self.video_settings.custom_dimension_value))
        
            # Update UI visibility - moved speed setting above this line
            self.on_video_codec_changed()
            self.on_audio_codec_manual_change()
            self.on_output_format_changed()
            self.on_resolution_changed()
            self.on_video_quality_changed()
            self.update_encoder_speed_options()  # This will now preserve the loaded speed
            self.update_video_codec_visibility()
            self.update_audio_quality_visibility()
            self.update_pairing_info_label()
            self.update_resolution_display()
            self.update_video_info_label()
        
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to apply preset: {e}")
            messagebox.showerror("Error", f"Failed to apply preset: {str(e)}")
    
    def apply_theme(self, theme_name: str) -> None:
        if theme_name not in self.themes:
            if ENABLE_LOGGING: logger.warning(f"Theme '{theme_name}' not found, using default")
            theme_name = self.DEFAULT_THEME
        
        theme = self.themes[theme_name]
        self.current_theme = theme_name
        
        for key, value in theme.items():
            setattr(self, key, value)
        
        if self.root:
            self.root.configure(bg=self.bg_color)
        
        if hasattr(self, 'theme_btn'):
            self.theme_btn.config(text=f"Theme: {self.current_theme.capitalize()}")
        
        if hasattr(self, 'style'):
            self.style.theme_use('clam')
            self.style.configure("custom.Horizontal.TProgressbar",
                               background=self.progress_fg,
                               troughcolor=self.progress_bg)
            self.style.configure("Custom.Treeview",
                                background=self.tree_bg,
                                foreground=self.tree_fg,
                                fieldbackground=self.tree_field_bg)
            self.style.configure("Custom.Treeview.Heading",
                                background=self.button_bg,
                                foreground=self.tree_fg)
            self.style.map("Custom.Treeview",
                          background=[('selected', self.tree_selected_bg)],
                          foreground=[('selected', self.tree_selected_fg)])
        
        if hasattr(self, 'progress_bar'):
            self.progress_bar.configure(style="custom.Horizontal.TProgressbar")
        
        if hasattr(self, 'progress_percentage'):
            self.progress_percentage.config(bg=self.progress_bg, fg=self.progress_text_fg)
        
        self.update_all_widgets()
    
    def toggle_theme(self) -> None:
        themes = list(self.themes.keys())
        current_index = themes.index(self.current_theme)
        next_index = (current_index + 1) % len(themes)
        self.apply_theme(themes[next_index])
    
    def update_all_widgets(self) -> None:
        if not hasattr(self, 'root') or not self.root:
            return
        
        try:
            self._update_widget_tree(self.root)
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error updating widgets: {e}")
    
    def _update_widget_tree(self, widget) -> None:
        try:
            wtype = widget.winfo_class()
            
            if wtype == "Tk":
                widget.configure(bg=self.bg_color)
            elif wtype in ("Frame", "TFrame", "Labelframe"):
                widget.configure(bg=self.bg_color)
            elif wtype in ("Label", "TLabel"):
                widget.configure(bg=self.bg_color, fg=self.fg_color)
            elif wtype in ("Button", "TButton"):
                self._update_button_theme(widget)
            elif wtype in ("Entry", "TEntry"):
                widget.configure(
                    bg=self.entry_bg,
                    fg=self.entry_fg,
                    insertbackground=self.entry_fg,
                    selectbackground=self.highlight_bg,
                    selectforeground=self.entry_fg
                )
            elif wtype in ("Scale", "TScale"):
                widget.configure(
                    bg=self.bg_color,
                    fg=self.fg_color,
                    troughcolor=self.entry_bg,
                    activebackground=self.highlight_bg
                )
            elif wtype == "Canvas":
                widget.configure(bg=self.canvas_bg)
            elif wtype == "Progressbar":
                widget.configure(style="custom.Horizontal.TProgressbar")
            elif wtype == "Menubutton":
                try:
                    if getattr(self, "load_menu_button", None) is widget:
                        widget.config(bg=self.accent_color2, fg=self.button_fg)
                    else:
                        widget.config(bg=self.dropdown_bg, fg=self.dropdown_fg)
                except Exception:
                    pass
            elif wtype == "TTreeview":
                widget.configure(style="Custom.Treeview")
                widget.tag_configure("current", background=self.accent_color, foreground="white")
            
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Failed to update widget {widget}: {e}")
        
        try:
            for child in widget.winfo_children():
                self._update_widget_tree(child)
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Failed to update widget children: {e}")
    
    def _update_button_theme(self, button: Button) -> None:
        try:
            btn_text = button.cget("text")

            # Preserve buttons that are explicitly marked as accent (if any)
            if getattr(button, "is_accent", False):
                button.config(fg=self.button_fg)
                return

            # Detect current background and theme button backgrounds
            try:
                current_bg = str(button.cget("bg")).strip()
            except Exception:
                current_bg = None

            theme_button_bgs = set()
            try:
                for t in self.themes:
                    b = self.themes[t].get("button_bg")
                    if b:
                        theme_button_bgs.add(str(b).strip())
                    h = self.themes[t].get("highlight_bg")
                    if h:
                        theme_button_bgs.add(str(h).strip())
            except Exception:
                pass                

            # If this button uses a custom bg (not one of the theme button_bgs), preserve it.
            if current_bg and theme_button_bgs and current_bg not in theme_button_bgs:
                button.config(fg=self.button_fg)
                return

            # Otherwise apply themed mappings (special cases first)
            if btn_text == "Load Video":
                button.configure(bg=self.accent_color2, fg=self.button_fg)
            elif btn_text == "CONVERT":
                button.configure(bg=self.accent_color, fg=self.button_fg)
            elif btn_text == "PAUSE":
                button.configure(bg="#FF9800", fg=self.button_fg)
            elif btn_text == "RESUME":
                button.configure(bg="#4CAF50", fg=self.button_fg)
            elif btn_text == "STOP":
                button.configure(bg="#f44336", fg=self.button_fg)
            elif "Set current" in btn_text:
                # Always set both background and foreground colors
                button.configure(bg=self.highlight_bg, fg=self.button_fg)
            elif getattr(self, "crop_btn", None) is button:
                button.configure(bg="#4CAF50", fg=self.button_fg)
            elif btn_text == "Clear Crop":
                button.configure(bg="#ff5a5a", fg=self.button_fg)
            elif btn_text == "Apply":
                button.configure(bg="#2196F3", fg=self.button_fg)
            else:
                button.configure(bg=self.button_bg, fg=self.button_fg)
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Failed to update widget {button}: {e}")
    def build_ui(self) -> None:
        self.style.configure("custom.Horizontal.TProgressbar",
                           background=self.progress_fg,
                           troughcolor=self.progress_bg)
        
        self.top_frame = Frame(self.root, bg=self.bg_color)
        self.top_frame.pack(pady=8, padx=10, fill=X)
        
        self.build_top_bar()
        
        self.canvas_container = Frame(self.root, bg=self.bg_color)
        self.canvas_container.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = Canvas(
            self.canvas_container,
            bg=self.canvas_bg,
            cursor="cross"
        )
        self.canvas.pack(fill=BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        
        self.build_second_line()
        
        self.build_codec_settings()
        
        self.build_status_bar()
    
    def build_second_line(self) -> None:
        second_line = Frame(self.root, bg=self.bg_color)
        second_line.pack(fill=X, padx=10, pady=(0, 5))
        
        playback_frame = Frame(second_line, bg=self.bg_color)
        playback_frame.pack(side=TOP, fill=X, expand=True, pady=(0, 0))
        
        self.timeline = Scale(
            playback_frame,
            from_=0,
            to=0,
            orient=HORIZONTAL,
            showvalue=0,
            bg=self.bg_color,
            fg=self.fg_color,
            troughcolor=self.entry_bg,
            highlightbackground=self.bg_color,
            sliderrelief=RAISED,
            activebackground=self.highlight_bg
        )
        self.timeline.pack(fill=X, expand=True)
        self.timeline.bind("<ButtonPress-1>", self.timeline_press)
        self.timeline.bind("<B1-Motion>", self.timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self.timeline_release)
        
        info_frame = Frame(second_line, bg=self.bg_color)
        info_frame.pack(side=TOP, fill=X, expand=True, pady=(2, 0))
        
        self.video_info_label = Label(
            info_frame,
            text="No video loaded",
            anchor=W,
            bg=self.bg_color,
            fg="#777777",
            font=("Arial", 9)
        )
        self.video_info_label.pack(side=LEFT, fill=X, expand=True)
        
        self.info = Label(
            info_frame,
            text="",
            anchor=E,
            bg=self.bg_color,
            fg="#aaaaaa",
            font=("Arial", 9)
        )
        self.info.pack(side=RIGHT, fill=X, expand=True)
    
    def build_status_bar(self) -> None:
        self.status_frame = Frame(
            self.root,
            bg=self.status_bg,
            relief=SUNKEN,
            bd=1,
            height=22
        )
        self.status_frame.pack(side=BOTTOM, fill=X, padx=0, pady=0)
        self.status_frame.pack_propagate(False)
        
        font = ("Arial", 9)
        max_width = 0
        for msg in self.LONGEST_STATUS_MESSAGES:
            label = Label(self.root, text=msg, font=font)
            width = label.winfo_reqwidth()
            max_width = max(max_width, width)
            label.destroy()
        
        max_width += 20
        
        self.status_frame.grid_columnconfigure(0, weight=0, minsize=max_width)
        self.status_frame.grid_columnconfigure(1, weight=1)
        self.status_frame.grid_columnconfigure(2, weight=0, minsize=250)
        
        self.status_label = Label(
            self.status_frame,
            text="Ready",
            anchor=W,
            bg=self.status_bg,
            fg=self.status_fg,
            font=font,
            width=len(max(self.LONGEST_STATUS_MESSAGES, key=len))
        )
        self.status_label.grid(row=0, column=0, sticky=W+E, padx=(10, 5), pady=0, columnspan=3)
        
        self.progress_container = Frame(self.status_frame, bg=self.status_bg)
        self.progress_container.grid(row=0, column=1, sticky=W+E, padx=5, pady=2)
        self.progress_container.grid_remove()
        
        self.progress_bar = Progressbar(
            self.progress_container,
            mode='determinate',
            style="custom.Horizontal.TProgressbar",
            maximum=100
        )
        self.progress_bar.pack(fill=BOTH, expand=True)
        
        self.progress_percentage = Label(
            self.progress_container,
            text="0.0%",
            bg=self.progress_bg,
            fg=self.progress_text_fg,
            font=("Arial", 8, "bold")
        )
        self.progress_percentage.place(relx=0.5, rely=0.5, anchor="center")
        
        self.time_label = Label(
            self.status_frame,
            text="",
            anchor=E,
            bg=self.status_bg,
            fg=self.status_fg,
            font=("Arial", 9),
            width=35
        )
        self.time_label.grid(row=0, column=2, sticky=E+W, padx=(5, 10), pady=0)
        self.time_label.grid_remove()
    
    def show_progress_in_status(self, show: bool = True) -> None:
        if show:
            self.status_label.grid_configure(columnspan=1)
            self.progress_container.grid()
            self.time_label.grid()
            self.progress_percentage.config(text="0.0%")
            self.progress_bar['value'] = 0
        else:
            self.hide_progress_in_status()
    
    def hide_progress_in_status(self) -> None:
        self.progress_container.grid_remove()
        self.time_label.grid_remove()
        self.progress_bar['value'] = 0
        self.progress_percentage.config(text="")
        self.time_label.config(text="")
        self.status_label.grid_configure(columnspan=3)
    
    def _add_paste_support(self, widget):
        widget.bind("<Control-v>", lambda e: self._handle_paste(e))
        widget.bind("<Command-v>", lambda e: self._handle_paste(e))
        
        self._add_context_menu(widget)
    
    def _add_context_menu(self, widget):
        menu = Menu(self.root, tearoff=0, bg=self.menu_bg, fg=self.menu_fg)
        
        menu.add_command(label="Cut", 
                         command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Copy", 
                         command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", 
                         command=lambda: self._handle_paste_for_menu(widget))
        menu.add_separator()
        menu.add_command(label="Select All", 
                         command=lambda: widget.select_range(0, END))
        
        widget.bind("<Button-3>", lambda e: self._show_context_menu(e, menu))
    
    def _show_context_menu(self, event, menu):
        try:
            self.current_context_menu = menu
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def close_context_menu(self, event=None):
        if self.current_context_menu:
            try:
                self.current_context_menu.unpost()
                self.current_context_menu = None
            except:
                pass
    
    def _handle_paste_for_menu(self, widget):
        try:
            clipboard_text = self.root.clipboard_get()
            if isinstance(widget, Entry):
                if widget.selection_present():
                    widget.delete("sel.first", "sel.last")
                widget.insert("insert", clipboard_text)
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Failed to paste from menu: {e}")
    
    def _add_enhanced_paste_support(self, widget, is_arguments_field=False):
        widget.bind("<Control-v>", lambda e: self._handle_enhanced_paste(e, is_arguments_field))
        widget.bind("<Command-v>", lambda e: self._handle_enhanced_paste(e, is_arguments_field))
        
        self._add_context_menu(widget)
    
    def _handle_enhanced_paste(self, event, is_arguments_field=False):
        widget = event.widget
        
        try:
            clipboard_text = self.root.clipboard_get()
            
            if is_arguments_field and clipboard_text:
                cleaned_text = clipboard_text.strip()
                cleaned_text = ' '.join(cleaned_text.split())
                
                if (cleaned_text.startswith('"') and cleaned_text.endswith('"')) or \
                   (cleaned_text.startswith("'") and cleaned_text.endswith("'")):
                    cleaned_text = cleaned_text[1:-1]
                
                for prefix in ['ffmpeg ', 'ffmpeg.exe ', '-c:v ', '-c:a ']:
                    if cleaned_text.startswith(prefix):
                        cleaned_text = cleaned_text[len(prefix):]
                
                clipboard_text = cleaned_text
            
            if isinstance(widget, Entry):
                if widget.selection_present():
                    widget.delete("sel.first", "sel.last")
                widget.insert("insert", clipboard_text)
                return "break"
                
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Failed to paste: {e}")
        
        return None
    
    def _handle_paste(self, event):
        widget = event.widget
        
        try:
            clipboard_text = self.root.clipboard_get()
            
            if isinstance(widget, Entry):
                if widget.selection_present():
                    widget.delete("sel.first", "sel.last")
                
                widget.insert("insert", clipboard_text)
                return "break"
                
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Failed to paste from clipboard: {e}")
        
        return None
    
    def build_top_bar(self) -> None:
        top = self.top_frame
        
        self.load_menu_button = Menubutton(
            top,
            text="Load Video ▼",
            bg=self.accent_color2,
            fg=self.button_fg,
            font=("Arial", 9, "bold"),
            width=12,
            height=1,
            relief=RAISED,
            bd=1
        )
        self.load_menu_button.pack(side=LEFT, padx=(0, 5))
        
        self.load_menu = Menu(self.load_menu_button, tearoff=0, bg=self.menu_bg, fg=self.menu_fg)
        self.load_menu.add_command(label="Load Video File(s)...", command=self.load_video_files)
        self.load_menu.add_command(label="Load Video Folder...", command=self.load_video_folder)
        self.load_menu_button.config(menu=self.load_menu)
        
        Separator(top, orient=VERTICAL).pack(side=LEFT, padx=5, fill=Y, pady=3)
        
        start_frame_compact = Frame(top, bg=self.bg_color)
        start_frame_compact.pack(side=LEFT, padx=5)
        
        start_label = Label(
            start_frame_compact,
            text="Start:",
            font=("Arial", 9),
            bg=self.bg_color,
            fg=self.fg_color
        )
        start_label.pack(side=LEFT)
        
        self.start_entry = Entry(
            start_frame_compact,
            width=7,
            font=("Arial", 9),
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1
        )
        self.start_entry.pack(side=LEFT, padx=2)
        self.start_entry.bind("<Return>", self.apply_start_entry)
        self.start_entry.bind("<FocusOut>", self.apply_start_entry)
        
        self._add_paste_support(self.start_entry)
        
        self.set_start_btn = Button(
            start_frame_compact,
            text="Set current",
            bg=self.button_bg,
            command=self.set_start,
            width=6,
            height=1,
            font=("Arial", 8),
            fg=self.button_fg,
            relief=RAISED,
            bd=1
        )
        self.set_start_btn.pack(side=LEFT, padx=(0, 5))
        
        end_frame_compact = Frame(top, bg=self.bg_color)
        end_frame_compact.pack(side=LEFT, padx=5)
        
        end_label = Label(
            end_frame_compact,
            text="End:",
            font=("Arial", 9),
            bg=self.bg_color,
            fg=self.fg_color
        )
        end_label.pack(side=LEFT)
        
        self.end_entry = Entry(
            end_frame_compact,
            width=7,
            font=("Arial", 9),
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1
        )
        self.end_entry.pack(side=LEFT, padx=2)
        self.end_entry.bind("<Return>", self.apply_end_entry)
        self.end_entry.bind("<FocusOut>", self.apply_end_entry)
        
        self._add_paste_support(self.end_entry)
        
        self.set_end_btn = Button(
            end_frame_compact,
            text="Set current",
            bg=self.button_bg,
            command=self.set_end,
            width=6,
            height=1,
            font=("Arial", 8),
            fg=self.button_fg,
            relief=RAISED,
            bd=1
        )
        self.set_end_btn.pack(side=LEFT, padx=(0, 5))
        
        Separator(top, orient=VERTICAL).pack(side=LEFT, padx=5, fill=Y, pady=3)
        
        self.play_btn = Button(
            top,
            text="▶ Play",
            command=self.toggle_play,
            width=7,
            height=1,
            bg=self.accent_color,
            fg=self.button_fg,
            relief=RAISED,
            bd=1
        )
        
        prev_frame_btn = Button(
            top,
            text="◀ Frame",
            command=self.prev_frame,
            width=6,
            height=1,
            bg=self.button_bg,
            fg=self.button_fg,
            relief=RAISED,
            bd=1
        )
        
        next_frame_btn = Button(
            top,
            text="Frame ▶",
            command=self.next_frame,
            width=6,
            height=1,
            bg=self.button_bg,
            fg=self.button_fg,
            relief=RAISED,
            bd=1
        )
        
        prev_frame_btn.pack(side=LEFT, padx=5)
        self.play_btn.pack(side=LEFT, padx=5)
        next_frame_btn.pack(side=LEFT, padx=(5, 5))
        
        Separator(top, orient=VERTICAL).pack(side=LEFT, padx=5, fill=Y, pady=3)
        
        self.queue_btn = Button(
            top,
            text="Add to Queue",
            command=self.add_to_queue,
            width=9,
            height=1,
            bg=self.accent_color2,
            fg=self.button_fg,
            font=("Arial", 9),
            relief=RAISED,
            bd=1
        )
        self.queue_btn.pack(side=LEFT, padx=5)
        
        self.queue_manager_btn = Button(
            top,
            text="Queue Manager",
            command=self.show_queue_manager,
            width=11,
            height=1,
            bg=self.accent_color2,
            fg=self.button_fg,
            font=("Arial", 9),
            relief=RAISED,
            bd=1
        )
        self.queue_manager_btn.pack(side=LEFT, padx=5)
        
        Separator(top, orient=VERTICAL).pack(side=LEFT, padx=5, fill=Y, pady=3)
        
        self.theme_btn = Button(
            top,
            text=f"Theme: {self.current_theme.capitalize()}",
            command=self.toggle_theme,
            width=9,
            height=1,
            bg=self.button_bg,
            fg=self.button_fg,
            font=("Arial", 9),
            relief=RAISED,
            bd=1
        )
        self.theme_btn.pack(side=LEFT, padx=5)
        
        Separator(top, orient=VERTICAL).pack(side=LEFT, padx=5, fill=Y, pady=3)
        
        self.crop_tools_frame = Frame(top, bg=self.bg_color)
        self.crop_tools_frame.pack(side=LEFT, padx=5)
        
        self.clear_crop_btn = Button(
            self.crop_tools_frame,
            text="Clear Crop",
            bg="#ff5a5a",
            command=self.clear_crop,
            width=7,
            height=1,
            fg=self.button_fg,
            relief=RAISED,
            bd=1
        )
        self.clear_crop_btn.pack_forget()
        
        self.crop_btn = Button(
            self.crop_tools_frame,
            text="Crop Tool",
            bg="#ba7ed6",
            fg=self.button_fg,
            command=self.toggle_crop_mode,
            width=7,
            height=1,
            relief=RAISED,
            bd=1
        )
        self.crop_btn.pack(side=LEFT)
        spacer = Label(top, text="", bg=self.bg_color)
        spacer.pack(side=LEFT, expand=True)
        
        title_label = Label(
            top,
            text="FAVencoder",
            font=("Arial", 8, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        )
        title_label.pack(side=RIGHT, padx=(0, 5))
    
    def load_video_files(self):
        try:
            paths = filedialog.askopenfilenames(
                title="Select Video File(s)",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mkv *.mov *.flv *.webm *.m4v *.wmv *.mpg *.mpeg"),
                    ("All files", "*.*")
                ]
            )
            if not paths:
                return
            
            if len(paths) == 1:
                self._load_single_video_preview(paths[0])
            else:
                self.process_multiple_videos(list(paths))
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load video files: {e}")
            messagebox.showerror("Error", f"Failed to load video files: {str(e)}")
    
    def load_video_folder(self):
        try:
            folder = filedialog.askdirectory(title="Select Folder with Videos")
            if not folder:
                return
            
            video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.m4v', '.wmv', '.mpg', '.mpeg')
            video_files = []
            
            for root_dir, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        video_files.append(os.path.join(root_dir, file))
            
            if not video_files:
                messagebox.showinfo("No Videos Found", "No video files found in the selected folder.")
                return
            
            if len(video_files) == 1:
                self._load_single_video_preview(video_files[0])
            else:
                self.process_multiple_videos(video_files)
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load video folder: {e}")
            messagebox.showerror("Error", f"Failed to load video folder: {str(e)}")
    
    def process_multiple_videos(self, video_paths):
        if not video_paths:
            return
        
        if len(video_paths) > 1:
            response = messagebox.askyesno(
                "Add Multiple Videos",
                f"Found {len(video_paths)} video(s).\n\n"
                "This will load all files in the folder with the current settings (video codec, audio codec, resolution, etc.) and add them to the conversion queue.\n"
                "The first video will be loaded for preview and editing.\n\n"
                "Note: Make sure to adjust all settings before loading the folder, as the same settings will be applied to all videos.\n\n"
                "Do you want to proceed?"
            )
            
            if not response:
                return
        
        if video_paths:
            self._load_single_video_preview(video_paths[0])
        
        added_count = 0
        for video_path in video_paths:
            if self.add_video_to_queue_from_path(video_path):
                added_count += 1
        
        self.status_label.config(text=f"Added {added_count} video(s) to queue")
        if added_count > 0 and len(video_paths) > 1:
            messagebox.showinfo("Videos Added", 
                              f"Successfully added {added_count} video(s) to the queue.\n\n"
                              f"The first video is loaded for preview and editing.\n"
                              f"Total jobs in queue: {len(self.conversion_queue)}")
    
    def _load_single_video_preview(self, path):
        try:
            self.stop_playback()
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.video_path = path
            self.cap = cv2.VideoCapture(path)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open video: {path}")
                self.video_path = None
                return
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            self.original_aspect_ratio = self.video_width / self.video_height if self.video_height > 0 else 1.0
            
            self.current_frame = 0
            self.output_settings.start_frame = None
            self.output_settings.end_frame = None
            
            if self.audio_settings.codec != AudioCodec.CUSTOM:
                self.audio_codec_manually_changed = False
            
            self.crop_rect = None
            self.crop_start = None
            self.crop_move_mode = False
            self.crop_resize_mode = False
            
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
                self.crop_rect_id = None
            
            for handle in self.crop_handles:
                self.canvas.delete(handle)
            self.crop_handles = []
            
            if self.clear_crop_btn.winfo_ismapped():
                self.clear_crop_btn.pack_forget()
            if not self.crop_btn.winfo_ismapped():
                self.crop_btn.pack(side=LEFT)
            self.crop_mode = False
            
            self.frame_cache.clear()
            
            self.start_entry.delete(0, END)
            self.end_entry.delete(0, END)
            
            self.timeline.config(to=max(0, self.total_frames - 1))
            self.timeline.set(0)
            
            self.state = AppState.VIDEO_LOADED
            
            self.aspect_ratio = None
            
            self.update_resolution_display()
            
            self.show_frame()
            self.update_video_info_label()
            
            self.status_label.config(text="Ready")
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load video: {e}")
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            
            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_path = None
            self.state = AppState.NO_VIDEO
    
    def add_video_to_queue_from_path(self, video_path):
        try:
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                if ENABLE_LOGGING: logger.warning(f"Failed to open video for queue: {video_path}")
                return False
            
            video_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = test_cap.get(cv2.CAP_PROP_FPS) or 25.0
            
            test_cap.release()
            
            output_path = self._get_output_filename_for_path(video_path)
            if not output_path:
                return False
            
            job = ConversionJob(
                id=self.next_job_id,
                input_path=video_path,
                output_path=output_path,
                video_settings=self.video_settings,
                audio_settings=self.audio_settings,
                output_settings=self.output_settings,
                crop_rect=self.crop_rect,
                start_frame=self.output_settings.start_frame,
                end_frame=self.output_settings.end_frame,
                status="Pending",
                video_width=video_width,
                video_height=video_height,
                video_fps=fps
            )
            
            self._update_current_settings_from_ui()
            job.ffmpeg_command = self._build_ffmpeg_command_for_job(job)
            
            self.conversion_queue.append(job)
            self.next_job_id += 1
            
            if self.queue_window and self.queue_window.winfo_exists():
                self._update_queue_list()
            
            self.save_queue()
            
            return True
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to add video to queue: {e}")
            return False
    
    def _get_output_filename_for_path(self, input_path):
        base = os.path.splitext(os.path.basename(input_path))[0]
        
        if self.video_settings.codec == VideoCodec.NO_VIDEO:
            audio_extensions = {
                AudioCodec.FLAC: ".flac",
                AudioCodec.PCM_16: ".wav",
                AudioCodec.PCM_24: ".wav",
                AudioCodec.PCM_32: ".wav",
                AudioCodec.AAC: ".m4a",
                AudioCodec.OPUS: ".opus",
                AudioCodec.MP3: ".mp3",
                AudioCodec.AC3: ".ac3",
                AudioCodec.DTS: ".dts",
                AudioCodec.VORBIS: ".ogg",
                AudioCodec.CUSTOM: ".mka"
            }
            ext = audio_extensions.get(self.audio_settings.codec, ".flac")
            default_name = f"{base}_audio{ext}"
            filetypes = [("Audio files", f"*{ext}"), ("All files", "*.*")]
        else:
            if self.output_settings.format == VideoFormat.CUSTOM and self.output_settings.custom_format:
                ext = self.output_settings.custom_format
                if not ext.startswith("."):
                    ext = f".{ext}"
            else:
                format_extensions = {
                    VideoFormat.MKV: ".mkv",
                    VideoFormat.MP4: ".mp4",
                    VideoFormat.MOV: ".mov",
                    VideoFormat.AVI: ".avi",
                    VideoFormat.WEBM: ".webm",
                    VideoFormat.FLV: ".flv",
                    VideoFormat.TS: ".ts"
                }
                fmt = VideoFormat(self.output_format_var.get())
                ext = format_extensions.get(fmt, ".mkv")
            
            try:
                cap = cv2.VideoCapture(input_path)
                if cap.isOpened():
                    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    out_width, out_height = self.video_settings.calculate_output_dimensions(
                        video_width, video_height, self.crop_rect
                    )
                    res_str = f"{out_width}x{out_height}"
                else:
                    res_str = "converted"
            except:
                res_str = "converted"
            
            default_name = f"{base}_{res_str}{ext}"
            filetypes = [(f"{fmt.value} files", f"*{ext}"), ("All files", "*.*")]
        
        input_dir = os.path.dirname(input_path)
        default_path = os.path.join(input_dir, default_name)
        
        return default_path
    
    def build_codec_settings(self) -> None:
        codec_frame = Frame(self.root, bg=self.bg_color)
        codec_frame.pack(pady=5, fill=X, padx=10)
        
        for i in range(6):
            codec_frame.columnconfigure(i, weight=1)
        
        self.build_video_settings(codec_frame)
        
        self.build_audio_settings(codec_frame)
        
        self.build_resolution_settings(codec_frame)
        
        self.build_output_settings(codec_frame)
        
        self.build_preset_controls(codec_frame)
        
        self.build_convert_controls(codec_frame)
    
    def build_video_settings(self, parent: Frame) -> None:
        video_settings = Frame(parent, bg=self.bg_color)
        video_settings.grid(row=0, column=0, padx=(10, 5), sticky="nsew")
        
        video_label = Label(
            video_settings,
            text="Video Codec:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        video_label.pack(anchor=W)
        
        self.video_codec_var = StringVar(value=self.video_settings.codec.value)
        self.video_codec_menu = OptionMenu(
            video_settings,
            self.video_codec_var,
            *[codec.value for codec in VideoCodec]
        )
        self._configure_dropdown(self.video_codec_menu)
        self.video_codec_menu.pack(fill=X)
        self.video_codec_var.trace('w', self.on_video_codec_changed)
        
        self.custom_video_frame = Frame(video_settings, bg=self.bg_color)
        
        self.video_encoder_label = Label(
            self.custom_video_frame,
            text="Encoder:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.video_encoder_label.pack(anchor=W)
        
        self.video_encoder_entry = Entry(
            self.custom_video_frame,
            width=20,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.video_encoder_var
        )
        self.video_encoder_entry.pack(fill=X, pady=(0, 5))
        self._add_paste_support(self.video_encoder_entry)
        self.video_encoder_var.trace('w', self.on_video_encoder_changed)
        
        self.video_args_label = Label(
            self.custom_video_frame,
            text="Additional Args:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.video_args_label.pack(anchor=W)
        
        self.video_args_entry = Entry(
            self.custom_video_frame,
            width=20,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.video_args_var
        )
        self.video_args_entry.pack(fill=X)
        self._add_enhanced_paste_support(self.video_args_entry, is_arguments_field=True)
        self.video_args_var.trace('w', self.on_video_args_changed)
        
        self.fps_frame = Frame(video_settings, bg=self.bg_color)
        self.fps_frame.pack(fill=X, pady=(5, 0))
        
        fps_label = Label(
            self.fps_frame,
            text="Output FPS:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        fps_label.pack(side=LEFT)
        
        self.fps_entry = Entry(
            self.fps_frame,
            width=6,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.fps_var
        )
        self.fps_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.fps_entry)
        self.fps_var.trace('w', self.on_fps_changed)
        
        fps_unit_label = Label(
            self.fps_frame,
            text="fps",
            bg=self.bg_color,
            fg="#aaaaaa",
            font=("Arial", 8)
        )
        fps_unit_label.pack(side=LEFT, padx=(0, 5))
        
        fps_note_label = Label(
            self.fps_frame,
            text="(leave empty for original)",
            bg=self.bg_color,
            fg="#888888",
            font=("Arial", 7)
        )
        fps_note_label.pack(side=LEFT)
        
        self.video_quality_frame = Frame(video_settings, bg=self.bg_color)
        
        quality_mode_frame = Frame(self.video_quality_frame, bg=self.bg_color)
        quality_mode_frame.pack(fill=X, pady=(5, 0))
        
        quality_label = Label(
            quality_mode_frame,
            text="Quality Mode:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        quality_label.pack(side=LEFT)
        
        self.video_quality_var = StringVar(value=QualityMode.CQ.value)
        self.video_quality_menu = OptionMenu(
            quality_mode_frame,
            self.video_quality_var,
            *[mode.value for mode in QualityMode]
        )
        self._configure_dropdown(self.video_quality_menu, width=6)
        self.video_quality_menu.pack(side=LEFT, padx=5)
        self.video_quality_var.trace('w', self.on_video_quality_changed)
        
        self.cq_frame = Frame(self.video_quality_frame, bg=self.bg_color)
        
        cq_label = Label(
            self.cq_frame,
            text="CQ Value:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        cq_label.pack(side=LEFT)
        
        self.cq_entry = Entry(
            self.cq_frame,
            width=5,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.cq_var
        )
        self.cq_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.cq_entry)
        self.cq_var.trace('w', self.on_cq_changed)
        
        self.cq_range_label = Label(
            self.cq_frame,
            text="(0-51)",
            bg=self.bg_color,
            fg="#aaaaaa",
            font=("Arial", 7)
        )
        self.cq_range_label.pack(side=LEFT, padx=(0, 5))
        
        self.bitrate_frame = Frame(self.video_quality_frame, bg=self.bg_color)
        
        bitrate_type_frame = Frame(self.bitrate_frame, bg=self.bg_color)
        bitrate_type_frame.pack(fill=X, pady=(0, 2))
        
        bitrate_type_label = Label(
            bitrate_type_frame,
            text="Bitrate Type:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        bitrate_type_label.pack(side=LEFT)
        
        self.bitrate_type_var = StringVar(value=BitrateType.VBR.value)
        self.bitrate_type_menu = OptionMenu(
            bitrate_type_frame,
            self.bitrate_type_var,
            *[bt.value for bt in BitrateType]
        )
        self._configure_dropdown(self.bitrate_type_menu, width=6)
        self.bitrate_type_menu.pack(side=LEFT, padx=5)
        self.bitrate_type_var.trace('w', self.on_bitrate_type_changed)
        
        bitrate_value_frame = Frame(self.bitrate_frame, bg=self.bg_color)
        bitrate_value_frame.pack(fill=X)
        
        bitrate_label = Label(
            bitrate_value_frame,
            text="Bitrate:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        bitrate_label.pack(side=LEFT)
        
        self.bitrate_entry = Entry(
            bitrate_value_frame,
            width=7,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.bitrate_var
        )
        self.bitrate_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.bitrate_entry)
        self.bitrate_var.trace('w', self.on_bitrate_changed)
        
        kbps_label = Label(
            bitrate_value_frame,
            text="kbps",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Arial", 8)
        )
        kbps_label.pack(side=LEFT)
        
        self.estimated_size_label = Label(
            bitrate_value_frame,
            text="",
            bg=self.bg_color,
            fg="#888888",
            font=("Arial", 7)
        )
        self.estimated_size_label.pack(side=LEFT, padx=(5, 0))
        
        self.speed_frame = Frame(video_settings, bg=self.bg_color)
        
        speed_label = Label(
            self.speed_frame,
            text="Encoder Speed:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        speed_label.pack(anchor=W)
        
        self.speed_var = StringVar(value="medium")
        self.speed_menu = OptionMenu(
            self.speed_frame,
            self.speed_var,
            "medium"
        )
        self._configure_dropdown(self.speed_menu, width=12)
        self.speed_menu.pack(fill=X)
        self.speed_var.trace('w', self.on_speed_changed)
    
    def update_estimated_size(self):
        if not self.video_path or self.video_settings.quality_mode != QualityMode.BITRATE:
            self.estimated_size_label.config(text="")
            return
        
        try:
            total_seconds = self.duration
            
            if self.output_settings.start_frame is not None or self.output_settings.end_frame is not None:
                start_frame = self.output_settings.start_frame if self.output_settings.start_frame is not None else 0
                end_frame = self.output_settings.end_frame if self.output_settings.end_frame is not None else self.total_frames
                total_seconds = (end_frame - start_frame) / self.fps
            
            video_bitrate_kbps = self.video_settings.bitrate
            video_bits = video_bitrate_kbps * 1000 * total_seconds
            
            audio_bits = 0
            if self.audio_settings.codec != AudioCodec.NO_AUDIO:
                audio_bitrate_kbps = self.audio_settings.bitrate
                audio_bits = audio_bitrate_kbps * 1000 * total_seconds
            
            total_bits = video_bits + audio_bits
            
            total_bytes = total_bits / 8
            
            if total_bytes < 1024:
                size_str = f"{total_bytes:.1f} B"
            elif total_bytes < 1024 * 1024:
                size_str = f"{total_bytes/1024:.1f} KB"
            elif total_bytes < 1024 * 1024 * 1024:
                size_str = f"{total_bytes/(1024*1024):.1f} MB"
            else:
                size_str = f"{total_bytes/(1024*1024*1024):.1f} GB"
            
            if self.video_settings.bitrate_type == BitrateType.VBR:
                self.estimated_size_label.config(text=f"~{size_str}")
            else:
                self.estimated_size_label.config(text=f"{size_str}")
                
        except Exception as e:
            if ENABLE_LOGGING: logger.warning(f"Failed to calculate estimated size: {e}")
            self.estimated_size_label.config(text="")
    
    def on_video_encoder_changed(self, *args):
        self.video_settings.custom_encoder = self.video_encoder_var.get()
    
    def on_video_args_changed(self, *args):
        self.video_settings.custom_encoder_args = self.video_args_var.get()
    
    def on_audio_encoder_changed(self, *args):
        self.audio_settings.custom_encoder = self.audio_encoder_var.get()
    
    def on_audio_args_changed(self, *args):
        self.audio_settings.custom_encoder_args = self.audio_args_var.get()
    
    def on_cq_changed(self, *args):
        try:
            self.video_settings.cq_value = int(self.cq_var.get())
        except ValueError:
            pass
    
    def on_bitrate_changed(self, *args):
        try:
            self.video_settings.bitrate = int(self.bitrate_var.get())
            self.update_estimated_size()
        except ValueError:
            pass
    
    def on_audio_bitrate_changed(self, *args):
        try:
            self.audio_settings.bitrate = int(self.audio_bitrate_var.get())
            self.update_estimated_size()
        except ValueError:
            pass
    
    def on_samplerate_changed(self, *args):
        try:
            self.audio_settings.samplerate = int(self.samplerate_var.get())
        except ValueError:
            pass
    
    def on_fps_changed(self, *args):
        try:
            fps_text = self.fps_var.get().strip()
            if fps_text:
                self.video_settings.output_fps = float(fps_text)
            else:
                self.video_settings.output_fps = None
        except ValueError:
            pass
    
    def on_speed_changed(self, *args):
        self.video_settings.speed_preset = self.speed_var.get()
    
    def on_bitrate_type_changed(self, *args):
        try:
            self.video_settings.bitrate_type = BitrateType(self.bitrate_type_var.get())
            self.update_estimated_size()
        except ValueError:
            pass
    
    def build_audio_settings(self, parent: Frame) -> None:
        audio_settings = Frame(parent, bg=self.bg_color)
        audio_settings.grid(row=0, column=1, padx=5, sticky="nsew")
        
        audio_label = Label(
            audio_settings,
            text="Audio Codec:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        audio_label.pack(anchor=W)
        
        self.audio_codec_var = StringVar(value=self.audio_settings.codec.value)
        self.audio_codec_menu = OptionMenu(
            audio_settings,
            self.audio_codec_var,
            *[codec.value for codec in AudioCodec]
        )
        self._configure_dropdown(self.audio_codec_menu, width=15)
        self.audio_codec_menu.pack(fill=X)
        self.audio_codec_var.trace('w', self.on_audio_codec_manual_change)
        
        self.custom_audio_frame = Frame(audio_settings, bg=self.bg_color)
        
        self.audio_encoder_label = Label(
            self.custom_audio_frame,
            text="Encoder:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.audio_encoder_label.pack(anchor=W)
        
        self.audio_encoder_entry = Entry(
            self.custom_audio_frame,
            width=20,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.audio_encoder_var)
        self.audio_encoder_entry.pack(fill=X, pady=(0, 5))
        self._add_paste_support(self.audio_encoder_entry)
        self.audio_encoder_var.trace('w', self.on_audio_encoder_changed)
        
        self.audio_args_label = Label(
            self.custom_audio_frame,
            text="Additional Args:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.audio_args_label.pack(anchor=W)
        
        self.audio_args_entry = Entry(
            self.custom_audio_frame,
            width=20,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.audio_args_var
        )
        self.audio_args_entry.pack(fill=X)
        self._add_enhanced_paste_support(self.audio_args_entry, is_arguments_field=True)
        self.audio_args_var.trace('w', self.on_audio_args_changed)
        
        self.pairing_info_label = Label(
            audio_settings,
            text="(Suggested: FLAC for FFV1)",
            bg=self.bg_color,
            fg="#888888",
            font=("Arial", 8, "italic")
        )
        self.pairing_info_label.pack(anchor=W, pady=(0, 2))
        
        self.reset_suggestion_frame = Frame(audio_settings, bg=self.bg_color)
        self.reset_suggestion_frame.pack(fill=X, pady=(0, 2))
        
        self.reset_suggestion_btn = Button(
            self.reset_suggestion_frame,
            text="Reset to Suggested",
            bg=self.button_bg,
            fg=self.button_fg,
            command=self.reset_to_suggested_codec,
            font=("Arial", 7),
            width=15,
            relief=RAISED,
            bd=1,
            state=DISABLED
        )
        self.reset_suggestion_btn.pack(side=LEFT)
        
        self.audio_quality_frame = Frame(audio_settings, bg=self.bg_color)
        
        audio_bitrate_frame = Frame(self.audio_quality_frame, bg=self.bg_color)
        audio_bitrate_frame.pack(fill=X, pady=(5, 0))
        
        audio_bitrate_label = Label(
            audio_bitrate_frame,
            text="Bitrate:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        audio_bitrate_label.pack(side=LEFT)
        
        self.audio_bitrate_entry = Entry(
            audio_bitrate_frame,
            width=7,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.audio_bitrate_var
        )
        self.audio_bitrate_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.audio_bitrate_entry)
        self.audio_bitrate_var.trace('w', self.on_audio_bitrate_changed)
        
        audio_kbps_label = Label(
            audio_bitrate_frame,
            text="kbps",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Arial", 8)
        )
        audio_kbps_label.pack(side=LEFT)
        
        samplerate_frame = Frame(self.audio_quality_frame, bg=self.bg_color)
        samplerate_frame.pack(fill=X, pady=(2, 0))
        
        samplerate_label = Label(
            samplerate_frame,
            text="Sample Rate:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        samplerate_label.pack(side=LEFT)
        
        self.samplerate_entry = Entry(
            samplerate_frame,
            width=7,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.samplerate_var
        )
        self.samplerate_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.samplerate_entry)
        self.samplerate_var.trace('w', self.on_samplerate_changed)
        
        hz_label = Label(
            samplerate_frame,
            text="Hz",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Arial", 8)
        )
        hz_label.pack(side=LEFT)
    
    def build_resolution_settings(self, parent: Frame) -> None:
        resolution_settings = Frame(parent, bg=self.bg_color)
        resolution_settings.grid(row=0, column=2, padx=5, sticky="nsew")
        
        resolution_label = Label(
            resolution_settings,
            text="Output Resolution:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        resolution_label.pack(anchor=W)
        
        self.resolution_var = StringVar(value=ResolutionMode.ORIGINAL.value)
        
        # Filter out GPU options on ARM64
        import platform
        machine = platform.machine().lower()
        if machine in ['aarch64', 'arm64', 'armv8l', 'armv7l']:
            resolution_modes = [mode for mode in ResolutionMode if not mode.value.startswith("AI") or "CPU" in mode.value]
        else:
            resolution_modes = list(ResolutionMode)
        self.resolution_menu = OptionMenu(
            resolution_settings,
            self.resolution_var,
            *[mode.value for mode in resolution_modes]
        )
        self._configure_dropdown(self.resolution_menu, width=26)
        self.resolution_menu.pack(fill=X)
        self.resolution_var.trace('w', self.on_resolution_changed)
        
        self.custom_res_frame = Frame(resolution_settings, bg=self.bg_color)
        
        self.custom_wh_frame = Frame(self.custom_res_frame, bg=self.bg_color)
        
        wh_label = Label(
            self.custom_wh_frame,
            text="Width x Height:",
            bg=self.bg_color,
            fg=self.fg_color,
            width=12
        )
        wh_label.pack(side=LEFT)
        
        self.width_entry = Entry(
            self.custom_wh_frame,
            width=6,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            state=DISABLED
        )
        self.width_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.width_entry)
        
        x_label = Label(
            self.custom_wh_frame,
            text="x",
            bg=self.bg_color,
            fg=self.fg_color
        )
        x_label.pack(side=LEFT)
        
        self.height_entry = Entry(
            self.custom_wh_frame,
            width=6,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            state=DISABLED
        )
        self.height_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.height_entry)
        
        self.custom_dim_frame = Frame(self.custom_res_frame, bg=self.bg_color)
        
        self.custom_dim_label = Label(
            self.custom_dim_frame,
            text="Dimension:",
            bg=self.bg_color,
            fg=self.fg_color,
            width=12
        )
        self.custom_dim_label.pack(side=LEFT)
        
        self.custom_dim_entry = Entry(
            self.custom_dim_frame,
            width=8,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            state=DISABLED
        )
        self.custom_dim_entry.pack(side=LEFT, padx=2)
        self._add_paste_support(self.custom_dim_entry)
        
        apply_button = Button(
            self.custom_res_frame,
            text="Apply",
            bg="#2196F3",
            fg=self.button_fg,
            command=self.apply_custom_resolution,
            width=8,
            height=1,
            font=("Arial", 9),
            relief=RAISED,
            bd=1
        )
        apply_button.pack(pady=(5, 0))
        
        self.current_res_label = Label(
            resolution_settings,
            text="Current: Original",
            bg=self.bg_color,
            fg="#888888",
            font=("Arial", 8)
        )
        self.current_res_label.pack(anchor=W, pady=(2, 0))
    
    def build_output_settings(self, parent: Frame) -> None:
        self.output_format_frame = Frame(parent, bg=self.bg_color)
        self.output_format_frame.grid(row=0, column=3, padx=5, sticky="nsew")
        
        format_label = Label(
            self.output_format_frame,
            text="Output Format:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        format_label.pack(anchor=W)
        
        self.output_format_var = StringVar(value=self.output_settings.format.value)
        self.output_format_menu = OptionMenu(
            self.output_format_frame,
            self.output_format_var,
            *[fmt.value for fmt in VideoFormat]
        )
        self._configure_dropdown(self.output_format_menu, width=7)
        self.output_format_menu.pack(fill=X)
        self.output_format_var.trace('w', self.on_output_format_changed)
        
        self.custom_output_frame = Frame(self.output_format_frame, bg=self.bg_color)
        
        custom_output_label = Label(
            self.custom_output_frame,
            text="Custom Format:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        custom_output_label.pack(anchor=W)
        
        self.custom_output_entry = Entry(
            self.custom_output_frame,
            width=15,
            bg=self.entry_bg,
            fg=self.entry_fg,
            insertbackground=self.entry_fg,
            relief=SUNKEN,
            bd=1,
            textvariable=self.custom_output_var
        )
        self.custom_output_entry.pack(fill=X)
        self._add_paste_support(self.custom_output_entry)
        self.custom_output_var.trace('w', self.on_custom_output_changed)
        
        custom_note_label = Label(
            self.custom_output_frame,
            text="Enter extension (e.g., 'avi', 'mp4')",
            bg=self.bg_color,
            fg="#888888",
            font=("Arial", 7)
        )
        custom_note_label.pack(anchor=W)
    
    def build_preset_controls(self, parent: Frame) -> None:
        preset_controls = Frame(parent, bg=self.bg_color)
        preset_controls.grid(row=0, column=4, padx=5, sticky="nsew")
        
        # Add padding to align with convert button
        preset_label = Label(
            preset_controls,
            text="Presets:",
            bg=self.bg_color,
            fg=self.fg_color
        )
        preset_label.pack(anchor=W, pady=(2, 0))  
        
        save_preset_btn = Button(
            preset_controls,
            text="Save Preset",
            bg=self.accent_color2,
            fg=self.button_fg,
            command=self.save_current_preset,
            width=12,
            height=1,
            relief=RAISED,
            bd=1
        )
        save_preset_btn.pack(pady=(0, 2))
        
        load_preset_btn = Button(
            preset_controls,
            text="Load Preset",
            bg=self.accent_color2,
            fg=self.button_fg,
            command=self.load_selected_preset,
            width=12,
            height=1,
            relief=RAISED,
            bd=1
        )
        load_preset_btn.pack(pady=2) 
    
    def on_output_format_changed(self, *args):
        try:
            self.output_settings.format = VideoFormat(self.output_format_var.get())
            
            if self.output_settings.format == VideoFormat.CUSTOM:
                if not self.custom_output_frame.winfo_ismapped():
                    self.custom_output_frame.pack(fill=X, pady=(5, 0))
            else:
                if self.custom_output_frame.winfo_ismapped():
                    self.custom_output_frame.pack_forget()
        except ValueError:
            pass
    
    def on_custom_output_changed(self, *args):
        self.output_settings.custom_format = self.custom_output_var.get()
    
    def build_convert_controls(self, parent: Frame) -> None:
        convert_frame = Frame(parent, bg=self.bg_color)
        convert_frame.grid(row=0, column=5, padx=(15, 10), sticky="nsew")
        
        # Adjust the convert button position to align with preset buttons
        self.convert_btn = Button(
            convert_frame,
            text="CONVERT",
            bg=self.accent_color,
            fg=self.button_fg,
            command=self.convert,
            font=("Arial", 12, "bold"),
            relief=RAISED,
            bd=2,
            padx=15,
            pady=10,
            width=8,
            height=2
        )
        self.convert_btn.pack(pady=(23, 0))  
        
        self.control_buttons_frame = Frame(convert_frame, bg=self.bg_color)
        
        self.pause_resume_btn = Button(
            self.control_buttons_frame,
            text="PAUSE",
            bg="#FF9800",
            fg=self.button_fg,
            command=self.toggle_pause_resume,
            font=("Arial", 10, "bold"),
            relief=RAISED,
            bd=1,
            padx=10,
            pady=8,
            width=6,
            height=1
        )
        self.pause_resume_btn.pack(side=LEFT, padx=(23, 5))
        
        self.stop_btn = Button(
            self.control_buttons_frame,
            text="STOP",
            bg="#f44336",
            fg=self.button_fg,
            command=self.stop_conversion,
            font=("Arial", 10, "bold"),
            relief=RAISED,
            bd=1,
            padx=10,
            pady=8,
            width=6,
            height=1
        )
        self.stop_btn.pack(side=LEFT)
    
    def _configure_dropdown(self, dropdown: OptionMenu, width: int = 20) -> None:
        if not hasattr(self, '_dropdown_widgets'):
            self._dropdown_widgets = []
        self._dropdown_widgets.append(dropdown)
        
        dropdown.config(
            bg=self.dropdown_bg,
            fg=self.dropdown_fg,
            activebackground=self.highlight_bg,
            activeforeground=self.dropdown_fg,
            highlightbackground=self.border_color,
            highlightthickness=1,
            relief=RAISED,
            bd=1,
            width=width
        )
        
        menu = dropdown['menu']
        menu.config(
            bg=self.menu_bg,
            fg=self.menu_fg,
            activebackground=self.menu_active_bg,
            activeforeground=self.menu_active_fg,
            bd=1,
            relief=RAISED
        )
    
    def _update_current_settings_from_ui(self):
        if self.video_settings.codec == VideoCodec.CUSTOM:
            self.video_settings.custom_encoder = self.video_encoder_var.get()
            self.video_settings.custom_encoder_args = self.video_args_var.get()
        
        if self.audio_settings.codec == AudioCodec.CUSTOM:
            self.audio_settings.custom_encoder = self.audio_encoder_var.get()
            self.audio_settings.custom_encoder_args = self.audio_args_var.get()
        
        if self.speed_var.get():
            self.video_settings.speed_preset = self.speed_var.get()
        
        if self.output_settings.format == VideoFormat.CUSTOM:
            self.output_settings.custom_format = self.custom_output_var.get()
    
    def add_to_queue(self) -> None:
        if not self.video_path:
            messagebox.showwarning("Warning", "No video loaded")
            return
        
        if not self._validate_conversion_settings():
            return
        
        output_path = self._get_output_filename()
        if not output_path:
            return
        
        self._update_current_settings_from_ui()
        
        job = ConversionJob(
            id=self.next_job_id,
            input_path=self.video_path,
            output_path=output_path,
            video_settings=self.video_settings,
            audio_settings=self.audio_settings,
            output_settings=self.output_settings,
            crop_rect=self.crop_rect,
            start_frame=self.output_settings.start_frame,
            end_frame=self.output_settings.end_frame,
            status="Pending",
            video_width=self.video_width,
            video_height=self.video_height,
            video_fps=self.fps
        )
        
        job.ffmpeg_command = self._build_ffmpeg_command_for_job(job)
        
        self.conversion_queue.append(job)
        self.next_job_id += 1
        
        if self.queue_window and self.queue_window.winfo_exists():
            self._update_queue_list()
        
        self.status_label.config(text=f"Added to queue: {os.path.basename(job.input_path)}")
        
        self.save_queue()
        
        messagebox.showinfo("Added to Queue", 
                           f"Conversion job added to queue.\n"
                           f"Total jobs in queue: {len(self.conversion_queue)}")
    
    def _build_ffmpeg_command_for_job(self, job: ConversionJob) -> str:
        cmd = self._build_ffmpeg_command_job(job)
        if cmd:
            return ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
        return ""
    
    def show_queue_manager(self) -> None:
        if self.queue_window and self.queue_window.winfo_exists():
            self.queue_window.lift()
            self.queue_window.focus_set()
            return
        
        self.queue_window = Toplevel(self.root)
        self.queue_window.title("Conversion Queue Manager")
        self.queue_window.geometry("700x550")
        self.queue_window.configure(bg=self.bg_color)
        self.queue_window.resizable(True, True)
        
        self.queue_window.transient(self.root)
        self.queue_window.grab_set()
        
        main_container = Frame(self.queue_window, bg=self.bg_color)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        header_frame = Frame(main_container, bg=self.bg_color)
        header_frame.pack(fill=X, pady=(0, 10))
        
        Label(
            header_frame,
            text="Conversion Queue",
            font=("Arial", 14, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        ).pack(side=LEFT)
        
        Label(
            header_frame,
            text=f"Total: {len(self.conversion_queue)}",
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#888888"
        ).pack(side=RIGHT)
        
        top_control_frame = Frame(main_container, bg=self.bg_color)
        top_control_frame.pack(fill=X, pady=(0, 10))
        
        Button(
            top_control_frame,
            text="Start Queue",
            bg=self.accent_color,
            fg=self.button_fg,
            command=self.start_queue_processing,
            width=12,
            height=1
        ).pack(side=LEFT, padx=(0, 5))
        
        Button(
            top_control_frame,
            text="Clear All",
            bg='#E02C00',
            fg=self.button_fg,
            command=self.clear_queue,
            width=10,
            height=1
        ).pack(side=LEFT, padx=(0, 5))
        
        self.show_command_btn = Button(
            top_control_frame,
            text="Show Command",
            bg=self.accent_color2,
            fg=self.button_fg,
            command=self.show_command_preview,
            width=12,
            height=1,
        )
        self.show_command_btn.pack(side=LEFT, padx=(0, 5))
        
        Button(
            top_control_frame,
            text="Close",
            bg=self.button_bg,
            fg=self.button_fg,
            command=self.queue_window.destroy,
            width=8,
            height=1
        ).pack(side=RIGHT)
        
        list_frame = Frame(main_container, bg=self.bg_color)
        list_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        tree_container = Frame(list_frame, bg=self.bg_color)
        tree_container.pack(fill=BOTH, expand=True)
        
        columns = ("ID", "File", "Status", "Progress", "Command")
        self.queue_tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show="headings",
            height=15,
            style="Custom.Treeview"
        )
        
        self.queue_tree.heading("ID", text="ID")
        self.queue_tree.heading("File", text="File")
        self.queue_tree.heading("Status", text="Status")
        self.queue_tree.heading("Progress", text="Progress")
        self.queue_tree.heading("Command", text="Command Preview")
        
        self.queue_tree.column("ID", width=50, anchor="center")
        self.queue_tree.column("File", width=200, anchor="w")
        self.queue_tree.column("Status", width=100, anchor="center")
        self.queue_tree.column("Progress", width=80, anchor="center")
        self.queue_tree.column("Command", width=200, anchor="w")
        
        scrollbar = ttk.Scrollbar(tree_container, orient=VERTICAL, command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=scrollbar.set)
        
        self.queue_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.queue_tree.bind("<<TreeviewSelect>>", self.on_queue_selection)
        
        item_control_frame = Frame(main_container, bg=self.bg_color)
        item_control_frame.pack(fill=X, pady=(0, 10))
        
        Button(
            item_control_frame,
            text="Remove Selected",
            bg="#ff5a5a",
            fg=self.button_fg,
            command=self.remove_selected_job,
            width=15,
            height=1
        ).pack(side=LEFT, padx=(0, 10))
        
        Button(
            item_control_frame,
            text="Move Up",
            bg=self.button_bg,
            fg=self.button_fg,
            command=self.move_job_up,
            width=10,
            height=1
        ).pack(side=LEFT, padx=(0, 10))
        
        Button(
            item_control_frame,
            text="Move Down",
            bg=self.button_bg,
            fg=self.button_fg,
            command=self.move_job_down,
            width=10,
            height=1
        ).pack(side=LEFT)
        
        info_frame = Frame(main_container, bg=self.bg_color)
        info_frame.pack(fill=X)
        
        Label(
            info_frame,
            text="Double-click a job to view full FFmpeg command | Select job and click 'Show Command' to preview FFmpeg command",
            font=("Arial", 9),
            bg=self.bg_color,
            fg="#888888"
        ).pack(side=LEFT)
        
        self._update_queue_list()
        
        self.queue_tree.bind("<Double-1>", lambda e: self.show_command_preview_double_click())
        
        self.queue_window.protocol("WM_DELETE_WINDOW", self.queue_window.destroy)
        
        self._update_widget_tree(self.queue_window)
    
    def show_command_preview_double_click(self):
        selection = self.queue_tree.selection()
        if not selection:
            return
        self.show_command_preview()
    
    def on_queue_selection(self, event):
        selection = self.queue_tree.selection()
        if selection:
            self.show_command_btn.config(state=NORMAL)
        else:
            self.show_command_btn.config(state=DISABLED)
    
    def show_command_preview(self):
        selection = self.queue_tree.selection()
        if not selection:
            return
        
        selected_item = selection[0]
        item_index = self.queue_tree.index(selected_item)
        
        if 0 <= item_index < len(self.conversion_queue):
            job = self.conversion_queue[item_index]
            
            preview_window = Toplevel(self.queue_window)
            preview_window.title(f"FFmpeg Command Preview - Job {job.id}")
            preview_window.geometry("900x500")
            preview_window.configure(bg=self.bg_color)
            
            header = Frame(preview_window, bg=self.bg_color)
            header.pack(fill=X, padx=10, pady=10)
            
            Label(
                header,
                text=f"FFmpeg Command for: {os.path.basename(job.input_path)}",
                font=("Arial", 12, "bold"),
                bg=self.bg_color,
                fg=self.fg_color
            ).pack(side=LEFT)
            
            text_frame = Frame(preview_window, bg=self.bg_color)
            text_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))
            
            text_widget = Text(
                text_frame,
                wrap=NONE,
                bg=self.entry_bg,
                fg=self.entry_fg,
                font=("Courier", 10),
                height=20
            )
            
            scrollbar_x = Scrollbar(text_frame, orient=HORIZONTAL, command=text_widget.xview)
            scrollbar_y = Scrollbar(text_frame, orient=VERTICAL, command=text_widget.yview)
            text_widget.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
            
            scrollbar_y.pack(side=RIGHT, fill=Y)
            scrollbar_x.pack(side=BOTTOM, fill=X)
            text_widget.pack(side=LEFT, fill=BOTH, expand=True)
            
            if job.ffmpeg_command:
                text_widget.insert(END, job.ffmpeg_command)
            else:
                text_widget.insert(END, "No FFmpeg command available for this job.")
            
            text_widget.configure(state=DISABLED)
            
            button_frame = Frame(preview_window, bg=self.bg_color)
            button_frame.pack(fill=X, padx=10, pady=(0, 10))
            
            Button(
                button_frame,
                text="Copy Command",
                bg=self.accent_color2,
                fg=self.button_fg,
                command=lambda: self.copy_command_to_clipboard(job.ffmpeg_command or ""),
                width=15
            ).pack(side=LEFT, padx=(0, 10))
            
            Button(
                button_frame,
                text="Edit Command",
                bg=self.accent_color,
                fg=self.button_fg,
                command=lambda: self.enable_command_editing(text_widget, job),
                width=15
            ).pack(side=LEFT, padx=(0, 10))
            
            Button(
                button_frame,
                text="Close",
                bg=self.button_bg,
                fg=self.button_fg,
                command=preview_window.destroy,
                width=10
            ).pack(side=RIGHT)
    
    def enable_command_editing(self, text_widget, job):
        text_widget.configure(state=NORMAL)
        messagebox.showinfo("Edit Mode", "You can now edit the FFmpeg command. Click 'Save Changes' when done.")
        
        parent = text_widget.master.master
        save_button = Button(
            parent.children['!frame2'],
            text="Save Changes",
            bg="#4CAF50",
            fg=self.button_fg,
            command=lambda: self.save_edited_command(text_widget, job),
            width=15
        )
        save_button.pack(side=LEFT, padx=(0, 10))
    
    def save_edited_command(self, text_widget, job):
        edited_command = text_widget.get("1.0", END).strip()
        job.ffmpeg_command = edited_command
        text_widget.configure(state=DISABLED)
        messagebox.showinfo("Saved", "FFmpeg command updated successfully!")
        
        if self.queue_window and self.queue_window.winfo_exists():
            self._update_queue_list()
        
        self.save_queue()
    
    def copy_command_to_clipboard(self, command):
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        self.root.update()
        messagebox.showinfo("Copied", "FFmpeg command copied to clipboard!")
    
    def _update_queue_list(self) -> None:
        if not hasattr(self, 'queue_tree') or not self.queue_tree:
            return
        
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)
        
        for i, job in enumerate(self.conversion_queue):
            progress_text = f"{job.progress:.1f}%" if job.progress > 0 else "-"
            
            cmd_preview = ""
            if job.ffmpeg_command:
                cmd_preview = job.ffmpeg_command[:100] + "..." if len(job.ffmpeg_command) > 100 else job.ffmpeg_command
            
            tags = ()
            if job == self.current_job:
                tags = ("current",)
            
            self.queue_tree.insert(
                "", "end",
                values=(job.id, job.get_filename(), job.status, progress_text, cmd_preview),
                tags=tags
            )
        
        self.queue_tree.tag_configure("current", background=self.accent_color, foreground="white")
    
    def remove_selected_job(self) -> None:
        selection = self.queue_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No job selected")
            return
        
        selected_item = selection[0]
        item_index = self.queue_tree.index(selected_item)
        
        if 0 <= item_index < len(self.conversion_queue):
            if self.conversion_queue[item_index] == self.current_job:
                messagebox.showwarning("Warning", "Cannot remove currently running job")
                return
            
            response = messagebox.askyesno(
                "Remove Job",
                f"Are you sure you want to remove job {self.conversion_queue[item_index].id}: {self.conversion_queue[item_index].get_filename()}?",
                icon=messagebox.WARNING
            )
            
            if response:
                removed_job = self.conversion_queue.pop(item_index)
                self._update_queue_list()
                
                if hasattr(self, 'queue_window') and self.queue_window.winfo_exists():
                    for widget in self.queue_window.winfo_children():
                        if isinstance(widget, Frame):
                            for child in widget.winfo_children():
                                if isinstance(child, Label) and "Total:" in child.cget("text"):
                                    child.config(text=f"Total: {len(self.conversion_queue)}")
                
                self.status_label.config(text=f"Removed from queue: {removed_job.get_filename()}")
                
                self.save_queue()
    
    def move_job_up(self) -> None:
        selection = self.queue_tree.selection()
        if not selection:
            return
        
        selected_item = selection[0]
        item_index = self.queue_tree.index(selected_item)
        
        if item_index > 0:
            self.conversion_queue[item_index], self.conversion_queue[item_index - 1] = \
                self.conversion_queue[item_index - 1], self.conversion_queue[item_index]
            self._update_queue_list()
            
            new_index = item_index - 1
            children = self.queue_tree.get_children()
            if new_index < len(children):
                self.queue_tree.selection_set(children[new_index])
            
            self.save_queue()
    
    def move_job_down(self) -> None:
        selection = self.queue_tree.selection()
        if not selection:
            return
        
        selected_item = selection[0]
        item_index = self.queue_tree.index(selected_item)
        
        if item_index < len(self.conversion_queue) - 1:
            self.conversion_queue[item_index], self.conversion_queue[item_index + 1] = \
                self.conversion_queue[item_index + 1], self.conversion_queue[item_index]
            self._update_queue_list()
            
            new_index = item_index + 1
            children = self.queue_tree.get_children()
            if new_index < len(children):
                self.queue_tree.selection_set(children[new_index])
            
            self.save_queue()
    
    def clear_queue(self) -> None:
        if not self.conversion_queue:
            return
        
        response = messagebox.askyesno(
            "Clear Queue",
            "Are you sure you want to clear all jobs from the queue?",
            icon=messagebox.WARNING
        )
        
        if response:
            if self.current_job and self.state in [AppState.CONVERTING, AppState.CONVERSION_PAUSED]:
                self.conversion_queue = [self.current_job]
            else:
                self.conversion_queue = []
            
            self.next_job_id = 1
            
            self._update_queue_list()
            
            if hasattr(self, 'queue_window') and self.queue_window.winfo_exists():
                for widget in self.queue_window.winfo_children():
                    if isinstance(widget, Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, Label) and "Total:" in child.cget("text"):
                                child.config(text=f"Total: {len(self.conversion_queue)}")
            
            self.status_label.config(text="Queue cleared")
            
            self.save_queue()
    
    def start_queue_processing(self) -> None:
        if not self.conversion_queue:
            messagebox.showinfo("Queue Empty", "No jobs in the queue")
            return
        
        if self.state in [AppState.CONVERTING, AppState.CONVERSION_PAUSED]:
            messagebox.showinfo("Already Converting", "A conversion is already in progress")
            return
        
        self._process_next_job()
    
    def _process_next_job(self) -> None:
        if not self.conversion_queue:
            self.status_label.config(text="Queue empty")
            return
        
        self.current_job = self.conversion_queue[0]
        self.current_job.status = "Processing"
        
        if self.queue_window and self.queue_window.winfo_exists():
            self._update_queue_list()
        
        if self.current_job.video_settings.is_ai_enhancement():
            self._convert_job_with_ai(self.current_job)
        else:
            self._convert_job(self.current_job)
    
    def _convert_job_with_ai(self, job: ConversionJob) -> None:
        self.video_path = job.input_path
        
        try:
            cap = cv2.VideoCapture(job.input_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {job.input_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load video for job {job.id}: {e}")
            job.status = "Failed"
            if self.queue_window and self.queue_window.winfo_exists():
                self._update_queue_list()
            self._job_completed()
            return
        
        total_duration = total_frames / fps
        if job.start_frame is not None or job.end_frame is not None:
            start_time = job.start_frame / fps if job.start_frame else 0
            end_time = job.end_frame / fps if job.end_frame else total_duration
            self.segment_duration = max(0.001, end_time - start_time)
        else:
            self.segment_duration = total_duration
            start_time = 0
        
        self.conversion_start_time_original = start_time
        self.conversion_start_time = time.time()
        self.last_progress_update_time = self.conversion_start_time
        self.last_percentage = 0.0
        
        # Check AI backend
        self._check_ai_availability()
        
        scale_factor = job.video_settings.get_ai_scale_factor()
        is_gpu = job.video_settings.is_gpu_ai_enhancement()
        
        backend = "Unknown"
        if is_gpu and self.realesrgan_available:
            backend = "GPU (Real-ESRGAN)"
            if self.realesrgan_vulkan_available:
                backend += " with Vulkan"
            else:
                backend += " (CPU fallback)"
        elif self.super_image_available:
            backend = "CPU (super-image)"
        else:
            backend = "CPU (simple resize)"
        
        self.root.after(0, lambda: messagebox.showinfo(
    "AI Enhancement",
    f"Starting AI {scale_factor}x enhancement using {backend}.\n\n"
    f"This process uses {'GPU acceleration' if is_gpu else 'CPU processing'}, "
    f"and the processing time depends on several factors:\n"
    f"• Video length and resolution\n"
    f"• AI scale factor ({scale_factor}x)\n"
    f"• Hardware acceleration: {'Enabled' if is_gpu and self.realesrgan_available else 'Not available'}\n\n"
    f"GPU acceleration will significantly speed up processing when available."
))
        
        self.state = AppState.CONVERTING
        self.conversion_paused = False
        self.conversion_pause_event.clear()
        self.convert_btn.config(state=DISABLED)
        
        self.convert_btn.pack_forget()
        self.control_buttons_frame.pack(pady=(0, 0))
        self.pause_resume_btn.config(text="PAUSE", bg="#FF9800")
        
        self.status_label.config(text=f"AI Enhancing: {job.get_filename()}")
        
        self.show_progress_in_status(True)
        self.progress_bar['value'] = 0
        self.progress_percentage.config(text="0.0%")
        self.time_label.config(text="AI Processing...")
        
        self.conversion_stop_event.clear()
        self.conversion_thread = threading.Thread(
            target=self._run_ai_conversion_job,
            args=(job,),
            daemon=True
        )
        self.conversion_thread.start()
    
    def _run_ai_conversion_job(self, job: ConversionJob) -> None:
        try:
            # Initialize AI model with appropriate backend
            is_gpu = job.video_settings.is_gpu_ai_enhancement()
            self._initialize_ai_model(use_gpu=is_gpu)
            
            video_dir = (os.path.dirname(job.input_path) if 'job' in locals() else (os.path.dirname(self.video_path) if hasattr(self, 'video_path') and self.video_path else os.getcwd()))
            with tempfile.TemporaryDirectory(prefix="favencoder_ai_", dir=video_dir) as temp_dir:
                frames_dir = os.path.join(temp_dir, "frames")
                enhanced_dir = os.path.join(temp_dir, "enhanced")
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(enhanced_dir, exist_ok=True)
                
                scale_factor = job.video_settings.get_ai_scale_factor()
                
                extract_cmd = ["ffmpeg"]
                
                if job.start_frame is not None:
                    start_time = job.start_frame / job.video_fps
                    extract_cmd.extend(["-ss", f"{start_time:.6f}"])
                
                extract_cmd.extend(["-i", job.input_path])
                
                if job.end_frame is not None and job.start_frame is not None:
                    duration = (job.end_frame - job.start_frame) / job.video_fps
                    extract_cmd.extend(["-t", f"{duration:.6f}"])
                elif job.end_frame is not None:
                    duration = job.end_frame / job.video_fps
                    extract_cmd.extend(["-t", f"{duration:.6f}"])
                
                if job.crop_rect:
                    extract_cmd.extend(["-vf", f"crop={job.crop_rect.width}:{job.crop_rect.height}:{job.crop_rect.x}:{job.crop_rect.y}"])
                
                extract_cmd.extend([
                    "-vsync", "0",
                    os.path.join(frames_dir, "frame_%06d.png")
                ])
                
                extract_process = subprocess.Popen(
                    extract_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                extract_process.wait()
                
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
                total_frames = len(frame_files)
                
                if total_frames == 0:
                    raise ValueError("No frames extracted from video")
                
                for i, frame_file in enumerate(frame_files):
                    if self.conversion_stop_event.is_set():
                        break
                    
                    progress_percentage = (i / total_frames) * 50
                    job.progress = progress_percentage
                    
                    self.root.after(0, lambda p=progress_percentage: self._update_ai_progress(p, "AI Processing Frames"))
                    
                    frame_path = os.path.join(frames_dir, frame_file)
                    img = Image.open(frame_path).convert('RGB')
                    
                    try:
                        enhanced_img = self._enhance_image_with_ai(img, scale_factor)
                    except Exception as e:
                        if ENABLE_LOGGING: logger.error(f"Failed to enhance frame {frame_file}: {e}")
                        new_width = img.width * scale_factor
                        new_height = img.height * scale_factor
                        enhanced_img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
                    
                    enhanced_path = os.path.join(enhanced_dir, frame_file)
                    enhanced_img.save(enhanced_path)
                
                if self.conversion_stop_event.is_set():
                    job.status = "Stopped"
                    self.root.after(0, self._job_completed, job, False, "Stopped by user")
                    return
                
                temp_video_path = os.path.join(temp_dir, "enhanced_video.mkv")
                
                video_cmd = [
                    "ffmpeg",
                    "-framerate", str(job.video_fps),
                    "-i", os.path.join(enhanced_dir, "frame_%06d.png"),
                    "-i", job.input_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-y", temp_video_path
                ]
                
                video_process = subprocess.Popen(
                    video_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                while not self.conversion_stop_event.is_set():
                    if self.conversion_pause_event.is_set():
                        while (self.conversion_pause_event.is_set() and 
                               not self.conversion_stop_event.is_set()):
                            time.sleep(0.1)
                        
                        if self.conversion_stop_event.is_set():
                            break
                    
                    line = video_process.stderr.readline()
                    if not line:
                        if video_process.poll() is not None:
                            break
                        continue
                    
                    progress = self._parse_ffmpeg_progress_with_eta(line)
                    if progress:
                        progress.percentage = 50 + (progress.percentage / 2)
                        job.progress = progress.percentage
                        self.root.after(0, self._update_progress_with_eta, progress, line)
                
                returncode = video_process.wait()
                
                if returncode == 0:
                    final_cmd = self._build_ffmpeg_command_for_enhanced_video(temp_video_path, job)
                    
                    if final_cmd:
                        final_process = subprocess.Popen(
                            final_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            bufsize=1
                        )
                        
                        while not self.conversion_stop_event.is_set():
                            if self.conversion_pause_event.is_set():
                                while (self.conversion_pause_event.is_set() and 
                                       not self.conversion_stop_event.is_set()):
                                    time.sleep(0.1)
                                
                                if self.conversion_stop_event.is_set():
                                    break
                            
                            line = final_process.stderr.readline()
                            if not line:
                                if final_process.poll() is not None:
                                    break
                                continue
                            
                            progress = self._parse_ffmpeg_progress_with_eta(line)
                            if progress:
                                self.root.after(0, self._update_progress_with_eta, progress, line)
                        
                        final_returncode = final_process.wait()
                        
                        if final_returncode == 0:
                            job.status = "Completed"
                            job.progress = 100.0
                            self.root.after(0, self._job_completed, job, True)
                        else:
                            job.status = "Failed"
                            self.root.after(0, self._job_completed, job, False, f"Final encoding failed: {final_returncode}")
                    else:
                        shutil.copy2(temp_video_path, job.output_path)
                        job.status = "Completed"
                        job.progress = 100.0
                        self.root.after(0, self._job_completed, job, True)
                else:
                    job.status = "Failed"
                    self.root.after(0, self._job_completed, job, False, f"Video creation failed: {returncode}")
                    
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"AI conversion error for job {job.id}: {e}")
            if not self.conversion_stop_event.is_set():
                job.status = "Failed"
                self.root.after(0, self._job_completed, job, False, str(e))
    
    def _build_ffmpeg_command_for_enhanced_video(self, enhanced_video_path: str, job: ConversionJob) -> Optional[List[str]]:
        try:
            cmd = ["ffmpeg", "-i", enhanced_video_path]
            
            self._add_video_params_job(cmd, job, 
                                      job.video_width * job.video_settings.get_ai_scale_factor(),
                                      job.video_height * job.video_settings.get_ai_scale_factor(),
                                      job.video_fps)
            
            self._add_audio_params_job(cmd, job)
            
            cmd.extend(["-map", "0:v:0"])
            if job.audio_settings.codec != AudioCodec.NO_AUDIO:
                cmd.extend(["-map", "0:a:0"])
            
            cmd.extend(["-y", job.output_path])
            
            if ENABLE_LOGGING: logger.info(f"Final FFmpeg command for AI job {job.id}: {' '.join(cmd)}")
            return cmd
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error building final FFmpeg command for AI job {job.id}: {e}")
            return None
    
    def _update_ai_progress(self, percentage: float, status: str) -> None:
        if not self.progress_container.winfo_ismapped():
            return
        
        self.progress_bar['value'] = min(100, percentage)
        self.progress_percentage.config(text=f"{percentage:.1f}%")
        self.time_label.config(text=status)
        self.root.update_idletasks()
    
    def _convert_job(self, job: ConversionJob) -> None:
        self.video_path = job.input_path
        
        try:
            cap = cv2.VideoCapture(job.input_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {job.input_path}")
            
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            cap.release()
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to load video for job {job.id}: {e}")
            job.status = "Failed"
            if self.queue_window and self.queue_window.winfo_exists():
                self._update_queue_list()
            self._job_completed()
            return
        
        total_duration = self.total_frames / self.fps
        
        if job.start_frame is not None or job.end_frame is not None:
            start_time = job.start_frame / self.fps if job.start_frame else 0
            end_time = job.end_frame / self.fps if job.end_frame else total_duration
            self.segment_duration = max(0.001, end_time - start_time)
        else:
            self.segment_duration = total_duration
            start_time = 0
        
        self.conversion_start_time_original = start_time
        self.conversion_start_time = time.time()
        self.last_progress_update_time = self.conversion_start_time
        self.last_percentage = 0.0
        
        cmd = self._build_ffmpeg_command_job(job)
        if not cmd:
            job.status = "Failed"
            if self.queue_window and self.queue_window.winfo_exists():
                self._update_queue_list()
            self._job_completed()
            return
        
        self.state = AppState.CONVERTING
        self.conversion_paused = False
        self.conversion_pause_event.clear()
        self.convert_btn.config(state=DISABLED)
        
        self.convert_btn.pack_forget()
        self.control_buttons_frame.pack(pady=(0, 0))
        self.pause_resume_btn.config(text="PAUSE", bg="#FF9800")
        
        self.status_label.config(text=f"Converting: {job.get_filename()}")
        
        self.show_progress_in_status(True)
        self.progress_bar['value'] = 0
        self.progress_percentage.config(text="0.0%")
        
        self.conversion_stop_event.clear()
        self.conversion_thread = threading.Thread(
            target=self._run_conversion_job,
            args=(cmd, job),
            daemon=True
        )
        self.conversion_thread.start()
    
    def _build_ffmpeg_command_job(self, job: ConversionJob) -> Optional[List[str]]:
        try:
            if job.video_width and job.video_height and job.video_fps:
                video_width = job.video_width
                video_height = job.video_height
                fps = job.video_fps
            else:
                cap = cv2.VideoCapture(job.input_path)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video: {job.input_path}")
                
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                cap.release()
            
            cmd = ["ffmpeg"]
            
            if job.start_frame is not None:
                start_time = job.start_frame / fps
                cmd.extend(["-ss", f"{start_time:.6f}"])
            
            cmd.extend(["-i", job.input_path])
            
            if job.end_frame is not None and job.start_frame is not None:
                duration = (job.end_frame - job.start_frame) / fps
                cmd.extend(["-t", f"{duration:.6f}"])
            elif job.end_frame is not None:
                duration = job.end_frame / fps
                cmd.extend(["-t", f"{duration:.6f}"])
            
            if job.video_settings.codec == VideoCodec.NO_VIDEO:
                cmd.extend(["-vn"])
                self._add_audio_params_job(cmd, job)
                cmd.extend(["-map", "0:a?"])
            
            else:
                self._add_video_params_job(cmd, job, video_width, video_height, fps)
                self._add_audio_params_job(cmd, job)
                
                cmd.extend(["-map", "0:v:0"])
                if job.audio_settings.codec != AudioCodec.NO_AUDIO:
                    cmd.extend(["-map", "0:a?"])
            
            cmd.extend(["-y", job.output_path])
            
            if ENABLE_LOGGING: logger.info(f"FFmpeg command for job {job.id}: {' '.join(cmd)}")
            return cmd
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error building FFmpeg command for job {job.id}: {e}")
            return None
    
    def _add_video_params_job(self, cmd: List[str], job: ConversionJob, video_width: int, video_height: int, fps: float) -> None:
        if job.video_settings.codec == VideoCodec.CUSTOM:
            if job.video_settings.custom_encoder:
                cmd.extend(["-c:v", job.video_settings.custom_encoder])
                
                if job.video_settings.custom_encoder_args:
                    import shlex
                    try:
                        args = shlex.split(job.video_settings.custom_encoder_args)
                        cmd.extend(args)
                    except Exception as e:
                        if ENABLE_LOGGING: logger.warning(f"Failed to parse custom video args for job {job.id}: {e}")
                        args = job.video_settings.custom_encoder_args.split()
                        cmd.extend(args)
            else:
                cmd.extend(["-c:v", "libx264"])
        else:
            encoder_map = {
                VideoCodec.FFV1: "ffv1",
                VideoCodec.H264_X264: "libx264",
                VideoCodec.H265_X265: "libx265",
                VideoCodec.AV1_SVT: "libsvtav1",
                VideoCodec.VP9: "libvpx-vp9",
                VideoCodec.PRORES: "prores",
                VideoCodec.DNXHD: "dnxhd",
                VideoCodec.MPEG2: "mpeg2video",
                VideoCodec.MJPEG: "mjpeg",
                VideoCodec.H264_NVENC: "h264_nvenc",
                VideoCodec.H265_NVENC: "hevc_nvenc",
                VideoCodec.H264_QSV: "h264_qsv",
                VideoCodec.H265_QSV: "hevc_qsv",
                VideoCodec.H264_AMF: "h264_amf",
                VideoCodec.H265_AMF: "hevc_amf",
                VideoCodec.AV1_AOM: "libaom-av1",
                VideoCodec.VP8: "libvpx",
                VideoCodec.MPEG4: "mpeg4",
                VideoCodec.DV: "dvvideo",
                VideoCodec.RAW: "rawvideo"
            }
            
            encoder = encoder_map.get(job.video_settings.codec)
            if not encoder:
                return
            
            cmd.extend(["-c:v", encoder])
            
            is_lossy = not any(lossless in job.video_settings.codec.value.lower() 
                             for lossless in ["lossless", "pcm", "raw", "ffv1"])
            
            if is_lossy and job.video_settings.speed_preset:
                preset = job.video_settings.speed_preset
                
                if "nvenc" in encoder:
                    cmd.extend(["-preset", preset])
                elif "qsv" in encoder:
                    cmd.extend(["-preset", preset])
                elif "amf" in encoder:
                    cmd.extend(["-quality", preset])
                elif "svtav1" in encoder:
                    cmd.extend(["-preset", preset])
                elif "libaom-av1" in encoder:
                    cmd.extend(["-cpu-used", preset])
                elif "libvpx" in encoder:
                    cmd.extend(["-cpu-used", preset])
                elif encoder in ["libx264", "libx265", "mpeg4", "mpeg2video", "mjpeg"]:
                    cmd.extend(["-preset", preset])
            
            if job.video_settings.quality_mode == QualityMode.CQ:
                cq_value = job.video_settings.cq_value
                
                if "nvenc" in encoder:
                    cmd.extend(["-cq", str(cq_value)])
                elif "qsv" in encoder:
                    cmd.extend(["-global_quality", str(cq_value)])
                elif "amf" in encoder:
                    cmd.extend(["-qp_i", str(cq_value), "-qp_p", str(cq_value), "-qp_b", str(cq_value)])
                elif "svtav1" in encoder:
                    cmd.extend(["-crf", str(cq_value)])
                    if job.video_settings.speed_preset:
                        cmd.extend(["-preset", str(job.video_settings.speed_preset)])
                elif encoder in ["libx264", "libx265", "libvpx-vp9", "libaom-av1"]:
                    cmd.extend(["-crf", str(cq_value)])
                else:
                    cmd.extend(["-crf", str(cq_value)])
                
            else:
                bitrate = job.video_settings.bitrate
                cmd.extend(["-b:v", f"{bitrate}k"])
                
                if job.video_settings.bitrate_type == BitrateType.CBR:
                    cmd.extend([
                        "-maxrate", f"{bitrate}k",
                        "-minrate", f"{bitrate}k",
                        "-bufsize", f"{bitrate * 2}k"
                    ])
        
        vfilters = []
        
        if job.crop_rect:
            vfilters.append(f"crop={job.crop_rect.width}:{job.crop_rect.height}:{job.crop_rect.x}:{job.crop_rect.y}")
        
        if job.video_settings.resolution_mode != ResolutionMode.ORIGINAL and not job.video_settings.is_ai_enhancement():
            out_width, out_height = job.video_settings.calculate_output_dimensions(
                video_width, video_height, job.crop_rect
            )
            scale_filter = f"scale={out_width}:{out_height}"
            vfilters.append(scale_filter)
        
        if vfilters:
            cmd.extend(["-vf", ",".join(vfilters)])
        
        if job.video_settings.output_fps:
            cmd.extend(["-r", str(job.video_settings.output_fps)])
    
    def _add_audio_params_job(self, cmd: List[str], job: ConversionJob) -> None:
        if job.audio_settings.codec == AudioCodec.NO_AUDIO:
            cmd.extend(["-an"])
            return
        
        if job.audio_settings.codec == AudioCodec.CUSTOM:
            if job.audio_settings.custom_encoder:
                cmd.extend(["-c:a", job.audio_settings.custom_encoder])
                
                if job.audio_settings.custom_encoder_args:
                    import shlex
                    try:
                        args = shlex.split(job.audio_settings.custom_encoder_args)
                        cmd.extend(args)
                    except Exception as e:
                        if ENABLE_LOGGING: logger.warning(f"Failed to parse custom audio args for job {job.id}: {e}")
                        args = job.audio_settings.custom_encoder_args.split()
                        cmd.extend(args)
            else:
                cmd.extend(["-c:a", "aac"])
        else:
            encoder_map = {
                AudioCodec.FLAC: "flac",
                AudioCodec.PCM_16: "pcm_s16le",
                AudioCodec.PCM_24: "pcm_s24le",
                AudioCodec.PCM_32: "pcm_s32le",
                AudioCodec.AAC: "aac",
                AudioCodec.OPUS: "libopus",
                AudioCodec.MP3: "libmp3lame",
                AudioCodec.AC3: "ac3",
                AudioCodec.DTS: "dca",
                AudioCodec.VORBIS: "libvorbis"
            }
            
            encoder = encoder_map.get(job.audio_settings.codec)
            if not encoder:
                return
            
            cmd.extend(["-c:a", encoder])
            
            if job.audio_settings.codec not in [AudioCodec.FLAC, 
                                               AudioCodec.PCM_16,
                                               AudioCodec.PCM_24,
                                               AudioCodec.PCM_32]:
                bitrate = job.audio_settings.bitrate
                cmd.extend(["-b:a", f"{bitrate}k"])
            
            samplerate = job.audio_settings.samplerate
            cmd.extend(["-ar", str(samplerate)])
    
    def _run_conversion_job(self, cmd: List[str], job: ConversionJob) -> None:
        try:
            self.conversion_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            while not self.conversion_stop_event.is_set():
                if self.conversion_pause_event.is_set():
                    while (self.conversion_pause_event.is_set() and 
                           not self.conversion_stop_event.is_set()):
                        time.sleep(0.1)
                    
                    if self.conversion_stop_event.is_set():
                        break
                
                line = self.conversion_process.stderr.readline()
                if not line:
                    if self.conversion_process.poll() is not None:
                        break
                    continue
                
                progress = self._parse_ffmpeg_progress_with_eta(line)
                if progress:
                    job.progress = progress.percentage
                    self.root.after(0, self._update_progress_with_eta, progress, line)
            
            returncode = self.conversion_process.wait()
            
            if returncode == 0:
                job.status = "Completed"
                job.progress = 100.0
                self.root.after(0, self._job_completed, job, True)
            else:
                if not self.conversion_stop_event.is_set():
                    job.status = "Failed"
                    self.root.after(0, self._job_completed, job, False, f"FFmpeg returned {returncode}")
                else:
                    job.status = "Stopped"
                    self.root.after(0, self._job_completed, job, False, "Stopped by user")
                    
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Conversion error for job {job.id}: {e}")
            if not self.conversion_stop_event.is_set():
                job.status = "Failed"
                self.root.after(0, self._job_completed, job, False, str(e))
            
        finally:
            self.conversion_process = None
    
    def _job_completed(self, job: Optional[ConversionJob] = None, success: bool = True, error: str = "") -> None:
        if job and success:
            if job in self.conversion_queue:
                self.conversion_queue.remove(job)
        
        self._reset_conversion_ui()
        self.state = AppState.VIDEO_LOADED
        
        if self.queue_window and self.queue_window.winfo_exists():
            self._update_queue_list()
        
        if success:
            if job:
                # Calculate time taken for queue jobs
                if hasattr(self, 'conversion_start_time'):
                    time_taken = time.time() - self.conversion_start_time
                    hours = int(time_taken // 3600)
                    minutes = int((time_taken % 3600) // 60)
                    seconds = int(time_taken % 60)
                    
                    if hours > 0:
                        time_str = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}"
                    elif minutes > 0:
                        time_str = f"{minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}"
                    else:
                        time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
                    
                    self.status_label.config(text=f"Completed after {time_str}: {job.get_filename()}")
                else:
                    self.status_label.config(text=f"Completed: {job.get_filename()}")
                if not self.conversion_queue:
                    messagebox.showinfo("Queue Complete", "All jobs in the queue have been completed!")
            else:
                self.status_label.config(text="Conversion complete")
            
            if self.conversion_queue:
                self.root.after(1000, self._process_next_job)
        else:
            if error:
                self.status_label.config(text=f"Failed: {error}")
                messagebox.showerror("Error", f"Conversion failed: {error}")
            
            self.current_job = None
        
        self.save_queue()
    
    def _update_progress_with_eta(self, progress: ConversionProgress, line: str) -> None:
        if not self.progress_container.winfo_ismapped():
            return
        
        if hasattr(progress, 'percentage') and progress.percentage is not None:
            self.progress_bar['value'] = min(100, progress.percentage)
            self.progress_percentage.config(text=f"{progress.percentage:.1f}%")
            
            elapsed_str = format_time_compact(progress.elapsed_time) if hasattr(progress, 'elapsed_time') else "00:00:00"
            if hasattr(progress, 'estimated_remaining') and progress.estimated_remaining and progress.estimated_remaining > 0:
                remaining_str = format_time_compact(progress.estimated_remaining)
                self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            else:
                self.time_label.config(text=f"Elapsed: {elapsed_str}")
        else:
            self.progress_bar['value'] = 0
            self.progress_percentage.config(text="0.0%")
            self.time_label.config(text="Converting...")
        
        self.root.update_idletasks()
    
    def _reset_conversion_ui(self) -> None:
        self.control_buttons_frame.pack_forget()
        self.convert_btn.pack(pady=(0, 0))
        self.convert_btn.config(state=NORMAL)
        
        self.hide_progress_in_status()
        
        if self.state in [AppState.CONVERTING, AppState.CONVERSION_PAUSED]:
            self.state = AppState.VIDEO_LOADED
        
        self.conversion_paused = False
        self.conversion_pause_event.clear()
        self.current_job = None
    
    def update_video_info_label(self) -> None:
        if not self.video_path or not self.cap:
            self.video_info_label.config(text="No video loaded")
            return
        
        aspect_str = simplify_aspect_ratio(self.video_width, self.video_height)
        
        if self.video_codec_var.get() == VideoCodec.NO_VIDEO.value:
            self.video_info_label.config(
                text=f"Audio Only | Duration: {format_time(self.duration)}"
            )
        else:
            out_width, out_height = self.video_settings.calculate_output_dimensions(
                self.video_width, self.video_height, self.crop_rect
            )
            
            info = (
                f"Original: {self.video_width}×{self.video_height} ({aspect_str}) | "
                f"Output: {out_width}×{out_height} | "
                f"FPS: {self.fps:.2f} | "
                f"Duration: {format_time(self.duration)}"
            )
            self.video_info_label.config(text=info)
        
        self.update_estimated_size()
    
    def show_frame(self, frame_num: Optional[int] = None) -> None:
        if not self.cap or not self.video_path:
            return
        
        if self.video_codec_var.get() == VideoCodec.NO_VIDEO.value:
            self.show_audio_only_message()
            return
        
        if frame_num is not None:
            self.current_frame = max(0, min(frame_num, self.total_frames - 1))
        
        with self.frame_lock:
            cached = self.frame_cache.get(self.current_frame)
        if cached is not None:
            self._display_image(cached)
            return
        
        try:
            with self.frame_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                
            if not ret:
                return
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            
            self.frame_cache.put(self.current_frame, img)
            
            self._display_image(img)
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error showing frame: {e}")
    
    def _display_image(self, img: Image.Image) -> None:
        orig_width, orig_height = img.size
        
        if self.aspect_ratio is None:
            self.aspect_ratio = orig_width / orig_height if orig_height > 0 else 1.0
        
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        
        if self.aspect_ratio > (cw / ch):
            display_width = cw
            display_height = int(cw / self.aspect_ratio)
        else:
            display_height = ch
            display_width = int(ch * self.aspect_ratio)
        
        x = (cw - display_width) // 2
        y = (ch - display_height) // 2
        
        self.display_img_x = x
        self.display_img_y = y
        self.display_img_w = display_width
        self.display_img_h = display_height
        
        if display_width <= 0 or display_height <= 0:
            if ENABLE_LOGGING: logger.warning("Invalid display dimensions")
            return
        
        if img.size != (display_width, display_height):
            img = img.resize((display_width, display_height), Image.Resampling.BILINEAR)
        
        self.tk_img = ImageTk.PhotoImage(img)
        
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor=NW, image=self.tk_img)
        
        if self.crop_rect and not self.playing:
            self._display_crop_rectangle()
        
        self.update_info_display()
    
    def show_audio_only_message(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text="Audio-Only Mode\nVideo preview disabled",
            fill="white",
            font=("Arial", 16),
            justify="center"
        )
        self.update_info_display()
    
    def toggle_play(self) -> None:
        if self.playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self) -> None:
        if not self.video_path:
            return
        
        if self.video_codec_var.get() == VideoCodec.NO_VIDEO.value:
            return
        
        self.playing = True
        self.state = AppState.PLAYING
        self.play_btn.config(text="⏸ Pause")
        
        try:
            self.player = VideoPlayer(
                self.video_path,
                self.fps,
                self.current_frame
            )
            self.player.start_playback()
            
            self.start_playback_update()
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Failed to start playback: {e}")
            self.playing = False
            self.state = AppState.VIDEO_LOADED
            self.play_btn.config(text="▶ Play")
    
    def stop_playback(self) -> None:
        self.playing = False
        self.state = AppState.VIDEO_LOADED
        
        if self.playback_update_id:
            self.root.after_cancel(self.playback_update_id)
            self.playback_update_id = None
        
        if self.player:
            self.player.stop_playback()
            self.player = None
        
        self.play_btn.config(text="▶ Play")
    
    def start_playback_update(self) -> None:
        if not self.playing or not self.player:
            return
        
        frame_data = self.player.get_frame()
        if frame_data:
            img, frame_num = frame_data
            self.current_frame = frame_num
            self.timeline.set(frame_num)
            self._display_image(img)
        
        self.playback_update_id = self.root.after(10, self.start_playback_update)
    
    def next_frame(self) -> None:
        if self.video_codec_var.get() == VideoCodec.NO_VIDEO.value:
            return
        
        self.stop_playback()
        self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.timeline.set(self.current_frame)
        self.show_frame()
    
    def prev_frame(self) -> None:
        if self.video_codec_var.get() == VideoCodec.NO_VIDEO.value:
            return
        
        self.stop_playback()
        self.current_frame = max(0, self.current_frame - 1)
        self.timeline.set(self.current_frame)
        self.show_frame()
    
    def timeline_press(self, event) -> None:
        self.was_playing = self.playing
        self.stop_playback()
        self.timeline_jump(event)
    
    def timeline_drag(self, event) -> None:
        self.timeline_jump(event)
    
    def timeline_release(self, event) -> None:
        if hasattr(self, 'was_playing') and self.was_playing:
            self.start_playback()
    
    def timeline_jump(self, event) -> None:
        w = self.timeline.winfo_width()
        if w <= 0:
            return
        
        frac = max(0.0, min(1.0, event.x / w))
        frame = int(frac * (self.total_frames - 1))
        self.current_frame = frame
        self.timeline.set(frame)
        self.show_frame()
    
    def apply_start_entry(self, event=None) -> None:
        try:
            if not self.start_entry.get():
                self.output_settings.start_frame = None
                return
            
            frame = int(self.start_entry.get())
            frame = max(0, min(frame, self.total_frames - 1))
            self.output_settings.start_frame = frame
            
            self.current_frame = frame
            self.timeline.set(frame)
            self.show_frame()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid frame number")
            self.start_entry.delete(0, END)
    
    def apply_end_entry(self, event=None) -> None:
        try:
            if not self.end_entry.get():
                self.output_settings.end_frame = None
                return
            
            frame = int(self.end_entry.get())
            frame = max(0, min(frame, self.total_frames - 1))
            self.output_settings.end_frame = frame
            
            self.current_frame = frame
            self.timeline.set(frame)
            self.show_frame()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid frame number")
            self.end_entry.delete(0, END)
    
    def set_start(self) -> None:
        self.output_settings.start_frame = self.current_frame
        self.start_entry.delete(0, END)
        self.start_entry.insert(0, str(self.current_frame))
    
    def set_end(self) -> None:
        self.output_settings.end_frame = self.current_frame
        self.end_entry.delete(0, END)
        self.end_entry.insert(0, str(self.current_frame))
    
    def toggle_crop_mode(self) -> None:
        if self.video_codec_var.get() == VideoCodec.NO_VIDEO.value:
            messagebox.showinfo("Info", "Crop tool is not available in audio-only mode")
            return
        
        self.crop_mode = not self.crop_mode
        
        if self.crop_mode:
            self.state = AppState.CROP_MODE
            self.status_label.config(text="Crop mode: Drag to select area. Use handles to resize.")
            
            if self.crop_rect is None and self.cap:
                self.crop_rect = CropRect(0, 0, self.video_width, self.video_height)
                self.crop_rect = self.crop_rect.ensure_even_dimensions()
            
            if self.crop_btn.winfo_ismapped():
                self.crop_btn.pack_forget()
            if not self.clear_crop_btn.winfo_ismapped():
                self.clear_crop_btn.pack(side=LEFT)
            
            self.show_frame()
            
        else:
            self.state = AppState.VIDEO_LOADED
            self.status_label.config(text="Ready")
            
            if self.clear_crop_btn.winfo_ismapped():
                self.clear_crop_btn.pack_forget()
            if not self.crop_btn.winfo_ismapped():
                self.crop_btn.pack(side=LEFT)
            self.show_frame()
    
    def cancel_crop_mode(self) -> None:
        if self.crop_mode:
            self.crop_mode = False
            self.state = AppState.VIDEO_LOADED
            self.status_label.config(text="Ready")
            
            if self.clear_crop_btn.winfo_ismapped():
                self.clear_crop_btn.pack_forget()
            if not self.crop_btn.winfo_ismapped():
                self.crop_btn.pack(side=LEFT)
            self.show_frame()
    
    def clear_crop(self) -> None:
        self.crop_rect = None
        self.crop_start = None
        self.crop_move_mode = False
        self.crop_resize_mode = False
        
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
        
        for handle in self.crop_handles:
            self.canvas.delete(handle)
        self.crop_handles = []
        
        if self.clear_crop_btn.winfo_ismapped():
            self.clear_crop_btn.pack_forget()
        if not self.crop_btn.winfo_ismapped():
            self.crop_btn.pack(side=LEFT)
        if self.crop_mode:
            self.status_label.config(text="Crop cleared")
            self.crop_mode = False
            self.state = AppState.VIDEO_LOADED
        else:
            self.status_label.config(text="Ready")
        
        self.show_frame()
    
    def _display_crop_rectangle(self) -> None:
        if not self.crop_rect or not self.cap:
            return
        
        orig_width = self.video_width
        orig_height = self.video_height
        
        if orig_width <= 0 or orig_height <= 0 or self.display_img_w <= 0 or self.display_img_h <= 0:
            return
        
        disp_x1 = int((self.crop_rect.x / orig_width) * self.display_img_w) + self.display_img_x
        disp_y1 = int((self.crop_rect.y / orig_height) * self.display_img_h) + self.display_img_y
        disp_x2 = int((self.crop_rect.x2 / orig_width) * self.display_img_w) + self.display_img_x
        disp_y2 = int((self.crop_rect.y2 / orig_height) * self.display_img_h) + self.display_img_y
        
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
        
        self.crop_rect_id = self.canvas.create_rectangle(
            disp_x1, disp_y1, disp_x2, disp_y2,
            outline="yellow",
            width=2,
            dash=(5, 5),
            tags="crop_rect"
        )
        
        self._create_crop_handles(disp_x1, disp_y1, disp_x2, disp_y2)
    
    def _create_crop_handles(self, x1: int, y1: int, x2: int, y2: int) -> None:
        for handle in self.crop_handles:
            self.canvas.delete(handle)
        self.crop_handles = []
        
        handle_size = 8
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        
        for x, y in corners:
            handle = self.canvas.create_rectangle(
                x - handle_size // 2,
                y - handle_size // 2,
                x + handle_size // 2,
                y + handle_size // 2,
                fill="yellow",
                outline="white",
                width=1,
                tags="crop_handle"
            )
            self.crop_handles.append(handle)
    
    def _get_handle_at_position(self, x: int, y: int) -> Optional[Tuple[int, str]]:
        for i, handle in enumerate(self.crop_handles):
            coords = self.canvas.coords(handle)
            if coords and len(coords) >= 4:
                if coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                    corner_names = ["nw", "ne", "sw", "se"]
                    return i, corner_names[i]
        return None
    
    def _is_over_crop_rect(self, x: int, y: int) -> bool:
        if not self.crop_rect_id:
            return False
        
        if self._get_handle_at_position(x, y) is not None:
            return False
        
        coords = self.canvas.coords(self.crop_rect_id)
        if coords and len(coords) >= 4:
            x1, y1, x2, y2 = coords
            return x1 <= x <= x2 and y1 <= y <= y2
        
        return False
    
    def on_canvas_click(self, event) -> None:
        if not self.crop_mode or not self.video_path:
            return
        
        x, y = event.x, event.y
        
        handle_info = self._get_handle_at_position(x, y)
        if handle_info is not None and self.crop_rect_id:
            _, corner = handle_info
            self.crop_resize_mode = True
            self.crop_resize_corner = corner
            self.crop_original_coords = self.canvas.coords(self.crop_rect_id)
            return
        
        if self._is_over_crop_rect(x, y) and self.crop_rect_id:
            self.crop_move_mode = True
            self.crop_original_coords = self.canvas.coords(self.crop_rect_id)
            self.move_start_x = x
            self.move_start_y = y
            return
        
        if (self.display_img_x <= x <= self.display_img_x + self.display_img_w and
            self.display_img_y <= y <= self.display_img_y + self.display_img_h):
            self.crop_start = (x, y)
            
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
                self.crop_rect_id = None
            
            for handle in self.crop_handles:
                self.canvas.delete(handle)
            self.crop_handles = []
    
    def on_canvas_drag(self, event) -> None:
        if not self.crop_mode:
            return
        
        x = max(self.display_img_x, min(event.x, self.display_img_x + self.display_img_w))
        y = max(self.display_img_y, min(event.y, self.display_img_y + self.display_img_h))
        
        if self.crop_resize_mode and self.crop_rect_id and self.crop_original_coords:
            self._handle_crop_resize(x, y)
        elif self.crop_move_mode and self.crop_rect_id:
            self._handle_crop_move(x, y)
        elif self.crop_start:
            self._handle_crop_draw(x, y)
    
    def _handle_crop_resize(self, x: int, y: int) -> None:
        x1, y1, x2, y2 = self.crop_original_coords
        
        if self.crop_resize_corner == "nw":
            x1 = x
            y1 = y
        elif self.crop_resize_corner == "ne":
            x2 = x
            y1 = y
        elif self.crop_resize_corner == "sw":
            x1 = x
            y2 = y
        elif self.crop_resize_corner == "se":
            x2 = x
            y2 = y
        
        min_size = self.MIN_CROP_SIZE
        if abs(x2 - x1) < min_size:
            if self.crop_resize_corner in ["nw", "sw"]:
                x1 = x2 - min_size
            else:
                x2 = x1 + min_size
        
        if abs(y2 - y1) < min_size:
            if self.crop_resize_corner in ["nw", "ne"]:
                y1 = y2 - min_size
            else:
                y2 = y1 + min_size
        
        x1 = max(self.display_img_x, min(x1, self.display_img_x + self.display_img_w - 1))
        y1 = max(self.display_img_y, min(y1, self.display_img_y + self.display_img_h - 1))
        x2 = max(self.display_img_x + 1, min(x2, self.display_img_x + self.display_img_w))
        y2 = max(self.display_img_y + 1, min(y2, self.display_img_y + self.display_img_h))
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        self.canvas.coords(self.crop_rect_id, x1, y1, x2, y2)
        self._create_crop_handles(x1, y1, x2, y2)
    
    def _handle_crop_move(self, x: int, y: int) -> None:
        current_coords = self.canvas.coords(self.crop_rect_id)
        if not current_coords or len(current_coords) < 4:
            return
        
        dx = x - self.move_start_x
        dy = y - self.move_start_y
        
        self.move_start_x = x
        self.move_start_y = y
        
        x1, y1, x2, y2 = current_coords
        rect_w = x2 - x1
        rect_h = y2 - y1
        
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = new_x1 + rect_w
        new_y2 = new_y1 + rect_h
        
        new_x1 = max(self.display_img_x, min(new_x1, self.display_img_x + self.display_img_w - rect_w))
        new_y1 = max(self.display_img_y, min(new_y1, self.display_img_y + self.display_img_h - rect_h))
        new_x2 = new_x1 + rect_w
        new_y2 = new_y1 + rect_h
        
        self.canvas.coords(self.crop_rect_id, new_x1, new_y1, new_x2, new_y2)
        self._create_crop_handles(new_x1, new_y1, new_x2, new_y2)
    
    def _handle_crop_draw(self, x: int, y: int) -> None:
        x1, y1 = self.crop_start
        
        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)
        
        self.crop_rect_id = self.canvas.create_rectangle(
            x1, y1, x, y,
            outline="yellow",
            width=2,
            dash=(5, 5),
            tags="crop_rect"
        )
    
    def on_canvas_release(self, event) -> None:
        if not self.crop_mode:
            return
        
        if self.crop_resize_mode:
            self.crop_resize_mode = False
            self.crop_resize_corner = None
            self._update_crop_from_display()
        
        elif self.crop_move_mode:
            self.crop_move_mode = False
            self._update_crop_from_display()
        
        elif self.crop_start:
            x1, y1 = self.crop_start
            x2, y2 = event.x, event.y
            
            x1 = max(self.display_img_x, min(x1, self.display_img_x + self.display_img_w))
            y1 = max(self.display_img_y, min(y1, self.display_img_y + self.display_img_h))
            x2 = max(self.display_img_x, min(x2, self.display_img_x + self.display_img_w))
            y2 = max(self.display_img_y, min(y2, self.display_img_y + self.display_img_h))
            
            crop_x1 = min(x1, x2)
            crop_y1 = min(y1, y2)
            crop_x2 = max(x1, x2)
            crop_y2 = max(y1, y2)
            
            min_size = self.MIN_CROP_SIZE
            if crop_x2 - crop_x1 < min_size:
                crop_x2 = crop_x1 + min_size
                if crop_x2 > self.display_img_x + self.display_img_w:
                    crop_x1 = crop_x2 - min_size
            
            if crop_y2 - crop_y1 < min_size:
                crop_y2 = crop_y1 + min_size
                if crop_y2 > self.display_img_y + self.display_img_h:
                    crop_y1 = crop_y2 - min_size
            
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
            
            self.crop_rect_id = self.canvas.create_rectangle(
                crop_x1, crop_y1, crop_x2, crop_y2,
                outline="yellow",
                width=3,
                tags="crop_rect"
            )
            
            self._create_crop_handles(crop_x1, crop_y1, crop_x2, crop_y2)
            self._update_crop_from_display()
            
            self.crop_start = None
    
    def on_canvas_motion(self, event) -> None:
        if not self.crop_mode:
            self.canvas.config(cursor="")
            return
        
        x, y = event.x, event.y
        
        handle_info = self._get_handle_at_position(x, y)
        if handle_info is not None:
            _, corner = handle_info
            if corner in ["nw", "se"]:
                self.canvas.config(cursor="top_left_corner")
            else:
                self.canvas.config(cursor="top_right_corner")
            return
        
        if self._is_over_crop_rect(x, y):
            self.canvas.config(cursor="fleur")
            return
        
        self.canvas.config(cursor="cross")
    
    def _update_crop_from_display(self) -> None:
        if not self.crop_rect_id or not self.cap:
            return
        
        coords = self.canvas.coords(self.crop_rect_id)
        if not coords or len(coords) < 4:
            return
        
        if self.display_img_w <= 0 or self.display_img_h <= 0:
            return
        
        crop_x1, crop_y1, crop_x2, crop_y2 = coords
        
        orig_width = self.video_width
        orig_height = self.video_height
        
        rel_x1 = crop_x1 - self.display_img_x
        rel_y1 = crop_y1 - self.display_img_y
        rel_x2 = crop_x2 - self.display_img_x
        rel_y2 = crop_y2 - self.display_img_y
        
        orig_crop_x1 = int((rel_x1 / self.display_img_w) * orig_width)
        orig_crop_y1 = int((rel_y1 / self.display_img_h) * orig_height)
        orig_crop_x2 = int((rel_x2 / self.display_img_w) * orig_width)
        orig_crop_y2 = int((rel_y2 / self.display_img_h) * orig_height)
        
        self.crop_rect = CropRect.from_coords(
            orig_crop_x1, orig_crop_y1, orig_crop_x2, orig_crop_y2
        ).ensure_even_dimensions()
        
        self.status_label.config(
            text=f"Crop: {self.crop_rect.width}x{self.crop_rect.height} "
                 f"at ({self.crop_rect.x},{self.crop_rect.y})"
        )
        
        self.update_info_display()
        self.update_video_info_label()
        self.update_resolution_display()
    
    def on_video_codec_changed(self, *args) -> None:
        try:
            codec_value = self.video_codec_var.get()
            self.video_settings.codec = VideoCodec(codec_value)
            
            if not self.audio_codec_manually_changed:
                self.suggest_audio_codec_for_video()
            
            self.update_pairing_info_label()
            
            suggested_codec = self.CODEC_PAIRINGS.get(self.video_settings.codec, self.DEFAULT_PAIRING)
            if self.audio_settings.codec != suggested_codec:
                self.reset_suggestion_btn.config(state=NORMAL)
            else:
                self.reset_suggestion_btn.config(state=DISABLED)
            
            self.update_video_codec_visibility()
            self.update_video_quality_visibility()
            self.update_video_quality_ui()
            self.update_encoder_speed_options()
            
            if self.video_settings.codec == VideoCodec.CUSTOM:
                self.show_custom_video_dialog()
            else:
                if self.custom_video_frame.winfo_ismapped():
                    self.custom_video_frame.pack_forget()
                self.video_settings.custom_encoder = None
                self.video_settings.custom_encoder_args = None
            
            if self.video_path:
                self.root.after(100, self.show_frame)
                
        except ValueError:
            if ENABLE_LOGGING: logger.warning(f"Invalid video codec: {self.video_codec_var.get()}")
    
    def on_audio_codec_manual_change(self, *args) -> None:
        try:
            codec_value = self.audio_codec_var.get()
            self.audio_settings.codec = AudioCodec(codec_value)
            
            self.audio_codec_manually_changed = True
            
            self.update_pairing_info_label()
            
            video_codec = self.video_settings.codec
            suggested_codec = self.CODEC_PAIRINGS.get(video_codec, self.DEFAULT_PAIRING)
            if self.audio_settings.codec != suggested_codec:
                self.reset_suggestion_btn.config(state=NORMAL)
            else:
                self.reset_suggestion_btn.config(state=DISABLED)
            
            if self.audio_settings.codec == AudioCodec.CUSTOM:
                self.show_custom_audio_dialog()
            else:
                if self.custom_audio_frame.winfo_ismapped():
                    self.custom_audio_frame.pack_forget()
                self.audio_settings.custom_encoder = None
                self.audio_settings.custom_encoder_args = None
            
            self.update_audio_quality_visibility()
            
            if self.video_path:
                self.root.after(100, self.show_frame)
                
        except ValueError:
            if ENABLE_LOGGING: logger.warning(f"Invalid audio codec: {self.audio_codec_var.get()}")
    
    def show_custom_video_dialog(self) -> None:
        explanation = (
            "Custom Video Codec (Advanced)\n\n"
            "This allows you to specify any FFmpeg video encoder.\n"
            "Format: Use FFmpeg encoder names (e.g., 'libx264', 'libvpx-vp9').\n\n"
            "Additional arguments can include:\n"
            "- Quality: -crf 23, -b:v 5000k, -qscale:v 2\n"
            "- Speed: -preset medium, -cpu-used 4\n"
            "- Other encoder-specific options\n\n"
            "Example:\n"
            "Encoder: libx264\n"
            "Args: -crf 23 -preset medium -tune film\n"
        )
        
        response = messagebox.askyesno(
            "Custom Video Codec",
            explanation + "\nDo you want to configure custom video settings?",
            icon=messagebox.INFO,
            detail="Click Yes to configure, No to keep defaults."
        )
        
        if response:
            if not self.custom_video_frame.winfo_ismapped():
                self.custom_video_frame.pack(fill=X, pady=(5, 0))
            
            if self.video_quality_frame.winfo_ismapped():
                self.video_quality_frame.pack_forget()
            if self.speed_frame.winfo_ismapped():
                self.speed_frame.pack_forget()
            if self.fps_frame.winfo_ismapped():
                self.fps_frame.pack_forget()
            
            if self.video_settings.custom_encoder:
                self.video_encoder_var.set(self.video_settings.custom_encoder)
            
            if self.video_settings.custom_encoder_args:
                self.video_args_var.set(self.video_settings.custom_encoder_args)
            
            self.video_encoder_entry.focus_set()
    
    def show_custom_audio_dialog(self) -> None:
        explanation = (
            "Custom Audio Codec (Advanced)\n\n"
            "This allows you to specify any FFmpeg audio encoder.\n"
            "Format: Use FFmpeg encoder names (e.g., 'aac', 'libopus').\n\n"
            "Additional arguments can include:\n"
            "- Bitrate: -b:a 192k\n"
            "- Quality: -qscale:a 2\n"
            "- Channels: -ac 2\n"
            "- Other encoder-specific options\n\n"
            "Example:\n"
            "Encoder: libopus\n"
            "Args: -b:a 128k -vbr on -compression_level 10\n"
        )
        
        response = messagebox.askyesno(
            "Custom Audio Codec",
            explanation + "\nDo you want to configure custom audio settings?",
            icon=messagebox.INFO,
            detail="Click Yes to configure, No to keep defaults."
        )
        
        if response:
            if not self.custom_audio_frame.winfo_ismapped():
                self.custom_audio_frame.pack(fill=X, pady=(5, 0))
            
            if self.audio_quality_frame.winfo_ismapped():
                self.audio_quality_frame.pack_forget()
            
            if self.audio_settings.custom_encoder:
                self.audio_encoder_var.set(self.audio_settings.custom_encoder)
            
            if self.audio_settings.custom_encoder_args:
                self.audio_args_var.set(self.audio_settings.custom_encoder_args)
            
            self.audio_encoder_entry.focus_set()
    
    def suggest_audio_codec_for_video(self) -> None:
        video_codec = self.video_settings.codec
        
        if video_codec == VideoCodec.CUSTOM:
            return
            
        suggested_codec = self.CODEC_PAIRINGS.get(video_codec, self.DEFAULT_PAIRING)
        
        self.audio_codec_var.set(suggested_codec.value)
        self.audio_settings.codec = suggested_codec
        
        if self.audio_settings.codec != AudioCodec.CUSTOM:
                self.audio_codec_manually_changed = False
        
        self.update_pairing_info_label()
        
        self.reset_suggestion_btn.config(state=DISABLED)
    
    def update_pairing_info_label(self) -> None:
        video_codec = self.video_settings.codec
        current_audio_codec = self.audio_settings.codec
        
        suggested_codec = self.CODEC_PAIRINGS.get(video_codec, self.DEFAULT_PAIRING)
        
        if video_codec == VideoCodec.CUSTOM:
            self.pairing_info_label.config(
                text="Using custom video encoder",
                fg="#888888"
            )
            return
        
        video_codec_name = video_codec.value
        if "(" in video_codec_name:
            video_codec_name = video_codec_name.split("(")[0].strip()
        
        if current_audio_codec == suggested_codec:
            self.pairing_info_label.config(
                text=f"✓ Paired optimally with {video_codec_name}",
                fg="#4CAF50"
            )
        else:
            suggested_name = suggested_codec.value
            if "(" in suggested_name:
                suggested_name = suggested_name.split("(")[0].strip()
            
            self.pairing_info_label.config(
                text=f"Suggested: {suggested_name} for {video_codec_name}",
                fg="#FF9800"
            )
    
    def reset_to_suggested_codec(self) -> None:
        video_codec = self.video_settings.codec
        suggested_codec = self.CODEC_PAIRINGS.get(video_codec, self.DEFAULT_PAIRING)
        
        self.audio_codec_var.set(suggested_codec.value)
        self.audio_settings.codec = suggested_codec
        
        if self.audio_settings.codec != AudioCodec.CUSTOM:
                self.audio_codec_manually_changed = False
        
        self.update_pairing_info_label()
        self.update_audio_quality_visibility()
        
        self.reset_suggestion_btn.config(state=DISABLED)
        
        if self.video_path:
            self.root.after(100, self.show_frame)
    
    def on_video_quality_changed(self, *args) -> None:
        try:
            quality_value = self.video_quality_var.get()
            self.video_settings.quality_mode = QualityMode(quality_value)
            self.update_video_quality_ui()
            self.update_estimated_size()
        except ValueError:
            if ENABLE_LOGGING: logger.warning(f"Invalid quality mode: {self.video_quality_var.get()}")
    
    def on_resolution_changed(self, *args) -> None:
        try:
            selected = self.resolution_var.get()
            resolution_mode = ResolutionMode(selected)
            self.video_settings.resolution_mode = resolution_mode
            
            if resolution_mode == ResolutionMode.CUSTOM:
                self.custom_wh_frame.pack(fill=X, pady=(5, 0))
                self.custom_dim_frame.pack_forget()
                
                self.width_entry.config(state=NORMAL)
                self.height_entry.config(state=NORMAL)
                self.custom_dim_entry.config(state=DISABLED)
                
                if not self.custom_res_frame.winfo_ismapped():
                    self.custom_res_frame.pack(fill=X, pady=(5, 0))
                
                if not self.width_entry.get() and self.video_width > 0:
                    self.width_entry.delete(0, END)
                    self.width_entry.insert(0, str(self.video_width - (self.video_width % 2)))
                
                if not self.height_entry.get() and self.video_height > 0:
                    self.height_entry.delete(0, END)
                    self.height_entry.insert(0, str(self.video_height - (self.video_height % 2)))
                    
            elif resolution_mode in [ResolutionMode.CUSTOM_WIDTH, ResolutionMode.CUSTOM_HEIGHT]:
                self.custom_wh_frame.pack_forget()
                self.custom_dim_frame.pack(fill=X, pady=(5, 0))
                
                if resolution_mode == ResolutionMode.CUSTOM_WIDTH:
                    self.custom_dim_label.config(text="Width:")
                else:
                    self.custom_dim_label.config(text="Height:")
                
                self.width_entry.config(state=DISABLED)
                self.height_entry.config(state=DISABLED)
                self.custom_dim_entry.config(state=NORMAL)
                
                if not self.custom_res_frame.winfo_ismapped():
                    self.custom_res_frame.pack(fill=X, pady=(5, 0))
                
                if not self.custom_dim_entry.get():
                    if resolution_mode == ResolutionMode.CUSTOM_WIDTH and self.video_width > 0:
                        self.custom_dim_entry.delete(0, END)
                        self.custom_dim_entry.insert(0, str(self.video_width - (self.video_width % 2)))
                    elif resolution_mode == ResolutionMode.CUSTOM_HEIGHT and self.video_height > 0:
                        self.custom_dim_entry.delete(0, END)
                        self.custom_dim_entry.insert(0, str(self.video_height - (self.video_height % 2)))
                        
            else:
                if self.custom_res_frame.winfo_ismapped():
                    self.custom_res_frame.pack_forget()
                
                self.width_entry.config(state=DISABLED)
                self.height_entry.config(state=DISABLED)
                self.custom_dim_entry.config(state=DISABLED)
            
            self.update_resolution_display()
            self.update_video_info_label()
            
        except ValueError:
            if ENABLE_LOGGING: logger.warning(f"Invalid resolution mode: {self.resolution_var.get()}")
    
    def apply_custom_resolution(self) -> None:
        try:
            resolution_mode = self.video_settings.resolution_mode
            
            if resolution_mode == ResolutionMode.CUSTOM:
                width_str = self.width_entry.get().strip()
                height_str = self.height_entry.get().strip()
                
                if not width_str or not height_str:
                    messagebox.showwarning("Warning", "Please enter both width and height")
                    return
                
                width = int(width_str)
                height = int(height_str)
                
                if width <= 0 or height <= 0:
                    messagebox.showerror("Error", "Width and height must be positive numbers")
                    return
                
                width = width - (width % 2)
                height = height - (height % 2)
                
                if width <= 0 or height <= 0:
                    messagebox.showerror("Error", "Dimensions must be at least 2 pixels")
                    return
                
                self.video_settings.custom_width = width
                self.video_settings.custom_height = height
                
                res_str = f"{width}x{height}"
                if res_str not in [f"{w}x{h}" for w, h in self.custom_resolutions_history]:
                    self.custom_resolutions_history.append((width, height))
                    if len(self.custom_resolutions_history) > self.max_custom_history:
                        self.custom_resolutions_history.pop(0)
                
            elif resolution_mode in [ResolutionMode.CUSTOM_WIDTH, ResolutionMode.CUSTOM_HEIGHT]:
                dim_str = self.custom_dim_entry.get().strip()
                
                if not dim_str:
                    messagebox.showwarning("Warning", "Please enter a dimension value")
                    return
                
                dim_value = int(dim_str)
                
                if dim_value <= 0:
                    messagebox.showerror("Error", "Dimension must be a positive number")
                    return
                
                dim_value = dim_value - (dim_value % 2)
                
                if dim_value <= 0:
                    messagebox.showerror("Error", "Dimension must be at least 2 pixels")
                    return
                
                self.video_settings.custom_dimension_value = dim_value
                
                if resolution_mode == ResolutionMode.CUSTOM_WIDTH:
                    self.video_settings.custom_width = dim_value
                    self.video_settings.custom_height = None
                else:
                    self.video_settings.custom_width = None
                    self.video_settings.custom_height = dim_value
                
            else:
                return
            
            self.update_resolution_display()
            self.update_video_info_label()
            
            messagebox.showinfo("Success", f"Custom resolution applied")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid value. Please enter a number.")
    
    def update_resolution_display(self) -> None:
        if not self.video_path:
            self.current_res_label.config(text="Current: No video loaded")
            return
        
        out_width, out_height = self.video_settings.calculate_output_dimensions(
            self.video_width, self.video_height, self.crop_rect
        )
        
        aspect_str = simplify_aspect_ratio(out_width, out_height)
        
        if self.video_settings.resolution_mode == ResolutionMode.ORIGINAL:
            self.current_res_label.config(
                text=f"Current: Original ({out_width}x{out_height}, {aspect_str})"
            )
        elif self.video_settings.resolution_mode == ResolutionMode.CUSTOM:
            if self.video_settings.custom_width and self.video_settings.custom_height:
                self.current_res_label.config(
                    text=f"Current: Custom ({out_width}x{out_height}, {aspect_str})"
                )
            else:
                self.current_res_label.config(text="Current: Custom (not set)")
        elif self.video_settings.resolution_mode in [ResolutionMode.CUSTOM_WIDTH, ResolutionMode.CUSTOM_HEIGHT]:
            if self.video_settings.custom_dimension_value:
                mode_name = "Custom Width" if self.video_settings.resolution_mode == ResolutionMode.CUSTOM_WIDTH else "Custom Height"
                self.current_res_label.config(
                    text=f"Current: {mode_name} ({out_width}x{out_height}, {aspect_str})"
                )
            else:
                mode_name = "Custom Width" if self.video_settings.resolution_mode == ResolutionMode.CUSTOM_WIDTH else "Custom Height"
                self.current_res_label.config(text=f"Current: {mode_name} (not set)")
        elif self.video_settings.resolution_mode in [
            ResolutionMode.AI_2X_CPU, ResolutionMode.AI_3X_CPU, ResolutionMode.AI_4X_CPU,
            ResolutionMode.AI_2X_GPU_ANIME, ResolutionMode.AI_3X_GPU_ANIME, ResolutionMode.AI_4X_GPU_ANIME, ResolutionMode.AI_4X_GPU_GENERAL
        ]:
            mode_name = self.video_settings.resolution_mode.value
            self.current_res_label.config(
                text=f"Current: {mode_name} ({out_width}x{out_height}, {aspect_str})"
            )
        else:
            mode_name = self.video_settings.resolution_mode.value
            self.current_res_label.config(
                text=f"Current: {mode_name} ({out_width}x{out_height}, {aspect_str})"
            )
    
    def update_encoder_speed_options(self) -> None:
        codec = self.video_settings.codec
        if codec == VideoCodec.CUSTOM:
            if self.speed_frame.winfo_ismapped():
                self.speed_frame.pack_forget()
            return
        presets = [str(p).strip() for p in self.ENCODER_PRESETS.get(codec, [])]
        if not presets:
            if self.speed_frame.winfo_ismapped():
                self.speed_frame.pack_forget()
            return
        is_lossless = any(l in codec.value.lower() for l in ("lossless","pcm","raw","ffv1"))
        if is_lossless or not presets:
            if self.speed_frame.winfo_ismapped():
                self.speed_frame.pack_forget()
            return
        menu = self.speed_menu["menu"]
        menu.delete(0, "end")
        for preset in presets:
            menu.add_command(label=preset, command=lambda p=preset: self.speed_var.set(p))
        
        # Preserve the current setting if valid
        current = str(self.speed_var.get()).strip()
        desired = str(getattr(self.video_settings, "speed_preset", "")).strip()
        
        if desired and desired in presets:
            # Use the video_settings value (e.g., from loaded preset)
            self.speed_var.set(desired)
        elif current and current in presets:
            # Keep current UI value if it's valid
            self.video_settings.speed_preset = current
        else:
            # Set to default
            default = "medium" if "medium" in presets else presets[len(presets)//2]
            self.speed_var.set(default)
            self.video_settings.speed_preset = default
        
        if not self.speed_frame.winfo_ismapped():
            self.speed_frame.pack(fill=X, pady=(5, 0))
    def update_video_codec_visibility(self) -> None:
        """Update visibility of video codec-related UI elements"""
        is_audio_only = self.video_settings.codec == VideoCodec.NO_VIDEO
        is_custom = self.video_settings.codec == VideoCodec.CUSTOM
        
        # Hide/show custom video frame
        if is_custom:
            if not self.custom_video_frame.winfo_ismapped():
                self.custom_video_frame.pack(fill=X, pady=(5, 0))
        else:
            if self.custom_video_frame.winfo_ismapped():
                self.custom_video_frame.pack_forget()
        
        # Hide/show FPS frame
        if is_audio_only:
            if self.fps_frame.winfo_ismapped():
                self.fps_frame.pack_forget()
        else:
            if not self.fps_frame.winfo_ismapped():
                self.fps_frame.pack(fill=X, pady=(5, 0))
    
    def update_video_quality_ui(self) -> None:
        if self.video_settings.quality_mode == QualityMode.CQ:
            self.cq_frame.pack(fill=X, pady=(2, 0))
            self.bitrate_frame.pack_forget()
            self.estimated_size_label.config(text="")
            
            if "AV1" in self.video_settings.codec.value:
                self.cq_range_label.config(text="(0-63, lower=better)")
            else:
                self.cq_range_label.config(text="(0-51, lower=better)")
        else:
            self.bitrate_frame.pack(fill=X, pady=(2, 0))
            self.cq_frame.pack_forget()
            self.update_estimated_size()
    
    def update_video_quality_visibility(self) -> None:
        is_audio_only = self.video_settings.codec == VideoCodec.NO_VIDEO
        is_custom = self.video_settings.codec == VideoCodec.CUSTOM
        
        if is_audio_only or is_custom:
            if self.video_quality_frame.winfo_ismapped():
                self.video_quality_frame.pack_forget()
            return
        
        is_lossy = not any(lossless in self.video_settings.codec.value.lower() 
                         for lossless in ["lossless", "pcm", "raw"])
        
        if is_lossy:
            if not self.video_quality_frame.winfo_ismapped():
                self.video_quality_frame.pack(fill=X, pady=(5, 0))
        else:
            if self.video_quality_frame.winfo_ismapped():
                self.video_quality_frame.pack_forget()
    
    def update_audio_quality_visibility(self) -> None:
        is_custom = self.audio_settings.codec == AudioCodec.CUSTOM
        
        if is_custom:
            if self.audio_quality_frame.winfo_ismapped():
                self.audio_quality_frame.pack_forget()
            return
            
        is_lossy = not any(lossless in self.audio_settings.codec.value.lower() 
                         for lossless in ["lossless", "pcm"]) and \
                   self.audio_settings.codec != AudioCodec.NO_AUDIO
        
        if is_lossy:
            if not self.audio_quality_frame.winfo_ismapped():
                self.audio_quality_frame.pack(fill=X, pady=(5, 0))
        else:
            if self.audio_quality_frame.winfo_ismapped():
                self.audio_quality_frame.pack_forget()
    
    def update_info_display(self) -> None:
        if not self.video_path:
            self.info.config(text="")
            return
        
        current_time = self.current_frame / self.fps
        
        info_parts = [
            f"Frame {self.current_frame}/{self.total_frames}",
            f"Time {format_time(current_time)}"
        ]
        
        if self.output_settings.start_frame is not None:
            info_parts.append(f"START: {self.output_settings.start_frame}")
        if self.output_settings.end_frame is not None:
            info_parts.append(f"END: {self.output_settings.end_frame}")
        
        if self.crop_rect:
            info_parts.append(f"Crop: {self.crop_rect.width}x{self.crop_rect.height}")
        
        if self.video_settings.resolution_mode != ResolutionMode.ORIGINAL and not self.video_settings.is_ai_enhancement():
            out_width, out_height = self.video_settings.calculate_output_dimensions(
                self.video_width, self.video_height, self.crop_rect
            )
            info_parts.append(f"Output: {out_width}x{out_height}")
        
        if self.video_settings.is_ai_enhancement():
            scale_factor = self.video_settings.get_ai_scale_factor()
            mode = "GPU" if self.video_settings.is_gpu_ai_enhancement() else "CPU"
            info_parts.append(f"AI {scale_factor}x ({mode})")
        
        if self.video_settings.codec == VideoCodec.NO_VIDEO:
            info_parts.append("Audio-Only Mode")
        elif self.video_settings.codec == VideoCodec.CUSTOM:
            info_parts.append("Custom Video Codec")
        
        if self.audio_settings.codec == AudioCodec.CUSTOM:
            info_parts.append("Custom Audio Codec")
        
        if self.playing:
            info_parts.append("Smooth Playback")
        
        if self.state == AppState.CONVERTING:
            info_parts.append("Converting...")
        elif self.state == AppState.CONVERSION_PAUSED:
            info_parts.append("Conversion Paused")
        
        self.info.config(text=" | ".join(info_parts))
    
    def convert(self) -> None:
        if not self.video_path:
            messagebox.showwarning("Warning", "No video loaded")
            return
        
        if not self._validate_conversion_settings():
            return
        
        output_path = self._get_output_filename()
        if not output_path:
            return
        
        if self.video_settings.is_ai_enhancement():
            self._convert_with_ai(output_path)
        else:
            self._convert_normal(output_path)
    
    def _convert_with_ai(self, output_path: str) -> None:
        total_duration = self.total_frames / self.fps
        
        if self.output_settings.start_frame is not None or self.output_settings.end_frame is not None:
            start_time = self.output_settings.start_frame / self.fps if self.output_settings.start_frame else 0
            end_time = self.output_settings.end_frame / self.fps if self.output_settings.end_frame else total_duration
            self.segment_duration = max(0.001, end_time - start_time)
        else:
            self.segment_duration = total_duration
            start_time = 0
        
        self.conversion_start_time_original = start_time
        
        self.conversion_start_time = time.time()
        self.last_progress_update_time = self.conversion_start_time
        self.last_percentage = 0.0
        
        # Check AI backend
        self._check_ai_availability()
        
        scale_factor = self.video_settings.get_ai_scale_factor()
        is_gpu = self.video_settings.is_gpu_ai_enhancement()
        
        backend = "Unknown"
        if is_gpu and self.realesrgan_available:
            backend = "GPU (Real-ESRGAN)"
            if self.realesrgan_vulkan_available:
                backend += " with Vulkan"
            else:
                backend += " (CPU fallback)"
        elif self.super_image_available:
            backend = "CPU (super-image)"
        else:
            backend = "CPU (simple resize)"
        
        messagebox.showinfo(
    "AI Enhancement",
    f"Starting AI {scale_factor}x enhancement using {backend}.\n\n"
    f"This process uses {'GPU acceleration' if is_gpu else 'CPU processing'}, "
    f"and the processing time depends on several factors:\n"
    f"• Video length and resolution\n"
    f"• AI scale factor ({scale_factor}x)\n"
    f"• Hardware acceleration: {'Enabled' if is_gpu and self.realesrgan_available else 'Not available'}\n\n"
    f"GPU acceleration will significantly speed up processing when available."
)
        
        self.state = AppState.CONVERTING
        self.conversion_paused = False
        self.conversion_pause_event.clear()
        self.convert_btn.config(state=DISABLED)
        
        self.convert_btn.pack_forget()
        self.control_buttons_frame.pack(pady=(0, 0))
        self.pause_resume_btn.config(text="PAUSE", bg="#FF9800")
        
        self.status_label.config(text="AI Enhancing...")
        
        self.show_progress_in_status(True)
        self.progress_bar['value'] = 0
        self.progress_percentage.config(text="0.0%")
        self.time_label.config(text="AI Processing...")
        
        self.conversion_stop_event.clear()
        self.conversion_thread = threading.Thread(
            target=self._run_ai_conversion,
            args=(output_path,),
            daemon=True
        )
        self.conversion_thread.start()
    
    def _run_ai_conversion(self, output_path: str) -> None:
        try:
            # Initialize AI model with appropriate backend
            is_gpu = self.video_settings.is_gpu_ai_enhancement()
            self._initialize_ai_model(use_gpu=is_gpu)
            
            video_dir = (os.path.dirname(job.input_path) if 'job' in locals() else (os.path.dirname(self.video_path) if hasattr(self, 'video_path') and self.video_path else os.getcwd()))
            with tempfile.TemporaryDirectory(prefix="favencoder_ai_", dir=video_dir) as temp_dir:
                frames_dir = os.path.join(temp_dir, "frames")
                enhanced_dir = os.path.join(temp_dir, "enhanced")
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(enhanced_dir, exist_ok=True)
                
                scale_factor = self.video_settings.get_ai_scale_factor()
                
                extract_cmd = ["ffmpeg"]
                
                if self.output_settings.start_frame is not None:
                    start_time = self.output_settings.start_frame / self.fps
                    extract_cmd.extend(["-ss", f"{start_time:.6f}"])
                
                extract_cmd.extend(["-i", self.video_path])
                
                if (self.output_settings.end_frame is not None and 
                    self.output_settings.start_frame is not None):
                    duration = (self.output_settings.end_frame - 
                              self.output_settings.start_frame) / self.fps
                    extract_cmd.extend(["-t", f"{duration:.6f}"])
                elif self.output_settings.end_frame is not None:
                    duration = self.output_settings.end_frame / self.fps
                    extract_cmd.extend(["-t", f"{duration:.6f}"])
                
                if self.crop_rect:
                    extract_cmd.extend(["-vf", f"crop={self.crop_rect.width}:{self.crop_rect.height}:{self.crop_rect.x}:{self.crop_rect.y}"])
                
                extract_cmd.extend([
                    "-vsync", "0",
                    os.path.join(frames_dir, "frame_%06d.png")
                ])
                
                extract_process = subprocess.Popen(
                    extract_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                extract_process.wait()
                
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
                total_frames = len(frame_files)
                
                if total_frames == 0:
                    raise ValueError("No frames extracted from video")
                
                for i, frame_file in enumerate(frame_files):
                    if self.conversion_stop_event.is_set():
                        break
                    
                    progress_percentage = (i / total_frames) * 50
                    
                    self.root.after(0, lambda p=progress_percentage: self._update_ai_progress(p, "AI Processing Frames"))
                    
                    frame_path = os.path.join(frames_dir, frame_file)
                    img = Image.open(frame_path).convert('RGB')
                    
                    try:
                        enhanced_img = self._enhance_image_with_ai(img, scale_factor)
                    except Exception as e:
                        if ENABLE_LOGGING: logger.error(f"Failed to enhance frame {frame_file}: {e}")
                        new_width = img.width * scale_factor
                        new_height = img.height * scale_factor
                        enhanced_img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
                    
                    enhanced_path = os.path.join(enhanced_dir, frame_file)
                    enhanced_img.save(enhanced_path)
                
                if self.conversion_stop_event.is_set():
                    self.root.after(0, self._conversion_stopped)
                    return
                
                temp_video_path = os.path.join(temp_dir, "enhanced_video.mkv")
                
                video_cmd = [
                    "ffmpeg",
                    "-framerate", str(self.fps),
                    "-i", os.path.join(enhanced_dir, "frame_%06d.png"),
                    "-i", self.video_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-y", temp_video_path
                ]
                
                video_process = subprocess.Popen(
                    video_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                while not self.conversion_stop_event.is_set():
                    if self.conversion_pause_event.is_set():
                        while (self.conversion_pause_event.is_set() and 
                               not self.conversion_stop_event.is_set()):
                            time.sleep(0.1)
                        
                        if self.conversion_stop_event.is_set():
                            break
                    
                    line = video_process.stderr.readline()
                    if not line:
                        if video_process.poll() is not None:
                            break
                        continue
                    
                    progress = self._parse_ffmpeg_progress_with_eta(line)
                    if progress:
                        progress.percentage = 50 + (progress.percentage / 2)
                        self.root.after(0, self._update_progress_with_eta, progress, line)
                
                returncode = video_process.wait()
                
                if returncode == 0:
                    final_cmd = self._build_ffmpeg_command_for_enhanced_video_immediate(temp_video_path, output_path)
                    
                    if final_cmd:
                        final_process = subprocess.Popen(
                            final_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            bufsize=1
                        )
                        
                        while not self.conversion_stop_event.is_set():
                            if self.conversion_pause_event.is_set():
                                while (self.conversion_pause_event.is_set() and 
                                       not self.conversion_stop_event.is_set()):
                                    time.sleep(0.1)
                                
                                if self.conversion_stop_event.is_set():
                                    break
                            
                            line = final_process.stderr.readline()
                            if not line:
                                if final_process.poll() is not None:
                                    break
                                continue
                            
                            progress = self._parse_ffmpeg_progress_with_eta(line)
                            if progress:
                                self.root.after(0, self._update_progress_with_eta, progress, line)
                        
                        final_returncode = final_process.wait()
                        
                        if final_returncode == 0:
                            self.root.after(0, self._conversion_complete, output_path)
                        else:
                            self.root.after(0, self._conversion_failed, f"Final encoding failed: {final_returncode}")
                    else:
                        shutil.copy2(temp_video_path, output_path)
                        self.root.after(0, self._conversion_complete, output_path)
                else:
                    self.root.after(0, self._conversion_failed, f"Video creation failed: {returncode}")
                    
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"AI conversion error: {e}")
            if not self.conversion_stop_event.is_set():
                self.root.after(0, self._conversion_failed, str(e))
    
    def _build_ffmpeg_command_for_enhanced_video_immediate(self, enhanced_video_path: str, output_path: str) -> Optional[List[str]]:
        try:
            cmd = ["ffmpeg", "-i", enhanced_video_path]
            
            self._add_video_params_immediate(cmd)
            
            self._add_audio_params_immediate(cmd)
            
            cmd.extend(["-map", "0:v:0"])
            if self.audio_settings.codec != AudioCodec.NO_AUDIO:
                cmd.extend(["-map", "0:a:0"])
            
            cmd.extend(["-y", output_path])
            
            if ENABLE_LOGGING: logger.info(f"Final FFmpeg command for AI enhancement: {' '.join(cmd)}")
            return cmd
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error building final FFmpeg command for AI enhancement: {e}")
            return None
    
    def _convert_normal(self, output_path: str) -> None:
        total_duration = self.total_frames / self.fps
        
        if self.output_settings.start_frame is not None or self.output_settings.end_frame is not None:
            start_time = self.output_settings.start_frame / self.fps if self.output_settings.start_frame else 0
            end_time = self.output_settings.end_frame / self.fps if self.output_settings.end_frame else total_duration
            self.segment_duration = max(0.001, end_time - start_time)
        else:
            self.segment_duration = total_duration
            start_time = 0
        
        self.conversion_start_time_original = start_time
        
        self.conversion_start_time = time.time()
        self.last_progress_update_time = self.conversion_start_time
        self.last_percentage = 0.0
        
        cmd = self._build_ffmpeg_command(output_path)
        if not cmd:
            return
        
        self.state = AppState.CONVERTING
        self.conversion_paused = False
        self.conversion_pause_event.clear()
        self.convert_btn.config(state=DISABLED)
        
        self.convert_btn.pack_forget()
        self.control_buttons_frame.pack(pady=(0, 0))
        self.pause_resume_btn.config(text="PAUSE", bg="#FF9800")
        
        self.status_label.config(text="Converting...")
        
        self.show_progress_in_status(True)
        self.progress_bar['value'] = 0
        self.progress_percentage.config(text="0.0%")
        
        self.conversion_stop_event.clear()
        self.conversion_thread = threading.Thread(
            target=self._run_conversion,
            args=(cmd, output_path),
            daemon=True
        )
        self.conversion_thread.start()
    
    def _parse_ffmpeg_progress_with_eta(self, line: str) -> Optional[ConversionProgress]:
        progress = ConversionProgress()
        
        time_patterns = [
            r'time=(\d{2}):(\d{2}):(\d{2}\.\d+)',
            r'time=(\d{2}):(\d{2}):(\d{2})',
            r'time=(\d{2}):(\d{2}):(\d{2}),',
            r'time=(\d{1,2}):(\d{2}):(\d{2}\.\d+)'
        ]
        
        time_match = None
        for pattern in time_patterns:
            time_match = re.search(pattern, line)
            if time_match:
                break
        
        if time_match:
            try:
                h, m, s = time_match.groups()
                segment_elapsed_time = int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))
                progress.current_time = segment_elapsed_time
                
                if hasattr(self, 'segment_duration') and self.segment_duration > 0:
                    percentage = min(100.0, (segment_elapsed_time / self.segment_duration) * 100.0)
                    progress.percentage = percentage
                    
                    if percentage > self.last_percentage:
                        self.last_percentage = percentage
                        if ENABLE_LOGGING: logger.debug(f"Progress: {percentage:.1f}% at time {segment_elapsed_time:.2f}s of {self.segment_duration:.2f}s")
                else:
                    progress.percentage = 0.0
                
                progress.total_time = self.segment_duration if hasattr(self, 'segment_duration') else 0
                
            except (ValueError, TypeError) as e:
                if ENABLE_LOGGING: logger.warning(f"Error parsing time from FFmpeg output: {e}")
                return None
        
        speed_match = re.search(r'speed=([\d\.]+)x', line)
        if speed_match:
            try:
                progress.speed = float(speed_match.group(1))
            except ValueError:
                progress.speed = 0.0
        
        fps_match = re.search(r'fps=([\d\.]+)', line)
        if fps_match:
            try:
                progress.fps = float(fps_match.group(1))
            except ValueError:
                progress.fps = 0.0
        
        current_time = time.time()
        if hasattr(self, 'conversion_start_time') and self.conversion_start_time:
            progress.elapsed_time = current_time - self.conversion_start_time
            
            if hasattr(progress, 'percentage') and progress.percentage > 0.1:
                progress.estimated_total_time = progress.elapsed_time / (progress.percentage / 100.0)
                progress.estimated_remaining = max(0.0, progress.estimated_total_time - progress.elapsed_time)
                
                if progress.percentage > 1.0 and progress.elapsed_time > 1.0:
                    if ENABLE_LOGGING: logger.debug(f"ETA: {progress.percentage:.1f}% in {progress.elapsed_time:.1f}s, estimated total: {progress.estimated_total_time:.1f}s")
        
        if hasattr(progress, 'current_time') and progress.current_time > 0:
            return progress
        
        return None
    
    def toggle_pause_resume(self) -> None:
        if self.state == AppState.CONVERTING and not self.conversion_paused:
            self.conversion_paused = True
            self.state = AppState.CONVERSION_PAUSED
            self.conversion_pause_event.set()
            self.pause_resume_btn.config(text="RESUME", bg="#4CAF50")
            self.status_label.config(text="Conversion Paused")
            if ENABLE_LOGGING: logger.info("Conversion paused by user")
            
            # Pause FFmpeg process
            if self.conversion_process and sys.platform != "win32":
                try:
                    self.conversion_process.send_signal(signal.SIGSTOP)
                    if ENABLE_LOGGING: logger.info("Sent SIGSTOP to ffmpeg process")
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Failed to send SIGSTOP: {e}")
            
            # Pause Real-ESRGAN process if it exists
            if hasattr(self, 'realesrgan_process') and self.realesrgan_process and sys.platform != "win32":
                try:
                    self.realesrgan_process.send_signal(signal.SIGSTOP)
                    if ENABLE_LOGGING: logger.info("Sent SIGSTOP to realesrgan process")
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Failed to send SIGSTOP to realesrgan: {e}")
            
            # Pause Real-ESRGAN process if it exists
            if hasattr(self, 'realesrgan_process') and self.realesrgan_process and sys.platform != "win32":
                try:
                    self.realesrgan_process.send_signal(signal.SIGSTOP)
                    if ENABLE_LOGGING: logger.info("Sent SIGSTOP to realesrgan process")
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Failed to send SIGSTOP to realesrgan: {e}")
            
        elif self.state == AppState.CONVERSION_PAUSED:
            self.conversion_paused = False
            self.state = AppState.CONVERTING
            self.conversion_pause_event.clear()
            self.pause_resume_btn.config(text="PAUSE", bg="#FF9800")
            self.status_label.config(text="Converting...")
            if ENABLE_LOGGING: logger.info("Conversion resumed by user")
            
            # Resume FFmpeg process
            if self.conversion_process and sys.platform != "win32":
                try:
                    self.conversion_process.send_signal(signal.SIGCONT)
                    if ENABLE_LOGGING: logger.info("Sent SIGCONT to ffmpeg process")
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Failed to send SIGCONT: {e}")
            
            # Resume Real-ESRGAN process if it exists
            if hasattr(self, 'realesrgan_process') and self.realesrgan_process and sys.platform != "win32":
                try:
                    self.realesrgan_process.send_signal(signal.SIGCONT)
                    if ENABLE_LOGGING: logger.info("Sent SIGCONT to realesrgan process")
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Failed to send SIGCONT to realesrgan: {e}")
            
            # Resume Real-ESRGAN process if it exists
            if hasattr(self, 'realesrgan_process') and self.realesrgan_process and sys.platform != "win32":
                try:
                    self.realesrgan_process.send_signal(signal.SIGCONT)
                    if ENABLE_LOGGING: logger.info("Sent SIGCONT to realesrgan process")
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Failed to send SIGCONT to realesrgan: {e}")
    
    def stop_conversion(self) -> None:
        if self.state in [AppState.CONVERTING, AppState.CONVERSION_PAUSED]:
            response = messagebox.askyesno(
                "Stop Conversion",
                "Are you sure you want to stop the conversion?\nThe partial output file will be incomplete.",
                icon=messagebox.WARNING
            )
            
            if not response:
                return
            
            self.conversion_stop_event.set()
            
            if self.conversion_paused:
                self.conversion_pause_event.clear()
            
            if self.conversion_process:
                try:
                    self.conversion_process.terminate()
                    if ENABLE_LOGGING: logger.info("Conversion process terminated")
                except Exception as e:
                    if ENABLE_LOGGING: logger.error(f"Error terminating process: {e}")
            
            self._reset_conversion_ui()
            self.status_label.config(text="Conversion stopped by user")
            
            self.current_job = None
            
            messagebox.showinfo("Conversion Stopped", "Conversion has been stopped.")
    
    def _validate_conversion_settings(self) -> bool:
        self._update_current_settings_from_ui()
        
        if self.video_settings.codec == VideoCodec.CUSTOM:
            encoder = self.video_encoder_var.get().strip()
            if not encoder:
                messagebox.showerror("Error", "Please specify a custom video encoder")
                return False
            
            self.video_settings.custom_encoder = encoder
            self.video_settings.custom_encoder_args = self.video_args_var.get().strip()
        
        if self.audio_settings.codec == AudioCodec.CUSTOM:
            encoder = self.audio_encoder_var.get().strip()
            if not encoder:
                messagebox.showerror("Error", "Please specify a custom audio encoder")
                return False
            
            self.audio_settings.custom_encoder = encoder
            self.audio_settings.custom_encoder_args = self.audio_args_var.get().strip()
        
        if self.output_settings.format == VideoFormat.CUSTOM:
            custom_format = self.custom_output_var.get().strip()
            if not custom_format:
                messagebox.showerror("Error", "Please specify a custom output format")
                return False
            self.output_settings.custom_format = custom_format
        
        if self.video_settings.quality_mode == QualityMode.CQ:
                # Skip CQ validation for FFV1 and other lossless codecs
                if "FFV1" not in self.video_settings.codec.value and "Lossless" not in self.video_settings.codec.value:
                    try:
                        cq_value = int(self.cq_var.get())
                        if "AV1" in self.video_settings.codec.value:
                            if not 0 <= cq_value <= 63:
                                messagebox.showerror("Error", "CQ value must be between 0 and 63 for AV1")
                                return False
                        else:
                            if not 0 <= cq_value <= 51:
                                messagebox.showerror("Error", "CQ value must be between 0 and 51")
                                return False
                    except ValueError:
                        messagebox.showerror("Error", "Invalid CQ value")
                        return False
        
        fps_text = self.fps_var.get().strip()
        if fps_text:
            try:
                fps = float(fps_text)
                if fps <= 0:
                    messagebox.showerror("Error", "FPS must be positive")
                    return False
            except ValueError:
                messagebox.showerror("Error", "Invalid FPS value")
                return False
        
        if self.video_settings.resolution_mode == ResolutionMode.CUSTOM:
            if not self.video_settings.custom_width or not self.video_settings.custom_height:
                messagebox.showerror("Error", "Please set custom width and height")
                return False
            
            if self.video_settings.custom_width <= 0 or self.video_settings.custom_height <= 0:
                messagebox.showerror("Error", "Resolution dimensions must be positive")
                return False
            
            if self.video_settings.custom_width % 2 != 0 or self.video_settings.custom_height % 2 != 0:
                messagebox.showerror("Error", "Resolution dimensions must be even numbers")
                return False
        
        elif self.video_settings.resolution_mode in [ResolutionMode.CUSTOM_WIDTH, ResolutionMode.CUSTOM_HEIGHT]:
            if not self.video_settings.custom_dimension_value:
                messagebox.showerror("Error", "Please set the custom dimension value")
                return False
            
            if self.video_settings.custom_dimension_value <= 0:
                messagebox.showerror("Error", "Dimension value must be positive")
                return False
            
            if self.video_settings.custom_dimension_value % 2 != 0:
                messagebox.showerror("Error", "Dimension value must be an even number")
                return False
        
        if self.audio_settings.codec != AudioCodec.CUSTOM:
            try:
                audio_bitrate = int(self.audio_bitrate_var.get())
                if audio_bitrate <= 0:
                    messagebox.showerror("Error", "Audio bitrate must be positive")
                    return False
            except ValueError:
                messagebox.showerror("Error", "Invalid audio bitrate")
                return False
            
            try:
                samplerate = int(self.samplerate_var.get())
                if samplerate <= 0:
                    messagebox.showerror("Error", "Sample rate must be positive")
                    return False
            except ValueError:
                messagebox.showerror("Error", "Invalid sample rate")
                return False
        
        return True
    
    def _get_output_filename(self) -> Optional[str]:
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        
        if self.video_settings.codec == VideoCodec.NO_VIDEO:
            audio_extensions = {
                AudioCodec.FLAC: ".flac",
                AudioCodec.PCM_16: ".wav",
                AudioCodec.PCM_24: ".wav",
                AudioCodec.PCM_32: ".wav",
                AudioCodec.AAC: ".m4a",
                AudioCodec.OPUS: ".opus",
                AudioCodec.MP3: ".mp3",
                AudioCodec.AC3: ".ac3",
                AudioCodec.DTS: ".dts",
                AudioCodec.VORBIS: ".ogg",
                AudioCodec.CUSTOM: ".mka"
            }
            ext = audio_extensions.get(self.audio_settings.codec, ".flac")
            default_name = f"{base}_audio{ext}"
            filetypes = [("Audio files", f"*{ext}"), ("All files", "*.*")]
        else:
            if self.output_settings.format == VideoFormat.CUSTOM and self.output_settings.custom_format:
                ext = self.output_settings.custom_format
                if not ext.startswith("."):
                    ext = f".{ext}"
            else:
                format_extensions = {
                    VideoFormat.MKV: ".mkv",
                    VideoFormat.MP4: ".mp4",
                    VideoFormat.MOV: ".mov",
                    VideoFormat.AVI: ".avi",
                    VideoFormat.WEBM: ".webm",
                    VideoFormat.FLV: ".flv",
                    VideoFormat.TS: ".ts"
                }
                fmt = VideoFormat(self.output_format_var.get())
                ext = format_extensions.get(fmt, ".mkv")
            
            out_width, out_height = self.video_settings.calculate_output_dimensions(
                self.video_width, self.video_height, self.crop_rect
            )
            res_str = f"{out_width}x{out_height}"
            default_name = f"{base}_{res_str}{ext}"
            filetypes = [(f"{fmt.value} files", f"*{ext}"), ("All files", "*.*")]
        
        return filedialog.asksaveasfilename(
            defaultextension=ext,
            initialfile=default_name,
            filetypes=filetypes
        )
    
    def _build_ffmpeg_command(self, output_path: str) -> Optional[List[str]]:
        try:
            cmd = ["ffmpeg"]
            
            if self.output_settings.start_frame is not None:
                start_time = self.output_settings.start_frame / self.fps
                cmd.extend(["-ss", f"{start_time:.6f}"])
            
            cmd.extend(["-i", self.video_path])
            
            if (self.output_settings.end_frame is not None and 
                self.output_settings.start_frame is not None):
                duration = (self.output_settings.end_frame - 
                          self.output_settings.start_frame) / self.fps
                cmd.extend(["-t", f"{duration:.6f}"])
            elif self.output_settings.end_frame is not None:
                duration = self.output_settings.end_frame / self.fps
                cmd.extend(["-t", f"{duration:.6f}"])
            
            if self.video_settings.codec == VideoCodec.NO_VIDEO:
                cmd.extend(["-vn"])
                self._add_audio_params_immediate(cmd)
                cmd.extend(["-map", "0:a?"])
            
            else:
                self._add_video_params_immediate(cmd)
                self._add_audio_params_immediate(cmd)
                
                cmd.extend(["-map", "0:v:0"])
                if self.audio_settings.codec != AudioCodec.NO_AUDIO:
                    cmd.extend(["-map", "0:a?"])
            
            cmd.extend(["-y", output_path])
            
            if ENABLE_LOGGING: logger.info(f"FFmpeg command: {' '.join(cmd)}")
            return cmd
            
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Error building FFmpeg command: {e}")
            messagebox.showerror("Error", f"Failed to build conversion command: {str(e)}")
            return None
    
    def _add_video_params_immediate(self, cmd: List[str]) -> None:
        if self.video_settings.codec == VideoCodec.CUSTOM:
            if self.video_settings.custom_encoder:
                cmd.extend(["-c:v", self.video_settings.custom_encoder])
                
                if self.video_settings.custom_encoder_args:
                    import shlex
                    try:
                        args = shlex.split(self.video_settings.custom_encoder_args)
                        cmd.extend(args)
                    except Exception as e:
                        if ENABLE_LOGGING: logger.warning(f"Failed to parse custom video args: {e}")
                        args = self.video_settings.custom_encoder_args.split()
                        cmd.extend(args)
            else:
                cmd.extend(["-c:v", "libx264"])
        else:
            encoder_map = {
                VideoCodec.FFV1: "ffv1",
                VideoCodec.H264_X264: "libx264",
                VideoCodec.H265_X265: "libx265",
                VideoCodec.AV1_SVT: "libsvtav1",
                VideoCodec.VP9: "libvpx-vp9",
                VideoCodec.PRORES: "prores",
                VideoCodec.DNXHD: "dnxhd",
                VideoCodec.MPEG2: "mpeg2video",
                VideoCodec.MJPEG: "mjpeg",
                VideoCodec.H264_NVENC: "h264_nvenc",
                VideoCodec.H265_NVENC: "hevc_nvenc",
                VideoCodec.H264_QSV: "h264_qsv",
                VideoCodec.H265_QSV: "hevc_qsv",
                VideoCodec.H264_AMF: "h264_amf",
                VideoCodec.H265_AMF: "hevc_amf",
                VideoCodec.AV1_AOM: "libaom-av1",
                VideoCodec.VP8: "libvpx",
                VideoCodec.MPEG4: "mpeg4",
                VideoCodec.DV: "dvvideo",
                VideoCodec.RAW: "rawvideo"
            }
            
            encoder = encoder_map.get(self.video_settings.codec)
            if not encoder:
                return
            
            cmd.extend(["-c:v", encoder])
            
            is_lossy = not any(lossless in self.video_settings.codec.value.lower() 
                             for lossless in ["lossless", "pcm", "raw", "ffv1"])
            
            if is_lossy and self.video_settings.speed_preset:
                preset = self.video_settings.speed_preset
                
                if "nvenc" in encoder:
                    cmd.extend(["-preset", preset])
                elif "qsv" in encoder:
                    cmd.extend(["-preset", preset])
                elif "amf" in encoder:
                    cmd.extend(["-quality", preset])
                elif "svtav1" in encoder:
                    cmd.extend(["-preset", preset])
                elif "libaom-av1" in encoder:
                    cmd.extend(["-cpu-used", preset])
                elif "libvpx" in encoder:
                    cmd.extend(["-cpu-used", preset])
                elif encoder in ["libx264", "libx265", "mpeg4", "mpeg2video", "mjpeg"]:
                    cmd.extend(["-preset", preset])
            
            if self.video_settings.quality_mode == QualityMode.CQ:
                cq_value = self.video_settings.cq_value
                
                if "nvenc" in encoder:
                    cmd.extend(["-cq", str(cq_value)])
                elif "qsv" in encoder:
                    cmd.extend(["-global_quality", str(cq_value)])
                elif "amf" in encoder:
                    cmd.extend(["-qp_i", str(cq_value), "-qp_p", str(cq_value), "-qp_b", str(cq_value)])
                elif "svtav1" in encoder:
                    cmd.extend(["-crf", str(cq_value)])
                    if self.video_settings.speed_preset:
                        cmd.extend(["-preset", str(self.video_settings.speed_preset)])
                elif encoder in ["libx264", "libx265", "libvpx-vp9", "libaom-av1"]:
                    cmd.extend(["-crf", str(cq_value)])
                else:
                    cmd.extend(["-crf", str(cq_value)])
                
            else:
                bitrate = self.video_settings.bitrate
                cmd.extend(["-b:v", f"{bitrate}k"])
                
                if self.video_settings.bitrate_type == BitrateType.CBR:
                    cmd.extend([
                        "-maxrate", f"{bitrate}k",
                        "-minrate", f"{bitrate}k",
                        "-bufsize", f"{bitrate * 2}k"
                    ])
        
        vfilters = []
        
        if self.crop_rect:
            vfilters.append(f"crop={self.crop_rect.width}:{self.crop_rect.height}:{self.crop_rect.x}:{self.crop_rect.y}")
        
        if self.video_settings.resolution_mode != ResolutionMode.ORIGINAL and not self.video_settings.is_ai_enhancement():
            out_width, out_height = self.video_settings.calculate_output_dimensions(
                self.video_width, self.video_height, self.crop_rect
            )
            scale_filter = f"scale={out_width}:{out_height}"
            vfilters.append(scale_filter)
        
        if vfilters:
            cmd.extend(["-vf", ",".join(vfilters)])
        
        if self.video_settings.output_fps:
            cmd.extend(["-r", str(self.video_settings.output_fps)])
    
    def _add_audio_params_immediate(self, cmd: List[str]) -> None:
        if self.audio_settings.codec == AudioCodec.NO_AUDIO:
            cmd.extend(["-an"])
            return
        
        if self.audio_settings.codec == AudioCodec.CUSTOM:
            if self.audio_settings.custom_encoder:
                cmd.extend(["-c:a", self.audio_settings.custom_encoder])
                
                if self.audio_settings.custom_encoder_args:
                    import shlex
                    try:
                        args = shlex.split(self.audio_settings.custom_encoder_args)
                        cmd.extend(args)
                    except Exception as e:
                        if ENABLE_LOGGING: logger.warning(f"Failed to parse custom audio args: {e}")
                        args = self.audio_settings.custom_encoder_args.split()
                        cmd.extend(args)
            else:
                cmd.extend(["-c:a", "aac"])
        else:
            encoder_map = {
                AudioCodec.FLAC: "flac",
                AudioCodec.PCM_16: "pcm_s16le",
                AudioCodec.PCM_24: "pcm_s24le",
                AudioCodec.PCM_32: "pcm_s32le",
                AudioCodec.AAC: "aac",
                AudioCodec.OPUS: "libopus",
                AudioCodec.MP3: "libmp3lame",
                AudioCodec.AC3: "ac3",
                AudioCodec.DTS: "dca",
                AudioCodec.VORBIS: "libvorbis"
            }
            
            encoder = encoder_map.get(self.audio_settings.codec)
            if not encoder:
                return
            
            cmd.extend(["-c:a", encoder])
            
            if self.audio_settings.codec not in [AudioCodec.FLAC, 
                                               AudioCodec.PCM_16,
                                               AudioCodec.PCM_24,
                                               AudioCodec.PCM_32]:
                bitrate = self.audio_settings.bitrate
                cmd.extend(["-b:a", f"{bitrate}k"])
            
            samplerate = self.audio_settings.samplerate
            cmd.extend(["-ar", str(samplerate)])
    
    def _run_conversion(self, cmd: List[str], output_path: str) -> None:
        try:
            self.conversion_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            while not self.conversion_stop_event.is_set():
                if self.conversion_pause_event.is_set():
                    while (self.conversion_pause_event.is_set() and 
                           not self.conversion_stop_event.is_set()):
                        time.sleep(0.1)
                    
                    if self.conversion_stop_event.is_set():
                        break
                
                line = self.conversion_process.stderr.readline()
                if not line:
                    if self.conversion_process.poll() is not None:
                        break
                    continue
                
                progress = self._parse_ffmpeg_progress_with_eta(line)
                if progress:
                    self.root.after(0, self._update_progress_with_eta, progress, line)
            
            returncode = self.conversion_process.wait()
            
            if returncode == 0:
                self.root.after(0, self._conversion_complete, output_path)
            else:
                if not self.conversion_stop_event.is_set():
                    self.root.after(0, self._conversion_failed, f"FFmpeg returned {returncode}")
                else:
                    self.root.after(0, self._conversion_stopped)
                    
        except Exception as e:
            if ENABLE_LOGGING: logger.error(f"Conversion error: {e}")
            if not self.conversion_stop_event.is_set():
                self.root.after(0, self._conversion_failed, str(e))
            
        finally:
            self.conversion_process = None
    
    def _conversion_complete(self, output_path: str) -> None:
        self._reset_conversion_ui()
        self.state = AppState.VIDEO_LOADED
        
        # Calculate time taken
        if hasattr(self, 'conversion_start_time'):
            time_taken = time.time() - self.conversion_start_time
            hours = int(time_taken // 3600)
            minutes = int((time_taken % 3600) // 60)
            seconds = int(time_taken % 60)
            
            if hours > 0:
                time_str = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}"
            elif minutes > 0:
                time_str = f"{minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}"
            else:
                time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
            
            self.status_label.config(text=f"Conversion complete after {time_str}")
            messagebox.showinfo("Success", f"Conversion complete after {time_str}!\nSaved to: {output_path}")
        else:
            self.status_label.config(text="Conversion complete")
            messagebox.showinfo("Success", f"Conversion complete!\nSaved to: {output_path}")
    
    def _conversion_failed(self, error: str) -> None:
        self._reset_conversion_ui()
        self.state = AppState.VIDEO_LOADED
        self.status_label.config(text="Conversion failed")
        
        messagebox.showerror("Error", f"Conversion failed: {error}")
    
    def _conversion_stopped(self) -> None:
        self._reset_conversion_ui()
        self.state = AppState.VIDEO_LOADED
        self.status_label.config(text="Conversion stopped")
        
        if ENABLE_LOGGING: logger.info("Conversion stopped by user")
    
    def on_window_resize(self, event) -> None:
        if event.widget == self.root and self.video_path:
            self.frame_cache.clear()
            self.root.after(100, self.show_frame)
    
    def cleanup(self) -> None:
        self.stop_playback()
        
        if (self.state in [AppState.CONVERTING, AppState.CONVERSION_PAUSED] and 
            self.conversion_process and self.conversion_process.poll() is None):
            self.conversion_stop_event.set()
            self.conversion_process.terminate()
            try:
                self.conversion_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.conversion_process.kill()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.frame_cache.clear()
        
        if self.playback_update_id:
            self.root.after_cancel(self.playback_update_id)
        
        if self.queue_window and self.queue_window.winfo_exists():
            self.queue_window.destroy()
        
        self.save_queue()

def main():
    parser = argparse.ArgumentParser(description='FAVencoder - Frame-accurate Video Encoder')
    parser.add_argument('--no-gui', '-n', action='store_true', help='Run in no-GUI mode')
    parser.add_argument('--gui', '-g', action='store_true', help='Run in GUI mode (default)')
    
    args = parser.parse_args()
    
    if args.no_gui:
        app = VideoCropper(root=None)
        try:
            app.run_no_gui_mode()
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user")
            print("App closed")
            sys.exit(0)
    else:
        root = Tk()
        app = VideoCropper(root)
        
        def on_closing():
            app.cleanup()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        def handle_keyboard_interrupt(signum, frame):
            print("\n\n⚠️  App interrupted by user")
            app.status_label.config(text="App interrupted by user")
            on_closing()
            sys.exit(0)
        
        try:
            import signal
            signal.signal(signal.SIGINT, lambda s, f: root.after(0, handle_keyboard_interrupt, s, f))
        except (ImportError, AttributeError):
            pass
        
        try:
            root.mainloop()
        except KeyboardInterrupt:
            print("\n\n⚠️  App interrupted by user")
            app.status_label.config(text="App interrupted by user")
            on_closing()

if __name__ == "__main__":
    main()
