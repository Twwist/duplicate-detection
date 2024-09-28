import numpy as np
import cv2
import tempfile
import subprocess
import torch
import os
import whisperx
import torchvision
import torchvision.transforms.functional as F
from PIL import Image


# IMAGE
class VideoEveryNFramesExtractor:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, video_path):
        frame_count = 0
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.n == 0:
                yield Image.fromarray(frame)

            frame_count += 1

        cap.release()


class VideoNEvenlySpacedExtractor:
    def __init__(self, n: int):
        self.n = n
        
    def __call__(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(self.n, total_frames)

        interval = total_frames / n

        for i in range(n):
            target_frame = int(i * interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret:
                yield Image.fromarray(frame)

        cap.release()


class VideoKeyFrameFFmpegExtractor:
    def __call__(self, video_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vf", "select='eq(pict_type,PICT_TYPE_I)'",
                "-vsync", "vfr",
                f"{temp_dir}/keyframe_%04d.jpg"
            ]

            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Error extracting keyframes: {e}")
                print(f"FFmpeg output: {e.stderr.decode()}")
                return

            keyframes = sorted(os.listdir(temp_dir))
            for keyframe in keyframes:
                frame_path = os.path.join(temp_dir, keyframe)
                frame = cv2.imread(frame_path)
                yield Image.fromarray(frame)

# AUDIO
# TODO Make class for audio splitting, because large files couldn't be loaded into vram


# TEXT
class TextExtractorWhisperx:
    def __init__(self, device='cpu', compute_type='int8'):
        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    def __call__(self, path, batch_size=64):
        audio = whisperx.load_audio(path)
        result = self.model.transcribe(audio, batch_size=batch_size)
        transcibe_text = ' '.join([i['text'] for i in result["segments"]])

        return transcibe_text