import numpy as np
import cv2
import tempfile
import subprocess
import torch
import os
import torchaudio
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

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
                yield {"image": Image.fromarray(frame)}

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
                yield {"image": Image.fromarray(frame)}

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
                yield {"image": Image.fromarray(frame)}


class AudioFull:
    def __call__(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] == 2:
            wav = wav.mean(0, keepdim=True)
        yield {"raw_audio": wav, "sample_rate": sr}


# TODO Make class for audio splitting, because large files couldn't be loaded into vram
class AudioNSecondsSplit:
    def __init__(self, seconds: float):
        self.seconds = seconds

    def __call__(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] == 2:
            wav = wav.mean(0, keepdim=True)
        num_frames = int(sr * self.seconds)
        for i in range(0, wav.shape[1], num_frames):
            yield {"raw_audio": wav[:, i: min(i + num_frames, wav.shape[1])], "sample_rate": sr}