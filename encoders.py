from typing import List, Union
import numpy as np

import torch
from torchaudio.functional import resample
import transformers
from transformers import CLIPProcessor, CLIPModel
from transformers import EncodecModel, AutoProcessor

from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class ColorHistogramEncoder:
    def __init__(self, n_buckets=256):
        self.n_buckets = n_buckets

    def encode(self, image: Image):
        """
        Takes a `PIL.Image` and returns a numpy array representing
        a color histogram.
        """
        arr = np.array(image)
        output = np.concatenate(
            [
                np.histogram(
                    arr[:, :, 0].flatten(),
                    bins=np.linspace(0, 255, self.n_buckets + 1),
                )[0],
                np.histogram(
                    arr[:, :, 1].flatten(),
                    bins=np.linspace(0, 255, self.n_buckets + 1),
                )[0],
                np.histogram(
                    arr[:, :, 2].flatten(),
                    bins=np.linspace(0, 255, self.n_buckets + 1),
                )[0],
            ]
        )
        return torch.tensor(output)


class TimmEncoder(torch.nn.Module):
    def __init__(self, name: str, device="cpu") -> None:
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self.model = timm.create_model(name, pretrained=True, num_classes=0).to(self.device)
        self.config = resolve_data_config(self.model.pretrained_cfg)
        self.transform = create_transform(**self.config)

    def encode(self, image: Image):
        """
        Transforms grabbed images into numeric representations.
        """
        image = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(image)
    
    # def to(self, device):
    #     self.device = device
    #     self.model = self.model.to(device)
    #     return self


class CLIPEncoder(torch.nn.Module):
    def __init__(self, name: str = "openai/clip-vit-large-patch14", device="cpu") -> None:
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(name)

    def encode(self, image: Image):
        """
        Transforms text into numeric representations.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs


class Wav2Vec2Encoder(torch.nn.Module):
    def __init__(self, name: str = "facebook/wav2vec2-base-960h", device="cpu") -> None:
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self.model = transformers.Wav2Vec2Model.from_pretrained(name).to(self.device)
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(name)
        self.sample_rate = self.processor.current_processor.sampling_rate

    def encode(self, raw_audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], sample_rate: int):
        raw_audio = resample(raw_audio.squeeze(), sample_rate, self.sample_rate)
        inputs = self.processor(audio=raw_audio, sampling_rate=self.sample_rate, return_tensors="pt").input_values.to(self.device)
        outputs = self.model(inputs)
        return outputs.last_hidden_state


class EnCodecEncoder(torch.nn.Module):
    def __init__(self, bandwidth=1.5, device="cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.processor.chunk_length_s = 10
        self.processor.overlap = 0
        self.bandwidth = bandwidth
        self.sampling_rate = self.processor.sampling_rate
        self.quantizer = self.model.quantizer
        self.sample_rate = self.processor.sampling_rate

    def encode(self, raw_audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], sample_rate: int):
        raw_audio = resample(raw_audio, sample_rate, self.sample_rate).squeeze()
        inputs = self.processor(raw_audio=raw_audio, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        encoder_outputs = self.model.encode(**inputs, bandwidth=self.bandwidth)
        codes = encoder_outputs.audio_codes
        codes = codes.transpose(0, 1)
        embeddings = []

        for codes in encoder_outputs.audio_codes:
            codes = codes.transpose(0, 1)
            emb = self.model.quantizer.decode(codes)
            emb = emb.transpose(-1, -2)
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)
        return embeddings