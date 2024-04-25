from pathlib import Path
from typing import List

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import vlm.contrib.s3d as s3d
from vlm.contrib.open_clip.transform import VICLIP_MEAN, VICLIP_STD, image_transform
from vlm.contrib.viclip import get_viclip


class TextEncoder(nn.Module):
    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Encode a batch of strings."""
        ...


class VideoEncoder(nn.Module):
    expected_n_frames: int

    def transform(self, frames: torch.Tensor) -> torch.Tensor:
        """Transform a batch of frames to the expected input format for the model. Input shape: (n_frames, c, h, w)"""
        ...

    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos. Input shape: (batch_size, n_frames, c, h, w)"""
        ...


class Encoder(TextEncoder, VideoEncoder): ...


class CLIP(Encoder):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        expected_n_frames: int = 32,
    ):
        super().__init__()

        self._model: open_clip.model.CLIP = open_clip.create_model(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )  # type: ignore
        assert isinstance(self._model, open_clip.model.CLIP)
        size = self._model.visual.image_size
        image_size: int = size if isinstance(size, int) else size[0]  # type: ignore
        self.transform = image_transform(image_size)  # type: ignore
        self.expected_n_frames = expected_n_frames

    @torch.inference_mode()
    def encode_text(self, x: List[str]) -> torch.Tensor:
        tokens = open_clip.tokenize(x)
        encoded = self._model.encode_text(
            tokens.to("cuda" if torch.cuda.is_available() else "cpu")
        ).float()
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)

        return encoded

    @torch.inference_mode()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._model.encode_image(x, normalize=True)
        return encoded

    @torch.inference_mode()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        _, n_frames, *_ = videos.shape
        videos = rearrange(videos, "b t c h w -> (b t) c h w")
        encoded_frames = self._model.encode_image(videos, normalize=True)
        window_embed = reduce(
            encoded_frames, "(b t) d -> b d", reduction="mean", t=n_frames
        )
        return window_embed


class ViCLIP(VideoEncoder):
    def __init__(self, model_cache_dir: str) -> None:
        super().__init__()
        model_name = "ViCLIP-L_InternVid-FLT-10M.pth"
        path = Path(model_cache_dir) / "viclip" / model_name
        self._model, self._tokenizer = get_viclip(
            "l", path.absolute().as_posix(), frames_per_video=8
        )
        size = self._model.inputs_image_res
        self.transform = image_transform(size, mean=VICLIP_MEAN, std=VICLIP_STD)  # type: ignore
        self.expected_n_frames = self._model.video_input_num_frames

    @torch.inference_mode()
    def encode_text(self, x: List[str]) -> torch.Tensor:
        result = [self._model.get_text_features(t, self._tokenizer) for t in x]
        result = torch.cat(result)
        return result

    @torch.inference_mode()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        _, n_frames, *_ = videos.shape
        assert n_frames >= self.expected_n_frames
        encoded_videos = self._model.get_vid_features(videos)

        return encoded_videos


class S3D(nn.Module):
    def __init__(self, model_cache_dir: str) -> None:
        super().__init__()
        self._model = s3d.S3D(f"{model_cache_dir}/s3d/s3d_dict.npy", 512)
        self._model.load_state_dict(
            torch.load(f"{model_cache_dir}/s3d/s3d_howto100m.pth")
        )
        self._model = self._model.eval()

        self.target_size = (224, 224)
        self.expected_n_frames = 32

    @torch.inference_mode()
    def encode_text(self, x: List[str]) -> torch.Tensor:
        return self._model.text_module(x)["text_embedding"]

    def _transform(self, img: torch.Tensor) -> torch.Tensor:
        if img.dtype not in (torch.float16, torch.float32, torch.float64):
            img = img.float() / 255
        img = F.interpolate(img, mode="bicubic", size=self.target_size)
        img = img.clamp(0, 1)
        return img

    @torch.inference_mode()
    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        _, n_frames, *_ = videos.shape

        assert n_frames >= self.expected_n_frames

        videos = rearrange(videos, "b t c h w -> b c t h w")
        encoded_videos = self._model(videos)["video_embedding"]

        return encoded_videos
