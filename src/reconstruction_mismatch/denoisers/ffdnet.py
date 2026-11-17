"""Pretrained FFDNet (grey-scale) from the KAIR model zoo -- optional extension.

Source  : https://github.com/cszn/KAIR (MIT License), release v1.0, file ffdnet_gray.pth
Model   : Zhang, Zuo, Zhang, "FFDNet: Toward a Fast and Flexible Solution for CNN-Based
          Image Denoising", IEEE TIP 2018. Trained by the KAIR authors on natural images.
This file re-implements the network topology of KAIR's models/network_ffdnet.py
(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R') so the released weights can be loaded;
no KAIR source code is copied. Key facts used:
  * 2x2 pixel-unshuffle -> concat noise map (sigma on the [0,1] scale) -> 15 conv layers
    (3x3, 64 ch, ReLU, zero padding) -> pixel-shuffle; output is the denoised image
    (not a residual). Input images in [0, 1].
Domain shift: the weights were fitted to natural photographs, not to our synthetic phantoms
or to CT slices.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch import nn

from ..utils.io import sha256_file

KAIR_FFDNET_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth"


class FFDNetGray(nn.Module):
    def __init__(self, nc: int = 64, nb: int = 15) -> None:
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        layers: list[nn.Module] = [nn.Conv2d(4 + 1, nc, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nb - 2):
            layers += [nn.Conv2d(nc, nc, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(nc, 4, 3, padding=1))
        self.model = nn.Sequential(*layers)
        self.up = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        h, w = x.shape[-2:]
        pb, pr = h % 2, w % 2
        if pb or pr:
            x = nn.functional.pad(x, (0, pr, 0, pb), mode="replicate")
        z = self.down(x)
        m = torch.full((z.shape[0], 1, *z.shape[-2:]), float(sigma), dtype=z.dtype, device=z.device)
        out = self.up(self.model(torch.cat([z, m], dim=1)))
        return out[..., :h, :w]


class FFDNetDenoiser:
    def __init__(self, model: FFDNetGray, chunk: int = 24) -> None:
        self.model = model.eval()
        self.chunk = chunk
        self.name = "ffdnet_kair"

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.cat([self.model(xc, sigma) for xc in torch.split(x, self.chunk)], dim=0)


def download_kair_ffdnet(dest_dir: Path, log=print) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "ffdnet_gray.pth"
    if not path.exists():
        log(f"[ffdnet] downloading {KAIR_FFDNET_URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(KAIR_FFDNET_URL, tmp)
        tmp.replace(path)
    return path


def load_kair_ffdnet(dest_dir: Path, device="cpu", log=print) -> tuple[FFDNetDenoiser, dict]:
    path = download_kair_ffdnet(dest_dir, log=log)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = FFDNetGray()
    model.load_state_dict(state, strict=True)  # raises on any key / shape mismatch
    model.to(device)
    info = {"denoiser": "ffdnet_kair", "url": KAIR_FFDNET_URL, "license": "MIT (KAIR repository)",
            "checkpoint_sha256": sha256_file(path)}
    return FFDNetDenoiser(model), info
