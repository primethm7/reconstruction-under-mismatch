"""Small noise-conditional residual CNN denoiser (written for this project).

Architecture (DnCNN-style residual learning, FFDNet-style noise-level map input):
    input  : [x, sigma * 1]  (2 channels; sigma is the Gaussian noise std on the [0,1] scale)
    body   : Conv3x3(2->C) + ReLU, (depth-2) x [Conv3x3(C->C) + ReLU], Conv3x3(C->1)
    output : x - body(input)   (the network predicts the noise)
Convolutions use *circular* padding, consistent with the periodic image model.
No batch-norm, so train/eval behaviour is identical.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SmallResidualDenoiser(nn.Module):
    def __init__(self, depth: int = 8, width: int = 32) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.depth = depth
        self.width = width
        layers: list[nn.Module] = [nn.Conv2d(2, width, 3, padding=1, padding_mode="circular"), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, 1, 3, padding=1, padding_mode="circular"))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(float(sigma), dtype=x.dtype, device=x.device)
        sigma = sigma.to(dtype=x.dtype, device=x.device)
        if sigma.dim() == 0:
            smap = sigma.expand(x.shape[0], 1, *x.shape[-2:])
        else:
            smap = sigma.view(-1, 1, 1, 1).expand(x.shape[0], 1, *x.shape[-2:])
        return x - self.body(torch.cat([x, smap], dim=1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {"arch": "SmallResidualDenoiser", "depth": self.depth, "width": self.width}


class CNNDenoiser:
    """Callable wrapper D(x, sigma) for PnP: eval mode, no grad, optional chunking."""

    def __init__(self, model: nn.Module, name: str = "small_cnn", chunk: int = 64) -> None:
        self.model = model.eval()
        self.name = name
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if x.shape[0] <= self.chunk:
            return self.model(x, sigma)
        return torch.cat([self.model(xc, sigma) for xc in torch.split(x, self.chunk)], dim=0)


def load_cnn_denoiser(path: str | Path, device: torch.device | str = "cpu") -> tuple[CNNDenoiser, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = SmallResidualDenoiser(depth=int(cfg["depth"]), width=int(cfg["width"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return CNNDenoiser(model, name="small_cnn"), cfg
