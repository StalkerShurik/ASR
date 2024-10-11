import torch_audiomentations
from torch import Tensor, nn
from torchaudio.utils import download_asset


class BackGroundNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddBackgroundNoise(
            sample_rate=16000, *args, **kwargs
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
