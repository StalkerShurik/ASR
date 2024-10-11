import torch_audiomentations
from torch import Tensor, nn
from torchaudio.utils import download_asset


class PitchShift(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(
            sample_rate=16000, p=0.25, *args, **kwargs
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
