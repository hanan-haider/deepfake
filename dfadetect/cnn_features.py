

from dataclasses import dataclass, field
from typing import List

import torch
import torchaudio


@dataclass
class CNNFeaturesSetting:
    frontend_algorithm: List[str]= field(default_factory=lambda: ["mfcc"])
    use_spectrogram: bool = True




LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=80,
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)


def prepare_feature_vector(
    audio: torch.Tensor,
    cnn_features_setting: CNNFeaturesSetting,
    win_length: int = 400,
    hop_length: int = 160,
) -> torch.Tensor:

    feature_vector = []

    if "mfcc" in cnn_features_setting.frontend_algorithm:
        mfcc_feature = MFCC_FN(audio)
        feature_vector.append(mfcc_feature)

    if "lfcc" in cnn_features_setting.frontend_algorithm:
        lfcc_feature = LFCC_FN(audio)
        feature_vector.append(lfcc_feature)

    if cnn_features_setting.use_spectrogram:
        stft_features = prepare_stft_features(audio, win_length, hop_length)
        feature_vector += stft_features  # abs_mel, abs_angle

    assert len(feature_vector) >= 1, "Feature vector must contain at least one feature!"

    feature_vector = torch.stack(feature_vector, dim=1)

    # [batch_size, feature_num, 80, frames], where feature_num in {1,2,3,4}
    return feature_vector

    