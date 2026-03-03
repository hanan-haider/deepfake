import random

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
# from torch.utils.data.dataset import T_co

# from dfadetect.datasets import AudioDataset, PadDataset


WAVE_FAKE_INTERFACE = True
WAVE_FAKE_SR = 16_000
WAVE_FAKE_TRIM = True
WAVE_FAKE_NORMALIZE = True
WAVE_FAKE_CELL_PHONE = False
WAVE_FAKE_PAD = True
WAVE_FAKE_CUT = 64_600


class SimpleAudioFakeDataset(Dataset):

    def __init__(self, fold_num, fold_subset, transform=None, return_label=True):
        print(f"[SimpleAudioFakeDataset.__init__] fold_num={fold_num}, fold_subset={fold_subset}, "
              f"return_label={return_label}")
        self.transform = transform
        self.samples = pd.DataFrame()

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = None
        self.bona_partition = None
        self.seed = None
        self.return_label = return_label

    def split_real_samples(self, samples_list):
        print(f"[split_real_samples] input type={type(samples_list)}, len={len(samples_list)}")
        if isinstance(samples_list, pd.DataFrame):
            print("[split_real_samples] treating as DataFrame; sorting and shuffling with seed", self.seed)
            samples_list = samples_list.sort_values(by=list(samples_list.columns))
            samples_list = samples_list.sample(frac=1, random_state=self.seed)
        else:
            print("[split_real_samples] treating as list; sorting and shuffling with seed", self.seed)
            samples_list = sorted(samples_list)
            random.seed(self.seed)
            random.shuffle(samples_list)

        p, s = self.bona_partition
        print(f"[split_real_samples] bona_partition={self.bona_partition}, fold_subset={self.fold_subset}")
        idx1 = int(p * len(samples_list))
        idx2 = int((p + s) * len(samples_list))
        subsets = np.split(samples_list, [idx1, idx2])
        sizes = {k: len(v) for k, v in zip(['train', 'test', 'val'], subsets)}
        print(f"[split_real_samples] subset sizes={sizes}")
        return dict(zip(['train', 'test', 'val'], subsets))[self.fold_subset]

    def df2tuples(self):
        print(f"[df2tuples] Converting samples DataFrame with {len(self.samples)} rows to list of tuples")
        tuple_samples = []
        for i, elem in self.samples.iterrows():
            tuple_samples.append((str(elem["path"]), elem["label"], elem["attack_type"]))

        self.samples = tuple_samples
        print(f"[df2tuples] Conversion done, new samples length={len(self.samples)}")
        return self.samples

    def __getitem__(self, index):
        # Show basic info every few items to avoid flooding logs
        if index % 1000 == 0:
            print(f"[__getitem__] Fetching index={index}, samples type={type(self.samples)}")

        if isinstance(self.samples, pd.DataFrame):
            sample = self.samples.iloc[index]
            path = str(sample["path"])
            label = sample["label"]
            attack_type = sample["attack_type"]
        else:
            path, label, attack_type = self.samples[index]

        if index % 1000 == 0:
            print(f"[__getitem__] path={path}, label={label}, attack_type={attack_type}")

        if WAVE_FAKE_INTERFACE:
            # TODO: apply normalization from torchaudio.load
            waveform, sample_rate = torchaudio.load(path, normalize=WAVE_FAKE_NORMALIZE)

            if index % 1000 == 0:
                print(f"[__getitem__] Loaded with torchaudio: waveform shape={waveform.shape}, "
                      f"sample_rate={sample_rate}")

            if sample_rate != WAVE_FAKE_SR:
                print(f"[__getitem__] Resampling from {sample_rate} to {WAVE_FAKE_SR}")
                waveform, sample_rate = AudioDataset.resample(path, WAVE_FAKE_SR, WAVE_FAKE_NORMALIZE)

            if waveform.dim() > 1 and waveform.shape[0] > 1:
                print(f"[__getitem__] Multi-channel waveform, taking first channel. Original shape={waveform.shape}")
                waveform = waveform[:1, ...]

            if WAVE_FAKE_TRIM:
                if index % 1000 == 0:
                    print("[__getitem__] Applying trim")
                waveform, sample_rate = AudioDataset.apply_trim(waveform, sample_rate)

            if WAVE_FAKE_CELL_PHONE:
                if index % 1000 == 0:
                    print("[__getitem__] Applying phone call processing")
                waveform, sample_rate = AudioDataset.process_phone_call(waveform, sample_rate)

            if WAVE_FAKE_PAD:
                if index % 1000 == 0:
                    print(f"[__getitem__] Applying pad/cut to length={WAVE_FAKE_CUT}")
                waveform = PadDataset.apply_pad(waveform, WAVE_FAKE_CUT)

            if self.return_label:
                numeric_label = 1 if label == "bonafide" else 0
                if index % 1000 == 0:
                    print(f"[__getitem__] Returning label mapped '{label}' -> {numeric_label}")
                return waveform, sample_rate, numeric_label
            else:
                return waveform, sample_rate

        # Fallback path if WAVE_FAKE_INTERFACE is False
        data, sr = sf.read(path)
        if index % 1000 == 0:
            print(f"[__getitem__] Loaded with soundfile: data shape={np.shape(data)}, sr={sr}")

        if self.transform:
            if index % 1000 == 0:
                print("[__getitem__] Applying transform")
            data = self.transform(data)

        return data, label, attack_type

    def __len__(self):
        length = len(self.samples)
        print(f"[__len__] Dataset length={length}")
        return length
