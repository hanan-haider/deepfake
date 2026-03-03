from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset

ASVSPOOF_KFOLD_SPLIT = {
    0: {
        "train": ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A19'],
        "test":  ['A05', 'A15', 'A16'],
        "val":   ['A06', 'A17', 'A18'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 42
    },
    1: {
        "train": ['A01', 'A02', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A15', 'A16', 'A17', 'A18', 'A19'],
        "test":  ['A03', 'A11', 'A12'],
        "val":   ['A04', 'A13', 'A14'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 43
    },
    2: {
        "train": ['A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
        "test":  ['A01', 'A07', 'A08'],
        "val":   ['A02', 'A09', 'A10'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 44
    }
}


class ASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_folder_name = "ASVspoof2019_LA_cm_protocols"
    subset_dir_prefix = "ASVspoof2019_LA_"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        print(f"[ASVSpoofDataset.__init__] path={path}, fold_num={fold_num}, fold_subset={fold_subset}")
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        split_cfg = ASVSPOOF_KFOLD_SPLIT[fold_num]
        self.allowed_attacks = split_cfg[fold_subset]
        self.bona_partition = split_cfg["bonafide_partition"]
        self.seed = split_cfg["seed"]

        print(f"[ASVSpoofDataset.__init__] allowed_attacks={self.allowed_attacks}")
        print(f"[ASVSpoofDataset.__init__] bona_partition={self.bona_partition}, seed={self.seed}")

        self.samples = pd.DataFrame()

        for subset in self.subsets:
            subset_dir = Path(self.path) / f"{self.subset_dir_prefix}{subset}"
            subset_protocol_path = self.get_protocol_path(subset)
            print(f"[ASVSpoofDataset.__init__] subset={subset}, subset_dir={subset_dir}, "
                  f"protocol_path={subset_protocol_path}")

            subset_samples = self.read_protocol(subset_dir, subset_protocol_path)
            print(f"[ASVSpoofDataset.__init__] subset={subset}, "
                  f"loaded {len(subset_samples)} samples")

            self.samples = pd.concat([self.samples, subset_samples], ignore_index=True)

        print(f"[ASVSpoofDataset.__init__] total samples after all subsets: {len(self.samples)}")
        # self.samples, self.attack_signatures = self.group_by_attack()
        self.transform = transform

    def get_protocol_path(self, subset):
        proto_dir = Path(self.path) / self.protocol_folder_name
        paths = list(proto_dir.glob("*.txt"))
        print(f"[get_protocol_path] Looking for subset='{subset}' in protocol dir={proto_dir}, "
              f"found {len(paths)} txt files")

        for path in paths:
            if subset in Path(path).stem:
                print(f"[get_protocol_path] Using protocol file: {path}")
                return path

        print(f"[get_protocol_path] WARNING: no protocol file found for subset='{subset}'")
        return None

    def read_protocol(self, subset_dir, protocol_path):
        print(f"[read_protocol] subset_dir={subset_dir}, protocol_path={protocol_path}")
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []

        if protocol_path is None:
            print("[read_protocol] ERROR: protocol_path is None, returning empty DataFrame")
            return pd.DataFrame(samples)

        with open(protocol_path, "r") as file:
            for line in file:
                attack_type = line.strip().split(" ")[3]

                if attack_type == "-":
                    real_samples.append(line)
                elif attack_type in self.allowed_attacks:
                    fake_samples.append(line)

                if attack_type not in self.allowed_attacks and attack_type != "-":
                    continue

        print(f"[read_protocol] fold_subset={self.fold_subset}, "
              f"collected real_samples={len(real_samples)}, fake_samples={len(fake_samples)} "
              f"before splitting")

        # Add spoof samples first
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        # Split real samples into train/test/val according to config
        real_samples = self.split_real_samples(real_samples)
        print(f"[read_protocol] real_samples after split for subset='{self.fold_subset}': "
              f"{len(real_samples)}")

        for line in real_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        df = pd.DataFrame(samples)
        print(f"[read_protocol] Returning DataFrame with {len(df)} rows for subset_dir={subset_dir}")
        return df

    @staticmethod
    def add_line_to_samples(samples, line, subset_dir):
        user_id, sample_name, _, attack_type, label = line.strip().split(" ")
        samples["user_id"].append(user_id)
        samples["sample_name"].append(sample_name)
        samples["attack_type"].append(attack_type)
        samples["label"].append(label)

        audio_path = subset_dir / "flac" / f"{sample_name}.flac"
        assert audio_path.exists(), f"[add_line_to_samples] Missing audio file: {audio_path}"
        samples["path"].append(audio_path)

        return samples


if __name__ == "__main__":
    ASVSPOOF_DATASET_PATH = ""  # set this to your ASVspoof base path

    dataset = ASVSpoofDataset(ASVSPOOF_DATASET_PATH, fold_num=1, fold_subset='test')
    print("[__main__] Unique attack types:", dataset.samples["attack_type"].unique())
    print("[__main__] Head of samples DataFrame:")
    print(dataset.samples.head())
