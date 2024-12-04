import numpy as np
import soundfile as sf
from pathlib import Path

from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as L

from .data_utils import pad_normal, pad_random

DEFAULT_MAX_LEN = 64600
DEFAULT_SAMPLING_RATE = 16000


class ASVspoof2019LADataset(Dataset):
    def __init__(
        self,
        base_dir,
        protocol_dir,
        pad="random",
        max_len=DEFAULT_MAX_LEN,
        feature=None,
        aug=None,
        config=None,
    ):
        """
        self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)

        example:
        LA_0098 LA_T_9779814 - A06 spoof
        LA_0098 LA_T_9779812 - - bonafide

        """
        self.list_IDs = []
        self.labels = []
        self.base_dir = base_dir
        self.protocol_dir = protocol_dir
        self.config = config
        self.parse_protocol()

        assert pad in ["normal", "random"]
        if pad == "random":
            self.pad = pad_random
        else:
            self.pad = pad_normal

        self.feature = feature
        self.cut = max_len
        self.aug = aug

    def parse_protocol(self):
        with open(self.protocol_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                _, key, _, _, label = line.strip().split(" ")
                self.list_IDs.append(key)

                if label == "bonafide":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = sf.read(str(self.base_dir / f"flac/{key}.flac"))

        # if augmentation function is provided
        if callable(self.aug):
            X = self.aug(X)

        X = Tensor(self.pad(X, self.cut))
        if callable(self.feature):
            X = self.feature(X)
        y = LongTensor([1]) if self.labels[index] == 1 else LongTensor([0])
        y = y.squeeze()
        return X, y, key


class ASVspoof2019LA(L.LightningDataModule):
    def __init__(self, base_dir, max_len=DEFAULT_MAX_LEN, **dataloaderArgs):
        """
        ASVspoof 2019 LA datamodule for training

        """
        super().__init__()
        self.max_len = max_len
        self.base_dir = Path(base_dir)
        self.protocol_dir = Path(base_dir) / "ASVspoof2019_LA_cm_protocols"
        self.train_cm = self.protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
        self.dev_cm = self.protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
        self.eval_cm = self.protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
        self.dev_asv_scores = (
            self.base_dir
            / "ASVspoof2019_LA_asv_scores"
            / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
        )
        self.eval_asv_scores = (
            self.base_dir
            / "ASVspoof2019_LA_asv_scores"
            / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
        )
        self.dataloaderArgs = dataloaderArgs

    def setup(self, stage: str):
        # read ASVSpoof 2019 LA protocol
        dataset_dir = self.base_dir / "ASVspoof2019_LA_train"
        self.train_data = ASVspoof2019LADataset(
            dataset_dir, self.train_cm, aug=True, pad="random", max_len=self.max_len
        )

        dataset_dir = self.base_dir / "ASVspoof2019_LA_dev"
        self.val_data = ASVspoof2019LADataset(
            dataset_dir, self.dev_cm, pad="normal", max_len=self.max_len
        )

        dataset_dir = self.base_dir / "ASVspoof2019_LA_eval"
        self.test_data = ASVspoof2019LADataset(
            dataset_dir, self.eval_cm, pad="normal", max_len=self.max_len
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.dataloaderArgs)

    def val_dataloader(self):
        # disable shuffle for validation, otherwise EER calculation will be wrong!
        return DataLoader(self.val_data, shuffle=False, **self.dataloaderArgs)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.dataloaderArgs)


class ASVspoofEvalDataset(Dataset):
    def __init__(self, base_dir, protocol_dir, max_len=DEFAULT_MAX_LEN):
        """
        ASVspoof 2019LA / 2021LA / 2021DF datamodule for evaluation
        """
        self.base_dir = base_dir
        self.protocol_dir = protocol_dir
        self.list_IDs = []
        self.pad = pad_normal
        self.cut = max_len
        self.parse_protocol()

    def parse_protocol(self):
        with open(self.protocol_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                _, key, *_ = line.strip().split(" ")
                self.list_IDs.append(key)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X = Tensor(self.pad(X, self.cut))
        return X, key


class ASVspoofEval(L.LightningDataModule):
    def __init__(
        self, base_dir, protocol_dir, max_len=DEFAULT_MAX_LEN, **dataloaderArgs
    ):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.protocol_dir = Path(protocol_dir)
        self.max_len = max_len
        self.dataloaderArgs = dataloaderArgs

    def setup(self, stage: str):
        self.testset = ASVspoofEvalDataset(self.base_dir, self.protocol_dir)

    def test_dataloader(self):
        return DataLoader(self.testset, shuffle=False, **self.dataloaderArgs)


class IntheWildDataset(Dataset):
    def __init__(self, base_dir, protocol_dir, max_len=DEFAULT_MAX_LEN):
        """
        In-the-Wild datamodule for evaluation
        """
        self.base_dir = base_dir
        self.protocol_dir = protocol_dir
        self.list_IDs = []
        self.labels = []
        self.pad = pad_normal
        self.cut = max_len
        self.parse_protocol()

    def parse_protocol(self):
        with open(self.protocol_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                _, key, _, _, label = line.strip().split(" ")
                self.list_IDs.append(key)
                if label == "bonafide":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = sf.read(str(self.base_dir / f"{key}.wav"))
        X = Tensor(self.pad(X, self.cut))
        return X, key


class IntheWild(L.LightningDataModule):
    def __init__(
        self, base_dir, protocol_dir, max_len=DEFAULT_MAX_LEN, **dataloaderArgs
    ):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.protocol_dir = Path(protocol_dir)
        self.max_len = max_len
        self.dataloaderArgs = dataloaderArgs

    def setup(self, stage: str):
        self.testset = IntheWildDataset(self.base_dir, self.protocol_dir)

    def test_dataloader(self):
        return DataLoader(self.testset, shuffle=False, **self.dataloaderArgs)
