import glob
import os
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset

logger = getLogger()


class DatasetKolmogorovFlow(Dataset):
    def __init__(
        self,
        data_dir_path: str,
        min_data_idx: int,
        max_data_idx: int,
        nt: int,
        ny: int,
        nx: int,
        mean: float,
        std: float,
        dtype: torch.dtype = torch.float32,
    ):

        lst = []
        for p in glob.glob(f"{data_dir_path}/*.npy"):
            name = os.path.basename(p).replace(".npy", "")
            idx = int(name.split("_")[-1])

            if min_data_idx <= idx <= max_data_idx:
                # add channel dim
                d = torch.from_numpy(np.load(p)).to(dtype)[:, None]
                # b, c, t, x, y -> b, c, t, y, x
                lst.append(d.permute(0, 1, 2, 4, 3).contiguous())

        self.all_data = torch.cat(lst, dim=0)
        logger.info(f"DatasetKolmogorovFlow: {self.all_data.shape=}\n")

        self.n_batches, self.n_channels, self.nt, self.ny, self.nx = self.all_data.shape
        assert self.n_channels == 1
        assert self.nt == nt and self.ny == ny and self.nx == nx

        self.mean = mean
        self.std = std

    def __len__(self):
        return self.n_batches

    def standardize(self, data):
        return (data - self.mean) / self.std

    def standardize_inversely(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self.all_data[idx]  # c, t, y, x dims
        standardized = self.standardize(data)
        return standardized.contiguous()
