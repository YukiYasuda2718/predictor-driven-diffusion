from logging import getLogger

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

logger = getLogger()


class DatasetLorenz96(Dataset):

    def __init__(
        self,
        path_to_dataarray: str,
        means: list[float],
        stds: list[float],
        min_data_idx: int,
        max_data_idx: int,
    ):
        assert 0 <= min_data_idx < max_data_idx

        self.data = xr.load_dataarray(path_to_dataarray)
        assert self.data.dims == ("batch", "channel", "time", "space")

        self.data = self.data.isel(batch=slice(min_data_idx, max_data_idx))
        assert self.data.shape[0] == max_data_idx - min_data_idx

        self.n_batches, self.n_channels, self.n_times, self.n_spaces = self.data.shape
        logger.info(f"Data shape = {self.data.shape}\n")
        assert self.n_channels == 2

        self.mean = np.array(means)
        self.std = np.array(stds)
        assert self.mean.shape == self.std.shape == (self.n_channels,)

        self.dtype = torch.float32

    def __len__(self):
        return self.n_batches

    def standardize(self, data):
        return (data - self.mean[:, None, None]) / self.std[:, None, None]

    def standardize_inversely(self, data):
        return data * self.std[:, None, None] + self.mean[:, None, None]

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self.data[idx].values  # channel x time x space
        standardized = self.standardize(data)
        ret = torch.tensor(standardized, dtype=self.dtype)
        assert ret.shape == (self.n_channels, self.n_times, self.n_spaces)
        return ret
