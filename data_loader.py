# data_loader.py

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from config import SAMPLING_RATE, ELECTRODES


@dataclass
class EEGDataset:
    """
    Represents a single EEG .csv file.

    - df: DataFrame version of the raw data (time, C3, C4, ...)
    - fs: Sampling frequency
    - time: Time vector as a NumPy array (seconds)
    - available_channels: Channels that actually exist in this CSV (C3, C4, ...)
    """
    filepath: str
    df: pd.DataFrame
    fs: float
    time: np.ndarray
    available_channels: List[str]

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        fs: Optional[float] = None,
    ) -> "EEGDataset":
        """
        Creates an EEGDataset object from a CSV file.

        :param filepath: Path to the CSV file
        :param fs: Sampling frequency (if None, uses config.SAMPLING_RATE)
        """

        if fs is None:
            fs = SAMPLING_RATE

        # Read the CSV file
        df = pd.read_csv(filepath)

        # If "time" column is missing, generate it manually
        if "time" in df.columns:
            time = df["time"].to_numpy()
        else:
            num_samples = len(df)
            time = np.arange(num_samples) / fs
            df.insert(0, "time", time)

        # Detect available channels
        available_channels = [
            ch for ch in ELECTRODES if ch in df.columns
        ]

        dataset = cls(
            filepath=filepath,
            df=df,
            fs=fs,
            time=time,
            available_channels=available_channels,
        )

        # We no longer store additional state data
        return dataset

    def get_segment(self, t_start: float, t_end: float) -> pd.DataFrame:
        """
        Returns the data within a specific time interval.
        """
        mask = (self.time >= t_start) & (self.time <= t_end)
        return self.df.loc[mask].copy()

    def __repr__(self) -> str:
        duration = self.time[-1] - self.time[0] if len(self.time) > 0 else 0
        return (
            f"EEGDataset(filepath='{self.filepath}', "
            f"fs={self.fs}, "
            f"duration={duration:.2f}s, "
            f"channels={self.available_channels})"
        )
