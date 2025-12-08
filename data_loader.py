# data_loader.py

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from config import SAMPLING_RATE, ELECTRODES


@dataclass
class EEGDataset:
    """
    Tek bir EEG .csv dosyasını temsil eden sınıf.

    - df: Ham verinin DataFrame hali (time, C3, C4, ...)
    - fs: Örnekleme frekansı
    - time: NumPy array olarak zaman vektörü (saniye)
    - available_channels: Bu CSV'de gerçekten var olan kanallar (C3, C4, ...)
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
        CSV dosyasından EEGDataset nesnesi oluşturur.

        :param filepath: CSV dosya yolu
        :param fs: Örnekleme frekansı (None ise config.SAMPLING_RATE kullanılır)
        """

        if fs is None:
            fs = SAMPLING_RATE

        # CSV'yi oku
        df = pd.read_csv(filepath)

        # time kolonu yoksa kendimiz oluşturalım
        if "time" in df.columns:
            time = df["time"].to_numpy()
        else:
            num_samples = len(df)
            time = np.arange(num_samples) / fs
            df.insert(0, "time", time)

        # Mevcut kanalları belirle
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

        # Artık state eklemiyoruz
        return dataset

    def get_segment(self, t_start: float, t_end: float) -> pd.DataFrame:
        """
        Belirli bir zaman aralığındaki veriyi döner.
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
