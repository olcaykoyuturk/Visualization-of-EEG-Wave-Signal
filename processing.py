# processing.py

"""
Signal processing functions:
- Notch and band-pass filters
- Filtering channels inside the dataset
- Computing frequency band powers
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from config import FREQUENCY_BANDS
from data_loader import EEGDataset


# ---------- Filter functions ----------

def design_notch_filter(fs: float, freq: float = 50.0, q: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Designs a notch filter.
    By default, it is used for 50 Hz.
    """
    # iirnotch returns (b, a)
    b, a = signal.iirnotch(w0=freq, Q=q, fs=fs)
    return b, a


def apply_notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """
    Applies a 50 Hz notch filter to a single channel.
    """
    b, a = design_notch_filter(fs, freq=freq, q=q)
    y = signal.filtfilt(b, a, x)
    return y


def design_bandpass_filter(fs: float,
                           lowcut: float = 5.0,
                           highcut: float = 35.0,
                           order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Designs a Butterworth band-pass filter.
    By default, it passes frequencies between 5–35 Hz.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(x: np.ndarray,
                          fs: float,
                          lowcut: float = 5.0,
                          highcut: float = 35.0,
                          order: int = 4) -> np.ndarray:
    """
    Applies a band-pass filter to a single channel.
    """
    b, a = design_bandpass_filter(fs, lowcut=lowcut, highcut=highcut, order=order)
    y = signal.filtfilt(b, a, x)
    return y


# ---------- Dataset-level filtering ----------

def filter_channels(dataset: EEGDataset,
                    channels: Optional[List[str]] = None,
                    apply_notch: bool = True,
                    bandpass: Tuple[float, float] = (5.0, 35.0),
                    order: int = 4) -> pd.DataFrame:
    """
    Applies filters to the selected channels inside an EEGDataset.

    :param dataset: EEGDataset object
    :param channels: List of channels to filter.
                     If None, dataset.available_channels is used.
    :param apply_notch: If True, 50 Hz notch filter is applied first.
    :param bandpass: (lowcut, highcut) frequency range
    :param order: Butterworth filter order
    :return: DataFrame containing filtered signals (time + selected channels)
    """
    if channels is None:
        channels = dataset.available_channels

    fs = dataset.fs
    df = dataset.df

    # Output DataFrame: time + filtered channels
    filtered_df = pd.DataFrame()
    filtered_df["time"] = df["time"].values

    lowcut, highcut = bandpass

    for ch in channels:
        if ch not in df.columns:
            # If the channel does not exist, skip quietly (UI can show a message)
            continue

        x = df[ch].to_numpy(dtype=float)

        # Apply notch filter first (if enabled)
        if apply_notch:
            x = apply_notch_filter(x, fs=fs)

        # Then apply band-pass filter
        x = apply_bandpass_filter(x, fs=fs, lowcut=lowcut, highcut=highcut, order=order)

        filtered_df[ch] = x

    return filtered_df


# ---------- Band power calculations ----------

def compute_band_powers_for_signal(x: np.ndarray,
                                   fs: float,
                                   bands: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS
                                   ) -> Dict[str, float]:
    """
    Computes the power of each frequency band for a single signal.

    Uses the Welch method.
    Returns a dictionary: { "delta": power, "theta": power, ... }
    """
    # Compute PSD using Welch
    f, psd = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))

    band_powers: Dict[str, float] = {}

    for band_name, (fmin, fmax) in bands.items():
        # Find the indices corresponding to the band
        idx = np.logical_and(f >= fmin, f <= fmax)
        # Integrate PSD within the band to obtain power
        band_power = np.trapezoid(psd[idx], f[idx])  # integral = area = band power
        band_powers[band_name] = float(band_power)

    return band_powers


def compute_band_powers_for_channels(filtered_df: pd.DataFrame,
                                     fs: float,
                                     channels: Optional[List[str]] = None,
                                     bands: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS
                                     ) -> Dict[str, Dict[str, float]]:
    """
    Computes band powers for multiple channels.

    :param filtered_df: DataFrame containing filtered signals (time + channels)
    :param fs: Sampling frequency
    :param channels: Channels to analyze (None → all except 'time')
    :param bands: Frequency band definitions
    :return: {"C3": {"delta": .., "theta": .., ...}, "C4": {...}, ...}
    """
    if channels is None:
        channels = [col for col in filtered_df.columns if col != "time"]

    results: Dict[str, Dict[str, float]] = {}

    for ch in channels:
        if ch not in filtered_df.columns:
            continue

        x = filtered_df[ch].to_numpy(dtype=float)
        band_powers = compute_band_powers_for_signal(x, fs=fs, bands=bands)
        results[ch] = band_powers

    return results


# ---------- Spectrogram ----------

def compute_spectrogram(
    x: np.ndarray,
    fs: float,
    nperseg: int = 512,
    noverlap: int = 384,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the time-frequency spectrogram of a single signal.

    For higher resolution:
    - nperseg = 512
    - noverlap = 384 (75% overlap)

    These settings produce smoother results for EEG.
    """

    # If the signal is shorter than nperseg, reduce segment size
    if len(x) < nperseg:
        nperseg = max(64, len(x) // 2)
        noverlap = nperseg // 2

    f, t, Sxx = signal.spectrogram(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    return f, t, Sxx
