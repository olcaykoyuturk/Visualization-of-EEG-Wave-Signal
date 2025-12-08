# processing.py

"""
Sinyal işleme fonksiyonları:
- Notch ve band geçiren filtreler
- Dataset'teki kanalların filtrelenmesi
- Frekans band güçlerinin hesaplanması
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from config import FREQUENCY_BANDS
from data_loader import EEGDataset


# ---------- Filtre fonksiyonları ----------

def design_notch_filter(fs: float, freq: float = 50.0, q: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Notch (çentik) filtresi tasarlar.
    Varsayılan olarak 50 Hz için kullanılır.
    """
    # iirnotch dönüşü (b, a) verir
    b, a = signal.iirnotch(w0=freq, Q=q, fs=fs)
    return b, a


def apply_notch_filter(x: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """
    Tek bir kanala 50 Hz notch filtresini uygular.
    """
    b, a = design_notch_filter(fs, freq=freq, q=q)
    y = signal.filtfilt(b, a, x)
    return y


def design_bandpass_filter(fs: float,
                           lowcut: float = 5.0,
                           highcut: float = 35.0,
                           order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Band geçiren Butterworth filtresi tasarlar.
    Varsayılan olarak 5-35 Hz aralığını geçirir.
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
    Tek bir kanala band geçiren filtre uygular.
    """
    b, a = design_bandpass_filter(fs, lowcut=lowcut, highcut=highcut, order=order)
    y = signal.filtfilt(b, a, x)
    return y


# ---------- Dataset seviyesinde filtreleme ----------

def filter_channels(dataset: EEGDataset,
                    channels: Optional[List[str]] = None,
                    apply_notch: bool = True,
                    bandpass: Tuple[float, float] = (5.0, 35.0),
                    order: int = 4) -> pd.DataFrame:
    """
    EEGDataset içindeki belirtilen kanallara filtre uygular.

    :param dataset: EEGDataset nesnesi
    :param channels: Filtrelenecek kanallar listesi.
                     None ise dataset.available_channels kullanılır.
    :param apply_notch: True ise önce 50 Hz notch filtresi uygulanır.
    :param bandpass: (lowcut, highcut) frekans bandı
    :param order: Butterworth filtre derecesi
    :return: Filtrelenmiş sinyalleri içeren DataFrame (time + seçilen kanallar)
    """
    if channels is None:
        channels = dataset.available_channels

    fs = dataset.fs
    df = dataset.df

    # Çıkış DataFrame'i: time + filtrelenmiş kanallar
    filtered_df = pd.DataFrame()
    filtered_df["time"] = df["time"].values

    lowcut, highcut = bandpass

    for ch in channels:
        if ch not in df.columns:
            # Kanal yoksa uyarı vermeden atla (UI tarafında mesajlanabilir)
            continue

        x = df[ch].to_numpy(dtype=float)

        # Önce notch (varsa)
        if apply_notch:
            x = apply_notch_filter(x, fs=fs)

        # Sonra band geçiren filtre
        x = apply_bandpass_filter(x, fs=fs, lowcut=lowcut, highcut=highcut, order=order)

        filtered_df[ch] = x

    return filtered_df


# ---------- Band power hesaplama ----------

def compute_band_powers_for_signal(x: np.ndarray,
                                   fs: float,
                                   bands: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS
                                   ) -> Dict[str, float]:
    """
    Tek bir sinyal için her bir frekans bandındaki gücü hesaplar.

    Welch yöntemini kullanır.
    Dönen sözlük: { "delta": güç, "theta": güç, ... }
    """
    # Welch ile güç spektral yoğunluğu (PSD) hesapla
    f, psd = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))

    band_powers: Dict[str, float] = {}

    for band_name, (fmin, fmax) in bands.items():
        # İlgili frekans aralığındaki indeksleri bul
        idx = np.logical_and(f >= fmin, f <= fmax)
        # Bu aralıktaki PSD toplamını band gücü olarak al
        band_power = np.trapz(psd[idx], f[idx])  # entegral: güç ~ alan
        band_powers[band_name] = float(band_power)

    return band_powers


def compute_band_powers_for_channels(filtered_df: pd.DataFrame,
                                     fs: float,
                                     channels: Optional[List[str]] = None,
                                     bands: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS
                                     ) -> Dict[str, Dict[str, float]]:
    """
    Birden fazla kanal için band power hesaplar.

    :param filtered_df: Filtrelenmiş sinyalleri içeren DataFrame (time + kanallar)
    :param fs: Örnekleme frekansı
    :param channels: Hesaplanacak kanallar listesi (None ise tüm kolonlardan time hariç olanlar)
    :param bands: Frekans band tanımları
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
    Tek bir sinyal için zaman-frekans spektrogramı hesaplar.

    Daha yüksek çözünürlük için:
    - nperseg = 512
    - noverlap = 384 (75% overlap)

    Bu değerler EEG için çok daha düzgün sonuç verir.
    """

    # Eğer sinyal kısa ise segment boyunu düşür
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
