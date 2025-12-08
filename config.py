# config.py

"""
Uygulama genelinde kullanılacak sabit ayarlar.
Burayı merkezi konfigürasyon dosyası gibi düşünebilirsin.
"""

# Varsayılan örnekleme frekansı (Hz)
SAMPLING_RATE = 250.0  # Gerekiyorsa bunu daha sonra gerçek fs'e göre güncelleriz

# Analiz penceresi (saniye cinsinden)
DEFAULT_ANALYSIS_WINDOW = (2.0, 6.0)  # Örn. 2-6 saniye aralığı

# EEG kanalları (senin projene göre bunu genişletebiliriz)
ELECTRODES = [
    "C3",
    "C4",
    "F3",
    "F4",
    "P3",
    "P4",
    "Pz",
    "Cz",
]

# EEG verilerinde kullanılabilecek elektrot yerleşimleri (10-20 sistemi 2D proje)
# C3 – C4 - Cz kullanan basit bir düzende idealdir.
ELECTRODE_POSITIONS = {
    "C3": (-0.5, 0),
    "C4": (0.5, 0),
    "Cz": (0.0, 0.3),
    "F3": (-0.4, 0.6),
    "F4": (0.4, 0.6),
    "P3": (-0.4, -0.5),
    "P4": (0.4, -0.5),
    "POz": (0, -0.7),
}

# Frekans band tanımları (Hz cinsinden)
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha_mu": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# İleride UI'dan da okunabilecek basit metinler
APP_NAME = "EEG Motor Activity Analyzer"
APP_VERSION = "0.1.0"
