# config.py
#
# This file is used to collect common settings used throughout the application.
# Keeping parameters here makes the project easier to manage.

# Sampling frequency (Hz)
# EEG recordings typically use 250–500 Hz, so this value works as a default.
SAMPLING_RATE = 250.0

# Time window used during analysis (in seconds)
# For example, to inspect the interval between 2–6 seconds.
DEFAULT_ANALYSIS_WINDOW = (2.0, 6.0)

# EEG electrode names
# A few channels from the 10–20 system are enough for this project.
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

# 2D positions of each electrode (for visualization in the UI)
# These coordinates represent a simplified layout.
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

# Classical EEG frequency bands (in Hz)
# Each band represents a different type of brain activity.
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha_mu": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Application information (for display in the UI)
APP_NAME = "EEG Motor Activity Analyzer"
APP_VERSION = "0.1.0"
