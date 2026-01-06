# gemini_api.py
"""
Gemini API integration for EEG signal analysis.
Sends filtered EEG data (P3, P4, C3, C4 channels) to Google's Gemini API
and returns the interpretation.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiEEGAnalyzer:
    """
    Handles EEG data analysis using Google's Gemini API.
    """
    
    # Target channels for motor cortex analysis
    TARGET_CHANNELS = ["P3", "P4", "C3", "C4"]
    
    # Available models (in order of preference)
    AVAILABLE_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini EEG Analyzer.
        
        :param api_key: Gemini API key. If None, tries to get from environment variable.
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Please install it with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = None
        
    def configure(self, api_key: str, model_name: str = "gemini-2.0-flash") -> None:
        """
        Configure the API with the provided key.
        
        :param api_key: Gemini API key
        :param model_name: Model to use (default: gemini-1.5-flash)
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
        
    def is_configured(self) -> bool:
        """Check if the API is properly configured."""
        return self._model is not None and self.api_key is not None
    
    def format_eeg_data(
        self,
        df: pd.DataFrame,
        channels: List[str],
        fs: float,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        band_powers: Optional[Dict[str, Dict[str, float]]] = None
    ) -> str:
        """
        Format EEG data into a compact structured text for Gemini API.
        Only sends statistics and band powers, no raw sample data.
        """
        # Filter time range if specified
        if t_start is not None and t_end is not None:
            if "time" in df.columns:
                mask = (df["time"] >= t_start) & (df["time"] <= t_end)
                df = df.loc[mask].copy()
        
        # Build compact formatted text
        lines = []
        lines.append("EEG Data Summary")
        lines.append(f"Fs:{fs}Hz, Channels:{','.join(channels)}")
        
        if "time" in df.columns and len(df) > 0:
            duration = df["time"].iloc[-1] - df["time"].iloc[0]
            lines.append(f"Duration:{duration:.1f}s, Samples:{len(df)}")
        
        # Compact channel statistics
        lines.append("\nStats(Mean/Std/Range):")
        for ch in channels:
            if ch in df.columns:
                data = df[ch].to_numpy()
                lines.append(f"{ch}: {np.mean(data):.2f}/{np.std(data):.2f}/{np.ptp(data):.2f}")
        
        # Compact band power information
        if band_powers:
            lines.append("\nBand Powers:")
            for ch, powers in band_powers.items():
                if ch in channels:
                    power_str = ", ".join([f"{b}:{p:.4f}" for b, p in powers.items()])
                    lines.append(f"{ch}: {power_str}")
        
        return "\n".join(lines)
    
    def analyze(
        self,
        formatted_data: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Send EEG data to Gemini API for analysis.
        
        :param formatted_data: Formatted EEG data string
        :param custom_prompt: Optional custom prompt to use
        :return: Gemini's analysis response
        """
        if not self.is_configured():
            raise ValueError("API is not configured. Call configure() first.")
        
        # Default prompt for EEG analysis
        default_prompt = """
You are an EEG signal analysis expert. Below is motor cortex EEG data from P3, P4, C3, and C4 electrodes.

Electrode positions:
- C3/C4: Central motor cortex (hand/arm movements). C3=right hand, C4=left hand.
- P3/P4: Parietal sensory-motor integration. P3=right side, P4=left side.

IMPORTANT INSTRUCTIONS:
1. Based on the asymmetry between left (C3, P3) and right (C4, P4) hemisphere signals, determine which limb (left hand, right hand, or both) the patient likely moved or imagined moving.
2. Write your analysis as flowing paragraphs. DO NOT use headings, bullet points, numbered lists, or markdown formatting.
3. Include: signal quality assessment, hemisphere comparison, frequency band analysis, motor activity inference, and which specific limb was likely active.

Data:
"""
        
        prompt = custom_prompt if custom_prompt else default_prompt
        full_prompt = prompt + "\n\n" + formatted_data
        
        try:
            response = self._model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"API Error: {str(e)}"
    
    def analyze_eeg(
        self,
        df: pd.DataFrame,
        channels: List[str],
        fs: float,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        band_powers: Optional[Dict[str, Dict[str, float]]] = None,
        custom_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Complete EEG analysis pipeline.
        
        :param df: DataFrame with EEG data
        :param channels: Channels to analyze
        :param fs: Sampling frequency
        :param t_start: Start time (optional)
        :param t_end: End time (optional)
        :param band_powers: Band powers (optional)
        :param custom_prompt: Custom prompt (optional)
        :return: Tuple of (formatted_data, analysis_result)
        """
        formatted_data = self.format_eeg_data(
            df, channels, fs, t_start, t_end, band_powers
        )
        analysis = self.analyze(formatted_data, custom_prompt)
        return formatted_data, analysis


def check_gemini_available() -> Tuple[bool, str]:
    """
    Check if Gemini API is available.
    
    :return: Tuple of (is_available, message)
    """
    if not GENAI_AVAILABLE:
        return False, "google-generativeai paketi yüklü değil. Lütfen 'pip install google-generativeai' komutunu çalıştırın."
    return True, "Gemini API kullanıma hazır."
