
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from scipy import signal

# Hardcoded phases (as per methodology) for visualization context
TRIAL_PHASES = [
    {"name": "Rest", "start": 0.0, "end": 2.0, "color": "#1a1a2e"},        # Dark blue
    {"name": "Movement", "start": 2.0, "end": 4.0, "color": "#16213e"},   # Darker blue
    {"name": "Recovery", "start": 4.0, "end": 7.0, "color": "#1a1a2e"},   # Dark blue
    {"name": "Extra", "start": 7.0, "end": 8.0, "color": "#0f0f1a"},      # Darkest
]

ELECTRODE_LABELS = {
    "C3": "C3 (Left)",
    "C4": "C4 (Right)",
    "P3": "P3 (Left)",
    "P4": "P4 (Right)",
}

class RealTimeVizWidget(QWidget):
    """
    PyQt Widget for Real-time EEG Brain Wave Visualization.
    Adapts logic from realtime_brainwaves.py to be embeddable in the main app.
    
    Usage:
        widget = RealTimeVizWidget()
        layout.addWidget(widget)
        widget.set_dataset(dataset) # Pass loaded EEG properties
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.sampling_rate = 250.0
        self.window_size = 2.0  # seconds
        self.update_interval = 40 # ms
        self.electrodes = ["C3", "C4", "P3", "P4"] # Default, can be updated
        
        # State
        self.dataset = None
        self.is_playing = False
        self.current_time = 0.0
        self.duration = 8.0
        self.animation = None
        self.lines = []
        self.time_markers = []
        
        self._setup_ui()
        self._setup_plot()
        
    def _setup_ui(self):
        """Setup the widget UI layout."""
        layout = QVBoxLayout(self)
        
        # Controls Layout
        controls_layout = QHBoxLayout()
        
        self.status_label = QLabel("Waiting for data...")
        self.status_label.setStyleSheet("font-weight: bold; color: gray;")
        
        self.btn_play = QPushButton("PLAY")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        
        self.btn_restart = QPushButton("RESTART")
        self.btn_restart.clicked.connect(self.restart)
        self.btn_restart.setEnabled(False)
        
        self.time_label = QLabel("0.0s / 0.0s")
        
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_restart)
        controls_layout.addWidget(self.time_label)
        
        layout.addLayout(controls_layout)
        
        # Plot Canvas
        # Use dark background style
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor('#0f0f1a')
        
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
    def _setup_plot(self):
        """Initialize matplotlib figure structure."""
        if not self.electrodes:
            return
            
        self.fig.clear()
        
        # Create subplots
        num_channels = len(self.electrodes)
        self.axes = self.fig.subplots(num_channels, 1, sharex=True)
        if num_channels == 1:
            self.axes = [self.axes]
            
        self.fig.subplots_adjust(hspace=0.15, left=0.10, right=0.96, top=0.95, bottom=0.08)
        
        # Colors
        colors = ["#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
        
        self.lines = []
        self.time_markers = []
        
        for i, (ax, electrode) in enumerate(zip(self.axes, self.electrodes)):
            color = colors[i % len(colors)]
            
            # Initialize empty line
            line, = ax.plot([], [], color=color, linewidth=1.5)
            self.lines.append(line)
            
            # Time marker (vertical line)
            marker = ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
            self.time_markers.append(marker)
            
            # Styling
            label = ELECTRODE_LABELS.get(electrode, electrode)
            ax.set_ylabel(label, fontsize=10, fontweight='bold', color=color, rotation=0, labelpad=40)
            ax.set_xlim(0, self.window_size)
            ax.set_ylim(-3.5, 3.5) # Normalized scale
            ax.set_yticks([-3.0, 0, 3.0])
            ax.grid(True, alpha=0.2)
            
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # Hide x tick labels for all except last
            if i < num_channels - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)", color='white')
                
        # Draw initial state
        self.canvas.draw()
        
    def set_dataset(self, dataset):
        """
        Receive dataset from Main Window.
        dataset object needs to have: fs (float), time (np array), df (pandas DataFrame)
        """
        if dataset is None:
            return
            
        self.dataset = dataset
        self.sampling_rate = dataset.fs
        
        # Filter available channels to prioritized ones or take first 4
        available = dataset.available_channels
        prioritized = ["C3", "C4", "P3", "P4"]
        self.electrodes = [ch for ch in prioritized if ch in available]
        if not self.electrodes:
            self.electrodes = available[:4]
            
        # Prepare Data
        self._prepare_data()
        
        # Reset UI
        self._setup_plot()
        self.duration = self.time_vector[-1]
        self.current_time = 0.0
        self.update_time_label()
        
        # Enable controls
        self.btn_play.setEnabled(True)
        self.btn_restart.setEnabled(True)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #4ECDC4;")
        
        # Setup Animation
        # We store animation to prevent garbage collection
        if self.animation:
            self.animation.event_source.stop()
            
        self.animation = FuncAnimation(
            self.fig, self.update_frame, 
            init_func=self.init_animation,
            interval=self.update_interval, 
            blit=False, 
            cache_frame_data=False
        )
        
    def _prepare_data(self):
        """Filter and normalize data similar to standalone script."""
        self.data_map = {}
        
        # Use entire dataset time vector
        self.time_vector = self.dataset.time
        
        for electrode in self.electrodes:
            if electrode in self.dataset.df:
                raw = self.dataset.df[electrode].to_numpy(dtype=float)
                
                # Apply light filtering (0.5 Hz Highpass only/mostly to remove DC)
                nyq = 0.5 * self.sampling_rate
                b_hp, a_hp = signal.butter(2, 0.5/nyq, btype='high')
                filtered = signal.filtfilt(b_hp, a_hp, raw)
                
                # Normalize
                # Skip first second for settling
                skip = int(self.sampling_rate * 1.0)
                if len(filtered) > skip:
                    stable = filtered[skip:]
                    mean_val = np.mean(stable)
                    std_val = np.std(stable)
                    
                    # Z-score normalization
                    if std_val > 0:
                        normalized = (filtered - mean_val) / std_val
                    else:
                        normalized = filtered - mean_val
                    self.data_map[electrode] = normalized
                else:
                    self.data_map[electrode] = filtered # Fallback
        
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("PAUSE")
            self.status_label.setText("Playing...")
        else:
            self.btn_play.setText("PLAY")
            self.status_label.setText("Paused")
            
    def restart(self):
        self.current_time = 0.0
        self.is_playing = True
        self.btn_play.setText("PAUSE")
        self.status_label.setText("Playing...")
        self.update_frame(0) # Force update
            
    def init_animation(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines + self.time_markers

    def update_frame(self, frame):
        if not self.is_playing or not self.dataset:
            return self.lines + self.time_markers
            
        # Advance time
        self.current_time += self.update_interval / 1000.0
        if self.current_time >= self.duration:
            self.current_time = self.duration
            self.is_playing = False
            self.btn_play.setText("PLAY")
            self.status_label.setText("Finished")
            
        self.update_time_label()
            
        # Calculate window
        window_end = self.current_time
        window_start = max(0, window_end - self.window_size)
        
        # Get mask
        mask = (self.time_vector >= window_start) & (self.time_vector <= window_end)
        time_window = self.time_vector[mask] - window_start 
        
        for i, electrode in enumerate(self.electrodes):
            if electrode in self.data_map:
                data_window = self.data_map[electrode][mask]
                
                # Check lengths match
                min_len = min(len(time_window), len(data_window))
                if min_len > 0:
                    self.lines[i].set_data(time_window[:min_len], data_window[:min_len])
                    
                    # Auto-scale Y-axis
                    current_min = np.min(data_window[:min_len])
                    current_max = np.max(data_window[:min_len])
                    
                    # Expand range slightly for padding
                    range_span = current_max - current_min
                    if range_span < 1.0: # Enforce minimum range to avoid zooming in on noise
                        mid = (current_max + current_min) / 2.0
                        ymin = mid - 1.5
                        ymax = mid + 1.5
                    else:
                        margin = range_span * 0.1 # 10% margin
                        ymin = current_min - margin
                        ymax = current_max + margin
                        
                    # Smoothly update limits (optional, but direct setting is fine for now)
                    # We only expand range if data goes out, or shrink slowly? 
                    # For simplicity, just follow the data window with some checks
                    
                    # Get current limits
                    # old_ymin, old_ymax = self.axes[i].get_ylim()
                    
                    # Instead of jumping every frame, we can just set it
                    self.axes[i].set_ylim(ymin, ymax)
                    # Use fixed number of ticks
                    ticks = [ymin, (ymin + ymax) / 2.0, ymax]
                    self.axes[i].set_yticks(ticks)
                    self.axes[i].set_yticklabels([f"{t:.1f}" for t in ticks])

                
                # Time marker moves and stays at end
                marker_pos = min(window_end - window_start, self.window_size)
                self.time_markers[i].set_xdata([marker_pos, marker_pos])
                
        return self.lines + self.time_markers

    def update_time_label(self):
        self.time_label.setText(f"{self.current_time:.1f}s / {self.duration:.1f}s")

