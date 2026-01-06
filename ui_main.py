from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QAction,
    QFileDialog,
    QTableView,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
    QHBoxLayout,
    QCheckBox,
    QDoubleSpinBox,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QSpinBox,
    QComboBox,
    QPushButton,
    QTextEdit,
    QScrollArea,
    QGroupBox,
    QLineEdit,
    QProgressBar,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.interpolate import griddata
from scipy.signal import welch

# Assuming config, data_loader, and processing modules exist and are available
from config import APP_NAME, APP_VERSION, ELECTRODE_POSITIONS
from data_loader import EEGDataset
from processing import (
    filter_channels,
    compute_band_powers_for_channels,
    compute_spectrogram,
)
from ui_realtime import RealTimeVizWidget

# Gemini API integration
try:
    from gemini_api import GeminiEEGAnalyzer, check_gemini_available
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ---------------------------------------------------------------------
# DataFrame -> Qt Table Model
# ---------------------------------------------------------------------
class PandasModel(QAbstractTableModel):
    """Simple model to display a Pandas DataFrame in a QTableView."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=QModelIndex()) -> int:
        # Return 0 if the index is valid (i.e., this is a sub-model) or the number of rows otherwise
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        # Return 0 if the index is valid (i.e., this is a sub-model) or the number of columns otherwise
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        # Display role is used to populate the table cells
        if role == Qt.DisplayRole:
            # Retrieve data from the DataFrame using integer-location indexing (.iat)
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        # Horizontal header displays column names
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        # Vertical header displays row indices
        else:
            return str(section)


# ---------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------
class MainWindow(QMainWindow):
    """
    Tabbed EEG analysis interface:
      - Raw Data
      - Filtered Signals
      - Band Power
      - Spectrograms
      - Band Timeline (Mu)
      - ERD/ERS (manual + trial-based)
      - Topographic Map (raw or band-selected)
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1300, 750)

        # Dataset state variables
        self.dataset: Optional[EEGDataset] = None
        self.selected_row_index: Optional[int] = None
        self.selected_time: Optional[float] = None

        # Filter settings (used for "Filtered Signals", "Spectrograms", "Mu Timeline")
        self.filter_lowcut: float = 5.0
        self.filter_highcut: float = 35.0
        self.filter_notch_enabled: bool = True
        self.filter_order: int = 4

        # ---- Main tab widget ----
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # -------- Raw Data tab setup --------
        self.raw_tab = QWidget()
        self.raw_layout = QVBoxLayout()

        raw_header = QHBoxLayout()
        raw_header_label = QLabel("Raw EEG Data (CSV table)")
        self.raw_info_btn = QPushButton("Info")
        self.raw_info_btn.clicked.connect(self._show_raw_info)
        raw_header.addWidget(raw_header_label)
        raw_header.addStretch()
        raw_header.addWidget(self.raw_info_btn)
        self.raw_layout.addLayout(raw_header)

        self.raw_table = QTableView()
        self.raw_layout.addWidget(self.raw_table)
        self.raw_selected_label = QLabel("No row selected.")
        self.raw_layout.addWidget(self.raw_selected_label)
        self.raw_tab.setLayout(self.raw_layout)
        self.tabs.addTab(self.raw_tab, "Raw Data")

        # -------- Filtered Signals tab setup --------
        self.filtered_tab = QWidget()
        self.filtered_layout = QVBoxLayout()

        filt_header = QHBoxLayout()
        filt_header_label = QLabel("Filtered Signals (notch + band-pass)")
        self.filtered_info_btn = QPushButton("Info")
        self.filtered_info_btn.clicked.connect(self._show_filtered_info)
        filt_header.addWidget(filt_header_label)
        filt_header.addStretch()
        filt_header.addWidget(self.filtered_info_btn)
        self.filtered_layout.addLayout(filt_header)

        # Matplotlib canvas for plotting filtered signals
        self.filtered_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.filtered_layout.addWidget(self.filtered_canvas)
        self.filtered_tab.setLayout(self.filtered_layout)
        self.tabs.addTab(self.filtered_tab, "Filtered Signals")

        # -------- Band Power tab setup --------
        self.bandpower_tab = QWidget()
        self.bandpower_layout = QVBoxLayout()

        bp_ctrl_layout = QHBoxLayout()
        # Checkbox to use a short window around the selected time point
        self.bandpower_use_window = QCheckBox("Use selected row window")
        self.bandpower_use_window.setChecked(False)

        # Spinbox for defining the window length in seconds
        self.bandpower_window_spin = QDoubleSpinBox()
        self.bandpower_window_spin.setRange(0.1, 10.0)
        self.bandpower_window_spin.setSingleStep(0.1)
        self.bandpower_window_spin.setValue(2.0)

        bp_ctrl_layout.addWidget(self.bandpower_use_window)
        bp_ctrl_layout.addWidget(QLabel("Window (s):"))
        bp_ctrl_layout.addWidget(self.bandpower_window_spin)

        self.bandpower_info_btn = QPushButton("Info")
        self.bandpower_info_btn.clicked.connect(self._show_bandpower_info)
        bp_ctrl_layout.addWidget(self.bandpower_info_btn)

        bp_ctrl_layout.addStretch()
        self.bandpower_layout.addLayout(bp_ctrl_layout)

        # Matplotlib canvas for plotting band power bar charts
        self.bandpower_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.bandpower_layout.addWidget(self.bandpower_canvas)
        self.bandpower_tab.setLayout(self.bandpower_layout)
        self.tabs.addTab(self.bandpower_tab, "Band Power")

        # -------- Spectrograms tab setup --------
        self.spectrogram_tab = QWidget()
        self.spectrogram_layout = QVBoxLayout()

        spec_header = QHBoxLayout()
        spec_header_label = QLabel("Spectrograms (time–frequency)")
        self.spectrogram_info_btn = QPushButton("Info")
        self.spectrogram_info_btn.clicked.connect(self._show_spectrogram_info)
        spec_header.addWidget(spec_header_label)
        spec_header.addStretch()
        spec_header.addWidget(self.spectrogram_info_btn)
        self.spectrogram_layout.addLayout(spec_header)

        # Matplotlib canvas for plotting spectrograms
        self.spectrogram_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.spectrogram_layout.addWidget(self.spectrogram_canvas)
        self.spectrogram_tab.setLayout(self.spectrogram_layout)
        self.tabs.addTab(self.spectrogram_tab, "Spectrograms")

        # -------- Band Timeline (Mu) tab setup --------
        self.timeline_tab = QWidget()
        self.timeline_layout = QVBoxLayout()

        tl_ctrl = QHBoxLayout()
        # Window length control for sliding window power calculation
        self.timeline_window_label = QLabel("Window (s):")
        self.timeline_window_spin = QDoubleSpinBox()
        self.timeline_window_spin.setRange(0.2, 10.0)
        self.timeline_window_spin.setSingleStep(0.2)
        self.timeline_window_spin.setValue(2.0)

        # Step size control for sliding window power calculation
        self.timeline_step_label = QLabel("Step (s):")
        self.timeline_step_spin = QDoubleSpinBox()
        self.timeline_step_spin.setRange(0.1, 5.0)
        self.timeline_step_spin.setSingleStep(0.1)
        self.timeline_step_spin.setValue(0.2)

        tl_ctrl.addWidget(self.timeline_window_label)
        tl_ctrl.addWidget(self.timeline_window_spin)
        tl_ctrl.addSpacing(10)
        tl_ctrl.addWidget(self.timeline_step_label)
        tl_ctrl.addWidget(self.timeline_step_spin)

        self.timeline_info_btn = QPushButton("Info")
        self.timeline_info_btn.clicked.connect(self._show_timeline_info)
        tl_ctrl.addWidget(self.timeline_info_btn)

        tl_ctrl.addStretch()

        self.timeline_layout.addLayout(tl_ctrl)

        # Matplotlib canvas for plotting the band power over time
        self.timeline_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.timeline_layout.addWidget(self.timeline_canvas)
        self.timeline_tab.setLayout(self.timeline_layout)
        self.tabs.addTab(self.timeline_tab, "Band Timeline (Mu 8–12 Hz)")

        # -------- ERD/ERS tab setup --------
        self.erd_tab = QWidget()
        self.erd_layout = QVBoxLayout()

        # ---- Manual ERD/ERS control block ----
        erd_ctrl = QHBoxLayout()
        # Spinboxes for defining the absolute time intervals for baseline and task
        self.erd_baseline_start = QDoubleSpinBox()
        self.erd_baseline_end = QDoubleSpinBox()
        self.erd_task_start = QDoubleSpinBox()
        self.erd_task_end = QDoubleSpinBox()

        for sb in (
            self.erd_baseline_start,
            self.erd_baseline_end,
            self.erd_task_start,
            self.erd_task_end,
        ):
            sb.setRange(0.0, 9999.0)
            sb.setDecimals(3)
            sb.setSingleStep(0.1)

        # Combobox to select the frequency band for ERD/ERS calculation
        self.erd_band_combo = QComboBox()

        erd_ctrl.addWidget(QLabel("Baseline start (s):"))
        erd_ctrl.addWidget(self.erd_baseline_start)
        erd_ctrl.addWidget(QLabel("Baseline end (s):"))
        erd_ctrl.addWidget(self.erd_baseline_end)
        erd_ctrl.addSpacing(10)
        erd_ctrl.addWidget(QLabel("Task start (s):"))
        erd_ctrl.addWidget(self.erd_task_start)
        erd_ctrl.addWidget(QLabel("Task end (s):"))
        erd_ctrl.addWidget(self.erd_task_end)
        erd_ctrl.addSpacing(10)
        erd_ctrl.addWidget(QLabel("Band:"))
        erd_ctrl.addWidget(self.erd_band_combo)

        self.erd_manual_info_btn = QPushButton("Info")
        self.erd_manual_info_btn.clicked.connect(self._show_erd_info)
        erd_ctrl.addWidget(self.erd_manual_info_btn)

        erd_ctrl.addStretch()

        self.erd_layout.addLayout(erd_ctrl)

        # Information label for manual ERD/ERS
        self.erd_info_label = QLabel(
            "Suggestion: Mu (8–12 Hz) and Beta (13–30 Hz) bands are typically used "
            "for motor imagery analysis.\n"
            "Positive ERD/ERS = power decrease (ERD), negative ERD/ERS = power increase (ERS)."
        )
        self.erd_info_label.setWordWrap(True)
        self.erd_info_label.setStyleSheet("color: gray; font-size: 10pt;")
        self.erd_layout.addWidget(self.erd_info_label)

        # Matplotlib canvas for plotting manual ERD/ERS bar chart
        self.erd_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.erd_layout.addWidget(self.erd_canvas)

        # ---- Trial-based ERD/ERS control block ----
        trial_title = QLabel("Trial-based ERD/ERS (scientific analysis)")
        trial_title.setStyleSheet("font-weight: bold; margin-top: 12px;")
        self.erd_layout.addWidget(trial_title)

        trial_ctrl = QHBoxLayout()

        # Spinboxes for defining trial and relative time windows
        self.trial_duration_spin = QDoubleSpinBox()
        self.trial_duration_spin.setRange(0.5, 60.0)
        self.trial_duration_spin.setSingleStep(0.5)
        self.trial_duration_spin.setValue(5.0)

        self.trial_base_start_spin = QDoubleSpinBox()
        self.trial_base_start_spin.setRange(0.0, 60.0)
        self.trial_base_start_spin.setSingleStep(0.1)
        self.trial_base_start_spin.setValue(0.0)

        self.trial_base_end_spin = QDoubleSpinBox()
        self.trial_base_end_spin.setRange(0.0, 60.0)
        self.trial_base_end_spin.setSingleStep(0.1)
        self.trial_base_end_spin.setValue(2.0)

        self.trial_task_start_spin = QDoubleSpinBox()
        self.trial_task_start_spin.setRange(0.0, 60.0)
        self.trial_task_start_spin.setSingleStep(0.1)
        self.trial_task_start_spin.setValue(2.0)

        self.trial_task_end_spin = QDoubleSpinBox()
        self.trial_task_end_spin.setRange(0.0, 60.0)
        self.trial_task_end_spin.setSingleStep(0.1)
        self.trial_task_end_spin.setValue(4.0)

        self.trial_compute_button = QPushButton("Compute Trial-based ERD/ERS")

        trial_ctrl.addWidget(QLabel("Trial duration (s):"))
        trial_ctrl.addWidget(self.trial_duration_spin)
        trial_ctrl.addSpacing(10)
        trial_ctrl.addWidget(QLabel("Baseline in trial (s):"))
        trial_ctrl.addWidget(self.trial_base_start_spin)
        trial_ctrl.addWidget(QLabel("→"))
        trial_ctrl.addWidget(self.trial_base_end_spin)
        trial_ctrl.addSpacing(10)
        trial_ctrl.addWidget(QLabel("Task in trial (s):"))
        trial_ctrl.addWidget(self.trial_task_start_spin)
        trial_ctrl.addWidget(QLabel("→"))
        trial_ctrl.addWidget(self.trial_task_end_spin)
        trial_ctrl.addSpacing(10)
        trial_ctrl.addWidget(self.trial_compute_button)

        self.erd_trial_info_btn = QPushButton("Info")
        self.erd_trial_info_btn.clicked.connect(self._show_trial_erd_info)
        trial_ctrl.addWidget(self.erd_trial_info_btn)

        trial_ctrl.addStretch()

        self.erd_layout.addLayout(trial_ctrl)

        # Information label for trial-based ERD/ERS
        self.erd_trial_info_label = QLabel(
            "Note: The recording is divided into consecutive trials. ERD/ERS is calculated for each trial, and "
            "the average is taken per channel. Trial windows are set based on relative times."
        )
        self.erd_trial_info_label.setWordWrap(True)
        self.erd_trial_info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.erd_layout.addWidget(self.erd_trial_info_label)

        # Matplotlib canvas for plotting trial-based ERD/ERS bar chart with error bars
        self.erd_trial_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.erd_layout.addWidget(self.erd_trial_canvas)

        self.erd_tab.setLayout(self.erd_layout)
        self.tabs.addTab(self.erd_tab, "ERD/ERS")

        # -------- Topographic Map tab setup --------
        self.topo_tab = QWidget()
        self.topo_layout = QVBoxLayout()

        topo_ctrl_layout = QHBoxLayout()
        topo_ctrl_layout.addWidget(QLabel("Topomap mode:"))
        self.topo_mode_combo = QComboBox()
        # Default mode is Raw amplitude
        self.topo_mode_combo.addItem("Raw amplitude", userData=None)
        topo_ctrl_layout.addWidget(self.topo_mode_combo)

        self.topo_info_btn = QPushButton("Info")
        self.topo_info_btn.clicked.connect(self._show_topomap_info)
        topo_ctrl_layout.addWidget(self.topo_info_btn)

        topo_ctrl_layout.addStretch()
        self.topo_layout.addLayout(topo_ctrl_layout)

        # Matplotlib canvas for plotting the topographic map
        self.topo_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.topo_layout.addWidget(self.topo_canvas)
        self.topo_tab.setLayout(self.topo_layout)
        self.tabs.addTab(self.topo_tab, "Topographic Map")

        # -------- Gemini Analysis tab setup --------
        self._setup_gemini_tab()

        # -------- Real-time Visualization tab setup --------
        self.realtime_tab = RealTimeVizWidget()
        self.tabs.addTab(self.realtime_tab, "Real-time Visualization")

        # Control connections (connect UI elements to update methods)
        self.bandpower_use_window.stateChanged.connect(
            lambda _: (self._update_band_power(), self._update_topomap())
        )
        self.bandpower_window_spin.valueChanged.connect(
            lambda _: (self._update_band_power(), self._update_topomap())
        )
        self.topo_mode_combo.currentIndexChanged.connect(
            lambda _: self._update_topomap()
        )

        self.timeline_window_spin.valueChanged.connect(
            lambda _: self._update_band_timeline()
        )
        self.timeline_step_spin.valueChanged.connect(
            lambda _: self._update_band_timeline()
        )

        # Connect manual ERD/ERS time control changes
        for sb in (
            self.erd_baseline_start,
            self.erd_baseline_end,
            self.erd_task_start,
            self.erd_task_end,
        ):
            sb.valueChanged.connect(lambda _: self._update_erd_ers())

        # Connect manual ERD/ERS band selection change
        self.erd_band_combo.currentIndexChanged.connect(
            lambda _: self._update_erd_ers()
        )

        # Connect trial-based ERD/ERS compute button
        self.trial_compute_button.clicked.connect(self._compute_trial_erd_ers)

        # Menu setup
        self._create_menu()

        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Menu setup
    # ------------------------------------------------------------------
    def _create_menu(self):
        menu = self.menuBar()

        # File Menu
        m_file = menu.addMenu("&File")
        act_open = QAction("&Open CSV...", self)
        act_open.triggered.connect(self.open_csv)
        m_file.addAction(act_open)

        m_file.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        # Analysis Menu
        m_analysis = menu.addMenu("&Analysis")
        act_filter = QAction("&Filter Settings...", self)
        act_filter.triggered.connect(self._dialog_filter_settings)
        m_analysis.addAction(act_filter)

        # Help Menu
        m_help = menu.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _show_about(self):
        """Displays the 'About' message box."""
        QMessageBox.information(
            self,
            "About",
            f"{APP_NAME} v{APP_VERSION}\n\nEEG motor activity analysis tool.",
        )

    # ------------------------------------------------------------------
    # Filter settings dialog
    # ------------------------------------------------------------------
    def _dialog_filter_settings(self):
        """Opens a dialog to configure the global filter settings."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Filter Settings")
        form = QFormLayout(dlg)

        # Lowcut frequency spinbox
        low_spin = QDoubleSpinBox()
        low_spin.setRange(0.1, 100.0)
        low_spin.setValue(self.filter_lowcut)

        # Highcut frequency spinbox
        high_spin = QDoubleSpinBox()
        high_spin.setRange(0.1, 200.0)
        high_spin.setValue(self.filter_highcut)

        # Filter order spinbox
        order_spin = QSpinBox()
        order_spin.setRange(1, 10)
        order_spin.setValue(self.filter_order)

        # Notch filter checkbox
        notch_cb = QCheckBox("Enable 50 Hz notch")
        notch_cb.setChecked(self.filter_notch_enabled)

        form.addRow("Lowcut (Hz):", low_spin)
        form.addRow("Highcut (Hz):", high_spin)
        form.addRow("Order:", order_spin)
        form.addRow(notch_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addRow(buttons)

        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec_() == QDialog.Accepted:
            # Update filter settings from dialog
            self.filter_lowcut = float(low_spin.value())
            self.filter_highcut = float(high_spin.value())
            self.filter_order = int(order_spin.value())
            self.filter_notch_enabled = notch_cb.isChecked()

            # Recalculate and redraw all views dependent on filter settings
            self._update_filtered_signals()
            self._update_spectrograms()
            self._update_band_timeline()
            self._update_band_power()
            self._update_erd_ers()
            self._update_topomap()

            self.statusBar().showMessage(
                f"Filter updated: notch={self.filter_notch_enabled}, "
                f"band={self.filter_lowcut}-{self.filter_highcut} Hz, "
                f"order={self.filter_order}"
            )

    # ------------------------------------------------------------------
    # Raw Data selection tracking
    # ------------------------------------------------------------------
    def _connect_raw_selection(self):
        """Connects the QTableView's selection model to the handler method."""
        sel = self.raw_table.selectionModel()
        if sel is None:
            return
        try:
            # Disconnect previous connection if exists
            sel.selectionChanged.disconnect()
        except TypeError:
            pass
        # Connect to the new selection handler
        sel.selectionChanged.connect(self._on_raw_selection_changed)

    def _on_raw_selection_changed(self, *_):
        """Handles changes in the selected row of the raw data table."""
        if self.dataset is None:
            return

        sel = self.raw_table.selectionModel()
        if sel is None:
            return
        indexes = sel.selectedIndexes()
        if not indexes:
            # No row selected, reset state and update views
            self.selected_row_index = None
            self.selected_time = None
            self.raw_selected_label.setText("No row selected.")
            self._update_filtered_signals()
            self._update_spectrograms()
            self._update_band_power()
            self._update_band_timeline()
            self._update_erd_ers()
            self._update_topomap()
            return

        # Get the row index of the first selected cell
        row_idx = indexes[0].row()
        self.selected_row_index = row_idx

        # Get the data for the selected row
        row = self.dataset.df.iloc[row_idx]
        if "time" in row.index:
            self.selected_time = float(row["time"])
        else:
            self.selected_time = None

        # Update the selection status label
        parts = [f"Row {row_idx}"]
        if self.selected_time is not None:
            parts.append(f"time={self.selected_time:.3f} s")

        for ch in self.dataset.available_channels:
            if ch in row.index:
                try:
                    parts.append(f"{ch}={float(row[ch]):.3f}")
                except Exception:
                    parts.append(f"{ch}={row[ch]}")

        self.raw_selected_label.setText(" | ".join(parts))

        # Update views that depend on the selected time point
        self._update_filtered_signals()
        self._update_spectrograms()
        self._update_band_power()
        self._update_band_timeline()
        self._update_erd_ers()
        self._update_topomap()

    # ------------------------------------------------------------------
    # Filtered Signals (global filter)
    # ------------------------------------------------------------------
    def _update_filtered_signals(self):
        """Filters the EEG data and plots the signals for selected channels."""
        if self.dataset is None:
            return

        # Prioritize C3 and C4 channels, otherwise use all available channels
        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels

        if not channels:
            self.statusBar().showMessage("No channels available to plot.")
            return

        # Apply filtering based on current settings
        df_filt = filter_channels(
            self.dataset,
            channels=channels,
            apply_notch=self.filter_notch_enabled,
            bandpass=(self.filter_lowcut, self.filter_highcut),
            order=self.filter_order,
        )

        fig = self.filtered_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Plot time series for each channel
        t = df_filt["time"].to_numpy()
        for ch in channels:
            ax.plot(t, df_filt[ch].to_numpy(), label=ch)

        # Draw a vertical line at the selected time, if any (synchronization)
        if self.selected_time is not None:
            ax.axvline(self.selected_time, color="red", linestyle="--", linewidth=1.0)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Filtered signals (notch={self.filter_notch_enabled}, "
            f"{self.filter_lowcut}-{self.filter_highcut} Hz)"
        )
        ax.legend()

        fig.tight_layout()
        self.filtered_canvas.draw()

    # ------------------------------------------------------------------
    # Band Power (from raw signal, separate subplot for each band)
    # ------------------------------------------------------------------
    def _update_band_power(self):
        """Computes and plots band power for selected channels/window."""
        if self.dataset is None:
            return

        # Prioritize C3-C4, otherwise use all available channels
        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels

        if not channels:
            self.statusBar().showMessage("No channels available for band power.")
            return

        ds = self.dataset
        window_info = "full signal"

        # Use a segment/window around selected row if checked
        if self.bandpower_use_window.isChecked() and self.selected_time is not None:
            win = float(self.bandpower_window_spin.value())
            half = win / 2.0
            t0 = max(ds.time[0], self.selected_time - half)
            t1 = min(ds.time[-1], self.selected_time + half)
            if t1 <= t0:
                self.statusBar().showMessage("Invalid window for band power.")
                return

            # Extract the segment and create a temporary EEGDataset for calculation
            seg = ds.get_segment(t0, t1)
            ds = EEGDataset(
                filepath=ds.filepath,
                df=seg.reset_index(drop=True),
                fs=ds.fs,
                time=seg["time"].to_numpy(),
                available_channels=ds.available_channels,
            )
            window_info = f"window around {self.selected_time:.3f} s (len={win:.2f} s)"

        # Compute band power from the raw signal segment
        df_in = ds.df[["time"] + channels].copy()
        band_powers = compute_band_powers_for_channels(
            df_in, fs=ds.fs, channels=channels
        )

        if not band_powers:
            self.statusBar().showMessage("No band power data computed.")
            return

        # Get list of band names (keys from the first channel's band power)
        band_names = list(next(iter(band_powers.values())).keys())
        n_bands = len(band_names)
        if n_bands == 0:
            self.statusBar().showMessage("No bands found for band power.")
            return

        fig = self.bandpower_canvas.figure
        fig.clear()

        # Create subplots side-by-side, one for each band
        axes = fig.subplots(1, n_bands, squeeze=False)[0]

        x = np.arange(len(channels)) # X-axis positions for channels

        for j, (band, ax) in enumerate(zip(band_names, axes)):
            # Get the power value for the current band across all channels
            vals = [band_powers[ch][band] for ch in channels]

            # Plot as a bar chart
            ax.bar(x, vals)
            ax.set_xticks(x)
            ax.set_xticklabels(channels) # Set channel names as X ticks

            ax.set_title(band)

            if j == 0:
                ax.set_ylabel("Power (a.u.)")
            else:
                # Clear Y-axis labels for subplots after the first one
                ax.set_yticklabels([])

        fig.suptitle(
            f"Band Power ({window_info})\n(computed from raw signal)",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90]) # Adjust layout to make space for suptitle
        self.bandpower_canvas.draw()

        self.statusBar().showMessage(
            f"Band power updated | channels={channels} | {window_info}"
        )

    # ------------------------------------------------------------------
    # Spectrograms (globally filtered)
    # ------------------------------------------------------------------
    def _update_spectrograms(self):
        """Computes and plots spectrograms for selected/first two channels."""
        if self.dataset is None:
            return

        # Prioritize C3 and C4, otherwise use the first two available channels
        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels[:2]

        if not channels:
            self.statusBar().showMessage("No channels available for spectrogram.")
            return

        # Filter the data first
        df_filt = filter_channels(
            self.dataset,
            channels=channels,
            apply_notch=self.filter_notch_enabled,
            bandpass=(self.filter_lowcut, self.filter_highcut),
            order=self.filter_order,
        )

        fig = self.spectrogram_canvas.figure
        fig.clear()

        n_ch = len(channels)
        for i, ch in enumerate(channels, start=1):
            # Create a subplot for each channel
            ax = fig.add_subplot(n_ch, 1, i)
            x = df_filt[ch].to_numpy(dtype=float)
            
            # Compute Spectrogram
            f, t, Sxx = compute_spectrogram(x, fs=self.dataset.fs)
            Sxx_db = 10 * np.log10(Sxx + 1e-12) # Convert power to dB

            # Plot the spectrogram using pseudocolor plot
            ax.pcolormesh(t, f, Sxx_db, shading="auto")
            ax.set_ylabel("Freq (Hz)")
            ax.set_title(f"Spectrogram - {ch}")
            ax.set_ylim(0, 120) # Limit y-axis to a common EEG range

            if len(t) > 1:
                ax.set_xlim(t[0], t[-1])

            # Draw a vertical line at the selected time, if within the time range
            if self.selected_time is not None and len(t) > 1:
                if t[0] <= self.selected_time <= t[-1]:
                    ax.axvline(
                        self.selected_time,
                        color="red",
                        linestyle="--",
                        linewidth=1.0,
                    )

        ax.set_xlabel("Time (s)")
        fig.tight_layout()
        self.spectrogram_canvas.draw()

    # ------------------------------------------------------------------
    # Band Timeline (Mu 8–12 Hz) – global filter + Welch
    # ------------------------------------------------------------------
    def _update_band_timeline(self):
        """Calculates and plots Mu band power over time using a sliding window."""
        if self.dataset is None:
            return

        # Prioritize C3 and C4, otherwise use all available channels
        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels

        if not channels:
            self.statusBar().showMessage("No channels available for band timeline.")
            return

        # Filter the entire signal first
        df_filt = filter_channels(
            self.dataset,
            channels=channels,
            apply_notch=self.filter_notch_enabled,
            bandpass=(self.filter_lowcut, self.filter_highcut),
            order=self.filter_order,
        )

        fs = self.dataset.fs
        t = df_filt["time"].to_numpy()

        win_len = float(self.timeline_window_spin.value()) # Window length in seconds
        step_len = float(self.timeline_step_spin.value())  # Step size in seconds

        if win_len <= 0 or step_len <= 0:
            self.statusBar().showMessage("Invalid window/step for band timeline.")
            return

        # Convert window/step from seconds to samples
        win_samples = int(win_len * fs)
        step_samples = int(step_len * fs)
        if win_samples < 8 or step_samples < 1:
            self.statusBar().showMessage("Window too small for band timeline.")
            return

        n_samples = len(t)
        if n_samples < win_samples:
            self.statusBar().showMessage("Signal is shorter than the window.")
            return

        mu_low, mu_high = 8.0, 12.0 # Mu band definition

        times = []
        powers_per_channel = {ch: [] for ch in channels}

        # Sliding window calculation loop
        start_idx = 0
        while start_idx + win_samples <= n_samples:
            end_idx = start_idx + win_samples
            seg_time = t[start_idx:end_idx]
            center_time = (seg_time[0] + seg_time[-1]) / 2.0 # Center time of the window
            times.append(center_time)

            for ch in channels:
                seg = df_filt[ch].to_numpy(dtype=float)[start_idx:end_idx]
                nper = min(win_samples, 256) # nperseg for Welch's method
                f, Pxx = welch(seg, fs=fs, nperseg=nper)
                
                # Sum power within the Mu band range (8–12 Hz)
                mask = (f >= mu_low) & (f <= mu_high)
                mu_power = Pxx[mask].sum()
                powers_per_channel[ch].append(mu_power)

            start_idx += step_samples # Move the window by the step size

        if not times:
            self.statusBar().showMessage("No timeline points computed.")
            return

        times = np.array(times)
        fig = self.timeline_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Plot Mu band power over time for each channel
        for ch in channels:
            p = np.array(powers_per_channel[ch])
            ax.plot(times, p, label=ch)

        # Draw a vertical line at the selected time
        if self.selected_time is not None:
            ax.axvline(self.selected_time, color="red", linestyle="--", linewidth=1.0)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mu band power (a.u.)")
        ax.set_title(
            f"Mu Band (8–12 Hz) Timeline | window={win_len:.2f}s, step={step_len:.2f}s"
        )
        ax.legend()

        fig.tight_layout()
        self.timeline_canvas.draw()

        self.statusBar().showMessage("Mu band timeline updated.")

    # ------------------------------------------------------------------
    # Refresh ERD/ERS band list (based on raw signal band power)
    # ------------------------------------------------------------------
    def _refresh_erd_band_list(self):
        """Updates the list of frequency bands available for ERD/ERS analysis."""
        if self.dataset is None:
            return

        channels = self._erd_channels()
        if not channels:
            return

        ds = self.dataset
        df_in = ds.df[["time"] + channels].copy()
        # Compute band power once to get the available band names
        band_powers = compute_band_powers_for_channels(
            df_in, fs=ds.fs, channels=channels
        )
        if not band_powers:
            return

        band_names = list(next(iter(band_powers.values())).keys())

        # Define user-friendly labels for common bands
        pretty_labels = {
            "delta": "Delta (0.5–4 Hz) – slow/artefact",
            "theta": "Theta (4–8 Hz)",
            "alpha_mu": "Mu (8–12 Hz) – motor imagery",
            "beta": "Beta (13–30 Hz) – motor imagery",
            "gamma": "Gamma (30–45 Hz)",
        }

        self.erd_band_combo.blockSignals(True) # Temporarily block signals during update
        self.erd_band_combo.clear()

        default_index = 0
        for i, name in enumerate(band_names):
            label = pretty_labels.get(name, name)
            self.erd_band_combo.addItem(label, userData=name) # Store internal band key as userData
            if name == "alpha_mu":
                default_index = i # Set Mu band as the default selection

        self.erd_band_combo.setCurrentIndex(default_index)
        self.erd_band_combo.blockSignals(False)

    def _erd_channels(self):
        """Returns the list of channels preferred for ERD/ERS analysis (C3/C4 or all)."""
        preferred = ["C3", "C4"]
        if self.dataset is None:
            return []
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels
        return channels

    # ------------------------------------------------------------------
    # ERD/ERS calculation and plotting (manual, raw signal band power)
    # ------------------------------------------------------------------
    def _update_erd_ers(self):
        """Calculates and plots manual ERD/ERS for the selected band and intervals."""
        if self.dataset is None:
            return
        if self.erd_band_combo.count() == 0:
            return

        idx = self.erd_band_combo.currentIndex()
        if idx < 0:
            return

        band_name = self.erd_band_combo.itemData(idx)  # Internal key (e.g., 'alpha_mu')
        band_label = self.erd_band_combo.currentText()  # User-friendly label
        if not band_name:
            return

        t_min = float(self.dataset.time[0])
        t_max = float(self.dataset.time[-1])

        # Get manually set absolute time intervals for baseline and task
        bs = float(self.erd_baseline_start.value())
        be = float(self.erd_baseline_end.value())
        ts = float(self.erd_task_start.value())
        te = float(self.erd_task_end.value())

        # Check for valid time intervals
        if not (t_min <= bs < be <= t_max and t_min <= ts < te <= t_max):
            self.statusBar().showMessage("Invalid baseline/task intervals for ERD/ERS.")
            return

        channels = self._erd_channels()
        if not channels:
            self.statusBar().showMessage("No channels for ERD/ERS.")
            return

        def band_power_interval(t0, t1):
            """Computes band power for a specific time interval."""
            seg = self.dataset.get_segment(t0, t1)
            # Create a temporary EEGDataset for the segment
            ds_seg = EEGDataset(
                filepath=self.dataset.filepath,
                df=seg.reset_index(drop=True),
                fs=self.dataset.fs,
                time=seg["time"].to_numpy(),
                available_channels=self.dataset.available_channels,
            )
            df_in = ds_seg.df[["time"] + channels].copy()
            bp = compute_band_powers_for_channels(
                df_in, fs=ds_seg.fs, channels=channels
            )
            
            # Extract the power for the selected band_name
            result = {}
            for ch in channels:
                ch_bp = bp.get(ch, {})
                if band_name in ch_bp:
                    result[ch] = float(ch_bp[band_name])
            return result

        # Compute baseline and task band power
        baseline_pw = band_power_interval(bs, be)
        task_pw = band_power_interval(ts, te)

        if not baseline_pw or not task_pw:
            self.statusBar().showMessage("Could not compute ERD/ERS band powers.")
            return

        # Calculate ERD/ERS for each channel
        erd_values = {}
        for ch in channels:
            pb = baseline_pw.get(ch, None)
            pt = task_pw.get(ch, None)
            if pb is None or pt is None or pb == 0:
                erd_values[ch] = np.nan
            else:
                # ERD% = (P_baseline - P_task) / P_baseline * 100.0 (Reduction is positive)
                erd_values[ch] = (pb - pt) / pb * 100.0

        fig = self.erd_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        ch_list = list(channels)
        x = np.arange(len(ch_list))
        vals = [erd_values[ch] for ch in ch_list]

        # Plot ERD/ERS as a bar chart
        bars = ax.bar(x, vals)
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0) # Zero line

        ax.set_xticks(x)
        ax.set_xticklabels(ch_list)
        ax.set_ylabel("ERD/ERS (%)")
        ax.set_title(
            f"Manual ERD/ERS – {band_label}\n"
            f"Baseline: {bs:.2f}-{be:.2f}s, Task: {ts:.2f}-{te:.2f}s"
        )

        # Add text labels for the percentage values above/below bars
        for xi, bar, ch in zip(x, bars, ch_list):
            value = bar.get_height()
            if np.isnan(value):
                label = "NaN"
                y = 0
                va = "bottom"
            else:
                label = f"{value:.1f}%"
                # Determine position and vertical alignment of the text label
                if value >= 0:
                    y = value + abs(value) * 0.02 + 1.0 # Offset slightly above bar
                    va = "bottom"
                else:
                    y = value - abs(value) * 0.02 - 1.0 # Offset slightly below bar
                    va = "top"
            ax.text(
                xi,
                y,
                label,
                ha="center",
                va=va,
                fontsize=9,
                color="black",
            )

        fig.tight_layout()
        self.erd_canvas.draw()

        # Update status bar with detailed results
        info_parts = []
        for ch in ch_list:
            pb = baseline_pw.get(ch, float("nan"))
            pt = task_pw.get(ch, float("nan"))
            ev = erd_values[ch]
            info_parts.append(
                f"{ch}: Pb={pb:.2f}, Pt={pt:.2f}, ERD={ev:.1f}%"
            )
        self.statusBar().showMessage(" | ".join(info_parts))

    # ------------------------------------------------------------------
    # Trial-based ERD/ERS (raw signal, multiple trials)
    # ------------------------------------------------------------------
    def _compute_trial_erd_ers(self):
        """Calculates the mean and standard deviation of ERD/ERS across consecutive trials."""
        if self.dataset is None:
            return
        if self.erd_band_combo.count() == 0:
            return

        idx = self.erd_band_combo.currentIndex()
        if idx < 0:
            return

        band_name = self.erd_band_combo.itemData(idx)
        band_label = self.erd_band_combo.currentText()
        if not band_name:
            return

        ds = self.dataset
        t0 = float(ds.time[0])
        t_end = float(ds.time[-1])
        total_duration = t_end - t0

        # Get trial parameters from spinboxes
        trial_dur = float(self.trial_duration_spin.value())
        b_rel_s = float(self.trial_base_start_spin.value()) # Baseline relative start
        b_rel_e = float(self.trial_base_end_spin.value())   # Baseline relative end
        t_rel_s = float(self.trial_task_start_spin.value()) # Task relative start
        t_rel_e = float(self.trial_task_end_spin.value())   # Task relative end

        # Validate trial parameters
        if trial_dur <= 0:
            self.statusBar().showMessage("Trial duration must be > 0.")
            return
        if not (0.0 <= b_rel_s < b_rel_e <= trial_dur):
            self.statusBar().showMessage("Invalid baseline window inside trial.")
            return
        if not (0.0 <= t_rel_s < t_rel_e <= trial_dur):
            self.statusBar().showMessage("Invalid task window inside trial.")
            return

        # Determine the number of complete trials
        n_trials = int(np.floor(total_duration / trial_dur))
        if n_trials < 1:
            self.statusBar().showMessage("Recording is too short for even one trial.")
            return

        channels = self._erd_channels()
        if not channels:
            self.statusBar().showMessage("No channels available for trial-based ERD/ERS.")
            return

        def band_power_interval_abs(t_start_abs, t_end_abs):
            """Computes band power for an absolute time interval using a segment."""
            seg = ds.get_segment(t_start_abs, t_end_abs)
            sub = EEGDataset(
                filepath=ds.filepath,
                df=seg.reset_index(drop=True),
                fs=ds.fs,
                time=seg["time"].to_numpy(),
                available_channels=ds.available_channels,
            )
            df_in = sub.df[["time"] + channels].copy()
            bp = compute_band_powers_for_channels(
                df_in, fs=sub.fs, channels=channels
            )
            result = {}
            for ch in channels:
                ch_bp = bp.get(ch, {})
                if band_name in ch_bp:
                    result[ch] = float(ch_bp[band_name])
            return result

        trial_erd = {ch: [] for ch in channels}

        # Loop through all complete trials
        for k in range(n_trials):
            trial_start = t0 + k * trial_dur
            # Calculate absolute time points for baseline and task windows
            base_s = trial_start + b_rel_s
            base_e = trial_start + b_rel_e
            task_s = trial_start + t_rel_s
            task_e = trial_start + t_rel_e

            base_pw = band_power_interval_abs(base_s, base_e)
            task_pw = band_power_interval_abs(task_s, task_e)

            # Calculate ERD/ERS for the current trial
            for ch in channels:
                pb = base_pw.get(ch, None)
                pt = task_pw.get(ch, None)
                if pb is None or pt is None or pb == 0:
                    trial_erd[ch].append(np.nan) # Append NaN if calculation fails
                else:
                    val = (pb - pt) / pb * 100.0
                    trial_erd[ch].append(val)

        # Calculate mean and standard deviation of ERD/ERS across trials
        mean_erd = {}
        std_erd = {}
        n_valid = {}
        for ch in channels:
            arr = np.array(trial_erd[ch], dtype=float)
            valid = ~np.isnan(arr) # Mask for valid (non-NaN) trials
            if not np.any(valid):
                mean_erd[ch] = np.nan
                std_erd[ch] = np.nan
                n_valid[ch] = 0
            else:
                # Use nanmean and nanstd to ignore NaN values
                mean_erd[ch] = float(np.nanmean(arr))
                std_erd[ch] = float(np.nanstd(arr))
                n_valid[ch] = int(valid.sum()) # Count of valid trials

        fig = self.erd_trial_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        ch_list = list(channels)
        x = np.arange(len(ch_list))
        means = [mean_erd[ch] for ch in ch_list]
        stds = [std_erd[ch] for ch in ch_list]

        # Plot mean ERD/ERS with standard deviation as error bars
        bars = ax.bar(x, means, yerr=stds, capsize=6)
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0) # Zero line

        ax.set_xticks(x)
        ax.set_xticklabels(ch_list)
        ax.set_ylabel("ERD/ERS (%)")
        ax.set_title(
            f"Trial-based ERD/ERS – {band_label}\n"
            f"Trial duration={trial_dur:.2f}s, "
            f"Baseline={b_rel_s:.2f}-{b_rel_e:.2f}s, "
            f"Task={t_rel_s:.2f}-{t_rel_e:.2f}s (relative to trial)"
        )

        # Add text labels for mean value and valid trial count (N)
        for xi, bar, ch in zip(x, bars, ch_list):
            value = bar.get_height()
            ntr = n_valid.get(ch, 0)
            if np.isnan(value) or ntr == 0:
                label = "NaN"
                y = 0
                va = "bottom"
            else:
                label = f"{value:.1f}% (N={ntr})"
                # Determine position and vertical alignment of the text label
                if value >= 0:
                    y = value + abs(value) * 0.02 + 1.0
                    va = "bottom"
                else:
                    y = value - abs(value) * 0.02 - 1.0
                    va = "top"
            ax.text(
                xi,
                y,
                label,
                ha="center",
                va=va,
                fontsize=9,
                color="black",
            )

        fig.tight_layout()
        self.erd_trial_canvas.draw()

        # Update status bar with detailed results
        info_parts = []
        for ch in ch_list:
            info_parts.append(
                f"{ch}: mean={mean_erd[ch]:.1f}%, std={std_erd[ch]:.1f}%, N={n_valid[ch]}"
            )
        self.statusBar().showMessage(
            f"Trial-based ERD/ERS computed over {n_trials} trials | "
            + " | ".join(info_parts)
        )

    # ------------------------------------------------------------------
    # Topomap band list (based on raw signal band power)
    # ------------------------------------------------------------------
    def _refresh_topo_band_list(self):
        """Updates the list of available modes (raw/band power) for the Topographic Map."""
        if self.dataset is None:
            return

        # Select only channels with defined electrode positions
        channels = [
            ch
            for ch in ELECTRODE_POSITIONS.keys()
            if ch in self.dataset.available_channels
        ]
        if not channels:
            return

        ds = self.dataset
        df_in = ds.df[["time"] + channels].copy()
        # Compute band power once to get the available band names
        band_powers = compute_band_powers_for_channels(
            df_in, fs=ds.fs, channels=channels
        )
        if not band_powers:
            return

        band_names = list(next(iter(band_powers.values())).keys())

        self.topo_mode_combo.blockSignals(True)
        self.topo_mode_combo.clear()
        self.topo_mode_combo.addItem("Raw amplitude", userData=None) # Default: raw amplitude
        # Add band power options
        for name in band_names:
            self.topo_mode_combo.addItem(f"Band power: {name}", userData=name)
        self.topo_mode_combo.setCurrentIndex(0)
        self.topo_mode_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Topographic Map (raw or band-selected, raw signal band power)
    # ------------------------------------------------------------------
    def _update_topomap(self):
        """Generates and plots a 2D topographic map of channel activity."""
        if self.dataset is None:
            return

        selected_band = None
        if hasattr(self, "topo_mode_combo"):
            idx = self.topo_mode_combo.currentIndex()
            if idx >= 0:
                selected_band = self.topo_mode_combo.itemData(idx) # Internal band key or None for raw

        channel_values = {}

        # RAW AMPLITUDE MODE
        if selected_band is None:
            if self.selected_row_index is None:
                self.statusBar().showMessage(
                    "Select a row for raw-amplitude topomap."
                )
                return

            # Use the raw amplitude value at the selected time point
            row = self.dataset.df.iloc[self.selected_row_index]
            for ch, (xx, yy) in ELECTRODE_POSITIONS.items():
                if ch in self.dataset.available_channels and ch in row.index:
                    val = row[ch]
                    if pd.isna(val):
                        continue
                    channel_values[ch] = float(val)

        # BAND POWER MODE (raw signal)
        else:
            # Select channels that have both data and defined positions
            channels = [
                ch
                for ch in ELECTRODE_POSITIONS.keys()
                if ch in self.dataset.available_channels
            ]
            if not channels:
                self.statusBar().showMessage("No electrodes available for topomap.")
                return

            ds = self.dataset
            
            # Use a time window if checked
            if self.bandpower_use_window.isChecked() and self.selected_time is not None:
                win = float(self.bandpower_window_spin.value())
                half = win / 2.0
                t0 = max(ds.time[0], self.selected_time - half)
                t1 = min(ds.time[-1], self.selected_time + half)
                if t1 <= t0:
                    self.statusBar().showMessage("Invalid window for topomap.")
                    return

                # Create temporary EEGDataset for the segment
                seg = ds.get_segment(t0, t1)
                ds = EEGDataset(
                    filepath=ds.filepath,
                    df=seg.reset_index(drop=True),
                    fs=ds.fs,
                    time=seg["time"].to_numpy(),
                    available_channels=ds.available_channels,
                )

            # Compute band power for the full signal or the window
            df_in = ds.df[["time"] + channels].copy()
            band_powers = compute_band_powers_for_channels(
                df_in, fs=ds.fs, channels=channels
            )
            if not band_powers:
                self.statusBar().showMessage("No band power data for topomap.")
                return

            band_key = str(selected_band)
            # Extract the power value for the selected band
            for ch in channels:
                ch_bp = band_powers.get(ch, {})
                if band_key in ch_bp:
                    channel_values[ch] = float(ch_bp[band_key])

        if not channel_values:
            self.statusBar().showMessage("No channel values to plot on topomap.")
            return

        # Prepare electrode positions (x, y) and values (z)
        xs, ys, zs = [], [], []
        for ch, (xx, yy) in ELECTRODE_POSITIONS.items():
            if ch in channel_values:
                xs.append(xx)
                ys.append(yy)
                zs.append(channel_values[ch])

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        zs = np.array(zs, dtype=float)

        # For raw amplitude, use absolute values for a heat map visualization
        if selected_band is None:
            zs = np.abs(zs) 
        
        if zs.size == 0:
            self.statusBar().showMessage("No valid values for topomap.")
            return
            
        zmin, zmax = zs.min(), zs.max()
        # Normalize Z values between 0 and 1
        if np.isclose(zmin, zmax):
            zs_norm = np.ones_like(zs) * 0.5
        else:
            zs_norm = (zs - zmin) / (zmax - zmin)

        # Create a grid for interpolation
        grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]

        # Interpolate the values onto the grid (using nearest neighbor for few points)
        if len(xs) < 4:
            zi = griddata((xs, ys), zs_norm, (grid_x, grid_y), method="nearest")
        else:
            try:
                zi = griddata((xs, ys), zs_norm, (grid_x, grid_y), method="linear")
            except Exception:
                zi = griddata((xs, ys), zs_norm, (grid_x, grid_y), method="nearest")

        # Mask data outside the head circle (r > 1.0)
        r = np.sqrt(grid_x**2 + grid_y**2)
        mask = r > 1.0
        zi_masked = np.ma.array(zi, mask=mask)

        if np.all(np.isnan(zi_masked)):
            self.statusBar().showMessage("Topomap: interpolation failed (all NaN).")
            return

        fig = self.topo_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Plot the interpolated data as a contour map
        cf = ax.contourf(
            grid_x,
            grid_y,
            zi_masked,
            levels=40,
            cmap="RdYlBu_r", # Red-Yellow-Blue reversed colormap
            vmin=0.0,
            vmax=1.0,
        )

        # Draw the head outline
        head = plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=2)
        ax.add_patch(head)

        # Draw the nose
        nose_x = [0.0, -0.08, 0.08]
        nose_y = [1.0, 1.12, 1.12]
        ax.plot(nose_x, nose_y, "k-", linewidth=2)

        # Draw the ears
        ear_y = 0.0
        ear_r = 0.08
        ax.add_patch(
            plt.Circle((-1.0, ear_y), ear_r, fill=False, color="black", linewidth=2)
        )
        ax.add_patch(
            plt.Circle((1.0, ear_y), ear_r, fill=False, color="black", linewidth=2)
        )

        # Plot electrode points and labels
        for ch, (xx, yy) in ELECTRODE_POSITIONS.items():
            if ch in channel_values:
                ax.plot(xx, yy, "wo", markersize=6, markeredgecolor="black") # White circle for electrode
                ax.text(xx + 0.03, yy + 0.03, ch, fontsize=9, color="black") # Electrode label

        ax.set_aspect("equal")
        ax.set_axis_off() # Hide axes

        # Add color bar
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Relative activity (a.u.)")

        fig.tight_layout()
        self.topo_canvas.draw()

        self.statusBar().showMessage("Topographic map updated.")

    # ------------------------------------------------------------------
    # Info windows (descriptions of each tab/feature)
    # ------------------------------------------------------------------
    def _show_raw_info(self):
        """Displays information about the Raw Data tab."""
        text = (
            "Raw Data tab:\n\n"
            "- Displays the raw EEG samples from the loaded .csv file as rows/columns.\n"
            "- The 'time' column represents the time axis in seconds.\n"
            "- Other columns are electrode channels (C3, C4, etc.).\n"
            "- When you select a row in the table:\n"
            "  • The label below shows the time and channel amplitudes of the row.\n"
            "  • The red vertical line in other tabs is aligned to this time point.\n"
            "- This tab allows you to select a specific moment in the recording to\n"
            "  examine the filtered signal, spectrogram, band power, and topomap relative to that moment."
        )
        QMessageBox.information(self, "Info – Raw Data", text)

    def _show_filtered_info(self):
        """Displays information about the Filtered Signals tab."""
        text = (
            "Filtered Signals tab:\n\n"
            "- Applies notch + band-pass filtering to the selected channels (preferably C3, C4).\n"
            "- Purpose: To suppress DC drift, very low frequencies, and high-frequency noise\n"
            "  in the raw EEG to better visualize the frequency range related to motor imagery.\n"
            "- Filter settings can be changed via the top menu: 'Analysis → Filter Settings'\n"
            "  (notch on/off, lowcut, highcut, order).\n"
            "- The red vertical line shows the time of the selected row (from the Raw Data tab).\n"
            "- This tab is used to interpret the filtered signal in the time domain."
        )
        QMessageBox.information(self, "Info – Filtered Signals", text)

    def _show_bandpower_info(self):
        """Displays information about the Band Power tab."""
        text = (
            "Band Power tab:\n\n"
            "- Shows the total power for the EEG frequency bands (delta, theta, mu, beta, gamma, etc.)\n"
            "  for the selected channels.\n"
            "- Power calculation is performed on the raw signal (not the filtered signal).\n"
            "- If 'Use selected row window' is checked:\n"
            "  • A window of the specified duration ('Window s') around the selected time is taken,\n"
            "    and band power is calculated only within this window.\n"
            "  • This allows comparing different moments like rest, task, or artifact periods.\n"
            "- If unchecked, band power is calculated over the entire recording.\n"
            "- Mu (8–12 Hz) and Beta (13–30 Hz) bands are particularly important in motor imagery studies\n"
            "  and form the basis for ERD/ERS analysis."
        )
        QMessageBox.information(self, "Info – Band Power", text)

    def _show_spectrogram_info(self):
        """Displays information about the Spectrograms tab."""
        text = (
            "Spectrograms tab:\n\n"
            "- Generates a time-frequency representation (spectrogram) for the selected channels.\n"
            "- Horizontal axis: Time (s), Vertical axis: Frequency (Hz), Color: Power (dB).\n"
            "- Filter settings (notch + band-pass) are also applied here; you can change the\n"
            "  filter from the top menu and recalculate.\n"
            "- The red vertical line shows the time of the selected row from the Raw Data tab.\n"
            "- Used in motor imagery analysis to visualize power changes in the mu/beta band\n"
            "  during specific time intervals.\n"
            "- The upper limit on the Y-axis is currently set up to 120 Hz."
        )
        QMessageBox.information(self, "Info – Spectrograms", text)

    def _show_timeline_info(self):
        """Displays information about the Band Timeline (Mu) tab."""
        text = (
            "Band Timeline (Mu) tab:\n\n"
            "- Shows the change in Mu band power (8–12 Hz) over time for the selected channels.\n"
            "- 'Window (s)': The length of the sliding window used for calculating each point.\n"
            "- 'Step (s)': The step size (slide) of the window; determines how frequently the calculation is performed.\n"
            "- E.g., if window=2s, step=0.2s, Mu band power is calculated every 0.2 seconds over a 2-second window.\n"
            "- Used to observe how Mu power on C3/C4 increases or decreases over time in motor imagery.\n"
            "- The red vertical line shows the selected row time and is synchronized with other tabs."
        )
        QMessageBox.information(self, "Info – Band Timeline (Mu)", text)

    def _show_erd_info(self):
        """Displays information about the Manual ERD/ERS tab."""
        text = (
            "ERD/ERS – Manual tab:\n\n"
            "- Manually select Baseline and Task intervals on the time axis to\n"
            "  calculate ERD/ERS (%) for each channel.\n"
            "- Formula used: ERD% = (P_baseline − P_task) / P_baseline × 100\n"
            "  • Positive value → Power decrease (ERD – event-related desynchronization)\n"
            "  • Negative value → Power increase (ERS – event-related synchronization)\n"
            "- You can select the frequency band to analyze from the 'Band' section:\n"
            "  • Mu (8–12 Hz) and Beta (13–30 Hz) are the most critical bands for motor imagery.\n"
            "- This tab is mainly used for quickly testing ERD/ERS for a single trial-like situation\n"
            "  and manually comparing different time intervals."
        )
        QMessageBox.information(self, "Info – Manual ERD/ERS", text)

    def _show_trial_erd_info(self):
        """Displays information about the Trial-based ERD/ERS tab."""
        text = (
            "Trial-based ERD/ERS tab:\n\n"
            "- Divides the recording into equal-length trials, calculates ERD/ERS for each trial,\n"
            "  and produces the mean + standard deviation per channel.\n"
            "- 'Trial duration (s)': The duration of each trial.\n"
            "- 'Baseline in trial (s)': The relative baseline window within the trial.\n"
            "- 'Task in trial (s)': The relative task window within the trial.\n"
            "- E.g., trial duration = 5 s, baseline = 0–2 s, task = 2–4 s:\n"
            "  • The recording is divided into 0–5, 5–10, 10–15 ... second blocks.\n"
            "  • In each block, 0–2 s is used as baseline and 2–4 s as task.\n"
            "- This method is close to the \"trial-based ERD/ERS\" approach used in the literature\n"
            "  for motor imagery studies.\n"
            "- In the graph: Bars show the mean ERD/ERS per channel, error bars show the standard deviation;\n"
            "  N (number of valid trials) information is displayed above them."
        )
        QMessageBox.information(self, "Info – Trial-based ERD/ERS", text)

    def _show_topomap_info(self):
        """Displays information about the Topographic Map tab."""
        text = (
            "Topographic Map tab:\n\n"
            "- Uses the approximate locations of electrodes (C3, C4, etc.) on the scalp to\n"
            "  display activity as a 2D brain map.\n"
            "- 'Topomap mode: Raw amplitude':\n"
            "  • Uses the instantaneous amplitudes at the selected row in the Raw Data tab.\n"
            "  • Warm colors (red/yellow) represent high activity, and cool colors (blue) represent low activity.\n"
            "- 'Topomap mode: Band power: ...':\n"
            "  • Uses the average band power over the entire recording (or the selected window).\n"
            "  • If 'Use selected row window' is active, band power is calculated over the window\n"
            "    around the selected time in Raw Data.\n"
            "- This tab is used to visualize which brain region (e.g., C3 vs C4) has relatively higher\n"
            "  activity.\n"
            "- The head outline, ears, and nose line are included for approximate orientation on the map."
        )
        QMessageBox.information(self, "Info – Topographic Map", text)

    # ------------------------------------------------------------------
    # Open CSV file
    # ------------------------------------------------------------------
    def open_csv(self):
        """Opens a file dialog, loads the selected CSV file, and initializes the views."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open EEG CSV file",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return

        try:
            # Load the dataset using the external data_loader module
            ds = EEGDataset.from_csv(file_path)
            self.dataset = ds

            # Set the Pandas model to the QTableView
            model = PandasModel(ds.df)
            self.raw_table.setModel(model)
            self._connect_raw_selection()

            duration = ds.time[-1] - ds.time[0]
            self.statusBar().showMessage(
                f"Loaded: {file_path} | fs={ds.fs} Hz | "
                f"duration={duration:.2f} s | channels={ds.available_channels}"
            )

            # Update max range for manual ERD/ERS time spinboxes
            for sb in (
                self.erd_baseline_start,
                self.erd_baseline_end,
                self.erd_task_start,
                self.erd_task_end,
            ):
                sb.setRange(0.0, float(duration))

            # Set default manual baseline/task intervals (e.g., 1–2s baseline, 2–3s task)
            t0 = float(ds.time[0])
            self.erd_baseline_start.setValue(t0 + 1.0)
            self.erd_baseline_end.setValue(t0 + 2.0)
            self.erd_task_start.setValue(t0 + 2.0)
            self.erd_task_end.setValue(t0 + 3.0)

            # Set default trial-based settings: 5s trial, 0–2s baseline, 2–4s task
            self.trial_duration_spin.setValue(5.0)
            self.trial_base_start_spin.setValue(0.0)
            self.trial_base_end_spin.setValue(2.0)
            self.trial_task_start_spin.setValue(2.0)
            self.trial_task_end_spin.setValue(4.0)

            self.tabs.setCurrentWidget(self.raw_tab)

            # Initial update of all analysis tabs
            self._update_filtered_signals()
            self._refresh_topo_band_list()
            self._refresh_erd_band_list()
            self._update_band_power()
            self._update_spectrograms()
            self._update_band_timeline()
            self._update_erd_ers()
            self._update_band_timeline()
            self._update_erd_ers()
            self._update_topomap()
            
            # Update Real-time Visualization tab
            if hasattr(self, 'realtime_tab'):
                self.realtime_tab.set_dataset(self.dataset)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open CSV:\n{file_path}\n\nError: {e}",
            )

    # ------------------------------------------------------------------
    # Gemini Analysis Tab Setup and Methods
    # ------------------------------------------------------------------
    def _setup_gemini_tab(self):
        """Sets up the Gemini Analysis tab for AI-powered EEG interpretation."""
        self.gemini_tab = QWidget()
        self.gemini_layout = QVBoxLayout()

        # Check if Gemini is available
        if not GEMINI_AVAILABLE:
            unavailable_label = QLabel(
                "Gemini API is not available.\n\n"
                "Please run 'pip install google-generativeai' to install it."
            )
            unavailable_label.setStyleSheet("color: red; font-size: 12pt; padding: 20px;")
            unavailable_label.setWordWrap(True)
            self.gemini_layout.addWidget(unavailable_label)
            self.gemini_tab.setLayout(self.gemini_layout)
            self.tabs.addTab(self.gemini_tab, "Gemini Analysis")
            return

        # Initialize Gemini analyzer
        self.gemini_analyzer = GeminiEEGAnalyzer()

        # Header
        header_label = QLabel("Gemini AI EEG Signal Analysis")
        header_label.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 10px;")
        self.gemini_layout.addWidget(header_label)

        # API Key section
        api_group = QGroupBox("API Settings")
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("Gemini API Key:"))
        self.gemini_api_key_input = QLineEdit()
        self.gemini_api_key_input.setPlaceholderText("Enter your API key here...")
        self.gemini_api_key_input.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.gemini_api_key_input)
        api_group.setLayout(api_layout)
        self.gemini_layout.addWidget(api_group)

        # Channel selection section
        channel_group = QGroupBox("Channel Selection (Motor Cortex)")
        channel_layout = QHBoxLayout()
        
        self.gemini_ch_p3 = QCheckBox("P3")
        self.gemini_ch_p3.setChecked(True)
        self.gemini_ch_p4 = QCheckBox("P4")
        self.gemini_ch_p4.setChecked(True)
        self.gemini_ch_c3 = QCheckBox("C3")
        self.gemini_ch_c3.setChecked(True)
        self.gemini_ch_c4 = QCheckBox("C4")
        self.gemini_ch_c4.setChecked(True)
        
        channel_layout.addWidget(self.gemini_ch_p3)
        channel_layout.addWidget(self.gemini_ch_p4)
        channel_layout.addWidget(self.gemini_ch_c3)
        channel_layout.addWidget(self.gemini_ch_c4)
        channel_layout.addStretch()
        channel_group.setLayout(channel_layout)
        self.gemini_layout.addWidget(channel_group)

        # Time range section
        time_group = QGroupBox("Time Range (seconds)")
        time_layout = QHBoxLayout()
        
        time_layout.addWidget(QLabel("Start:"))
        self.gemini_time_start = QDoubleSpinBox()
        self.gemini_time_start.setRange(0.0, 9999.0)
        self.gemini_time_start.setDecimals(2)
        self.gemini_time_start.setValue(0.0)
        time_layout.addWidget(self.gemini_time_start)
        
        time_layout.addWidget(QLabel("End:"))
        self.gemini_time_end = QDoubleSpinBox()
        self.gemini_time_end.setRange(0.0, 9999.0)
        self.gemini_time_end.setDecimals(2)
        self.gemini_time_end.setValue(10.0)
        time_layout.addWidget(self.gemini_time_end)
        
        self.gemini_use_full = QCheckBox("Use full signal")
        self.gemini_use_full.setChecked(True)
        time_layout.addWidget(self.gemini_use_full)
        time_layout.addStretch()
        time_group.setLayout(time_layout)
        self.gemini_layout.addWidget(time_group)

        # Analyze button and progress
        button_layout = QHBoxLayout()
        self.gemini_analyze_btn = QPushButton("Analyze")
        self.gemini_analyze_btn.setStyleSheet(
            "font-size: 12pt; font-weight: bold; padding: 10px 20px; "
            "background-color: #4285f4; color: white; border-radius: 5px;"
        )
        self.gemini_analyze_btn.clicked.connect(self._run_gemini_analysis)
        button_layout.addWidget(self.gemini_analyze_btn)
        
        self.gemini_progress = QProgressBar()
        self.gemini_progress.setVisible(False)
        self.gemini_progress.setRange(0, 0)  # Indeterminate
        button_layout.addWidget(self.gemini_progress)
        button_layout.addStretch()
        self.gemini_layout.addLayout(button_layout)

        # Result display
        result_group = QGroupBox("Gemini Analysis Result")
        result_layout = QVBoxLayout()
        
        self.gemini_result_text = QTextEdit()
        self.gemini_result_text.setReadOnly(True)
        self.gemini_result_text.setPlaceholderText(
            "Analysis result will appear here...\n\n"
            "Usage:\n"
            "1. Load an EEG CSV file\n"
            "2. Enter your API key\n"
            "3. Select the channels to analyze\n"
            "4. Click 'Analyze' button"
        )
        self.gemini_result_text.setMinimumHeight(300)
        result_layout.addWidget(self.gemini_result_text)
        result_group.setLayout(result_layout)
        self.gemini_layout.addWidget(result_group)

        self.gemini_tab.setLayout(self.gemini_layout)
        self.tabs.addTab(self.gemini_tab, "Gemini Analysis")

    def _run_gemini_analysis(self):
        """Runs the Gemini API analysis on selected EEG channels."""
        if not GEMINI_AVAILABLE:
            QMessageBox.warning(
                self,
                "Error",
                "Gemini API is not available. Please run 'pip install google-generativeai'."
            )
            return

        if self.dataset is None:
            QMessageBox.warning(
                self,
                "Error",
                "Please load an EEG CSV file first."
            )
            return

        # Get API key
        api_key = self.gemini_api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(
                self,
                "Error",
                "Please enter your Gemini API key.\n\n"
                "Get an API key at: https://aistudio.google.com/app/apikey"
            )
            return

        # Get selected channels
        selected_channels = []
        if self.gemini_ch_p3.isChecked():
            selected_channels.append("P3")
        if self.gemini_ch_p4.isChecked():
            selected_channels.append("P4")
        if self.gemini_ch_c3.isChecked():
            selected_channels.append("C3")
        if self.gemini_ch_c4.isChecked():
            selected_channels.append("C4")

        # Filter to only available channels
        available_selected = [ch for ch in selected_channels if ch in self.dataset.available_channels]
        
        if not available_selected:
            QMessageBox.warning(
                self,
                "Error",
                f"Selected channels ({', '.join(selected_channels)}) are not available in this dataset.\n\n"
                f"Available channels: {', '.join(self.dataset.available_channels)}"
            )
            return

        # Get time range
        t_start = None
        t_end = None
        if not self.gemini_use_full.isChecked():
            t_start = float(self.gemini_time_start.value())
            t_end = float(self.gemini_time_end.value())
            if t_end <= t_start:
                QMessageBox.warning(
                    self,
                    "Error",
                    "End time must be greater than start time."
                )
                return

        # Show progress
        self.gemini_progress.setVisible(True)
        self.gemini_analyze_btn.setEnabled(False)
        self.gemini_result_text.setPlainText("Connecting to Gemini API and running analysis...\nThis may take a few seconds.")
        QApplication.processEvents()

        try:
            # Configure the analyzer
            self.gemini_analyzer.configure(api_key)

            # Compute band powers for the analysis
            df_filtered = filter_channels(
                self.dataset,
                channels=available_selected,
                apply_notch=self.filter_notch_enabled,
                bandpass=(self.filter_lowcut, self.filter_highcut),
                order=self.filter_order,
            )
            
            band_powers = compute_band_powers_for_channels(
                df_filtered, 
                fs=self.dataset.fs, 
                channels=available_selected
            )

            # Run the analysis
            formatted_data, analysis_result = self.gemini_analyzer.analyze_eeg(
                df=self.dataset.df,
                channels=available_selected,
                fs=self.dataset.fs,
                t_start=t_start,
                t_end=t_end,
                band_powers=band_powers
            )

            # Display the result
            result_text = f"=== Gemini EEG Analysis ===\n\n"
            result_text += f"Analyzed channels: {', '.join(available_selected)}\n"
            if t_start is not None and t_end is not None:
                result_text += f"Time range: {t_start:.2f}s - {t_end:.2f}s\n"
            else:
                result_text += f"Time range: Full signal\n"
            result_text += f"\n{'='*50}\n\n"
            result_text += analysis_result

            self.gemini_result_text.setPlainText(result_text)
            self.statusBar().showMessage("Gemini analysis completed!")

        except Exception as e:
            error_msg = f"An error occurred during analysis:\n\n{str(e)}"
            self.gemini_result_text.setPlainText(error_msg)
            QMessageBox.critical(
                self,
                "Error",
                error_msg
            )

        finally:
            self.gemini_progress.setVisible(False)
            self.gemini_analyze_btn.setEnabled(True)
