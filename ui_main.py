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
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.interpolate import griddata
from scipy.signal import welch

from config import APP_NAME, APP_VERSION, ELECTRODE_POSITIONS
from data_loader import EEGDataset
from processing import (
    filter_channels,
    compute_band_powers_for_channels,
    compute_spectrogram,
)


# ---------------------------------------------------------------------
# DataFrame -> Qt Table modeli
# ---------------------------------------------------------------------
class PandasModel(QAbstractTableModel):
    """Pandas DataFrame'i QTableView'de göstermek için basit model."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(section)


# ---------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------
class MainWindow(QMainWindow):
    """
    Sekmeli EEG analiz arayüzü:
      - Raw Data
      - Filtered Signals
      - Band Power
      - Spectrograms
      - Band Timeline (Mu)
      - ERD/ERS (manuel + trial-based)
      - Topographic Map (raw veya band-seçimli)
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1300, 750)

        # Dataset durumu
        self.dataset: Optional[EEGDataset] = None
        self.selected_row_index: Optional[int] = None
        self.selected_time: Optional[float] = None

        # Filtre ayarları (yalnızca "Filtered Signals", "Spectrograms", "Mu Timeline" için)
        self.filter_lowcut: float = 5.0
        self.filter_highcut: float = 35.0
        self.filter_notch_enabled: bool = True
        self.filter_order: int = 4

        # ---- Ana sekme widget'ı ----
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # -------- Raw Data tab --------
        self.raw_tab = QWidget()
        self.raw_layout = QVBoxLayout()

        raw_header = QHBoxLayout()
        raw_header_label = QLabel("Raw EEG Data (CSV tablosu)")
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

        # -------- Filtered Signals tab --------
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

        self.filtered_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.filtered_layout.addWidget(self.filtered_canvas)
        self.filtered_tab.setLayout(self.filtered_layout)
        self.tabs.addTab(self.filtered_tab, "Filtered Signals")

        # -------- Band Power tab --------
        self.bandpower_tab = QWidget()
        self.bandpower_layout = QVBoxLayout()

        bp_ctrl_layout = QHBoxLayout()
        self.bandpower_use_window = QCheckBox("Use selected row window")
        self.bandpower_use_window.setChecked(False)

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

        self.bandpower_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.bandpower_layout.addWidget(self.bandpower_canvas)
        self.bandpower_tab.setLayout(self.bandpower_layout)
        self.tabs.addTab(self.bandpower_tab, "Band Power")

        # -------- Spectrograms tab --------
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

        self.spectrogram_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.spectrogram_layout.addWidget(self.spectrogram_canvas)
        self.spectrogram_tab.setLayout(self.spectrogram_layout)
        self.tabs.addTab(self.spectrogram_tab, "Spectrograms")

        # -------- Band Timeline (Mu) tab --------
        self.timeline_tab = QWidget()
        self.timeline_layout = QVBoxLayout()

        tl_ctrl = QHBoxLayout()
        self.timeline_window_label = QLabel("Window (s):")
        self.timeline_window_spin = QDoubleSpinBox()
        self.timeline_window_spin.setRange(0.2, 10.0)
        self.timeline_window_spin.setSingleStep(0.2)
        self.timeline_window_spin.setValue(2.0)

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

        self.timeline_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.timeline_layout.addWidget(self.timeline_canvas)
        self.timeline_tab.setLayout(self.timeline_layout)
        self.tabs.addTab(self.timeline_tab, "Band Timeline (Mu 8–12 Hz)")

        # -------- ERD/ERS tab --------
        self.erd_tab = QWidget()
        self.erd_layout = QVBoxLayout()

        # ---- Manuel ERD/ERS kontrol bloğu ----
        erd_ctrl = QHBoxLayout()
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

        # Manuel ERD açıklama etiketi
        self.erd_info_label = QLabel(
            "Öneri: Motor imagery analizi için genelde Mu (8–12 Hz) ve Beta (13–30 Hz) "
            "bantları kullanılır.\n"
            "Pozitif ERD/ERS = güçte azalma (ERD), negatif ERD/ERS = güçte artış (ERS)."
        )
        self.erd_info_label.setWordWrap(True)
        self.erd_info_label.setStyleSheet("color: gray; font-size: 10pt;")
        self.erd_layout.addWidget(self.erd_info_label)

        self.erd_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.erd_layout.addWidget(self.erd_canvas)

        # ---- Trial-based ERD/ERS kontrol bloğu ----
        trial_title = QLabel("Trial-based ERD/ERS (bilimsel analiz)")
        trial_title.setStyleSheet("font-weight: bold; margin-top: 12px;")
        self.erd_layout.addWidget(trial_title)

        trial_ctrl = QHBoxLayout()

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

        self.erd_trial_info_label = QLabel(
            "Not: Kayıt, ardışık trial'lara bölünür. Her trial için ERD/ERS hesaplanır ve "
            "kanal bazında ortalaması alınır. Trial pencereleri relative sürelere göre ayarlanır."
        )
        self.erd_trial_info_label.setWordWrap(True)
        self.erd_trial_info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.erd_layout.addWidget(self.erd_trial_info_label)

        self.erd_trial_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.erd_layout.addWidget(self.erd_trial_canvas)

        self.erd_tab.setLayout(self.erd_layout)
        self.tabs.addTab(self.erd_tab, "ERD/ERS")

        # -------- Topographic Map tab --------
        self.topo_tab = QWidget()
        self.topo_layout = QVBoxLayout()

        topo_ctrl_layout = QHBoxLayout()
        topo_ctrl_layout.addWidget(QLabel("Topomap mode:"))
        self.topo_mode_combo = QComboBox()
        self.topo_mode_combo.addItem("Raw amplitude", userData=None)
        topo_ctrl_layout.addWidget(self.topo_mode_combo)

        self.topo_info_btn = QPushButton("Info")
        self.topo_info_btn.clicked.connect(self._show_topomap_info)
        topo_ctrl_layout.addWidget(self.topo_info_btn)

        topo_ctrl_layout.addStretch()
        self.topo_layout.addLayout(topo_ctrl_layout)

        self.topo_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.topo_layout.addWidget(self.topo_canvas)
        self.topo_tab.setLayout(self.topo_layout)
        self.tabs.addTab(self.topo_tab, "Topographic Map")

        # Kontrol bağlantıları
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

        for sb in (
            self.erd_baseline_start,
            self.erd_baseline_end,
            self.erd_task_start,
            self.erd_task_end,
        ):
            sb.valueChanged.connect(lambda _: self._update_erd_ers())

        self.erd_band_combo.currentIndexChanged.connect(
            lambda _: self._update_erd_ers()
        )

        self.trial_compute_button.clicked.connect(self._compute_trial_erd_ers)

        # Menü
        self._create_menu()

        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Menü
    # ------------------------------------------------------------------
    def _create_menu(self):
        menu = self.menuBar()

        # File
        m_file = menu.addMenu("File")
        act_open = QAction("Open CSV...", self)
        act_open.triggered.connect(self.open_csv)
        m_file.addAction(act_open)

        m_file.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        # Analysis
        m_analysis = menu.addMenu("Analysis")
        act_filter = QAction("Filter Settings...", self)
        act_filter.triggered.connect(self._dialog_filter_settings)
        m_analysis.addAction(act_filter)

        # Help
        m_help = menu.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _show_about(self):
        QMessageBox.information(
            self,
            "About",
            f"{APP_NAME} v{APP_VERSION}\n\nEEG motor aktivite analiz aracı.",
        )

    # ------------------------------------------------------------------
    # Filter settings dialog
    # ------------------------------------------------------------------
    def _dialog_filter_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Filter Settings")
        form = QFormLayout(dlg)

        low_spin = QDoubleSpinBox()
        low_spin.setRange(0.1, 100.0)
        low_spin.setValue(self.filter_lowcut)

        high_spin = QDoubleSpinBox()
        high_spin.setRange(0.1, 200.0)
        high_spin.setValue(self.filter_highcut)

        order_spin = QSpinBox()
        order_spin.setRange(1, 10)
        order_spin.setValue(self.filter_order)

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
            self.filter_lowcut = float(low_spin.value())
            self.filter_highcut = float(high_spin.value())
            self.filter_order = int(order_spin.value())
            self.filter_notch_enabled = notch_cb.isChecked()

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
    # Raw Data seçim takibi
    # ------------------------------------------------------------------
    def _connect_raw_selection(self):
        sel = self.raw_table.selectionModel()
        if sel is None:
            return
        try:
            sel.selectionChanged.disconnect()
        except TypeError:
            pass
        sel.selectionChanged.connect(self._on_raw_selection_changed)

    def _on_raw_selection_changed(self, *_):
        if self.dataset is None:
            return

        sel = self.raw_table.selectionModel()
        if sel is None:
            return
        indexes = sel.selectedIndexes()
        if not indexes:
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

        row_idx = indexes[0].row()
        self.selected_row_index = row_idx

        row = self.dataset.df.iloc[row_idx]
        if "time" in row.index:
            self.selected_time = float(row["time"])
        else:
            self.selected_time = None

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

        self._update_filtered_signals()
        self._update_spectrograms()
        self._update_band_power()
        self._update_band_timeline()
        self._update_erd_ers()
        self._update_topomap()

    # ------------------------------------------------------------------
    # Filtered Signals (global filtre)
    # ------------------------------------------------------------------
    def _update_filtered_signals(self):
        if self.dataset is None:
            return

        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels

        if not channels:
            self.statusBar().showMessage("No channels available to plot.")
            return

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

        t = df_filt["time"].to_numpy()
        for ch in channels:
            ax.plot(t, df_filt[ch].to_numpy(), label=ch)

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
    # Band Power (ham sinyal üzerinden, her band için ayrı subplot)
    # ------------------------------------------------------------------
    def _update_band_power(self):
        if self.dataset is None:
            return

        # Öncelikli olarak C3–C4, yoksa mevcut tüm kanallar
        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels

        if not channels:
            self.statusBar().showMessage("No channels available for band power.")
            return

        ds = self.dataset
        window_info = "full signal"

        # İstenirse seçili satır etrafında pencere kullan
        if self.bandpower_use_window.isChecked() and self.selected_time is not None:
            win = float(self.bandpower_window_spin.value())
            half = win / 2.0
            t0 = max(ds.time[0], self.selected_time - half)
            t1 = min(ds.time[-1], self.selected_time + half)
            if t1 <= t0:
                self.statusBar().showMessage("Invalid window for band power.")
                return

            seg = ds.get_segment(t0, t1)
            ds = EEGDataset(
                filepath=ds.filepath,
                df=seg.reset_index(drop=True),
                fs=ds.fs,
                time=seg["time"].to_numpy(),
                available_channels=ds.available_channels,
            )
            window_info = f"window around {self.selected_time:.3f} s (len={win:.2f} s)"

        # Ham sinyalden band power hesabı
        df_in = ds.df[["time"] + channels].copy()
        band_powers = compute_band_powers_for_channels(
            df_in, fs=ds.fs, channels=channels
        )

        if not band_powers:
            self.statusBar().showMessage("No band power data computed.")
            return

        band_names = list(next(iter(band_powers.values())).keys())
        n_bands = len(band_names)
        if n_bands == 0:
            self.statusBar().showMessage("No bands found for band power.")
            return

        fig = self.bandpower_canvas.figure
        fig.clear()

        # Her band için ayrı eksen – yan yana sub-plotlar
        axes = fig.subplots(1, n_bands, squeeze=False)[0]

        x = np.arange(len(channels))

        for j, (band, ax) in enumerate(zip(band_names, axes)):
            vals = [band_powers[ch][band] for ch in channels]

            ax.bar(x, vals)
            ax.set_xticks(x)
            ax.set_xticklabels(channels)

            ax.set_title(band)

            if j == 0:
                ax.set_ylabel("Power (a.u.)")
            else:
                # Y eksen yazılarını temizle, sadece ilk grafikte kalsın
                ax.set_yticklabels([])

        fig.suptitle(
            f"Band Power ({window_info})\n(computed from raw signal)",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        self.bandpower_canvas.draw()

        self.statusBar().showMessage(
            f"Band power updated | channels={channels} | {window_info}"
        )

    # ------------------------------------------------------------------
    # Spectrograms (global filtreli)
    # ------------------------------------------------------------------
    def _update_spectrograms(self):
        if self.dataset is None:
            return

        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels[:2]

        if not channels:
            self.statusBar().showMessage("No channels available for spectrogram.")
            return

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
            ax = fig.add_subplot(n_ch, 1, i)
            x = df_filt[ch].to_numpy(dtype=float)
            f, t, Sxx = compute_spectrogram(x, fs=self.dataset.fs)
            Sxx_db = 10 * np.log10(Sxx + 1e-12)

            ax.pcolormesh(t, f, Sxx_db, shading="auto")
            ax.set_ylabel("Freq (Hz)")
            ax.set_title(f"Spectrogram - {ch}")
            ax.set_ylim(0, 120)

            if len(t) > 1:
                ax.set_xlim(t[0], t[-1])

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
    # Band Timeline (Mu 8–12 Hz) – global filtre + Welch
    # ------------------------------------------------------------------
    def _update_band_timeline(self):
        if self.dataset is None:
            return

        preferred = ["C3", "C4"]
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels

        if not channels:
            self.statusBar().showMessage("No channels available for band timeline.")
            return

        df_filt = filter_channels(
            self.dataset,
            channels=channels,
            apply_notch=self.filter_notch_enabled,
            bandpass=(self.filter_lowcut, self.filter_highcut),
            order=self.filter_order,
        )

        fs = self.dataset.fs
        t = df_filt["time"].to_numpy()

        win_len = float(self.timeline_window_spin.value())
        step_len = float(self.timeline_step_spin.value())

        if win_len <= 0 or step_len <= 0:
            self.statusBar().showMessage("Invalid window/step for band timeline.")
            return

        win_samples = int(win_len * fs)
        step_samples = int(step_len * fs)
        if win_samples < 8 or step_samples < 1:
            self.statusBar().showMessage("Window too small for band timeline.")
            return

        n_samples = len(t)
        if n_samples < win_samples:
            self.statusBar().showMessage("Signal is shorter than the window.")
            return

        mu_low, mu_high = 8.0, 12.0

        times = []
        powers_per_channel = {ch: [] for ch in channels}

        start_idx = 0
        while start_idx + win_samples <= n_samples:
            end_idx = start_idx + win_samples
            seg_time = t[start_idx:end_idx]
            center_time = (seg_time[0] + seg_time[-1]) / 2.0
            times.append(center_time)

            for ch in channels:
                seg = df_filt[ch].to_numpy(dtype=float)[start_idx:end_idx]
                nper = min(win_samples, 256)
                f, Pxx = welch(seg, fs=fs, nperseg=nper)
                mask = (f >= mu_low) & (f <= mu_high)
                mu_power = Pxx[mask].sum()
                powers_per_channel[ch].append(mu_power)

            start_idx += step_samples

        if not times:
            self.statusBar().showMessage("No timeline points computed.")
            return

        times = np.array(times)
        fig = self.timeline_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        for ch in channels:
            p = np.array(powers_per_channel[ch])
            ax.plot(times, p, label=ch)

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
    # ERD/ERS band listesini güncelle (ham sinyal üzerinden)
    # ------------------------------------------------------------------
    def _refresh_erd_band_list(self):
        """ERD/ERS için band listesini (güzel etiketlerle) günceller."""
        if self.dataset is None:
            return

        channels = self._erd_channels()
        if not channels:
            return

        ds = self.dataset
        df_in = ds.df[["time"] + channels].copy()
        band_powers = compute_band_powers_for_channels(
            df_in, fs=ds.fs, channels=channels
        )
        if not band_powers:
            return

        band_names = list(next(iter(band_powers.values())).keys())

        pretty_labels = {
            "delta": "Delta (0.5–4 Hz) – slow/artefact",
            "theta": "Theta (4–8 Hz)",
            "alpha_mu": "Mu (8–12 Hz) – motor imagery",
            "beta": "Beta (13–30 Hz) – motor imagery",
            "gamma": "Gamma (30–45 Hz)",
        }

        self.erd_band_combo.blockSignals(True)
        self.erd_band_combo.clear()

        default_index = 0
        for i, name in enumerate(band_names):
            label = pretty_labels.get(name, name)
            self.erd_band_combo.addItem(label, userData=name)
            if name == "alpha_mu":
                default_index = i

        self.erd_band_combo.setCurrentIndex(default_index)
        self.erd_band_combo.blockSignals(False)

    def _erd_channels(self):
        preferred = ["C3", "C4"]
        if self.dataset is None:
            return []
        channels = [c for c in preferred if c in self.dataset.available_channels]
        if not channels:
            channels = self.dataset.available_channels
        return channels

    # ------------------------------------------------------------------
    # ERD/ERS hesaplama ve çizim (manuel, ham sinyal band power)
    # ------------------------------------------------------------------
    def _update_erd_ers(self):
        if self.dataset is None:
            return
        if self.erd_band_combo.count() == 0:
            return

        idx = self.erd_band_combo.currentIndex()
        if idx < 0:
            return

        band_name = self.erd_band_combo.itemData(idx)  # internal key
        band_label = self.erd_band_combo.currentText()  # kullanıcıya gösterilen text
        if not band_name:
            return

        t_min = float(self.dataset.time[0])
        t_max = float(self.dataset.time[-1])

        bs = float(self.erd_baseline_start.value())
        be = float(self.erd_baseline_end.value())
        ts = float(self.erd_task_start.value())
        te = float(self.erd_task_end.value())

        if not (t_min <= bs < be <= t_max and t_min <= ts < te <= t_max):
            self.statusBar().showMessage("Invalid baseline/task intervals for ERD/ERS.")
            return

        channels = self._erd_channels()
        if not channels:
            self.statusBar().showMessage("No channels for ERD/ERS.")
            return

        def band_power_interval(t0, t1):
            seg = self.dataset.get_segment(t0, t1)
            ds = EEGDataset(
                filepath=self.dataset.filepath,
                df=seg.reset_index(drop=True),
                fs=self.dataset.fs,
                time=seg["time"].to_numpy(),
                available_channels=self.dataset.available_channels,
            )
            df_in = ds.df[["time"] + channels].copy()
            bp = compute_band_powers_for_channels(
                df_in, fs=ds.fs, channels=channels
            )
            result = {}
            for ch in channels:
                ch_bp = bp.get(ch, {})
                if band_name in ch_bp:
                    result[ch] = float(ch_bp[band_name])
            return result

        baseline_pw = band_power_interval(bs, be)
        task_pw = band_power_interval(ts, te)

        if not baseline_pw or not task_pw:
            self.statusBar().showMessage("Could not compute ERD/ERS band powers.")
            return

        erd_values = {}
        for ch in channels:
            pb = baseline_pw.get(ch, None)
            pt = task_pw.get(ch, None)
            if pb is None or pt is None or pb == 0:
                erd_values[ch] = np.nan
            else:
                erd_values[ch] = (pb - pt) / pb * 100.0  # azalma pozitif

        fig = self.erd_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        ch_list = list(channels)
        x = np.arange(len(ch_list))
        vals = [erd_values[ch] for ch in ch_list]

        bars = ax.bar(x, vals)
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(ch_list)
        ax.set_ylabel("ERD/ERS (%)")
        ax.set_title(
            f"Manual ERD/ERS – {band_label}\n"
            f"Baseline: {bs:.2f}-{be:.2f}s, Task: {ts:.2f}-{te:.2f}s"
        )

        for xi, bar, ch in zip(x, bars, ch_list):
            value = bar.get_height()
            if np.isnan(value):
                label = "NaN"
                y = 0
                va = "bottom"
            else:
                label = f"{value:.1f}%"
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
        self.erd_canvas.draw()

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
    # Trial-based ERD/ERS (ham sinyal, birden çok trial)
    # ------------------------------------------------------------------
    def _compute_trial_erd_ers(self):
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

        trial_dur = float(self.trial_duration_spin.value())
        b_rel_s = float(self.trial_base_start_spin.value())
        b_rel_e = float(self.trial_base_end_spin.value())
        t_rel_s = float(self.trial_task_start_spin.value())
        t_rel_e = float(self.trial_task_end_spin.value())

        if trial_dur <= 0:
            self.statusBar().showMessage("Trial duration must be > 0.")
            return
        if not (0.0 <= b_rel_s < b_rel_e <= trial_dur):
            self.statusBar().showMessage("Invalid baseline window inside trial.")
            return
        if not (0.0 <= t_rel_s < t_rel_e <= trial_dur):
            self.statusBar().showMessage("Invalid task window inside trial.")
            return

        n_trials = int(np.floor(total_duration / trial_dur))
        if n_trials < 1:
            self.statusBar().showMessage("Recording is too short for even one trial.")
            return

        channels = self._erd_channels()
        if not channels:
            self.statusBar().showMessage("No channels available for trial-based ERD/ERS.")
            return

        def band_power_interval_abs(t_start_abs, t_end_abs):
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

        for k in range(n_trials):
            trial_start = t0 + k * trial_dur
            base_s = trial_start + b_rel_s
            base_e = trial_start + b_rel_e
            task_s = trial_start + t_rel_s
            task_e = trial_start + t_rel_e

            base_pw = band_power_interval_abs(base_s, base_e)
            task_pw = band_power_interval_abs(task_s, task_e)

            for ch in channels:
                pb = base_pw.get(ch, None)
                pt = task_pw.get(ch, None)
                if pb is None or pt is None or pb == 0:
                    trial_erd[ch].append(np.nan)
                else:
                    val = (pb - pt) / pb * 100.0
                    trial_erd[ch].append(val)

        mean_erd = {}
        std_erd = {}
        n_valid = {}
        for ch in channels:
            arr = np.array(trial_erd[ch], dtype=float)
            valid = ~np.isnan(arr)
            if not np.any(valid):
                mean_erd[ch] = np.nan
                std_erd[ch] = np.nan
                n_valid[ch] = 0
            else:
                mean_erd[ch] = float(np.nanmean(arr))
                std_erd[ch] = float(np.nanstd(arr))
                n_valid[ch] = int(valid.sum())

        fig = self.erd_trial_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        ch_list = list(channels)
        x = np.arange(len(ch_list))
        means = [mean_erd[ch] for ch in ch_list]
        stds = [std_erd[ch] for ch in ch_list]

        bars = ax.bar(x, means, yerr=stds, capsize=6)
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(ch_list)
        ax.set_ylabel("ERD/ERS (%)")
        ax.set_title(
            f"Trial-based ERD/ERS – {band_label}\n"
            f"Trial duration={trial_dur:.2f}s, "
            f"Baseline={b_rel_s:.2f}-{b_rel_e:.2f}s, "
            f"Task={t_rel_s:.2f}-{t_rel_e:.2f}s (relative to trial)"
        )

        for xi, bar, ch in zip(x, bars, ch_list):
            value = bar.get_height()
            ntr = n_valid.get(ch, 0)
            if np.isnan(value) or ntr == 0:
                label = "NaN"
                y = 0
                va = "bottom"
            else:
                label = f"{value:.1f}% (N={ntr})"
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
    # Topomap için band listesi (ham sinyal üzerinden)
    # ------------------------------------------------------------------
    def _refresh_topo_band_list(self):
        if self.dataset is None:
            return

        channels = [
            ch
            for ch in ELECTRODE_POSITIONS.keys()
            if ch in self.dataset.available_channels
        ]
        if not channels:
            return

        ds = self.dataset
        df_in = ds.df[["time"] + channels].copy()
        band_powers = compute_band_powers_for_channels(
            df_in, fs=ds.fs, channels=channels
        )
        if not band_powers:
            return

        band_names = list(next(iter(band_powers.values())).keys())

        self.topo_mode_combo.blockSignals(True)
        self.topo_mode_combo.clear()
        self.topo_mode_combo.addItem("Raw amplitude", userData=None)
        for name in band_names:
            self.topo_mode_combo.addItem(f"Band power: {name}", userData=name)
        self.topo_mode_combo.setCurrentIndex(0)
        self.topo_mode_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Topographic Map (raw veya band-seçimli, ham sinyal band power)
    # ------------------------------------------------------------------
    def _update_topomap(self):
        if self.dataset is None:
            return

        selected_band = None
        if hasattr(self, "topo_mode_combo"):
            idx = self.topo_mode_combo.currentIndex()
            if idx >= 0:
                selected_band = self.topo_mode_combo.itemData(idx)

        channel_values = {}

        # RAW AMPLITUDE MODU
        if selected_band is None:
            if self.selected_row_index is None:
                self.statusBar().showMessage(
                    "Select a row for raw-amplitude topomap."
                )
                return

            row = self.dataset.df.iloc[self.selected_row_index]
            for ch, (xx, yy) in ELECTRODE_POSITIONS.items():
                if ch in self.dataset.available_channels and ch in row.index:
                    val = row[ch]
                    if pd.isna(val):
                        continue
                    channel_values[ch] = float(val)

        # BAND POWER MODU (ham sinyal)
        else:
            channels = [
                ch
                for ch in ELECTRODE_POSITIONS.keys()
                if ch in self.dataset.available_channels
            ]
            if not channels:
                self.statusBar().showMessage("No electrodes available for topomap.")
                return

            ds = self.dataset
            if self.bandpower_use_window.isChecked() and self.selected_time is not None:
                win = float(self.bandpower_window_spin.value())
                half = win / 2.0
                t0 = max(ds.time[0], self.selected_time - half)
                t1 = min(ds.time[-1], self.selected_time + half)
                if t1 <= t0:
                    self.statusBar().showMessage("Invalid window for topomap.")
                    return

                seg = ds.get_segment(t0, t1)
                ds = EEGDataset(
                    filepath=ds.filepath,
                    df=seg.reset_index(drop=True),
                    fs=ds.fs,
                    time=seg["time"].to_numpy(),
                    available_channels=ds.available_channels,
                )

            df_in = ds.df[["time"] + channels].copy()
            band_powers = compute_band_powers_for_channels(
                df_in, fs=ds.fs, channels=channels
            )
            if not band_powers:
                self.statusBar().showMessage("No band power data for topomap.")
                return

            band_key = str(selected_band)
            for ch in channels:
                ch_bp = band_powers.get(ch, {})
                if band_key in ch_bp:
                    channel_values[ch] = float(ch_bp[band_key])

        if not channel_values:
            self.statusBar().showMessage("No channel values to plot on topomap.")
            return

        xs, ys, zs = [], [], []
        for ch, (xx, yy) in ELECTRODE_POSITIONS.items():
            if ch in channel_values:
                xs.append(xx)
                ys.append(yy)
                zs.append(channel_values[ch])

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        zs = np.array(zs, dtype=float)

        zs = np.abs(zs)
        if zs.size == 0:
            self.statusBar().showMessage("No valid values for topomap.")
            return
        zmin, zmax = zs.min(), zs.max()
        if np.isclose(zmin, zmax):
            zs_norm = np.ones_like(zs) * 0.5
        else:
            zs_norm = (zs - zmin) / (zmax - zmin)

        grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]

        if len(xs) < 4:
            zi = griddata((xs, ys), zs_norm, (grid_x, grid_y), method="nearest")
        else:
            try:
                zi = griddata((xs, ys), zs_norm, (grid_x, grid_y), method="linear")
            except Exception:
                zi = griddata((xs, ys), zs_norm, (grid_x, grid_y), method="nearest")

        r = np.sqrt(grid_x**2 + grid_y**2)
        mask = r > 1.0
        zi_masked = np.ma.array(zi, mask=mask)

        if np.all(np.isnan(zi_masked)):
            self.statusBar().showMessage("Topomap: interpolation failed (all NaN).")
            return

        fig = self.topo_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        cf = ax.contourf(
            grid_x,
            grid_y,
            zi_masked,
            levels=40,
            cmap="RdYlBu_r",
            vmin=0.0,
            vmax=1.0,
        )

        head = plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=2)
        ax.add_patch(head)

        nose_x = [0.0, -0.08, 0.08]
        nose_y = [1.0, 1.12, 1.12]
        ax.plot(nose_x, nose_y, "k-", linewidth=2)

        ear_y = 0.0
        ear_r = 0.08
        ax.add_patch(
            plt.Circle((-1.0, ear_y), ear_r, fill=False, color="black", linewidth=2)
        )
        ax.add_patch(
            plt.Circle((1.0, ear_y), ear_r, fill=False, color="black", linewidth=2)
        )

        for ch, (xx, yy) in ELECTRODE_POSITIONS.items():
            if ch in channel_values:
                ax.plot(xx, yy, "wo", markersize=6, markeredgecolor="black")
                ax.text(xx + 0.03, yy + 0.03, ch, fontsize=9, color="black")

        ax.set_aspect("equal")
        ax.set_axis_off()

        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Relative activity (a.u.)")

        fig.tight_layout()
        self.topo_canvas.draw()

        self.statusBar().showMessage("Topographic map updated.")

    # ------------------------------------------------------------------
    # Info pencereleri
    # ------------------------------------------------------------------
    def _show_raw_info(self):
        text = (
            "Raw Data sekmesi:\n\n"
            "- Yüklediğin .csv dosyasındaki ham EEG örneklerini satır/sütun olarak gösterir.\n"
            "- 'time' sütunu saniye cinsinden zaman eksenini temsil eder.\n"
            "- Diğer sütunlar elektrot kanallarıdır (C3, C4, vb.).\n"
            "- Tablo üzerinde bir satır seçtiğinde:\n"
            "  • Alttaki etiket satırın zamanını ve kanal genliklerini gösterir.\n"
            "  • Diğer sekmelerde kırmızı dikey çizgi bu zamana hizalanır.\n"
            "- Bu sekme aynı zamanda kayıt içinden belli bir anı seçip, o ana göre\n"
            "  filtreli sinyal, spektrogram, band power ve topomap'i incelemeni sağlar."
        )
        QMessageBox.information(self, "Info – Raw Data", text)

    def _show_filtered_info(self):
        text = (
            "Filtered Signals sekmesi:\n\n"
            "- Seçilen kanallar (tercihen C3, C4) üzerine notch + band-pass filtrasyonu uygular.\n"
            "- Amaç: Ham EEG'deki DC drift, çok düşük frekanslar ve yüksek frekans gürültüsünü\n"
            "  bastırıp motor imagery ile ilgili frekans aralığını daha net görmek.\n"
            "- Filtre ayarlarını üst menüdeki 'Analysis → Filter Settings' kısmından\n"
            "  (notch açık/kapalı, lowcut, highcut, order) değiştirebilirsin.\n"
            "- Kırmızı dikey çizgi seçili satırın (Raw Data sekmesindeki) zamanını gösterir.\n"
            "- Bu sekme, sinyalin zaman alanında filtrelenmiş halini yorumlamak için kullanılır."
        )
        QMessageBox.information(self, "Info – Filtered Signals", text)

    def _show_bandpower_info(self):
        text = (
            "Band Power sekmesi:\n\n"
            "- Seçilen kanallar için EEG frekans bantlarına (delta, theta, mu, beta, gamma vb.)\n"
            "  ait toplam güçleri gösterir.\n"
            "- Güç hesaplama ham sinyal üzerinden yapılır (filtreli sinyal değil).\n"
            "- 'Use selected row window' işaretliyse:\n"
            "  • Seçili zamanın etrafında belirlediğin süre (Window s) kadar bir pencere alınır,\n"
            "    band power sadece bu pencere içinde hesaplanır.\n"
            "  • Böylece dinlenme, görev, artefakt gibi farklı anları karşılaştırabilirsin.\n"
            "- İşaretli değilse, band power tüm kayıt üzerinden hesaplanır.\n"
            "- Mu (8–12 Hz) ve Beta (13–30 Hz) bantları özellikle motor imagery çalışmalarında\n"
            "  önemli olup ERD/ERS analizine temel oluşturur."
        )
        QMessageBox.information(self, "Info – Band Power", text)

    def _show_spectrogram_info(self):
        text = (
            "Spectrograms sekmesi:\n\n"
            "- Seçilen kanallar için zaman-frekans gösterimi (spektrogram) üretir.\n"
            "- Yatay eksen: Zaman (s), dikey eksen: Frekans (Hz), renk: Güç (dB).\n"
            "- Filtre ayarları (notch + band-pass) burada da geçerlidir; üst menüden\n"
            "  filtreyi değiştirerek yeniden hesaplayabilirsin.\n"
            "- Kırmızı dikey çizgi, Raw Data sekmesindeki seçili satırın zamanını gösterir.\n"
            "- Motor imagery analizinde, belirli zaman aralıklarında mu/beta bandındaki\n"
            "  güç değişimlerini görselleştirmek için kullanılır.\n"
            "- Y eksenindeki üst sınır şu anda 120 Hz'e kadar ayarlanmıştır."
        )
        QMessageBox.information(self, "Info – Spectrograms", text)

    def _show_timeline_info(self):
        text = (
            "Band Timeline (Mu) sekmesi:\n\n"
            "- Seçilen kanallar için Mu bandı (8–12 Hz) gücünün zamana göre değişimini gösterir.\n"
            "- 'Window (s)': Her noktayı hesaplarken kullanılan pencere uzunluğudur.\n"
            "- 'Step (s)': Pencerenin kayma adımıdır; ne kadar sık hesap yapılacağını belirler.\n"
            "- Örn. window=2s, step=0.2s ise, her 0.2 saniyede bir 2 saniyelik pencere üzerinden\n"
            "  Mu bandı gücü hesaplanır.\n"
            "- Motor imagery'de C3/C4 üzerindeki Mu gücünün zaman içinde nasıl azalıp arttığını\n"
            "  gözlemlemek için kullanılır.\n"
            "- Kırmızı dikey çizgi seçili satır zamanını gösterir ve diğer sekmelerle senkronizedir."
        )
        QMessageBox.information(self, "Info – Band Timeline (Mu)", text)

    def _show_erd_info(self):
        text = (
            "ERD/ERS – Manuel sekmesi:\n\n"
            "- Baseline ve Task aralıklarını zaman ekseninden manuel olarak seçerek,\n"
            "  her kanal için ERD/ERS (%) hesaplar.\n"
            "- Kullanılan formül: ERD% = (P_baseline − P_task) / P_baseline × 100\n"
            "  • Pozitif değer → Güçte azalma (ERD – event-related desynchronization)\n"
            "  • Negatif değer → Güçte artış (ERS – event-related synchronization)\n"
            "- 'Band' kısmından analiz edilecek frekans bandını seçebilirsin:\n"
            "  • Mu (8–12 Hz) ve Beta (13–30 Hz) motor imagery için en kritik bantlardır.\n"
            "- Bu sekme özellikle tek bir trial benzeri durum için hızlı ERD/ERS denemeleri\n"
            "  yapmak ve farklı zaman aralıklarını manuel kıyaslamak için kullanılır."
        )
        QMessageBox.information(self, "Info – Manual ERD/ERS", text)

    def _show_trial_erd_info(self):
        text = (
            "Trial-based ERD/ERS sekmesi:\n\n"
            "- Kayıt süresini eşit uzunluklu trial'lara böler ve her trial için ayrı ERD/ERS\n"
            "  hesaplayıp kanal bazında ortalama + standart sapma üretir.\n"
            "- 'Trial duration (s)': Her bir trial'ın süresi.\n"
            "- 'Baseline in trial (s)': Trial içindeki relatif baseline penceresi.\n"
            "- 'Task in trial (s)': Trial içindeki relatif görev penceresi.\n"
            "- Örn. trial duration = 5 s, baseline = 0–2 s, task = 2–4 s:\n"
            "  • Kayıt 0–5, 5–10, 10–15 ... saniyelik bloklara ayrılır.\n"
            "  • Her blokta 0–2 s baseline, 2–4 s task olarak kullanılır.\n"
            "- Bu yöntem literatürdeki motor imagery çalışmalarında kullanılan\n"
            "  \"trial-based ERD/ERS\" yaklaşımına yakındır.\n"
            "- Grafikte: Barlar kanal başına ortalama ERD/ERS, error bar'lar standart sapmayı\n"
            "  gösterir; üstlerinde N (geçerli trial sayısı) bilgisi yer alır."
        )
        QMessageBox.information(self, "Info – Trial-based ERD/ERS", text)

    def _show_topomap_info(self):
        text = (
            "Topographic Map sekmesi:\n\n"
            "- Kafa üzerinde elektrotların (C3, C4 vb.) yaklaşık konumlarını kullanarak\n"
            "  aktiviteyi 2D bir beyin haritası şeklinde gösterir.\n"
            "- 'Topomap mode: Raw amplitude':\n"
            "  • Raw Data sekmesinde seçili satırdaki anlık genlikleri kullanır.\n"
            "  • Sıcak renkler (kırmızı/sarı) yüksek aktiviteyi, soğuk renkler (mavi) düşük\n"
            "    aktiviteyi temsil eder.\n"
            "- 'Topomap mode: Band power: ...':\n"
            "  • Tüm kayıt (veya seçili pencere) içerisindeki ortalama band gücünü kullanır.\n"
            "  • 'Use selected row window' açıksa, Raw Data'daki seçili zamanın etrafındaki\n"
            "    pencere üzerinden band power hesaplanır.\n"
            "- Bu sekme, hangi beynin hangi bölgesinde (ör. C3 vs C4) göreceli olarak daha fazla\n"
            "  aktivite olduğunu görselleştirmek için kullanılır.\n"
            "- Haritada kafa sınırı, kulaklar ve burun çizgisi yaklaşık oryantasyon için eklenmiştir."
        )
        QMessageBox.information(self, "Info – Topographic Map", text)

    # ------------------------------------------------------------------
    # CSV açma
    # ------------------------------------------------------------------
    def open_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open EEG CSV file",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return

        try:
            ds = EEGDataset.from_csv(file_path)
            self.dataset = ds

            model = PandasModel(ds.df)
            self.raw_table.setModel(model)
            self._connect_raw_selection()

            duration = ds.time[-1] - ds.time[0]
            self.statusBar().showMessage(
                f"Loaded: {file_path} | fs={ds.fs} Hz | "
                f"duration={duration:.2f} s | channels={ds.available_channels}"
            )

            for sb in (
                self.erd_baseline_start,
                self.erd_baseline_end,
                self.erd_task_start,
                self.erd_task_end,
            ):
                sb.setRange(0.0, float(duration))

            # Varsayılan manuel baseline / task (1–2 s baseline, 2–3 s task)
            t0 = float(ds.time[0])
            self.erd_baseline_start.setValue(t0 + 1.0)
            self.erd_baseline_end.setValue(t0 + 2.0)
            self.erd_task_start.setValue(t0 + 2.0)
            self.erd_task_end.setValue(t0 + 3.0)

            # Trial-based default: 5 s trial, 0–2 s baseline, 2–4 s task
            self.trial_duration_spin.setValue(5.0)
            self.trial_base_start_spin.setValue(0.0)
            self.trial_base_end_spin.setValue(2.0)
            self.trial_task_start_spin.setValue(2.0)
            self.trial_task_end_spin.setValue(4.0)

            self.tabs.setCurrentWidget(self.raw_tab)

            self._update_filtered_signals()
            self._refresh_topo_band_list()
            self._refresh_erd_band_list()
            self._update_band_power()
            self._update_spectrograms()
            self._update_band_timeline()
            self._update_erd_ers()
            self._update_topomap()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open CSV:\n{file_path}\n\nError: {e}",
            )
