"""
AudioScrubber Live Denoiser UI
===============================
PyQt5-based GUI for real-time FM audio denoising with:
- Model/algorithm selection (DL and classical)
- Real-time waveform and spectrum plots
- SNR comparison (with/without denoising)
- Multi-algorithm comparison mode
- Performance metrics and resource monitoring

Usage:
    python src/ui/live_denoiser_app.py
    
Dependencies:
    pip install PyQt5 pyqtgraph numpy sounddevice torch psutil
"""

import sys
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from queue import Queue
from threading import Thread, Event
import time
import psutil

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QComboBox, QPushButton, QLabel, QSpinBox, QCheckBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QSplitter, QStatusBar,
    QProgressBar, QFrame, QGridLayout, QDoubleSpinBox, QMessageBox,
    QFileDialog, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor

# PyQtGraph for fast plotting
import pyqtgraph as pg

# Audio
import sounddevice as sd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Project imports
try:
    import torch
    from config import AudioSettings, Paths
    from src.fm.model_loader import FMModelLoader, get_model_for_inference
    from src.fm.classical import DenoiserRegistry, BaseDenoiser, DenoiseResult
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    TORCH_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AudioBuffer:
    """Circular buffer for audio history."""
    max_samples: int = 441000  # 10 seconds @ 44.1kHz
    data: np.ndarray = field(default_factory=lambda: np.zeros(441000, dtype=np.float32))
    write_pos: int = 0
    
    def append(self, chunk: np.ndarray):
        """Append audio chunk to buffer."""
        n = len(chunk)
        if self.write_pos + n <= self.max_samples:
            self.data[self.write_pos:self.write_pos + n] = chunk
            self.write_pos += n
        else:
            # Wrap around
            remaining = self.max_samples - self.write_pos
            self.data[self.write_pos:] = chunk[:remaining]
            self.data[:n - remaining] = chunk[remaining:]
            self.write_pos = n - remaining
    
    def get_last(self, n: int) -> np.ndarray:
        """Get last n samples."""
        if self.write_pos >= n:
            return self.data[self.write_pos - n:self.write_pos].copy()
        else:
            # Wrap around
            return np.concatenate([
                self.data[self.max_samples - (n - self.write_pos):],
                self.data[:self.write_pos]
            ])


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    processing_time_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    snr_input: float = 0.0
    snr_output: float = 0.0
    latency_ms: float = 0.0
    chunks_processed: int = 0
    dropped_chunks: int = 0


# ============================================================================
# Signal Bridge for Thread Communication
# ============================================================================

class SignalBridge(QObject):
    """Qt signals for thread-safe UI updates."""
    update_waveform = pyqtSignal(np.ndarray, np.ndarray)  # noisy, clean
    update_spectrum = pyqtSignal(np.ndarray, np.ndarray)  # noisy_fft, clean_fft
    update_metrics = pyqtSignal(PerformanceMetrics)
    update_snr = pyqtSignal(float, float)  # snr_before, snr_after
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)


# ============================================================================
# Deep Learning Denoiser Wrapper
# ============================================================================

class DLDenoiserWrapper(BaseDenoiser):
    """Wrapper to make DL models compatible with BaseDenoiser interface."""
    
    def __init__(self, model, model_info: Dict, device: str = 'cpu'):
        self.model = model
        self.model_info = model_info
        self.device = torch.device(device)
        self._name = f"DL: {model_info.get('mode', 'unknown')} ({model_info.get('architecture', 'unknown')})"
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            # Prepare input
            tensor = torch.from_numpy(audio.astype(np.float32))
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Forward pass
            output = self.model(tensor)
            
            # Convert back
            result = output.squeeze().cpu().numpy()
            
            # Ensure same length
            if len(result) > len(audio):
                result = result[:len(audio)]
            elif len(result) < len(audio):
                result = np.pad(result, (0, len(audio) - len(result)))
            
            return result.astype(np.float32)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def category(self) -> str:
        return 'deep_learning'


# ============================================================================
# Audio Processing Engine
# ============================================================================

class AudioEngine:
    """
    Real-time audio processing engine.
    Handles audio I/O and denoising pipeline.
    """
    
    def __init__(self, signals: SignalBridge):
        self.signals = signals
        self.running = Event()
        self.sample_rate = 44100
        self.chunk_size = 8192
        
        # Denoisers
        self.current_denoiser: Optional[BaseDenoiser] = None
        self.comparison_denoisers: List[BaseDenoiser] = []
        
        # Buffers
        self.input_buffer = AudioBuffer()
        self.output_buffer = AudioBuffer()
        self.in_queue = Queue(maxsize=20)
        self.out_queue = Queue(maxsize=20)
        
        # Metrics
        self.metrics = PerformanceMetrics()
        self._last_output = np.zeros(self.chunk_size, dtype=np.float32)
        
        # SNR history for plotting
        self.snr_history_input = deque(maxlen=100)
        self.snr_history_output = deque(maxlen=100)
        
        # Thread
        self.ai_thread: Optional[Thread] = None
        
        # Audio devices
        self.input_device = None
        self.output_device = None
    
    def set_denoiser(self, denoiser: BaseDenoiser):
        """Set the active denoiser."""
        self.current_denoiser = denoiser
        self.signals.status_message.emit(f"Denoiser: {denoiser.name}")
    
    def add_comparison_denoiser(self, denoiser: BaseDenoiser):
        """Add denoiser for comparison mode."""
        self.comparison_denoisers.append(denoiser)
    
    def clear_comparison_denoisers(self):
        """Clear comparison denoisers."""
        self.comparison_denoisers.clear()
    
    def set_devices(self, input_device, output_device):
        """Set audio devices."""
        self.input_device = input_device
        self.output_device = output_device
    
    def _find_device_id(self, name: str, kind: str = 'input') -> Optional[int]:
        """Find device ID by name."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if name.lower() in device['name'].lower():
                if device[f'max_{kind}_channels'] > 0:
                    return i
        return None
    
    def _ai_worker(self):
        """AI processing thread."""
        process = psutil.Process()
        
        while self.running.is_set():
            if self.in_queue.empty():
                time.sleep(0.001)
                continue
            
            try:
                noisy_chunk = self.in_queue.get(timeout=0.1)
            except:
                continue
            
            self.metrics.chunks_processed += 1
            
            # Process with denoiser
            if self.current_denoiser is not None:
                start_time = time.perf_counter()
                
                try:
                    clean_chunk = self.current_denoiser.denoise(noisy_chunk)
                except Exception as e:
                    self.signals.error_occurred.emit(f"Denoise error: {e}")
                    clean_chunk = noisy_chunk.copy()
                
                end_time = time.perf_counter()
                
                # Update metrics
                self.metrics.processing_time_ms = (end_time - start_time) * 1000
                self.metrics.cpu_percent = process.cpu_percent()
                self.metrics.memory_mb = process.memory_info().rss / 1024 / 1024
                
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    self.metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                clean_chunk = noisy_chunk.copy()
            
            # Estimate SNR
            noise_est = noisy_chunk - clean_chunk
            signal_power = np.mean(clean_chunk ** 2) + 1e-10
            noise_power = np.mean(noise_est ** 2) + 1e-10
            snr_improvement = 10 * np.log10(signal_power / noise_power)
            
            self.snr_history_output.append(snr_improvement)
            
            # Store in buffers
            self.input_buffer.append(noisy_chunk)
            self.output_buffer.append(clean_chunk)
            
            # Emit signals
            self.signals.update_waveform.emit(noisy_chunk, clean_chunk)
            self.signals.update_metrics.emit(self.metrics)
            
            # FFT for spectrum
            noisy_fft = np.abs(np.fft.rfft(noisy_chunk))
            clean_fft = np.abs(np.fft.rfft(clean_chunk))
            self.signals.update_spectrum.emit(noisy_fft, clean_fft)
            
            # Queue for output
            try:
                self.out_queue.put_nowait(clean_chunk)
                self._last_output = clean_chunk.copy()
            except:
                self.metrics.dropped_chunks += 1
    
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """Audio stream callback."""
        if status:
            print(f"Audio status: {status}")
        
        # Enqueue input
        try:
            self.in_queue.put_nowait(indata[:, 0].copy())
        except:
            pass
        
        # Dequeue output
        try:
            clean = self.out_queue.get_nowait()
            outdata[:, 0] = clean
        except:
            outdata[:, 0] = self._last_output
    
    def start(self):
        """Start audio processing."""
        if self.running.is_set():
            return
        
        # Find devices
        input_id = self._find_device_id(self.input_device or 'CABLE Output', 'input')
        output_id = self._find_device_id(self.output_device, 'output') if self.output_device else None
        
        if input_id is None:
            self.signals.error_occurred.emit("Input device not found!")
            return
        
        self.running.set()
        
        # Start AI thread
        self.ai_thread = Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()
        
        # Start audio stream
        try:
            self.stream = sd.Stream(
                device=(input_id, output_id),
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
            self.signals.status_message.emit("🔊 Audio stream started")
        except Exception as e:
            self.signals.error_occurred.emit(f"Failed to start audio: {e}")
            self.running.clear()
    
    def stop(self):
        """Stop audio processing."""
        self.running.clear()
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        
        self.signals.status_message.emit("⏹️ Audio stream stopped")
    
    def get_history(self, seconds: float = 5.0) -> tuple:
        """Get audio history for plotting."""
        n = int(seconds * self.sample_rate)
        return (
            self.input_buffer.get_last(n),
            self.output_buffer.get_last(n)
        )


# ============================================================================
# Main Application Window
# ============================================================================

class LiveDenoiserApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioScrubber - Live FM Denoiser")
        self.setMinimumSize(1400, 900)
        
        # Signal bridge
        self.signals = SignalBridge()
        self.signals.update_waveform.connect(self._on_waveform_update)
        self.signals.update_spectrum.connect(self._on_spectrum_update)
        self.signals.update_metrics.connect(self._on_metrics_update)
        self.signals.status_message.connect(self._on_status_message)
        self.signals.error_occurred.connect(self._on_error)
        
        # Audio engine
        self.engine = AudioEngine(self.signals)
        
        # Available denoisers
        self.denoisers: Dict[str, BaseDenoiser] = {}
        self._load_denoisers()
        
        # Frozen plot data (for persistence after stop)
        self.frozen_waveform_data = None
        self.frozen_spectrum_data = None
        
        # SNR history for comparison plot
        self.snr_times = deque(maxlen=200)
        self.snr_noisy = deque(maxlen=200)
        self.snr_denoised = deque(maxlen=200)
        self.snr_start_time = None
        
        # Comparison results
        self.comparison_results: Dict[str, List[float]] = {}
        
        # Build UI
        self._build_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._periodic_update)
        self.update_timer.start(100)  # 10 Hz
        
        # Resource monitor timer
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self._update_resources)
        self.resource_timer.start(1000)  # 1 Hz
    
    def _load_denoisers(self):
        """Load all available denoisers."""
        # Classical denoisers
        for info in DenoiserRegistry.list_denoisers():
            try:
                denoiser = DenoiserRegistry.get(info['key'])
                self.denoisers[info['key']] = denoiser
            except Exception as e:
                print(f"Failed to load {info['key']}: {e}")
        
        # DL models
        if TORCH_AVAILABLE:
            try:
                models = FMModelLoader.list_available_models()
                for path, info in models.items():
                    key = f"dl_{info['mode']}_{info['architecture']}"
                    try:
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        model, model_info = FMModelLoader.load_model(
                            model_path=path,
                            architecture=info['architecture'],
                            device=device
                        )
                        self.denoisers[key] = DLDenoiserWrapper(model, model_info, device)
                    except Exception as e:
                        print(f"Failed to load DL model {path}: {e}")
            except Exception as e:
                print(f"Failed to enumerate DL models: {e}")
    
    def _build_ui(self):
        """Build the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Top control bar
        control_bar = self._build_control_bar()
        main_layout.addWidget(control_bar)
        
        # Main content area with tabs
        tabs = QTabWidget()
        
        # Tab 1: Live View
        live_tab = self._build_live_tab()
        tabs.addTab(live_tab, "🎵 Live View")
        
        # Tab 2: SNR Comparison
        snr_tab = self._build_snr_tab()
        tabs.addTab(snr_tab, "📊 SNR Comparison")
        
        # Tab 3: Multi-Algorithm Comparison
        compare_tab = self._build_comparison_tab()
        tabs.addTab(compare_tab, "🔬 Algorithm Comparison")
        
        # Tab 4: Performance
        perf_tab = self._build_performance_tab()
        tabs.addTab(perf_tab, "⚡ Performance")
        
        main_layout.addWidget(tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select a denoiser and click Start.")
    
    def _build_control_bar(self) -> QWidget:
        """Build top control bar."""
        group = QGroupBox("Controls")
        layout = QHBoxLayout(group)
        
        # Device selection
        layout.addWidget(QLabel("Input:"))
        self.input_combo = QComboBox()
        self.input_combo.addItems(self._get_audio_devices('input'))
        self.input_combo.setCurrentText('CABLE Output')
        layout.addWidget(self.input_combo)
        
        layout.addWidget(QLabel("Output:"))
        self.output_combo = QComboBox()
        self.output_combo.addItem("Default")
        self.output_combo.addItems(self._get_audio_devices('output'))
        layout.addWidget(self.output_combo)
        
        layout.addSpacing(20)
        
        # Denoiser selection
        layout.addWidget(QLabel("Denoiser:"))
        self.denoiser_combo = QComboBox()
        self.denoiser_combo.setMinimumWidth(250)
        
        # Group by category
        classical = [k for k, v in self.denoisers.items() if v.category == 'classical']
        dl = [k for k, v in self.denoisers.items() if v.category == 'deep_learning']
        baseline = [k for k, v in self.denoisers.items() if v.category == 'baseline']
        
        for key in baseline:
            self.denoiser_combo.addItem(f"[Baseline] {self.denoisers[key].name}", key)
        for key in classical:
            self.denoiser_combo.addItem(f"[Classical] {self.denoisers[key].name}", key)
        for key in dl:
            self.denoiser_combo.addItem(f"[DL] {self.denoisers[key].name}", key)
        
        self.denoiser_combo.currentIndexChanged.connect(self._on_denoiser_changed)
        layout.addWidget(self.denoiser_combo)
        
        layout.addSpacing(20)
        
        # Chunk size
        layout.addWidget(QLabel("Chunk:"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1024, 16384)
        self.chunk_spin.setSingleStep(1024)
        self.chunk_spin.setValue(8192)
        layout.addWidget(self.chunk_spin)
        
        layout.addStretch()
        
        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        # Save button
        self.save_btn = QPushButton("💾 Save Plots")
        self.save_btn.clicked.connect(self._on_save_plots)
        layout.addWidget(self.save_btn)
        
        return group
    
    def _build_live_tab(self) -> QWidget:
        """Build live view tab with waveform and spectrum."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Set dark theme for pyqtgraph
        pg.setConfigOptions(antialias=True)
        
        # Waveform plot
        waveform_group = QGroupBox("Waveform")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Samples')
        self.waveform_plot.addLegend()
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.noisy_curve = self.waveform_plot.plot([], [], pen=pg.mkPen('orange', width=1), name='Noisy')
        self.clean_curve = self.waveform_plot.plot([], [], pen=pg.mkPen('cyan', width=1), name='Denoised')
        
        waveform_layout.addWidget(self.waveform_plot)
        layout.addWidget(waveform_group)
        
        # Spectrum plot
        spectrum_group = QGroupBox("Frequency Spectrum")
        spectrum_layout = QVBoxLayout(spectrum_group)
        
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Magnitude (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spectrum_plot.addLegend()
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.noisy_spectrum = self.spectrum_plot.plot([], [], pen=pg.mkPen('orange', width=1), name='Noisy')
        self.clean_spectrum = self.spectrum_plot.plot([], [], pen=pg.mkPen('cyan', width=1), name='Denoised')
        
        spectrum_layout.addWidget(self.spectrum_plot)
        layout.addWidget(spectrum_group)
        
        return widget
    
    def _build_snr_tab(self) -> QWidget:
        """Build SNR comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # SNR over time plot
        snr_group = QGroupBox("SNR Over Time (Without DL vs With DL)")
        snr_layout = QVBoxLayout(snr_group)
        
        self.snr_plot = pg.PlotWidget()
        self.snr_plot.setLabel('left', 'SNR (dB)')
        self.snr_plot.setLabel('bottom', 'Time (s)')
        self.snr_plot.addLegend()
        self.snr_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.snr_noisy_curve = self.snr_plot.plot([], [], pen=pg.mkPen('red', width=2), name='Without Processing')
        self.snr_clean_curve = self.snr_plot.plot([], [], pen=pg.mkPen('green', width=2), name='With Denoising')
        
        snr_layout.addWidget(self.snr_plot)
        layout.addWidget(snr_group)
        
        # Current SNR values
        values_group = QGroupBox("Current Values")
        values_layout = QHBoxLayout(values_group)
        
        self.snr_before_label = QLabel("SNR Before: -- dB")
        self.snr_before_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.snr_before_label.setStyleSheet("color: red;")
        values_layout.addWidget(self.snr_before_label)
        
        values_layout.addStretch()
        
        self.snr_after_label = QLabel("SNR After: -- dB")
        self.snr_after_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.snr_after_label.setStyleSheet("color: green;")
        values_layout.addWidget(self.snr_after_label)
        
        values_layout.addStretch()
        
        self.snr_gain_label = QLabel("Improvement: -- dB")
        self.snr_gain_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.snr_gain_label.setStyleSheet("color: cyan;")
        values_layout.addWidget(self.snr_gain_label)
        
        layout.addWidget(values_group)
        
        return widget
    
    def _build_comparison_tab(self) -> QWidget:
        """Build multi-algorithm comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Algorithm selection
        select_group = QGroupBox("Select Algorithms to Compare")
        select_layout = QGridLayout(select_group)
        
        self.compare_checkboxes: Dict[str, QCheckBox] = {}
        row, col = 0, 0
        for key, denoiser in self.denoisers.items():
            cb = QCheckBox(denoiser.name)
            self.compare_checkboxes[key] = cb
            select_layout.addWidget(cb, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        layout.addWidget(select_group)
        
        # Run comparison button
        self.run_compare_btn = QPushButton("🔬 Run Comparison (5 seconds)")
        self.run_compare_btn.clicked.connect(self._on_run_comparison)
        layout.addWidget(self.run_compare_btn)
        
        # Results table
        results_group = QGroupBox("Comparison Results")
        results_layout = QVBoxLayout(results_group)
        
        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(5)
        self.compare_table.setHorizontalHeaderLabels([
            "Algorithm", "Avg SNR Gain (dB)", "Processing Time (ms)", "CPU %", "Memory (MB)"
        ])
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.compare_table)
        
        layout.addWidget(results_group)
        
        # Comparison bar chart
        chart_group = QGroupBox("SNR Improvement Comparison")
        chart_layout = QVBoxLayout(chart_group)
        
        self.compare_plot = pg.PlotWidget()
        self.compare_plot.setLabel('left', 'SNR Improvement (dB)')
        self.compare_plot.setLabel('bottom', 'Algorithm')
        chart_layout.addWidget(self.compare_plot)
        
        layout.addWidget(chart_group)
        
        return widget
    
    def _build_performance_tab(self) -> QWidget:
        """Build performance metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Real-time metrics
        metrics_group = QGroupBox("Real-Time Performance")
        metrics_layout = QGridLayout(metrics_group)
        
        # Processing time
        metrics_layout.addWidget(QLabel("Processing Time:"), 0, 0)
        self.proc_time_label = QLabel("-- ms")
        self.proc_time_label.setFont(QFont("Arial", 12, QFont.Bold))
        metrics_layout.addWidget(self.proc_time_label, 0, 1)
        
        self.proc_time_bar = QProgressBar()
        self.proc_time_bar.setRange(0, 200)  # Max 200ms
        metrics_layout.addWidget(self.proc_time_bar, 0, 2)
        
        # CPU usage
        metrics_layout.addWidget(QLabel("CPU Usage:"), 1, 0)
        self.cpu_label = QLabel("-- %")
        self.cpu_label.setFont(QFont("Arial", 12, QFont.Bold))
        metrics_layout.addWidget(self.cpu_label, 1, 1)
        
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        metrics_layout.addWidget(self.cpu_bar, 1, 2)
        
        # Memory
        metrics_layout.addWidget(QLabel("Memory:"), 2, 0)
        self.mem_label = QLabel("-- MB")
        self.mem_label.setFont(QFont("Arial", 12, QFont.Bold))
        metrics_layout.addWidget(self.mem_label, 2, 1)
        
        self.mem_bar = QProgressBar()
        self.mem_bar.setRange(0, 1000)  # Max 1GB
        metrics_layout.addWidget(self.mem_bar, 2, 2)
        
        # GPU memory (if available)
        metrics_layout.addWidget(QLabel("GPU Memory:"), 3, 0)
        self.gpu_label = QLabel("-- MB")
        self.gpu_label.setFont(QFont("Arial", 12, QFont.Bold))
        metrics_layout.addWidget(self.gpu_label, 3, 1)
        
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setRange(0, 8000)  # Max 8GB
        metrics_layout.addWidget(self.gpu_bar, 3, 2)
        
        # Chunks processed
        metrics_layout.addWidget(QLabel("Chunks Processed:"), 4, 0)
        self.chunks_label = QLabel("0")
        self.chunks_label.setFont(QFont("Arial", 12, QFont.Bold))
        metrics_layout.addWidget(self.chunks_label, 4, 1)
        
        # Dropped chunks
        metrics_layout.addWidget(QLabel("Dropped Chunks:"), 5, 0)
        self.dropped_label = QLabel("0")
        self.dropped_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.dropped_label.setStyleSheet("color: red;")
        metrics_layout.addWidget(self.dropped_label, 5, 1)
        
        layout.addWidget(metrics_group)
        
        # Processing time history
        history_group = QGroupBox("Processing Time History")
        history_layout = QVBoxLayout(history_group)
        
        self.perf_plot = pg.PlotWidget()
        self.perf_plot.setLabel('left', 'Time (ms)')
        self.perf_plot.setLabel('bottom', 'Chunk #')
        self.perf_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.perf_curve = self.perf_plot.plot([], [], pen=pg.mkPen('yellow', width=2))
        self.perf_history = deque(maxlen=200)
        
        history_layout.addWidget(self.perf_plot)
        layout.addWidget(history_group)
        
        return widget
    
    def _get_audio_devices(self, kind: str) -> List[str]:
        """Get list of audio devices."""
        devices = sd.query_devices()
        result = []
        for device in devices:
            if device[f'max_{kind}_channels'] > 0:
                result.append(device['name'])
        return result
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    def _on_denoiser_changed(self, index):
        """Handle denoiser selection change."""
        key = self.denoiser_combo.currentData()
        if key and key in self.denoisers:
            self.engine.set_denoiser(self.denoisers[key])
    
    def _on_start(self):
        """Start audio processing."""
        # Set devices
        input_dev = self.input_combo.currentText()
        output_dev = self.output_combo.currentText()
        if output_dev == "Default":
            output_dev = None
        
        # Check if audio source is available (VB-CABLE from SDR# or Airspy)
        if 'CABLE' in input_dev.upper():
            # Show warning that user needs SDR# or Airspy running
            reply = QMessageBox.warning(
                self,
                "⚠️ Audio Source Required",
                "Make sure your FM receiver is running and outputting to VB-CABLE:\n\n"
                "1. Open SDR# (or Airspy) and tune to an FM station\n"
                "2. Set audio output to 'CABLE Input (VB-Audio Virtual Cable)'\n"
                "3. Verify you see audio levels in the SDR software\n\n"
                "Click OK when ready, or Cancel to abort.",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok
            )
            if reply == QMessageBox.Cancel:
                return
        
        self.engine.input_device = input_dev
        self.engine.output_device = output_dev
        self.engine.chunk_size = self.chunk_spin.value()
        
        # Set denoiser
        key = self.denoiser_combo.currentData()
        if key and key in self.denoisers:
            self.engine.set_denoiser(self.denoisers[key])
        
        # Start
        self.engine.start()
        self.snr_start_time = time.time()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.denoiser_combo.setEnabled(False)
    
    def _on_stop(self):
        """Stop audio processing and freeze plots."""
        self.engine.stop()
        
        # Freeze current plot data
        self.frozen_waveform_data = (
            self.noisy_curve.getData(),
            self.clean_curve.getData()
        )
        self.frozen_spectrum_data = (
            self.noisy_spectrum.getData(),
            self.clean_spectrum.getData()
        )
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.denoiser_combo.setEnabled(True)
    
    def _on_save_plots(self):
        """Save current plots to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plots", f"denoiser_plots_{timestamp}.png", "PNG Files (*.png)"
        )
        if filename:
            # Export plots as images
            # TODO: Implement proper multi-plot export
            self.waveform_plot.grab().save(filename.replace('.png', '_waveform.png'))
            self.spectrum_plot.grab().save(filename.replace('.png', '_spectrum.png'))
            self.snr_plot.grab().save(filename.replace('.png', '_snr.png'))
            self.status_bar.showMessage(f"Plots saved to {filename}")
    
    def _on_run_comparison(self):
        """Run multi-algorithm comparison."""
        # Get selected algorithms
        selected = [k for k, cb in self.compare_checkboxes.items() if cb.isChecked()]
        
        if len(selected) < 2:
            QMessageBox.warning(self, "Warning", "Select at least 2 algorithms to compare.")
            return
        
        # Collect 5 seconds of audio
        self.status_bar.showMessage("Recording 5 seconds of audio for comparison...")
        QApplication.processEvents()
        
        # Use existing history or record new
        if self.engine.running.is_set():
            time.sleep(5)
            noisy, _ = self.engine.get_history(5.0)
        else:
            QMessageBox.information(self, "Info", "Please start the audio stream first.")
            return
        
        # Run each algorithm
        results = {}
        self.compare_table.setRowCount(len(selected))
        
        for i, key in enumerate(selected):
            denoiser = self.denoisers[key]
            
            # Process in chunks
            chunk_size = 8192
            times = []
            snr_gains = []
            
            for start in range(0, len(noisy) - chunk_size, chunk_size):
                chunk = noisy[start:start + chunk_size]
                
                result = denoiser.denoise_with_metrics(chunk)
                times.append(result.processing_time_ms)
                
                # Estimate SNR gain
                noise_est = chunk - result.audio
                signal_power = np.mean(result.audio ** 2) + 1e-10
                noise_power = np.mean(noise_est ** 2) + 1e-10
                snr_gain = 10 * np.log10(signal_power / noise_power)
                snr_gains.append(snr_gain)
            
            avg_time = np.mean(times)
            avg_snr = np.mean(snr_gains)
            
            results[key] = {
                'name': denoiser.name,
                'snr_gain': avg_snr,
                'time_ms': avg_time,
                'cpu': np.mean([result.cpu_percent for _ in range(3)]),
                'memory': result.memory_mb
            }
            
            # Update table
            self.compare_table.setItem(i, 0, QTableWidgetItem(denoiser.name))
            self.compare_table.setItem(i, 1, QTableWidgetItem(f"{avg_snr:.2f}"))
            self.compare_table.setItem(i, 2, QTableWidgetItem(f"{avg_time:.2f}"))
            self.compare_table.setItem(i, 3, QTableWidgetItem(f"{results[key]['cpu']:.1f}"))
            self.compare_table.setItem(i, 4, QTableWidgetItem(f"{results[key]['memory']:.1f}"))
        
        # Update bar chart
        self.compare_plot.clear()
        x = np.arange(len(results))
        heights = [r['snr_gain'] for r in results.values()]
        names = [r['name'][:15] for r in results.values()]
        
        bar = pg.BarGraphItem(x=x, height=heights, width=0.6, brush='cyan')
        self.compare_plot.addItem(bar)
        
        # Set x-axis labels
        ax = self.compare_plot.getAxis('bottom')
        ax.setTicks([[(i, name) for i, name in enumerate(names)]])
        
        self.status_bar.showMessage("Comparison complete!")
    
    # ========================================================================
    # Signal Handlers
    # ========================================================================
    
    def _on_waveform_update(self, noisy: np.ndarray, clean: np.ndarray):
        """Update waveform plot."""
        x = np.arange(len(noisy))
        self.noisy_curve.setData(x, noisy)
        self.clean_curve.setData(x, clean)
    
    def _on_spectrum_update(self, noisy_fft: np.ndarray, clean_fft: np.ndarray):
        """Update spectrum plot."""
        freqs = np.fft.rfftfreq(len(noisy_fft) * 2 - 1, 1 / 44100)
        
        # Convert to dB
        noisy_db = 20 * np.log10(noisy_fft + 1e-10)
        clean_db = 20 * np.log10(clean_fft + 1e-10)
        
        self.noisy_spectrum.setData(freqs, noisy_db)
        self.clean_spectrum.setData(freqs, clean_db)
    
    def _on_metrics_update(self, metrics: PerformanceMetrics):
        """Update metrics display."""
        # Processing time
        self.proc_time_label.setText(f"{metrics.processing_time_ms:.1f} ms")
        self.proc_time_bar.setValue(int(min(metrics.processing_time_ms, 200)))
        
        # CPU
        self.cpu_label.setText(f"{metrics.cpu_percent:.1f} %")
        self.cpu_bar.setValue(int(min(metrics.cpu_percent, 100)))
        
        # Memory
        self.mem_label.setText(f"{metrics.memory_mb:.1f} MB")
        self.mem_bar.setValue(int(min(metrics.memory_mb, 1000)))
        
        # GPU
        self.gpu_label.setText(f"{metrics.gpu_memory_mb:.1f} MB")
        self.gpu_bar.setValue(int(min(metrics.gpu_memory_mb, 8000)))
        
        # Chunks
        self.chunks_label.setText(str(metrics.chunks_processed))
        self.dropped_label.setText(str(metrics.dropped_chunks))
        
        # Performance history
        self.perf_history.append(metrics.processing_time_ms)
        self.perf_curve.setData(list(range(len(self.perf_history))), list(self.perf_history))
    
    def _on_status_message(self, msg: str):
        """Update status bar."""
        self.status_bar.showMessage(msg)
    
    def _on_error(self, msg: str):
        """Handle error."""
        QMessageBox.critical(self, "Error", msg)
        self.status_bar.showMessage(f"Error: {msg}")
    
    def _periodic_update(self):
        """Periodic UI update."""
        if not self.engine.running.is_set():
            return
        
        # Update SNR plot
        if self.snr_start_time:
            elapsed = time.time() - self.snr_start_time
            
            # Get current SNR values from history
            if self.engine.snr_history_output:
                current_snr = self.engine.snr_history_output[-1]
                
                self.snr_times.append(elapsed)
                self.snr_noisy.append(0)  # Baseline
                self.snr_denoised.append(current_snr)
                
                # Update curves
                self.snr_noisy_curve.setData(list(self.snr_times), list(self.snr_noisy))
                self.snr_clean_curve.setData(list(self.snr_times), list(self.snr_denoised))
                
                # Update labels
                self.snr_before_label.setText(f"SNR Before: 0.0 dB (baseline)")
                self.snr_after_label.setText(f"SNR After: {current_snr:.1f} dB")
                self.snr_gain_label.setText(f"Improvement: {current_snr:.1f} dB")
    
    def _update_resources(self):
        """Update resource usage display."""
        # System-wide resources
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        
        # Could add system-wide display here if needed
    
    def closeEvent(self, event):
        """Handle window close."""
        self.engine.stop()
        self.update_timer.stop()
        self.resource_timer.stop()
        event.accept()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = LiveDenoiserApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
