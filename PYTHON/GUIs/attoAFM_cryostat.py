#!/usr/bin/env python3

"""
attoAFM Cryostat Control GUI for Lake Shore 335 Temperature Controller

This application provides a comprehensive graphical interface for controlling
the Lake Shore 335 Temperature Controller with specialized labeling for
VTI (Output 1/Sensor A) and AFM (Output 2/Sensor B) applications.

Author: Assistant
Date: 2024
License: MIT
"""

import sys
import os
import time
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

import pyvisa
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit, QFileDialog,
    QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QFrame, QSizePolicy
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QSettings
from PyQt6.QtGui import QFont, QIcon, QPalette
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import numpy as np

# Import our Lake Shore controller
from control.lakeshore import LakeShore335


class ConnectionDialog(QDialog):
    """Dialog for selecting and configuring instrument connection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Lake Shore 335")
        self.setFixedSize(500, 350)
        self.controller = None
        self.setup_ui()
        self.scan_instruments()

    def setup_ui(self):
        """Setup the connection dialog UI."""
        layout = QFormLayout(self)

        # Backend selection
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["@py (pyvisa-py)", "@ivi (National Instruments)"])
        self.backend_combo.setCurrentText("@py (pyvisa-py)")
        layout.addRow("VISA Backend:", self.backend_combo)

        # Instrument selection
        self.resource_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.scan_instruments)
        resource_layout = QHBoxLayout()
        resource_layout.addWidget(self.resource_combo)
        resource_layout.addWidget(self.refresh_btn)
        layout.addRow("Instrument:", resource_layout)

        # Communication parameters
        self.baud_rate = QComboBox()
        self.baud_rate.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_rate.setCurrentText("57600")
        layout.addRow("Baud Rate:", self.baud_rate)

        self.data_bits = QComboBox()
        self.data_bits.addItems(["7", "8"])
        self.data_bits.setCurrentText("7")
        layout.addRow("Data Bits:", self.data_bits)

        self.parity = QComboBox()
        self.parity.addItems(["None", "Odd", "Even"])
        self.parity.setCurrentText("Odd")
        layout.addRow("Parity:", self.parity)

        # Connection test
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self.test_connection)
        layout.addRow("", self.test_btn)

        self.status_label = QLabel("Select instrument and test connection")
        layout.addRow("Status:", self.status_label)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_connection)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def get_backend(self):
        """Get the selected backend."""
        backend_text = self.backend_combo.currentText()
        if "@py" in backend_text:
            return "@py"
        else:
            return "@ivi"

    def scan_instruments(self):
        """Scan for available instruments."""
        self.resource_combo.clear()
        self.status_label.setText("Scanning for instruments...")
        try:
            backend = self.get_backend()
            rm = pyvisa.ResourceManager(backend)
            resources = rm.list_resources()
            # Filter for likely serial/USB resources
            filtered_resources = [
                r for r in resources if 'ASRL' in r or 'USB' in r or 'COM' in r
            ]
            if filtered_resources:
                self.resource_combo.addItems(filtered_resources)
                self.status_label.setText(
                    f"Found {len(filtered_resources)} potential instruments"
                )
            else:
                self.resource_combo.addItem("No instruments found")
                self.status_label.setText("No instruments detected")
        except Exception as e:
            self.status_label.setText(f"Error scanning: {str(e)}")

    def test_connection(self):
        """Test connection to selected instrument."""
        if self.resource_combo.currentText() == "No instruments found":
            self.status_label.setText("No instrument selected")
            return

        self.status_label.setText("Testing connection...")
        self.test_btn.setEnabled(False)
        try:
            # Get connection parameters
            resource = self.resource_combo.currentText()
            baud_rate = int(self.baud_rate.currentText())
            data_bits = int(self.data_bits.currentText())
            parity_map = {"None": 0, "Odd": 1, "Even": 2}
            parity = parity_map[self.parity.currentText()]
            backend = self.get_backend()

            # Try to connect
            controller = LakeShore335(
                resource, baud_rate=baud_rate, data_bits=data_bits,
                parity=parity, backend=backend
            )
            # Test communication
            id_string = controller.get_identification()
            controller.close()
            self.status_label.setText(f"✓ Connected: {id_string}")
            self.controller = None  # Will be created in main app
        except Exception as e:
            self.status_label.setText(f"✗ Connection failed: {str(e)}")
        self.test_btn.setEnabled(True)

    def accept_connection(self):
        """Accept the connection if test was successful."""
        if "✓ Connected" in self.status_label.text():
            self.accept()
        else:
            QMessageBox.warning(
                self, "Connection Error",
                "Please test connection successfully before proceeding."
            )

    def get_connection_params(self):
        """Get the connection parameters."""
        parity_map = {"None": 0, "Odd": 1, "Even": 2}
        return {
            'resource': self.resource_combo.currentText(),
            'baud_rate': int(self.baud_rate.currentText()),
            'data_bits': int(self.data_bits.currentText()),
            'parity': parity_map[self.parity.currentText()],
            'backend': self.get_backend()
        }


class DataLogger:
    """Handles CSV data logging functionality."""

    def __init__(self):
        self.logging_active = False
        self.log_file = None
        self.csv_writer = None
        self.log_settings = {
            'directory': os.path.expanduser('~'),
            'filename': 'cryostat_log.csv',
            'interval': 1000,  # ms
            'log_vti_temp': True,
            'log_vti_setpoint': True,
            'log_afm_temp': True,
            'log_afm_setpoint': True,
            'log_vti_power': True,
            'log_afm_power': True,
            'log_vti_range': False,
            'log_afm_range': False
        }

    def start_logging(self):
        """Start data logging."""
        try:
            filepath = os.path.join(
                self.log_settings['directory'], self.log_settings['filename']
            )
            self.log_file = open(filepath, 'w', newline='')

            # Create CSV writer and write header
            self.csv_writer = csv.writer(self.log_file)
            header = ['Timestamp']
            if self.log_settings['log_vti_temp']:
                header.append('VTI_Temperature_K')
            if self.log_settings['log_vti_setpoint']:
                header.append('VTI_Setpoint_K')
            if self.log_settings['log_afm_temp']:
                header.append('AFM_Temperature_K')
            if self.log_settings['log_afm_setpoint']:
                header.append('AFM_Setpoint_K')
            if self.log_settings['log_vti_power']:
                header.append('VTI_Power_Percent')
            if self.log_settings['log_afm_power']:
                header.append('AFM_Power_Percent')
            if self.log_settings['log_vti_range']:
                header.append('VTI_Range')
            if self.log_settings['log_afm_range']:
                header.append('AFM_Range')

            self.csv_writer.writerow(header)
            self.log_file.flush()
            self.logging_active = True
        except Exception as e:
            raise Exception(f"Failed to start logging: {str(e)}")

    def stop_logging(self):
        """Stop data logging."""
        self.logging_active = False
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None

    def log_data(self, data: dict):
        """Log a data point."""
        if not self.logging_active or not self.csv_writer:
            return

        try:
            row = [datetime.now().isoformat()]
            if self.log_settings['log_vti_temp']:
                row.append(data.get('vti_temp', ''))
            if self.log_settings['log_vti_setpoint']:
                row.append(data.get('vti_setpoint', ''))
            if self.log_settings['log_afm_temp']:
                row.append(data.get('afm_temp', ''))
            if self.log_settings['log_afm_setpoint']:
                row.append(data.get('afm_setpoint', ''))
            if self.log_settings['log_vti_power']:
                row.append(data.get('vti_power', ''))
            if self.log_settings['log_afm_power']:
                row.append(data.get('afm_power', ''))
            if self.log_settings['log_vti_range']:
                row.append(data.get('vti_range', ''))
            if self.log_settings['log_afm_range']:
                row.append(data.get('afm_range', ''))

            self.csv_writer.writerow(row)
            self.log_file.flush()  # type: ignore
        except Exception as e:
            print(f"Logging error: {e}")


class DataCollectionThread(QThread):
    """Thread for collecting data from the instrument."""

    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, controller, refresh_interval=1000):
        super().__init__()
        self.controller = controller
        self.refresh_interval = refresh_interval
        self.running = False

    def set_refresh_interval(self, interval_ms):
        """Set the refresh interval in milliseconds."""
        self.refresh_interval = max(100, interval_ms)

    def run(self):
        """Run the data collection loop."""
        self.running = True
        while self.running:
            try:
                # Collect data from controller
                vti_temp = self.controller.fetch_T('A', 'K')
                afm_temp = self.controller.fetch_T('B', 'K')
                vti_setpoint = self.controller.get_setpoint(1)
                afm_setpoint = self.controller.get_setpoint(2)
                vti_power = self.controller.get_heater_output(1)
                afm_power = self.controller.get_heater_output(2)
                vti_range = self.controller.get_heater_range(1)
                afm_range = self.controller.get_heater_range(2)

                data = {
                    'timestamp': time.time(),
                    'vti_temp': vti_temp,
                    'afm_temp': afm_temp,
                    'vti_setpoint': vti_setpoint,
                    'afm_setpoint': afm_setpoint,
                    'vti_power': vti_power,
                    'afm_power': afm_power,
                    'vti_range': vti_range,
                    'afm_range': afm_range
                }
                self.data_ready.emit(data)
            except Exception as e:
                self.error_occurred.emit(str(e))
            self.msleep(self.refresh_interval)

    def stop(self):
        """Stop the data collection thread."""
        self.running = False
        self.wait()


class OverviewTab(QWidget):
    """The main overview tab with plots and controls."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.data_logger = DataLogger()
        self.time_window_minutes = 60  # Default to 60 minutes
        self.setup_ui()
        self.setup_data_collection()
        # Load instrument settings after UI is setup
        QTimer.singleShot(500, self.load_instrument_settings)

    def setup_ui(self):
        """Setup the overview tab UI."""
        layout = QGridLayout(self)

        # Upper left: Temperature plot
        self.plot_widget = self.create_plot_widget()
        layout.addWidget(self.plot_widget, 0, 0)

        # Upper right: Logging controls
        self.logging_group = self.create_logging_controls()
        layout.addWidget(self.logging_group, 0, 1)

        # Lower left: VTI controls
        self.vti_group = self.create_control_group("VTI Controls", 1)
        layout.addWidget(self.vti_group, 1, 0)

        # Lower right: AFM controls
        self.afm_group = self.create_control_group("AFM Controls", 2)
        layout.addWidget(self.afm_group, 1, 1)

        # Set column stretch
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

    def create_plot_widget(self):
        """Create the temperature plotting widget."""
        plot_widget = PlotWidget()
        plot_widget.setLabel('left', 'Temperature', units='K')
        plot_widget.setLabel('bottom', 'Time', units='s')
        plot_widget.setTitle('Temperature Monitor')
        plot_widget.showGrid(x=True, y=True)
        plot_widget.addLegend()
        
        # Set white background
        plot_widget.setBackground('w')

        # Create plot curves
        self.vti_temp_curve = plot_widget.plot(
            [], [], pen='b', name='VTI Temperature'
        )
        self.afm_temp_curve = plot_widget.plot(
            [], [], pen='r', name='AFM Temperature'
        )
        self.vti_setpoint_curve = plot_widget.plot(
            [], [], pen=pg.mkPen('b', style=Qt.PenStyle.DashLine), name='VTI Setpoint'
        )
        self.afm_setpoint_curve = plot_widget.plot(
            [], [], pen=pg.mkPen('r', style=Qt.PenStyle.DashLine), name='AFM Setpoint'
        )

        # Create control widgets for the plot
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # First row: Refresh rate control
        refresh_layout = QHBoxLayout()
        refresh_layout.addWidget(QLabel("Refresh (ms):"))
        refresh_layout.addSpacing(10)  # Add spacing to prevent text cutoff
        self.refresh_rate_spin = QSpinBox()
        self.refresh_rate_spin.setRange(100, 10000)
        self.refresh_rate_spin.setValue(1000)
        self.refresh_rate_spin.valueChanged.connect(self.update_refresh_rate)
        refresh_layout.addWidget(self.refresh_rate_spin)
        refresh_layout.addStretch()
        controls_layout.addLayout(refresh_layout)
        
        # Second row: Time window control
        time_window_layout = QHBoxLayout()
        time_window_layout.addWidget(QLabel("Time Window (min):"))
        time_window_layout.addSpacing(10)  # Add spacing to prevent text overlap
        self.time_window_spin = QDoubleSpinBox()
        self.time_window_spin.setRange(0.1, 1440)  # 0.1 min to 24 hours
        self.time_window_spin.setDecimals(1)
        self.time_window_spin.setValue(self.time_window_minutes)
        time_window_layout.addWidget(self.time_window_spin)
        
        self.set_time_window_btn = QPushButton("Set")
        self.set_time_window_btn.clicked.connect(self.set_time_window_clicked)
        time_window_layout.addWidget(self.set_time_window_btn)
        time_window_layout.addStretch()
        controls_layout.addLayout(time_window_layout)
        
        # Third row: Clear graph button
        clear_layout = QHBoxLayout()
        self.clear_graph_btn = QPushButton("Clear Graph")
        self.clear_graph_btn.clicked.connect(self.clear_graph_clicked)
        clear_layout.addWidget(self.clear_graph_btn)
        clear_layout.addStretch()
        controls_layout.addLayout(clear_layout)

        # Create displays widget for temperature and power
        displays_widget = QWidget()
        displays_layout = QHBoxLayout(displays_widget)
        
        # Temperature display
        temp_widget = QWidget()
        temp_layout = QVBoxLayout(temp_widget)
        temp_layout.addWidget(QLabel("Temperature:"))
        self.vti_temp_label = QLabel("VTI: 0.000 K")
        self.afm_temp_label = QLabel("AFM: 0.000 K")
        temp_layout.addWidget(self.vti_temp_label)
        temp_layout.addWidget(self.afm_temp_label)
        
        # Power display
        power_widget = QWidget()
        power_layout = QVBoxLayout(power_widget)
        power_layout.addWidget(QLabel("Power:"))
        self.vti_power_label = QLabel("VTI: 0.0%")
        self.afm_power_label = QLabel("AFM: 0.0%")
        power_layout.addWidget(self.vti_power_label)
        power_layout.addWidget(self.afm_power_label)
        
        # Add both to displays layout
        displays_layout.addWidget(temp_widget)
        displays_layout.addSpacing(20)  # Space between temperature and power
        displays_layout.addWidget(power_widget)
        displays_layout.addStretch()

        # Position widgets on plot
        controls_widget.setFixedSize(200, 100)
        displays_widget.setFixedSize(250, 80)

        # Add widgets to plot layout
        plot_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(controls_widget)
        top_layout.addStretch()
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(displays_widget)
        plot_layout.addLayout(top_layout)
        plot_layout.addWidget(plot_widget)
        plot_layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(plot_layout)

        # Initialize data arrays
        self.time_data = []
        self.vti_temp_data = []
        self.afm_temp_data = []
        self.vti_setpoint_data = []
        self.afm_setpoint_data = []

        return container

    def create_logging_controls(self):
        """Create the data logging control group."""
        group = QGroupBox("Data Logging")
        layout = QVBoxLayout(group)

        # File settings
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Directory:"))
        self.log_directory_edit = QLineEdit(self.data_logger.log_settings['directory'])
        file_layout.addWidget(self.log_directory_edit)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_directory)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)

        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Filename:"))
        self.log_filename_edit = QLineEdit(self.data_logger.log_settings['filename'])
        filename_layout.addWidget(self.log_filename_edit)
        layout.addLayout(filename_layout)

        # Logging interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Log Interval (ms):"))
        self.log_interval_spin = QSpinBox()
        self.log_interval_spin.setRange(100, 60000)
        self.log_interval_spin.setValue(self.data_logger.log_settings['interval'])
        interval_layout.addWidget(self.log_interval_spin)
        layout.addLayout(interval_layout)

        # Logging checkboxes
        self.log_checkboxes = {}
        checkbox_names = [
            ('log_vti_temp', 'VTI Temperature'),
            ('log_vti_setpoint', 'VTI Setpoint'),
            ('log_afm_temp', 'AFM Temperature'),
            ('log_afm_setpoint', 'AFM Setpoint'),
            ('log_vti_power', 'VTI Power'),
            ('log_afm_power', 'AFM Power'),
            ('log_vti_range', 'VTI Range'),
            ('log_afm_range', 'AFM Range')
        ]

        for key, name in checkbox_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(self.data_logger.log_settings[key])
            self.log_checkboxes[key] = checkbox
            layout.addWidget(checkbox)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_log_btn = QPushButton("Start Logging")
        self.start_log_btn.clicked.connect(self.toggle_logging)
        button_layout.addWidget(self.start_log_btn)
        layout.addLayout(button_layout)

        return group

    def create_control_group(self, title, output_num):
        """Create a control group for VTI or AFM."""
        group = QGroupBox(title)
        layout = QFormLayout(group)

        # Setpoint with Set button
        setpoint_layout = QHBoxLayout()
        setpoint_spin = QDoubleSpinBox()
        setpoint_spin.setRange(0, 500)
        setpoint_spin.setDecimals(3)
        setpoint_spin.setSuffix(" K")
        setpoint_btn = QPushButton("Set")
        setpoint_layout.addWidget(setpoint_spin)
        setpoint_layout.addWidget(setpoint_btn)
        layout.addRow("Setpoint:", setpoint_layout)

        # Ramp rate and control with Set button
        ramp_layout = QVBoxLayout()
        
        # First row: ramp rate input and set button
        ramp_rate_layout = QHBoxLayout()
        ramp_rate_spin = QDoubleSpinBox()
        ramp_rate_spin.setRange(0.1, 100)
        ramp_rate_spin.setDecimals(1)
        ramp_rate_spin.setSuffix(" K/min")
        ramp_rate_btn = QPushButton("Set Rate")
        ramp_rate_layout.addWidget(ramp_rate_spin)
        ramp_rate_layout.addWidget(ramp_rate_btn)
        ramp_layout.addLayout(ramp_rate_layout)
        
        # Second row: enable checkbox
        ramp_enable_cb = QCheckBox("Enable Ramping")
        ramp_layout.addWidget(ramp_enable_cb)
        
        layout.addRow("Ramp Rate:", ramp_layout)

        # Power range
        range_combo = QComboBox()
        range_combo.addItems(["OFF", "LOW", "MED", "HIGH"])
        layout.addRow("Power Range:", range_combo)

        # Control mode
        mode_combo = QComboBox()
        mode_combo.addItems(["PID", "Manual"])
        layout.addRow("Control Mode:", mode_combo)

        # Manual power
        manual_spin = QDoubleSpinBox()
        manual_spin.setRange(0, 100)
        manual_spin.setDecimals(2)
        manual_spin.setSuffix(" %")
        layout.addRow("Manual Power:", manual_spin)

        # PID parameters with Set button
        pid_layout = QVBoxLayout()
        
        pid_p_layout = QHBoxLayout()
        pid_p_spin = QDoubleSpinBox()
        pid_p_spin.setRange(0.1, 1000)
        pid_p_spin.setDecimals(1)
        pid_p_layout.addWidget(QLabel("P:"))
        pid_p_layout.addWidget(pid_p_spin)
        pid_layout.addLayout(pid_p_layout)

        pid_i_layout = QHBoxLayout()
        pid_i_spin = QDoubleSpinBox()
        pid_i_spin.setRange(0.1, 1000)
        pid_i_spin.setDecimals(1)
        pid_i_layout.addWidget(QLabel("I:"))
        pid_i_layout.addWidget(pid_i_spin)
        pid_layout.addLayout(pid_i_layout)

        pid_d_layout = QHBoxLayout()
        pid_d_spin = QDoubleSpinBox()
        pid_d_spin.setRange(0, 200)
        pid_d_spin.setDecimals(1)
        pid_d_spin.setSuffix(" %")
        pid_d_layout.addWidget(QLabel("D:"))
        pid_d_layout.addWidget(pid_d_spin)
        pid_layout.addLayout(pid_d_layout)

        pid_btn = QPushButton("Set PID")
        pid_layout.addWidget(pid_btn)
        layout.addRow("PID Control:", pid_layout)

        # Store widgets for later access
        widgets = {
            'setpoint': setpoint_spin,
            'setpoint_btn': setpoint_btn,
            'ramp_rate': ramp_rate_spin,
            'ramp_rate_btn': ramp_rate_btn,
            'ramp_enable': ramp_enable_cb,
            'range': range_combo,
            'mode': mode_combo,
            'manual_power': manual_spin,
            'pid_p': pid_p_spin,
            'pid_i': pid_i_spin,
            'pid_d': pid_d_spin,
            'pid_btn': pid_btn
        }

        if output_num == 1:
            self.vti_widgets = widgets
        else:
            self.afm_widgets = widgets

        # Connect signals (only for non-instrument settings)
        ramp_enable_cb.toggled.connect(
            lambda v, o=output_num: self.update_ramp_enable(o, v)
        )
        range_combo.currentIndexChanged.connect(
            lambda i, o=output_num: self.update_range(o, i)
        )
        mode_combo.currentTextChanged.connect(
            lambda t, o=output_num: self.update_mode(o, t)
        )
        manual_spin.valueChanged.connect(
            lambda v, o=output_num: self.update_manual_power(o, v)
        )

        # Connect set buttons
        setpoint_btn.clicked.connect(lambda: self.set_setpoint_clicked(output_num))
        ramp_rate_btn.clicked.connect(lambda: self.set_ramp_rate_clicked(output_num))
        pid_btn.clicked.connect(lambda: self.set_pid_clicked(output_num))

        return group

    def setup_data_collection(self):
        """Setup the data collection thread and logging timer."""
        self.data_thread = DataCollectionThread(self.controller, 1000)
        self.data_thread.data_ready.connect(self.update_plot_data)
        self.data_thread.error_occurred.connect(self.handle_data_error)

        # Logging timer
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.log_current_data)

        # Start data collection
        self.data_thread.start()

    def load_instrument_settings(self):
        """Load current settings from the instrument."""
        try:
            print("Loading instrument settings...")

            # Load VTI settings
            vti_setpoint = self.controller.get_setpoint(1)
            vti_ramp_rate = self.controller.get_ramp_rate(1)
            vti_ramp_enabled = self.controller.get_ramp_status(1)
            vti_range = self.controller.get_heater_range(1)
            vti_pid = self.controller.get_PID(1)
            vti_manual = self.controller.get_manual_output(1)

            # Load AFM settings
            afm_setpoint = self.controller.get_setpoint(2)
            afm_ramp_rate = self.controller.get_ramp_rate(2)
            afm_ramp_enabled = self.controller.get_ramp_status(2)
            afm_range = self.controller.get_heater_range(2)
            afm_pid = self.controller.get_PID(2)
            afm_manual = self.controller.get_manual_output(2)

            # Set VTI values (disconnect signals temporarily to avoid loops)
            self.vti_widgets['setpoint'].blockSignals(True)
            self.vti_widgets['setpoint'].setValue(vti_setpoint)
            self.vti_widgets['setpoint'].blockSignals(False)
            self.vti_widgets['ramp_rate'].setValue(vti_ramp_rate)
            self.vti_widgets['ramp_enable'].setChecked(vti_ramp_enabled)
            self.vti_widgets['range'].setCurrentIndex(vti_range)
            self.vti_widgets['manual_power'].setValue(vti_manual)
            self.vti_widgets['pid_p'].setValue(vti_pid[0])
            self.vti_widgets['pid_i'].setValue(vti_pid[1])
            self.vti_widgets['pid_d'].setValue(vti_pid[2])

            # Set AFM values
            self.afm_widgets['setpoint'].blockSignals(True)
            self.afm_widgets['setpoint'].setValue(afm_setpoint)
            self.afm_widgets['setpoint'].blockSignals(False)
            self.afm_widgets['ramp_rate'].setValue(afm_ramp_rate)
            self.afm_widgets['ramp_enable'].setChecked(afm_ramp_enabled)
            self.afm_widgets['range'].setCurrentIndex(afm_range)
            self.afm_widgets['manual_power'].setValue(afm_manual)
            self.afm_widgets['pid_p'].setValue(afm_pid[0])
            self.afm_widgets['pid_i'].setValue(afm_pid[1])
            self.afm_widgets['pid_d'].setValue(afm_pid[2])

            print("Instrument settings loaded successfully")

        except Exception as e:
            print(f"Failed to load some settings: {str(e)}")
            QMessageBox.warning(
                self, "Load Settings Error",
                f"Failed to load some settings from instrument:\n{str(e)}"
            )

    def set_setpoint_clicked(self, output):
        """Handle setpoint set button click."""
        try:
            if output == 1:
                setpoint = self.vti_widgets['setpoint'].value()
            else:
                setpoint = self.afm_widgets['setpoint'].value()

            self.controller.set_T(output, setpoint)
            print(f"Set output {output} setpoint to {setpoint} K")
        except Exception as e:
            QMessageBox.warning(
                self, "Setpoint Error", f"Failed to set setpoint: {str(e)}"
            )

    def set_ramp_rate_clicked(self, output):
        """Handle ramp rate set button click."""
        try:
            if output == 1:
                ramp_rate = self.vti_widgets['ramp_rate'].value()
            else:
                ramp_rate = self.afm_widgets['ramp_rate'].value()

            self.controller.set_ramp(output, ramp_rate)
            print(f"Set output {output} ramp rate to {ramp_rate} K/min")
        except Exception as e:
            QMessageBox.warning(
                self, "Ramp Rate Error", f"Failed to set ramp rate: {str(e)}"
            )

    def set_pid_clicked(self, output):
        """Handle PID set button click."""
        try:
            if output == 1:
                widgets = self.vti_widgets
            else:
                widgets = self.afm_widgets

            p_value = widgets['pid_p'].value()
            i_value = widgets['pid_i'].value()
            d_value = widgets['pid_d'].value()

            self.controller.set_PID(output, p_value, i_value, d_value)
            print(f"Set output {output} PID to P={p_value}, I={i_value}, D={d_value}")
        except Exception as e:
            QMessageBox.warning(
                self, "PID Error", f"Failed to set PID values: {str(e)}"
            )

    def clear_graph_clicked(self):
        """Handle clear graph button click."""
        self.time_data = []
        self.vti_temp_data = []
        self.afm_temp_data = []
        self.vti_setpoint_data = []
        self.afm_setpoint_data = []
        
        # Clear the plot curves
        self.vti_temp_curve.setData([], [])
        self.afm_temp_curve.setData([], [])
        self.vti_setpoint_curve.setData([], [])
        self.afm_setpoint_curve.setData([], [])
        
        print("Graph cleared")

    def set_time_window_clicked(self):
        """Handle set time window button click."""
        # Clear the graph first
        self.clear_graph_clicked()
        
        # Update the time window
        self.time_window_minutes = self.time_window_spin.value()
        print(f"Time window set to {self.time_window_minutes} minutes")

    def trim_data_to_time_window(self):
        """Remove data older than the time window."""
        if not self.time_data:
            return
            
        current_time = time.time()
        time_window_seconds = self.time_window_minutes * 60
        cutoff_time = current_time - time_window_seconds
        
        # Find the index where data is still within the time window
        keep_indices = []
        for i, timestamp in enumerate(self.time_data):
            if timestamp >= cutoff_time:
                keep_indices.append(i)
        
        if keep_indices:
            # Keep only data within the time window
            start_idx = keep_indices[0]
            self.time_data = self.time_data[start_idx:]
            self.vti_temp_data = self.vti_temp_data[start_idx:]
            self.afm_temp_data = self.afm_temp_data[start_idx:]
            self.vti_setpoint_data = self.vti_setpoint_data[start_idx:]
            self.afm_setpoint_data = self.afm_setpoint_data[start_idx:]
        else:
            # All data is too old, clear everything
            self.clear_graph_clicked()

    def update_refresh_rate(self, value):
        """Update the data collection refresh rate."""
        self.data_thread.set_refresh_interval(value)

    def browse_directory(self):
        """Browse for log directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Log Directory")
        if directory:
            self.log_directory_edit.setText(directory)

    def toggle_logging(self):
        """Toggle data logging on/off."""
        if not self.data_logger.logging_active:
            try:
                # Update logging settings
                self.data_logger.log_settings['directory'] = self.log_directory_edit.text()
                self.data_logger.log_settings['filename'] = self.log_filename_edit.text()
                self.data_logger.log_settings['interval'] = self.log_interval_spin.value()

                for key, checkbox in self.log_checkboxes.items():
                    self.data_logger.log_settings[key] = checkbox.isChecked()

                self.data_logger.start_logging()
                self.log_timer.start(self.data_logger.log_settings['interval'])
                self.start_log_btn.setText("Stop Logging")
            except Exception as e:
                QMessageBox.warning(self, "Logging Error", str(e))
        else:
            self.data_logger.stop_logging()
            self.log_timer.stop()
            self.start_log_btn.setText("Start Logging")

    def log_current_data(self):
        """Log current data point."""
        if hasattr(self, 'current_data'):
            self.data_logger.log_data(self.current_data)

    def update_plot_data(self, data):
        """Update plot with new data."""
        self.current_data = data

        # Add data to arrays
        self.time_data.append(data['timestamp'])
        self.vti_temp_data.append(data['vti_temp'])
        self.afm_temp_data.append(data['afm_temp'])
        self.vti_setpoint_data.append(data['vti_setpoint'])
        self.afm_setpoint_data.append(data['afm_setpoint'])

        # Trim data to time window (removes old data and saves memory)
        self.trim_data_to_time_window()

        # Convert time to relative seconds
        if self.time_data:
            time_relative = [(t - self.time_data[0]) for t in self.time_data]

            # Update plot curves
            self.vti_temp_curve.setData(time_relative, self.vti_temp_data)
            self.afm_temp_curve.setData(time_relative, self.afm_temp_data)
            self.vti_setpoint_curve.setData(time_relative, self.vti_setpoint_data)
            self.afm_setpoint_curve.setData(time_relative, self.afm_setpoint_data)

        # Update temperature labels with 3 decimal places
        self.vti_temp_label.setText(f"VTI: {data['vti_temp']:.3f} K")
        self.afm_temp_label.setText(f"AFM: {data['afm_temp']:.3f} K")
        
        # Update power labels
        self.vti_power_label.setText(f"VTI: {data['vti_power']:.1f}%")
        self.afm_power_label.setText(f"AFM: {data['afm_power']:.1f}%")

    def handle_data_error(self, error_msg):
        """Handle data collection errors."""
        print(f"Data collection error: {error_msg}")

    # Control update methods
    def update_ramp_enable(self, output, enabled):
        try:
            if enabled:
                self.controller.ramp_on(output)
            else:
                self.controller.ramp_off(output)
        except Exception as e:
            print(f"Failed to set ramp enable: {e}")

    def update_range(self, output, index):
        try:
            self.controller.set_heater_range(output, index)
        except Exception as e:
            print(f"Failed to set heater range: {e}")

    def update_mode(self, output, mode):
        # This would need to be implemented based on your control needs
        pass

    def update_manual_power(self, output, value):
        try:
            self.controller.set_manual_output(output, value)
        except Exception as e:
            print(f"Failed to set manual power: {e}")

    def closeEvent(self, event):
        """Handle window close event."""
        if hasattr(self, 'data_thread'):
            self.data_thread.stop()
        if self.data_logger.logging_active:
            self.data_logger.stop_logging()
        event.accept()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("attoAFM Cryostat Control - Lake Shore 335")
        self.setGeometry(100, 100, 1200, 800)

        # Create single tab (overview only)
        self.overview_tab = OverviewTab(controller)
        self.setCentralWidget(self.overview_tab)

    def closeEvent(self, event):
        """Handle application close."""
        if hasattr(self.overview_tab, 'data_thread'):
            self.overview_tab.data_thread.stop()
        if self.overview_tab.data_logger.logging_active:
            self.overview_tab.data_logger.stop_logging()
        self.controller.close()
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("attoAFM Cryostat Control")

    # Show connection dialog
    conn_dialog = ConnectionDialog()
    if conn_dialog.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)

    # Get connection parameters and create controller
    conn_params = conn_dialog.get_connection_params()
    try:
        controller = LakeShore335(**conn_params)

        # Create and show main window
        main_window = MainWindow(controller)
        main_window.show()

        # Run application
        sys.exit(app.exec())

    except Exception as e:
        QMessageBox.critical(
            None, "Connection Error",
            f"Failed to connect to instrument:\n{str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()