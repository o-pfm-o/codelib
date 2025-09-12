#!/usr/bin/env python3
"""
Lake Shore Model 335 Temperature Controller Interface

This module provides a Python interface for controlling the Lake Shore Model 335
Temperature Controller using PyVISA over USB or IEEE-488.

Author: Philipp Fahler-Muenzer
Date: 2024
License: MIT
"""

import pyvisa
from typing import Dict, List, Tuple, Union, Optional, Any
import time


class LakeShore335:
    """
    Interface class for Lake Shore Model 335 Temperature Controller.

    This class provides methods to communicate with the Lake Shore 335 via PyVISA,
    allowing control of temperature setpoints, PID parameters, ramping, and reading
    of all instrument parameters.

    Attributes:
        instrument: PyVISA instrument object for communication
        inputs: List of available sensor inputs ['A', 'B']
        outputs: List of available control outputs [1, 2]
    """

    def __init__(self, resource: str, read_termination: str = '\r\n',
                 write_termination: str = '\r\n', baud_rate: int = 57600,
                 data_bits: int = 7, parity: int = 1, backend: str = '@py'):
        """
        Initialize connection to Lake Shore 335.

        Args:
            resource: PyVISA resource string (e.g., 'ASRL3::INSTR' for COM3)
            read_termination: Read termination character(s)
            write_termination: Write termination character(s)
            baud_rate: Communication baud rate (default 57600)
            data_bits: Number of data bits (default 7)
            parity: Parity setting (0=none, 1=odd, 2=even)
            backend: PyVISA backend ('@py' or '@ivi')

        Raises:
            pyvisa.errors.VisaIOError: If connection cannot be established
        """
        rm = pyvisa.ResourceManager(backend)
        self.instrument = rm.open_resource(resource)

        # Configure serial communication parameters
        self.instrument.baud_rate = baud_rate
        self.instrument.data_bits = data_bits
        self.instrument.parity = pyvisa.constants.Parity(parity)
        self.instrument.stop_bits = pyvisa.constants.StopBits.one
        self.instrument.flow_control = pyvisa.constants.VI_ASRL_FLOW_NONE

        # Set termination characters
        self.instrument.read_termination = read_termination
        self.instrument.write_termination = write_termination

        # Set timeout (30 seconds)
        self.instrument.timeout = 30000

        # Available inputs and outputs
        self.inputs = ['A', 'B']
        self.outputs = [1, 2]

        # Test connection
        try:
            self.get_identification()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Lake Shore 335: {e}")

    def close(self):
        """Close the connection to the instrument."""
        if hasattr(self, 'instrument'):
            self.instrument.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ==================== Utility Methods ====================

    def _query(self, command: str) -> str:
        """Send query command and return response."""
        return self.instrument.query(command).strip()

    def _write(self, command: str):
        """Send command without expecting response."""
        self.instrument.write(command)

    def _validate_input(self, input_channel: str):
        """Validate input channel is A or B."""
        if input_channel.upper() not in self.inputs:
            raise ValueError(f"Input must be one of {self.inputs}")
        return input_channel.upper()

    def _validate_output(self, output: int):
        """Validate output channel is 1 or 2."""
        if output not in self.outputs:
            raise ValueError(f"Output must be one of {self.outputs}")
        return output

    # ==================== Identification ====================

    def get_identification(self) -> str:
        """
        Get instrument identification string.

        Returns:
            Identification string containing manufacturer, model, serial, firmware
        """
        return self._query("*IDN?")

    # ==================== Temperature Reading Methods ====================

    def fetch_T(self, sensor: str, units: str = 'K') -> float:
        """
        Read temperature from specified sensor.

        Args:
            sensor: Input channel ('A' or 'B')
            units: Temperature units ('K' for Kelvin, 'C' for Celsius)

        Returns:
            Temperature reading as float

        Raises:
            ValueError: If invalid sensor or units specified
        """
        sensor = self._validate_input(sensor)
        if units.upper() == 'K':
            response = self._query(f"KRDG? {sensor}")
        elif units.upper() == 'C':
            response = self._query(f"CRDG? {sensor}")
        else:
            raise ValueError("Units must be 'K' or 'C'")
        return float(response)

    def read_T(self, units: str = 'K') -> Tuple[float, float]:
        """
        Read temperature from both sensors A and B.

        Args:
            units: Temperature units ('K' for Kelvin, 'C' for Celsius)

        Returns:
            Tuple of (temp_A, temp_B)
        """
        temp_a = self.fetch_T('A', units)
        temp_b = self.fetch_T('B', units)
        return (temp_a, temp_b)

    # ==================== Setpoint Control ====================

    def set_T(self, output: int, temperature: float):
        """
        Set temperature setpoint for specified output.

        Args:
            output: Output channel (1 or 2)
            temperature: Target temperature
        """
        output = self._validate_output(output)
        self._write(f"SETP {output},{temperature}")

    def get_setpoint(self, output: int) -> float:
        """
        Get temperature setpoint for specified output.

        Args:
            output: Output channel (1 or 2)

        Returns:
            Current setpoint temperature
        """
        output = self._validate_output(output)
        response = self._query(f"SETP? {output}")
        return float(response)

    # ==================== Ramp Control ====================

    def set_ramp(self, output: int, rate: float):
        """
        Set temperature ramp rate for specified output.

        Args:
            output: Output channel (1 or 2)
            rate: Ramp rate in K/min (0.1 to 100)
        """
        output = self._validate_output(output)
        # Enable ramping and set rate
        self._write(f"RAMP {output},1,{rate}")

    def ramp_on(self, output: int):
        """
        Enable ramping for specified output (keeps current rate).

        Args:
            output: Output channel (1 or 2)
        """
        output = self._validate_output(output)
        # Get current rate and enable ramping
        current_rate = self.get_ramp_rate(output)
        self._write(f"RAMP {output},1,{current_rate}")

    def ramp_off(self, output: int):
        """
        Disable ramping for specified output.

        Args:
            output: Output channel (1 or 2)
        """
        output = self._validate_output(output)
        current_rate = self.get_ramp_rate(output)
        self._write(f"RAMP {output},0,{current_rate}")

    def get_ramp_rate(self, output: int) -> float:
        """
        Get current ramp rate for specified output.

        Args:
            output: Output channel (1 or 2)

        Returns:
            Ramp rate in K/min
        """
        output = self._validate_output(output)
        response = self._query(f"RAMP? {output}")
        parts = response.split(',')
        return float(parts[1])  # Rate is second parameter

    def get_ramp_status(self, output: int) -> bool:
        """
        Get ramping enable status for specified output.

        Args:
            output: Output channel (1 or 2)

        Returns:
            True if ramping is enabled, False otherwise
        """
        output = self._validate_output(output)
        response = self._query(f"RAMP? {output}")
        parts = response.split(',')
        return bool(int(parts[0]))  # Enable is first parameter

    # ==================== PID Control ====================

    def get_PID(self, output: int) -> Tuple[float, float, float]:
        """
        Get PID parameters for specified output.

        Args:
            output: Output channel (1 or 2)

        Returns:
            Tuple of (P, I, D) values
        """
        output = self._validate_output(output)
        response = self._query(f"PID? {output}")
        parts = response.split(',')
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    def set_PID(self, output: int, P: float, I: float, D: float):
        """
        Set PID parameters for specified output.

        Args:
            output: Output channel (1 or 2)
            P: Proportional gain (0.1 to 1000)
            I: Integral reset (0.1 to 1000)
            D: Derivative rate (0 to 200%)
        """
        output = self._validate_output(output)
        self._write(f"PID {output},{P},{I},{D}")

    def autotune_PID(self, output: int, mode: int = 1):
        """
        Initiate PID autotuning for specified output.

        Args:
            output: Output channel (1 or 2)
            mode: Autotune mode (0=P only, 1=P and I, 2=P, I, and D)
        """
        output = self._validate_output(output)
        if mode not in [0, 1, 2]:
            raise ValueError("Mode must be 0 (P only), 1 (PI), or 2 (PID)")
        self._write(f"ATUNE {output},{mode}")

    def get_autotune_status(self) -> Tuple[bool, int, bool, int]:
        """
        Get autotuning status.

        Returns:
            Tuple of (tuning_active, output_being_tuned, error_occurred, stage)
        """
        response = self._query("TUNEST?")
        parts = response.split(',')
        return (bool(int(parts[0])), int(parts[1]),
                bool(int(parts[2])), int(parts[3]))

    # ==================== Zone Control ====================

    def get_zones(self, output: int) -> List[Tuple[float, float]]:
        """
        Get zone boundaries for specified output.

        Args:
            output: Output channel (1 or 2)

        Returns:
            List of tuples (lower_boundary, upper_boundary) for each zone
        """
        output = self._validate_output(output)
        zones = []
        lower_bound = 0.0
        for zone in range(1, 11):  # Zones 1-10
            try:
                response = self._query(f"ZONE? {output},{zone}")
                parts = response.split(',')
                upper_bound = float(parts[0])
                # Only add zone if it has a meaningful upper boundary
                if upper_bound > 0:
                    zones.append((lower_bound, upper_bound))
                    lower_bound = upper_bound
                else:
                    break
            except:
                break
        return zones

    def set_zones(self, output: int, zones: List[Tuple[float, float]]):
        """
        Set zone boundaries for specified output.

        Args:
            output: Output channel (1 or 2)
            zones: List of (lower_boundary, upper_boundary) tuples
        """
        output = self._validate_output(output)
        # Clear existing zones by setting them to 0
        for zone in range(1, 11):
            self._write(f"ZONE {output},{zone},0,0,0,0,0,0,0,0")
        # Set new zones
        for i, (lower, upper) in enumerate(zones[:10], 1):
            # Set zone with default PID values - user should customize these
            self._write(f"ZONE {output},{i},{upper},50,20,0,0,2,0,0.1")

    # ==================== Getter Methods for All Parameters ====================

    def get_alarm_parameters(self, input_channel: str) -> Dict[str, Any]:
        """Get alarm parameters for input."""
        input_channel = self._validate_input(input_channel)
        response = self._query(f"ALARM? {input_channel}")
        parts = response.split(',')
        return {
            'enabled': bool(int(parts[0])),
            'high_setpoint': float(parts[1]),
            'low_setpoint': float(parts[2]),
            'deadband': float(parts[3]),
            'latch_enable': bool(int(parts[4])),
            'audible': bool(int(parts[5])),
            'visible': bool(int(parts[6]))
        }

    def get_display_brightness(self) -> int:
        """Get display brightness (0-3)."""
        response = self._query("BRIGT?")
        return int(response)

    def get_filter_parameters(self, input_channel: str) -> Dict[str, Any]:
        """Get input filter parameters."""
        input_channel = self._validate_input(input_channel)
        response = self._query(f"FILTER? {input_channel}")
        parts = response.split(',')
        return {
            'enabled': bool(int(parts[0])),
            'points': int(parts[1]),
            'window': int(parts[2])
        }

    def get_heater_output(self, output: int) -> float:
        """Get heater output percentage."""
        output = self._validate_output(output)
        response = self._query(f"HTR? {output}")
        return float(response)

    def get_heater_setup(self, output: int) -> Dict[str, Any]:
        """Get heater setup parameters."""
        output = self._validate_output(output)
        response = self._query(f"HTRSET? {output}")
        parts = response.split(',')
        return {
            'type': int(parts[0]),  # 0=Current, 1=Voltage (Output 2 only)
            'resistance': int(parts[1]),  # 1=25Ω, 2=50Ω
            'max_current': int(parts[2]),
            'max_user_current': float(parts[3]),
            'display_mode': int(parts[4])  # 1=Current, 2=Power
        }

    def get_input_curve(self, input_channel: str) -> int:
        """Get input curve number."""
        input_channel = self._validate_input(input_channel)
        response = self._query(f"INCRV? {input_channel}")
        return int(response)

    def get_input_name(self, input_channel: str) -> str:
        """Get input sensor name."""
        input_channel = self._validate_input(input_channel)
        return self._query(f"INNAME? {input_channel}")

    def get_input_type(self, input_channel: str) -> Dict[str, Any]:
        """Get input type parameters."""
        input_channel = self._validate_input(input_channel)
        response = self._query(f"INTYPE? {input_channel}")
        parts = response.split(',')
        return {
            'sensor_type': int(parts[0]),  # 0=Disabled, 1=Diode, 2=PTC RTD, 3=NTC RTD, 4=Thermocouple
            'autorange': bool(int(parts[1])),
            'range': int(parts[2]),
            'compensation': bool(int(parts[3])),
            'units': int(parts[4])  # 1=Kelvin, 2=Celsius, 3=Sensor
        }

    def get_manual_output(self, output: int) -> float:
        """Get manual output percentage."""
        output = self._validate_output(output)
        response = self._query(f"MOUT? {output}")
        return float(response)

    def get_output_mode(self, output: int) -> Dict[str, Any]:
        """Get output mode parameters."""
        output = self._validate_output(output)
        response = self._query(f"OUTMODE? {output}")
        parts = response.split(',')
        return {
            'mode': int(parts[0]),  # 0=Off, 1=Closed Loop PID, 2=Zone, 3=Open Loop, 4=Monitor Out, 5=Warmup Supply
            'input': int(parts[1]),  # 0=None, 1=A, 2=B
            'powerup_enable': bool(int(parts[2]))
        }

    def get_heater_range(self, output: int) -> int:
        """Get heater range setting."""
        output = self._validate_output(output)
        response = self._query(f"RANGE? {output}")
        return int(response)

    def get_temperature_limit(self, input_channel: str) -> float:
        """Get temperature limit for input."""
        input_channel = self._validate_input(input_channel)
        response = self._query(f"TLIMIT? {input_channel}")
        return float(response)

    def get_sensor_reading_raw(self, input_channel: str) -> float:
        """Get raw sensor reading in sensor units."""
        input_channel = self._validate_input(input_channel)
        response = self._query(f"SRDG? {input_channel}")
        return float(response)

    def get_min_max_data(self, input_channel: str) -> Tuple[float, float]:
        """Get minimum and maximum recorded data."""
        input_channel = self._validate_input(input_channel)
        response = self._query(f"MDAT? {input_channel}")
        parts = response.split(',')
        return (float(parts[0]), float(parts[1]))  # (min, max)

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Read all parameters stored in the Lake Shore 335 and return as dictionary.

        This comprehensive method queries all available parameters from the instrument
        including input configurations, output settings, PID values, alarm settings,
        temperature readings, and system status.

        Returns:
            Dictionary containing all instrument parameters organized by category.
        """
        all_params = {
            'identification': {},
            'inputs': {},
            'outputs': {},
            'control': {},
            'alarms': {},
            'system': {}
        }

        try:
            # Identification
            all_params['identification']['id_string'] = self.get_identification()

            # Input parameters for each channel
            for input_ch in self.inputs:
                input_data = {}
                input_data['temperature_K'] = self.fetch_T(input_ch, 'K')
                input_data['temperature_C'] = self.fetch_T(input_ch, 'C')
                input_data['sensor_reading'] = self.get_sensor_reading_raw(input_ch)
                input_data['min_max'] = self.get_min_max_data(input_ch)
                input_data['type_parameters'] = self.get_input_type(input_ch)
                input_data['curve_number'] = self.get_input_curve(input_ch)
                input_data['name'] = self.get_input_name(input_ch)
                input_data['filter'] = self.get_filter_parameters(input_ch)
                input_data['temperature_limit'] = self.get_temperature_limit(input_ch)
                all_params['inputs'][input_ch] = input_data

            # Output parameters for each channel
            for output_ch in self.outputs:
                output_data = {}
                output_data['setpoint'] = self.get_setpoint(output_ch)
                output_data['heater_output'] = self.get_heater_output(output_ch)
                output_data['heater_range'] = self.get_heater_range(output_ch)
                output_data['manual_output'] = self.get_manual_output(output_ch)
                output_data['mode'] = self.get_output_mode(output_ch)
                output_data['heater_setup'] = self.get_heater_setup(output_ch)
                output_data['ramp_rate'] = self.get_ramp_rate(output_ch)
                output_data['ramp_enabled'] = self.get_ramp_status(output_ch)
                all_params['outputs'][output_ch] = output_data

            # Control parameters
            all_params['control']['pid'] = {}
            all_params['control']['zones'] = {}
            for output_ch in self.outputs:
                all_params['control']['pid'][output_ch] = {
                    'P': self.get_PID(output_ch)[0],
                    'I': self.get_PID(output_ch)[1],
                    'D': self.get_PID(output_ch)[2]
                }
                all_params['control']['zones'][output_ch] = self.get_zones(output_ch)
            all_params['control']['autotune_status'] = self.get_autotune_status()

            # Alarm parameters
            all_params['alarms'] = {}
            for input_ch in self.inputs:
                all_params['alarms'][input_ch] = self.get_alarm_parameters(input_ch)

            # System parameters
            all_params['system']['display_brightness'] = self.get_display_brightness()

        except Exception as e:
            # If any parameter fails to read, include the error in the results
            all_params['_errors'] = str(e)

        return all_params

    # ==================== Setter Methods ====================

    def set_alarm_parameters(self, input_channel: str, enabled: bool = False,
                           high_setpoint: float = 1000.0, low_setpoint: float = 0.0,
                           deadband: float = 1.0, latch_enable: bool = False,
                           audible: bool = True, visible: bool = True):
        """Set alarm parameters for input."""
        input_channel = self._validate_input(input_channel)
        self._write(f"ALARM {input_channel},{int(enabled)},{high_setpoint},"
                   f"{low_setpoint},{deadband},{int(latch_enable)},"
                   f"{int(audible)},{int(visible)}")

    def set_display_brightness(self, brightness: int):
        """Set display brightness (0-3: 25%, 50%, 75%, 100%)."""
        if brightness not in range(4):
            raise ValueError("Brightness must be 0-3")
        self._write(f"BRIGT {brightness}")

    def set_filter_parameters(self, input_channel: str, enabled: bool,
                            points: int = 8, window: int = 10):
        """Set input filter parameters."""
        input_channel = self._validate_input(input_channel)
        if not (2 <= points <= 64):
            raise ValueError("Filter points must be 2-64")
        if not (1 <= window <= 10):
            raise ValueError("Filter window must be 1-10%")
        self._write(f"FILTER {input_channel},{int(enabled)},{points},{window}")

    def set_heater_range(self, output: int, range_setting: int):
        """Set heater range (0=Off, 1=Low, 2=Med, 3=High)."""
        output = self._validate_output(output)
        if range_setting not in range(4):
            raise ValueError("Range must be 0-3")
        self._write(f"RANGE {output},{range_setting}")

    def set_input_curve(self, input_channel: str, curve_number: int):
        """Set input curve number (0=none, 1-20=standard, 21-59=user)."""
        input_channel = self._validate_input(input_channel)
        self._write(f"INCRV {input_channel},{curve_number}")

    def set_input_name(self, input_channel: str, name: str):
        """Set input sensor name (max 15 characters)."""
        input_channel = self._validate_input(input_channel)
        if len(name) > 15:
            raise ValueError("Name must be 15 characters or less")
        self._write(f'INNAME {input_channel},"{name}"')

    def set_manual_output(self, output: int, percentage: float):
        """Set manual output percentage (0-100%)."""
        output = self._validate_output(output)
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage must be 0-100")
        self._write(f"MOUT {output},{percentage}")

    def set_output_mode(self, output: int, mode: int, input_channel: int = 0,
                       powerup_enable: bool = False):
        """
        Set output mode.

        Args:
            output: Output channel (1 or 2)
            mode: 0=Off, 1=Closed Loop PID, 2=Zone, 3=Open Loop, 4=Monitor Out, 5=Warmup Supply
            input_channel: Control input (0=None, 1=A, 2=B)
            powerup_enable: Enable output after power cycle
        """
        output = self._validate_output(output)
        if mode not in range(6):
            raise ValueError("Mode must be 0-5")
        if input_channel not in range(3):
            raise ValueError("Input channel must be 0-2")
        self._write(f"OUTMODE {output},{mode},{input_channel},{int(powerup_enable)}")

    def set_temperature_limit(self, input_channel: str, limit: float):
        """Set temperature limit for input (0 = disabled)."""
        input_channel = self._validate_input(input_channel)
        self._write(f"TLIMIT {input_channel},{limit}")

    # ==================== Utility Methods ====================

    def reset_min_max(self):
        """Reset minimum and maximum data for all inputs."""
        self._write("MNMXRST")

    def all_heaters_off(self):
        """Turn off all heater outputs."""
        for output in self.outputs:
            self.set_heater_range(output, 0)

    def wait_for_temperature(self, output: int, target_temp: float,
                           tolerance: float = 0.1, timeout: int = 600,
                           check_interval: float = 1.0) -> bool:
        """
        Wait for temperature to reach target within tolerance.

        Args:
            output: Output channel to monitor
            target_temp: Target temperature
            tolerance: Acceptable deviation from target
            timeout: Maximum wait time in seconds
            check_interval: Time between checks in seconds

        Returns:
            True if target reached, False if timeout
        """
        # Get the control input for this output
        mode_info = self.get_output_mode(output)
        input_map = {0: None, 1: 'A', 2: 'B'}
        control_input = input_map[mode_info['input']]

        if control_input is None:
            raise ValueError(f"Output {output} has no control input assigned")

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            current_temp = self.fetch_T(control_input)
            if abs(current_temp - target_temp) <= tolerance:
                return True
            time.sleep(check_interval)

        return False



if __name__ == "__main__":
    print("Module for the VISA interface using PyVISA.")