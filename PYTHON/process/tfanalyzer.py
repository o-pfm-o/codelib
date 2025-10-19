"""
TF Analyzer 3000 Data Parser
============================

Improved and PEP 8 compliant parser for TF Analyzer ferroelectric measurement data files.

This module provides functions to read and parse data from TF Analyzer 3000
ferroelectric measurement files (.dat format) with improved structure, 
comprehensive documentation, and robust error handling.

Author: User/Assistant Collaboration
Date: October 2025
License: MIT
"""

import pandas as pd
from typing import Dict, Any, Union


def read_TFAnalyzer_3000(file_path: str) -> Dict[str, Any]:
    """
    Read and parse TF Analyzer 3000 ferroelectric measurement data.

    This function parses TF Analyzer 3000 data files containing dynamic hysteresis
    measurements. The function processes the hierarchical structure of the data file
    and organizes it into a nested dictionary with metadata and measurement tables.

    Parameters
    ----------
    file_path : str
        Path to the TF Analyzer .dat file to be parsed.

    Returns
    -------
    Dict[str, Any]
        Nested dictionary structure containing:
        - Top level: measurement type keys (e.g., 'DynamicHysteresis')
        - Second level: 
            * General metadata (Program, TimeStamp, etc.)
            * Table keys (e.g., 'Table 1', 'Table 2', etc.)
        - Third level (for tables): 
            * Measurement metadata fields
            * 'table': pandas DataFrame with time-series data

    Examples
    --------
    >>> data = read_TFAnalyzer_3000('measurement_data.dat')
    >>> 
    >>> # Access general file metadata
    >>> program_version = data['DynamicHysteresis']['Program']
    >>> timestamp = data['DynamicHysteresis']['TimeStamp']
    >>> 
    >>> # Access specific table's metadata
    >>> table1_metadata = data['DynamicHysteresis']['Table 1']
    >>> frequency = table1_metadata['Hysteresis Frequency [Hz]']
    >>> 
    >>> # Access time-series data
    >>> time_series = data['DynamicHysteresis']['Table 1']['table']
    >>> voltage_data = time_series['V+ [V]']
    >>> polarization_data = time_series['P1 [uC/cm2]']

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist
    ValueError
        If the file format is not recognized or cannot be parsed

    Notes
    -----
    The function expects a specific TF Analyzer file format with:
    1. Measurement type headers (e.g., 'DynamicHysteresis')
    2. General metadata blocks with key: value pairs
    3. Table sections with individual measurement metadata
    4. Tab-delimited time-series data with headers

    The time-series data tables typically contain columns:
    - Time [s]: Time points during measurement
    - V+ [V], V- [V]: Positive and negative voltage measurements
    - I1 [A], I2 [A], I3 [A]: Current measurements from multiple channels
    - P1 [uC/cm2], P2 [uC/cm2], P3 [uC/cm2]: Polarization measurements
    """
    try:
        # Read all lines from the file with explicit encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")

    # Initialize the main data dictionary
    data_dict = {}
    main_key = ''  # Main key for measurement type (e.g., 'DynamicHysteresis')
    sub_key = ''   # Sub key for individual tables (e.g., 'Table 1')
    skips = 0      # Counter for lines to skip in the main loop

    # Process each line in the file
    for i, line in enumerate(lines):
        # Skip lines if indicated by previous operations
        if skips:
            skips -= 1
            continue

        # Check for main measurement type headers
        if line.strip() == 'DynamicHysteresis':
            main_key = line.strip()
            data_dict[main_key] = {}

            # Parse general header block following the main key
            j = i + 1
            while j < len(lines) and lines[j].strip() != '':
                skips += 1

                # Split header line into key-value pairs
                header_parts = lines[j].strip().split(': ', 1)
                if len(header_parts) == 2:
                    key, val = header_parts
                    # Convert to numeric if possible, otherwise keep as string
                    data_dict[main_key][key] = _convert_to_numeric(val)
                j += 1

            # Process table blocks within this measurement type
            while j < len(lines) and not lines[j].startswith('DynamicHysteresisResult'):
                skips += 1

                # Check for new table marker (empty line followed by table name)
                if lines[j].strip() == '':
                    # Look ahead for table identifier
                    if (j + 1 < len(lines) and 
                        lines[j + 1].strip().startswith('Table')):
                        sub_key = lines[j + 1].strip()
                        data_dict[main_key][sub_key] = {}
                        j += 2
                        skips += 1
                        continue

                # Check for beginning of time-series data table
                elif 'Time [s]' in lines[j]:
                    header_line_index = j

                    # Find the end of the data table (next empty line)
                    data_start = j + 1
                    while j < len(lines) and lines[j].strip() != '':
                        j += 1

                    # Calculate number of data rows
                    num_rows = j - data_start

                    # Read the data table using pandas
                    if num_rows > 0 and sub_key:
                        try:
                            data_dict[main_key][sub_key]['table'] = pd.read_csv(
                                file_path,
                                header=header_line_index,
                                delimiter='\t',
                                usecols=range(9),  # Limit to 9 columns to avoid empty ones
                                nrows=num_rows,
                                skip_blank_lines=False,
                                encoding='utf-8'
                            )
                        except Exception as e:
                            print(f"Warning: Could not parse data table in {sub_key}: {e}")
                            data_dict[main_key][sub_key]['table'] = None

                    continue

                # Parse metadata entries within table sections
                else:
                    metadata_parts = lines[j].strip().split(': ', 1)
                    if len(metadata_parts) == 2 and sub_key:
                        key, val = metadata_parts
                        data_dict[main_key][sub_key][key] = _convert_to_numeric(val)

                j += 1

                # Safety check to prevent infinite loops
                if j >= len(lines):
                    break

    if not data_dict:
        raise ValueError("No valid measurement data found in file")

    return data_dict


def _convert_to_numeric(value: str) -> Union[float, str]:
    """
    Attempt to convert a string value to a numeric type.

    This helper function tries to convert string values to floats,
    preserving the original string if conversion fails.

    Parameters
    ----------
    value : str
        String value to convert

    Returns
    -------
    Union[float, str]
        Float value if conversion successful, original string otherwise

    Examples
    --------
    >>> _convert_to_numeric("123.45")
    123.45
    >>> _convert_to_numeric("text_value")
    'text_value'
    >>> _convert_to_numeric("1.23e-10")
    1.23e-10
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def display_data_structure(data_dict: Dict[str, Any], max_depth: int = 2) -> None:
    """
    Display the hierarchical structure of the parsed TF Analyzer data.

    This function provides a clear overview of the data structure,
    showing measurement types, tables, and data availability.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Parsed data dictionary from read_TFAnalyzer_3000
    max_depth : int, optional
        Maximum depth to display in the structure, by default 2

    Examples
    --------
    >>> data = read_TFAnalyzer_3000('measurement.dat')
    >>> display_data_structure(data)
    """
    print("TF Analyzer Data Structure:")
    print("=" * 50)

    for main_key, main_value in data_dict.items():
        print(f"\n{main_key}:")

        if isinstance(main_value, dict):
            # Display general metadata first
            general_metadata = []
            table_keys = []

            for key in main_value.keys():
                if key.startswith('Table'):
                    table_keys.append(key)
                else:
                    general_metadata.append(key)

            # Show general metadata
            if general_metadata:
                print(f"  General metadata: {len(general_metadata)} fields")
                for meta_key in general_metadata[:5]:  # Show first 5
                    print(f"    {meta_key}: {main_value[meta_key]}")
                if len(general_metadata) > 5:
                    print(f"    ... and {len(general_metadata) - 5} more")

            # Show table information
            for sub_key in sorted(table_keys):
                sub_value = main_value[sub_key]
                print(f"  {sub_key}:")

                if isinstance(sub_value, dict):
                    # Count metadata fields (excluding 'table')
                    metadata_count = sum(1 for k in sub_value.keys() if k != 'table')
                    print(f"    Metadata fields: {metadata_count}")

                    # Display table info if present
                    if 'table' in sub_value and sub_value['table'] is not None:
                        table = sub_value['table']
                        print(f"    Data table: {table.shape[0]} rows × {table.shape[1]} columns")
                        print(f"    Columns: {list(table.columns)}")
                    else:
                        print(f"    Data table: Not available")


def get_measurement_summary(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of measurements in the TF Analyzer data.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Parsed data dictionary from read_TFAnalyzer_3000

    Returns
    -------
    Dict[str, Any]
        Summary information including:
        - measurement_types: List of measurement types found
        - total_tables: Total number of tables
        - tables_with_data: Number of tables with successfully parsed data
        - table_details: Per-measurement-type breakdown

    Examples
    --------
    >>> data = read_TFAnalyzer_3000('measurement.dat')
    >>> summary = get_measurement_summary(data)
    >>> print(f"Found {summary['total_tables']} tables")
    """
    summary = {
        'measurement_types': [],
        'total_tables': 0,
        'tables_with_data': 0,
        'table_details': {}
    }

    for main_key, main_value in data_dict.items():
        summary['measurement_types'].append(main_key)

        if isinstance(main_value, dict):
            table_count = 0
            data_count = 0

            for sub_key, sub_value in main_value.items():
                if sub_key.startswith('Table'):
                    table_count += 1
                    summary['total_tables'] += 1

                    # Check if table has successfully parsed data
                    if (isinstance(sub_value, dict) and 
                        'table' in sub_value and 
                        sub_value['table'] is not None):
                        data_count += 1
                        summary['tables_with_data'] += 1

            summary['table_details'][main_key] = {
                'tables': table_count,
                'with_data': data_count
            }

    return summary


def extract_measurement_parameters(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract key measurement parameters into a summary DataFrame.

    This function creates a DataFrame with one row per table, containing
    the most important measurement parameters for easy analysis.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Parsed data dictionary from read_TFAnalyzer_3000

    Returns
    -------
    pd.DataFrame
        DataFrame with measurement parameters, one row per table

    Examples
    --------
    >>> data = read_TFAnalyzer_3000('measurement.dat')
    >>> params_df = extract_measurement_parameters(data)
    >>> print(params_df[['Table', 'Hysteresis Frequency [Hz]', 'Vc+ [V]']])
    """
    rows = []

    for main_key, main_value in data_dict.items():
        if isinstance(main_value, dict):
            for sub_key, sub_value in main_value.items():
                if sub_key.startswith('Table') and isinstance(sub_value, dict):
                    row = {'Measurement_Type': main_key, 'Table': sub_key}

                    # Extract key parameters if they exist
                    key_params = [
                        'Timestamp', 'Hysteresis Frequency [Hz]', 
                        'Hysteresis Amplitude [V]', 'Vc+ [V]', 'Vc- [V]',
                        'Pr+ [uC/cm2]', 'Pr- [uC/cm2]', 'Psw [uC/cm2]',
                        'Wloss [uJ/cm2]', 'Area [mm2]', 'Thickness [nm]'
                    ]

                    for param in key_params:
                        row[param] = sub_value.get(param, None)

                    # Add data availability flag
                    row['Has_Data'] = ('table' in sub_value and 
                                     sub_value['table'] is not None)

                    rows.append(row)

    return pd.DataFrame(rows)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage and testing of the improved TF Analyzer parser.
    """
    try:
        print("=== TF Analyzer 3000 Parser - Example Usage ===\n")

        # Parse the data file
        data = read_TFAnalyzer_3000('TFAnalyzer-testdata.dat')

        # Display file structure
        display_data_structure(data)

        # Generate and display summary
        summary = get_measurement_summary(data)
        print(f"\n\nSUMMARY:")
        print("=" * 50)
        print(f"Measurement types: {summary['measurement_types']}")
        print(f"Total tables found: {summary['total_tables']}")
        print(f"Tables with data: {summary['tables_with_data']}")

        for mtype, details in summary['table_details'].items():
            success_rate = details['with_data'] / details['tables'] * 100
            print(f"{mtype}: {details['with_data']}/{details['tables']} "
                  f"tables have data ({success_rate:.1f}%)")

        # Extract measurement parameters
        params_df = extract_measurement_parameters(data)
        print(f"\n\nMEASUREMENT PARAMETERS:")
        print("=" * 50)
        key_cols = ['Table', 'Hysteresis Frequency [Hz]', 
                   'Hysteresis Amplitude [V]', 'Vc+ [V]', 'Vc- [V]']
        available_cols = [col for col in key_cols if col in params_df.columns]
        if available_cols and not params_df.empty:
            print(params_df[available_cols].head().to_string(index=False))

        # Show example data access
        print(f"\n\nEXAMPLE DATA ACCESS:")
        print("=" * 50)

        # Access general file information
        if 'DynamicHysteresis' in data:
            dh_data = data['DynamicHysteresis']
            print(f"Program: {dh_data.get('Program', 'N/A')}")
            print(f"Timestamp: {dh_data.get('TimeStamp', 'N/A')}")
            print(f"TFA Version: {dh_data.get('TfaVersion', 'N/A')}")

            # Access specific table data
            table_key = 'Table 1'
            if table_key in dh_data and 'table' in dh_data[table_key]:
                table_data = dh_data[table_key]['table']
                if table_data is not None:
                    print(f"\n{table_key} time-series data:")
                    print(f"  Shape: {table_data.shape}")
                    print(f"  Time range: {table_data['Time [s]'].min():.4f} - "
                          f"{table_data['Time [s]'].max():.4f} seconds")
                    print(f"  Voltage range: {table_data['V+ [V]'].min():.2f} - "
                          f"{table_data['V+ [V]'].max():.2f} V")

        print("\n✓ All operations completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
