'''
Module for input and output of data, e.g. reading files.
Created on 2025-05-20.
Creator: PFM
'''

from . import tfanalyzer
from . import dtmanag
import os
import pickle
import tkinter as tk
from tkinter import filedialog
from typing import Union, Tuple, Optional, Any, List, Dict
import copy
import pandas as pd
import numpy as np



def directory_query(
    mode: Optional[str] = None,
    title: str = "Select file or directory",
    initialdir: str = ".",
    filetypes: Tuple[Tuple[str, str], ...] = (("All files", "*.*"),)
) -> Union[str, Tuple[str, ...], None]:
    """
    Opens a Tkinter file or directory dialog.

    Parameters
    ----------
    mode : str, optional
        Determines the dialog type:
        * None: Open a single file.
        * 'files': Open multiple files.
        * 'dir': Choose a directory.
        * 'save': Save a file.
    title : str, optional
        Window title for the dialog.
    initialdir : str, optional
        Initial directory the dialog opens in.
    filetypes : tuple, optional
        Tuple of supported file types.

    Returns
    -------
    str or tuple or None
        The selected path(s) or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)

    dialog_kwargs = {"title": title, "initialdir": initialdir, "filetypes": filetypes}

    if mode == "dir":
        result = filedialog.askdirectory(**dialog_kwargs)
    elif mode == "files":
        result = filedialog.askopenfilenames(**dialog_kwargs)
    elif mode == "save":
        result = filedialog.asksaveasfilename(**dialog_kwargs)
    else:
        result = filedialog.askopenfilename(**dialog_kwargs)

    root.destroy()
    return result

class Args:
    """
    A helper class for managing positional arguments.

    This class wraps positional arguments into a single object,
    allowing storage in NumPy arrays or other data containers.
    It automatically merges nested iterables (lists, tuples)
    into a flattened internal list.

    Example:
        >>> a = Args(1, [2, 3], (4, 5))
        >>> a.get()
        [1, 2, 3, 4, 5]
    """

    def __init__(self, *args: Any):
        """
        Initialize the Args object by merging positional arguments.

        Args:
            *args: Variable number of positional arguments. Each argument
                can be a single value, list, or tuple.
        """
        merged: List[Any] = []

        # Flatten nested lists/tuples into a single list.
        for arg in args:
            if isinstance(arg, (list, tuple)):
                merged.extend(arg)
            else:
                merged.append(arg)

        self.merged = merged

    def get(self) -> List[Any]:
        """
        Retrieve the merged list of arguments.

        Returns:
            list: A list containing all merged positional arguments.
        """
        return self.merged
    
class Kwargs:
    """
    A helper class for managing keyword arguments.

    This class merges multiple dictionaries of keyword arguments into one.
    It supports nested dictionary merging for overlapping keys with `dict`
    values, and overwrites non-dict key collisions.

    Example:
        >>> k = Kwargs({'a': 1, 'b': {'x': 10}}, {'b': {'y': 20}, 'c': 3})
        >>> k.get()
        {'a': 1, 'b': {'x': 10, 'y': 20}, 'c': 3}
    """

    def __init__(self, *kwargs: Dict[str, Any]):
        """
        Initialize the Kwargs object by merging provided dictionaries.

        Args:
            *kwargs: Variable number of dictionaries. Each dictionary may
                contain any combination of key–value pairs.
        """
        kwargs_copy = copy.deepcopy(kwargs)
        self.merged: Dict[str, Any] = {}

        # Iterate through provided kwargs dictionaries
        for kwarg in kwargs_copy:
            # Identify keys that overlap (intersect) and those that are new
            intersect = set(kwarg.keys()).intersection(self.merged.keys())
            difference = set(kwarg.keys()) - self.merged.keys()

            # Merge overlapping keys
            for key in intersect:
                current_val = self.merged[key]
                new_val = kwarg[key]

                # If both values are dicts, update the existing dictionary
                if isinstance(new_val, dict) and isinstance(current_val, dict):
                    current_val.update(new_val)
                else:
                    # Otherwise, overwrite existing value
                    self.merged[key] = new_val

            # Add keys not previously present
            for key in difference:
                self.merged[key] = kwarg[key]

    def get(self) -> Dict[str, Any]:
        """
        Retrieve the merged dictionary of keyword arguments.

        Returns:
            dict: A merged dictionary combining all provided keyword arguments.
        """
        return self.merged
    
def read_sspfm_forcecurve(path: str, amplification: float = 1) -> pd.DataFrame:
    """
    Extract and process force curve data from Asylum Hysteresis CSV files.
    
    This function reads force plot data from a CSV file, separates amplitude-on 
    (AmpOn) from amplitude-off (AmpOff) measurements, and processes the data into 
    a structured DataFrame. Specifically designed for Asylum Hysteresis curves 
    with 2 cycles.
    
    Parameters
    ----------
    path : str
        File path to the CSV file containing force curve data.
    amplification : float, optional
        Multiplication factor to apply to the Bias column (default is 1).
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing statistical descriptions of both AmpOn and AmpOff 
        measurements for each cycle. Columns are named with the pattern:
        '{AmpOn/AmpOff}_{ColumnName}_{StatisticName}'
    
    Notes
    -----
    - The input CSV is expected to have specific columns: Raw, Deflection, 
      Amplitude, Phase_1, Phase_2, Frequency, ZSnsr, and Bias.
    - The function detects transitions between AmpOff and AmpOn states by 
      identifying bias jumps greater than 1.
    - If the number of AmpOff and AmpOn segments differ, the extra AmpOn 
      segments are discarded to ensure equal lengths.
    
    Examples
    --------
    >>> fc_data = self.read_forcecurve('forcecurve_data.csv', amplification=2.0)
    >>> print(fc_data.shape)
    (n_cycles, n_features)
    """
    
    # Read the CSV file with appropriate parameters
    fc_raw = pd.read_csv(
        path,
        header=None,
        skiprows=1,
        index_col=0,
        names=('Raw', 'Deflection', 'Amplitude', 'Phase_1', 
               'Phase_2', 'Frequency', 'ZSnsr', 'Bias')
    )
    
    # Remove any rows with missing values
    fc_raw = fc_raw.dropna()
    
    # Apply amplification factor to the Bias column
    fc_raw['Bias'] = fc_raw['Bias'] * amplification
    
    # Initialize lists to store AmpOn and AmpOff data segments
    amp_on = []
    amp_off = []
    
    # Initialize tracking variables for segment detection
    ref_index = 0  # Starting index of current segment
    ref_bias = 0  # Bias value at the start of current segment
    bias_on = False  # Flag to track current state (AmpOff or AmpOn)
    first = True  # Flag to handle the first transition specially
    
    # Iterate through the dataframe to detect state transitions
    for index, row in fc_raw.iterrows():
        # Update reference bias at the start of each segment
        if ref_index == index:
            ref_bias = row['Bias']
        
        # Detect state transition: bias jump greater than 1
        if np.abs(row['Bias'] - ref_bias) > 1:
            # Handle the first transition (initial calibration segment)
            if first:
                first = False
                segment_stats = fc_raw.loc[ref_index:(index - 1), :].describe()  # type: ignore
                amp_on.append(segment_stats)
                amp_off.append(segment_stats)
            else:
                # Classify segment based on current state
                segment_stats = fc_raw.loc[ref_index:(index - 1), :].describe() # type: ignore
                if bias_on:
                    amp_on.append(segment_stats)
                else:
                    amp_off.append(segment_stats)
            
            # Toggle state and update reference values
            bias_on = not bias_on
            ref_index = index
            ref_bias = row['Bias']
    
    # Ensure equal lengths by truncating the longer list
    amp_off_len = len(amp_off)
    amp_on_len = len(amp_on)
    if amp_off_len < amp_on_len:
        amp_on = amp_on[0:amp_off_len]
    
    # Generate column headers for the output DataFrame
    columns = amp_on[0].columns.values
    rows = amp_on[0].index.values
    headers = []
    for bias in ['AmpOn', 'AmpOff']:
        for column in columns:
            for row in rows:
                headers.append('_'.join((bias, column, row)))
    
    # Construct the output matrix by combining AmpOn and AmpOff statistics
    matrix = []
    for index, on in enumerate(amp_on):
        off = amp_off[index]
        line_on = []
        line_off = []
        
        # Extract statistics for each column and row combination
        for column in columns:
            for row in rows:
                line_on.append(on.loc[row, column])
                line_off.append(off.loc[row, column])
        
        # Combine AmpOn and AmpOff data for this cycle
        line_on.extend(line_off)
        matrix.append(line_on)
    
    # Convert matrix to DataFrame with descriptive headers
    matrix = pd.DataFrame(data=matrix, columns=headers)
    
    return matrix


