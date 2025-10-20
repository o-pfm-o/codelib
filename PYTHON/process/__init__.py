'''
Module for input and output of data, e.g. reading files.
Created on 2025-05-20.
Creator: PFM
'''

from . import tfanalyzer
import os
import pickle
import tkinter as tk
from tkinter import filedialog
from typing import Union, Tuple, Optional, Any, List, Dict
import copy


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

