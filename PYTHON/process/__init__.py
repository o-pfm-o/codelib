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
from typing import Union, Tuple, Optional


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
