'''
Generalized Data I/O Handler Module

A comprehensive solution for saving and loading data in multiple file formats
with support for NumPy arrays, Pandas DataFrames, lists, dictionaries, and 3D voxel data.

Supported formats:
- Pickle (.pkl, .pickle)
- JSON (.json)
- CSV (.csv)
- Text (.txt)
- HDF5 (.h5, .hdf5)
- TIFF (.tif, .tiff) - for 3D voxel data with ImageJ compatibility
- Excel (.xlsx, .xls) - compatible with Origin, LabPlot
- NumPy binary (.npy, .npz)
- NIfTI (.nii, .nii.gz) - for medical/neuroimaging data

Author: Enhanced from original POLDER data handler
License: MIT
Python: 3.9+

Usage:
    from data_io_handler import save_file, load_file
    
    # Save data
    save_file(data_array, 'output.h5')
    
    # Load data
    data = load_file('output.h5')
'''

import os
import pickle
import json
import csv
import warnings
from pathlib import Path
from typing import Any, Union, Dict, List, Optional, TYPE_CHECKING

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    import h5py

# Optional imports with graceful degradation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy not available. NumPy array support disabled.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("Pandas not available. DataFrame support disabled.")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not available. HDF5 support disabled.")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    warnings.warn("tifffile not available. TIFF support for voxel data disabled.")

try:
    from openpyxl import load_workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    warnings.warn("openpyxl not available. Excel append mode disabled.")

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel not available. NIfTI support disabled.")


# ========== Module Constants ==========

SUPPORTED_FORMATS = {
    'pickle': ['.pkl', '.pickle'],
    'json': ['.json'],
    'csv': ['.csv'],
    'txt': ['.txt'],
    'hdf5': ['.h5', '.hdf5'],
    'tiff': ['.tiff', '.tif'],
    'excel': ['.xlsx', '.xls'],
    'numpy': ['.npy', '.npz'],
    'nifti': ['.nii', '.nii.gz']
}


# ========== Helper Functions (Private) ==========

def _separate_file_folder(filename: Union[str, Path]) -> tuple[str, str]:
    '''
    Separate filename and folder path.
    
    Parameters
    ----------
    filename : str or Path
        Full path or just filename
        
    Returns
    -------
    tuple[str, str]
        Tuple of (filename, path)
        
    Examples
    --------
    >>> _separate_file_folder('/data/test.h5')
    ('test.h5', '/data')
    '''
    path_obj = Path(filename)
    return path_obj.name, str(path_obj.parent)


def _ensure_directory(path: Union[str, Path]) -> None:
    '''
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str or Path
        Directory path to create
        
    Notes
    -----
    Creates parent directories as needed (equivalent to mkdir -p)
    '''
    if path and path != '.':
        Path(path).mkdir(parents=True, exist_ok=True)


def _detect_format(filename: str) -> str:
    '''
    Detect file format from extension.
    
    Parameters
    ----------
    filename : str
        Filename with extension
        
    Returns
    -------
    str
        Detected format name
        
    Raises
    ------
    ValueError
        If file extension is not supported
    '''
    ext = Path(filename).suffix.lower()
    
    # Handle compressed NIfTI
    if filename.endswith('.nii.gz'):
        return 'nifti'
    
    for format_name, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return format_name
    
    raise ValueError(
        f"Unsupported file format: {ext}. "
        f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
    )


def _save_dict_to_hdf5(h5group: Any, data: Dict[str, Any], compression: str) -> None:
    '''
    Recursively save dictionary to HDF5 group.
    
    Parameters
    ----------
    h5group : h5py.Group
        HDF5 group to save to
    data : dict
        Dictionary to save
    compression : str
        Compression type (e.g., 'gzip')
    '''
    for key, value in data.items():
        safe_key = str(key)
        if isinstance(value, dict):
            subgroup = h5group.create_group(safe_key)
            _save_dict_to_hdf5(subgroup, value, compression)
        elif HAS_NUMPY and isinstance(value, np.ndarray):
            h5group.create_dataset(safe_key, data=value, compression=compression)
        else:
            try:
                h5group.create_dataset(safe_key, data=value)
            except (TypeError, ValueError):
                # Store as string if direct storage fails
                h5group.create_dataset(safe_key, data=str(value))


def _load_hdf5_recursive(h5group: Any) -> Dict[str, Any]:
    '''
    Recursively load HDF5 group structure into dictionary.
    
    Parameters
    ----------
    h5group : h5py.Group
        HDF5 group to load from
        
    Returns
    -------
    dict
        Dictionary containing loaded data
    '''
    result: Dict[str, Any] = {}
    for key in h5group.keys():
        item = h5group[key]
        if hasattr(item, 'keys'):  # It's a Group
            result[key] = _load_hdf5_recursive(item)
        else:  # It's a Dataset
            result[key] = item[()]  # Use [()] instead of [:] for better compatibility
    return result


# ========== Format-Specific Save Functions ==========

def _save_pickle(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save object using pickle (Python-specific, fastest).'''
    protocol = kwargs.get('protocol', pickle.HIGHEST_PROTOCOL)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def _save_json(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save object as JSON (human-readable, universal).'''
    indent = kwargs.get('indent', 2)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, cls=NumpyJSONEncoder, **kwargs)


def _save_csv(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save object as CSV (universal, tabular data).'''
    if HAS_PANDAS and isinstance(obj, pd.DataFrame):
        obj.to_csv(filepath, index=kwargs.get('index', False))
    elif HAS_NUMPY and isinstance(obj, np.ndarray):
        delimiter = kwargs.get('delimiter', ',')
        np.savetxt(filepath, obj, delimiter=delimiter, **kwargs)
    elif isinstance(obj, (list, tuple)):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if obj and isinstance(obj[0], (list, tuple)):
                writer.writerows(obj)
            else:
                writer.writerow(obj)
    else:
        raise TypeError(
            f"Cannot save {type(obj).__name__} to CSV. "
            "Supported types: pd.DataFrame, np.ndarray, list"
        )


def _save_txt(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save object as plain text.'''
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        np.savetxt(filepath, obj, **kwargs)
    elif isinstance(obj, str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(obj)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(obj))


def _save_hdf5(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save object to HDF5 (efficient for large datasets).'''
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install h5py"
        )
    
    dataset_name = kwargs.get('dataset_name', 'data')
    compression = kwargs.get('compression', 'gzip')
    
    with h5py.File(filepath, 'w') as f:
        if HAS_PANDAS and isinstance(obj, pd.DataFrame):
            # Save DataFrame columns separately
            for col in obj.columns:
                f.create_dataset(
                    str(col),
                    data=obj[col].values,
                    compression=compression
                )
            # Store metadata as attributes
            f.attrs['columns'] = [str(c) for c in obj.columns]
            if not isinstance(obj.index, pd.RangeIndex):
                f.attrs['index'] = obj.index.tolist()
        elif HAS_NUMPY and isinstance(obj, np.ndarray):
            f.create_dataset(dataset_name, data=obj, compression=compression)
        elif isinstance(obj, dict):
            _save_dict_to_hdf5(f, obj, compression)
        else:
            try:
                f.create_dataset(dataset_name, data=obj, compression=compression)
            except TypeError:
                # Fallback for objects that can't be directly saved
                f.create_dataset(dataset_name, data=str(obj))


def _save_tiff(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''
    Save 3D voxel data as TIFF (ImageJ compatible).
    
    Notes
    -----
    Use imagej=True and provide metadata for ImageJ compatibility.
    Common metadata keys: 'spacing', 'unit', 'axes', 'fps'
    '''
    if not HAS_TIFFFILE:
        raise ImportError(
            "tifffile is required for TIFF support. "
            "Install with: pip install tifffile"
        )
    
    if not HAS_NUMPY or not isinstance(obj, np.ndarray):
        raise TypeError(
            f"TIFF format requires NumPy array. Got {type(obj).__name__}"
        )
    
    # tifffile.imwrite handles all kwargs including imagej, metadata, etc.
    tifffile.imwrite(filepath, obj, **kwargs)


def _save_excel(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save object to Excel file (compatible with Origin, LabPlot).'''
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for Excel support. "
            "Install with: pip install pandas openpyxl"
        )
    
    sheet_name = kwargs.get('sheet_name', 'Sheet1')
    
    if isinstance(obj, pd.DataFrame):
        # Single DataFrame
        obj.to_excel(
            filepath,
            sheet_name=sheet_name,
            index=kwargs.get('index', False),
            engine='openpyxl'
        )
    elif isinstance(obj, dict):
        # Multiple DataFrames as sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for name, df in obj.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(
                        writer,
                        sheet_name=str(name),
                        index=kwargs.get('index', False)
                    )
                else:
                    raise TypeError(
                        f"All dict values must be DataFrames, "
                        f"got {type(df).__name__} for key '{name}'"
                    )
    else:
        raise TypeError(
            f"Cannot save {type(obj).__name__} to Excel. "
            "Supported types: pd.DataFrame, dict of DataFrames"
        )


def _save_numpy(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save as NumPy binary format (fastest for NumPy arrays).'''
    if not HAS_NUMPY:
        raise ImportError(
            "numpy is required for .npy/.npz format. "
            "Install with: pip install numpy"
        )
    
    if filepath.endswith('.npz'):
        # Compressed archive for multiple arrays
        if isinstance(obj, dict):
            np.savez_compressed(filepath, **obj)
        else:
            np.savez_compressed(filepath, data=obj)
    else:
        # Single array
        if not isinstance(obj, np.ndarray):
            raise TypeError(
                f".npy format requires NumPy array, got {type(obj).__name__}"
            )
        np.save(filepath, obj)


def _save_nifti(obj: Any, filepath: str, **kwargs: Any) -> None:
    '''Save 3D/4D data as NIfTI (neuroimaging/medical imaging standard).'''
    if not HAS_NIBABEL:
        raise ImportError(
            "nibabel is required for NIfTI support. "
            "Install with: pip install nibabel"
        )
    
    if not HAS_NUMPY or not isinstance(obj, np.ndarray):
        raise TypeError(
            f"NIfTI format requires NumPy array. Got {type(obj).__name__}"
        )
    
    affine = kwargs.get('affine', np.eye(4) if HAS_NUMPY else None)
    
    # Use proper public API for nibabel
    nifti_img = nib.nifti1.Nifti1Image(obj, affine)
    nib.loadsave.save(nifti_img, filepath)


# ========== Format-Specific Load Functions ==========

def _load_pickle(filepath: str, **kwargs: Any) -> Any:
    '''Load object from pickle file.'''
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _load_json(filepath: str, **kwargs: Any) -> Any:
    '''Load object from JSON file.'''
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_csv(filepath: str, **kwargs: Any) -> Any:
    '''Load data from CSV file.'''
    if HAS_PANDAS:
        return pd.read_csv(filepath, **kwargs)
    elif HAS_NUMPY:
        delimiter = kwargs.get('delimiter', ',')
        return np.loadtxt(filepath, delimiter=delimiter, **kwargs)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.reader(f))


def _load_txt(filepath: str, **kwargs: Any) -> Any:
    '''Load data from text file.'''
    if kwargs.get('as_array', False) and HAS_NUMPY:
        return np.loadtxt(filepath, **kwargs)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


def _load_hdf5(filepath: str, **kwargs: Any) -> Any:
    '''Load data from HDF5 file.'''
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install h5py"
        )
    
    dataset_name = kwargs.get('dataset_name', 'data')
    
    with h5py.File(filepath, 'r') as f:
        # Check if it's a saved DataFrame
        if 'columns' in f.attrs and HAS_PANDAS:
            # Safe attribute access with proper type handling
            columns_attr = f.attrs.get('columns')
            if columns_attr is not None:
                columns = [str(c) for c in columns_attr]
                data = {col: f[col][()] for col in columns} # type: ignore
                df = pd.DataFrame(data)
                
                # Handle index if present
                index_attr = f.attrs.get('index')
                if index_attr is not None:
                    df.index = pd.Index(index_attr)
                return df
        
        if dataset_name in f:
            # Load specific dataset - use [()] for proper array loading
            return f[dataset_name][()] # type: ignore
        else:
            # Load entire file structure as dict
            return _load_hdf5_recursive(f)


def _load_tiff(filepath: str, **kwargs: Any) -> Any:
    '''Load TIFF file (including 3D stacks).'''
    if not HAS_TIFFFILE:
        raise ImportError(
            "tifffile is required for TIFF support. "
            "Install with: pip install tifffile"
        )
    return tifffile.imread(filepath, **kwargs)


def _load_excel(filepath: str, **kwargs: Any) -> Any:
    '''Load Excel file.'''
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for Excel support. "
            "Install with: pip install pandas openpyxl"
        )
    
    sheet_name = kwargs.get('sheet_name', 0)
    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)


def _load_numpy(filepath: str, **kwargs: Any) -> Any:
    '''Load NumPy binary file.'''
    if not HAS_NUMPY:
        raise ImportError(
            "numpy is required for .npy/.npz format. "
            "Install with: pip install numpy"
        )
    
    data = np.load(filepath, **kwargs)
    
    # For .npz files, return the NpzFile object
    # User can access arrays with data['array_name']
    return data


def _load_nifti(filepath: str, **kwargs: Any) -> Any:
    '''Load NIfTI file (returns data array).'''
    if not HAS_NIBABEL:
        raise ImportError(
            "nibabel is required for NIfTI support. "
            "Install with: pip install nibabel"
        )
    
    # Use proper public API for nibabel
    nifti_img = nib.loadsave.load(filepath)
    
    if kwargs.get('return_img', False):
        # Return full NIfTI image object (includes header, affine, etc.)
        return nifti_img
    else:
        # Return just the data array using proper method
        if hasattr(nifti_img, 'get_fdata'):
            return nifti_img.get_fdata() # type: ignore
        else:
            # Fallback for older nibabel versions
            return nifti_img.get_data()  # type: ignore


# ========== Public API Functions ==========

def save_file(
    obj: Any,
    filename: Union[str, Path],
    path: Union[str, Path] = '',
    file_format: Optional[str] = None,
    **kwargs: Any
) -> None:
    '''
    Save object to file in various formats.
    
    Parameters
    ----------
    obj : Any
        Object to save (np.ndarray, pd.DataFrame, list, dict, etc.)
    filename : str or Path
        Filename with extension
    path : str or Path, optional
        Directory path (default: '')
    file_format : str or None, optional
        Explicit format specification (default: auto-detect from extension)
    **kwargs : Any
        Additional keyword arguments for specific savers:
        - protocol : int (pickle) - Protocol version
        - indent : int (json) - Indentation level
        - dataset_name : str (hdf5) - Dataset name
        - compression : str (hdf5, npz) - Compression type
        - imagej : bool (tiff) - Save with ImageJ metadata
        - sheet_name : str (excel) - Excel sheet name
        
    Examples
    --------
    >>> # Save NumPy array to HDF5
    >>> data = np.random.rand(100, 100, 100)
    >>> save_file(data, 'voxel_data.h5', dataset_name='volume')
    
    >>> # Save DataFrame to CSV
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> save_file(df, 'data.csv')
    
    >>> # Save with ImageJ metadata for 3D microscopy
    >>> save_file(stack, 'volume.tif', imagej=True, 
    ...          metadata={'spacing': 3.0, 'unit': 'um'})
    '''
    # Separate filename and path if needed
    if path == '':
        filename_str, path = _separate_file_folder(filename)
    else:
        filename_str = str(Path(filename).name)
    
    # Ensure directory exists
    _ensure_directory(path)
    
    # Detect format if not specified
    if file_format is None:
        file_format = _detect_format(filename_str)
    
    # Construct full path
    full_path = os.path.join(path, filename_str) if path else filename_str
    
    print(f'Saving file: {filename_str}')
    
    # Dispatch to appropriate save method
    save_methods = {
        'pickle': _save_pickle,
        'json': _save_json,
        'csv': _save_csv,
        'txt': _save_txt,
        'hdf5': _save_hdf5,
        'tiff': _save_tiff,
        'excel': _save_excel,
        'numpy': _save_numpy,
        'nifti': _save_nifti
    }
    
    if file_format in save_methods:
        save_methods[file_format](obj, full_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    
    print('Done!')


def load_file(
    filename: Union[str, Path],
    path: Union[str, Path] = '',
    file_format: Optional[str] = None,
    **kwargs: Any
) -> Any:
    '''
    Load object from file.
    
    Parameters
    ----------
    filename : str or Path
        Filename with extension
    path : str or Path, optional
        Directory path (default: '')
    file_format : str or None, optional
        Explicit format specification (default: auto-detect)
    **kwargs : Any
        Additional keyword arguments for specific loaders:
        - dataset_name : str (hdf5) - Specific dataset to load
        - sheet_name : str or int (excel) - Excel sheet to load
        - as_array : bool (txt) - Load text file as NumPy array
        
    Returns
    -------
    Any
        Loaded object (type depends on file format)
        
    Examples
    --------
    >>> # Load HDF5 dataset
    >>> data = load_file('voxel_data.h5', dataset_name='volume')
    
    >>> # Load CSV as DataFrame
    >>> df = load_file('data.csv')
    
    >>> # Load specific Excel sheet
    >>> df = load_file('results.xlsx', sheet_name='Sheet1')
    '''
    # Detect format if not specified
    filename_str = str(Path(filename).name)
    if file_format is None:
        file_format = _detect_format(filename_str)
    
    # Construct full path
    full_path = os.path.join(path, filename_str) if path else filename_str
    
    print(f'Loading file: {filename_str}')
    
    # Dispatch to appropriate load method
    load_methods = {
        'pickle': _load_pickle,
        'json': _load_json,
        'csv': _load_csv,
        'txt': _load_txt,
        'hdf5': _load_hdf5,
        'tiff': _load_tiff,
        'excel': _load_excel,
        'numpy': _load_numpy,
        'nifti': _load_nifti
    }
    
    if file_format in load_methods:
        obj = load_methods[file_format](full_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    
    print('Done!')
    return obj


# ========== Utility Classes ==========

class NumpyJSONEncoder(json.JSONEncoder):
    '''
    Custom JSON encoder for NumPy types.
    
    Enables JSON serialization of NumPy arrays and scalar types.
    Handles: np.ndarray, np.integer, np.floating, np.bool_, np.complexfloating
    
    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> data = {'array': np.array([1, 2, 3]), 'value': np.float32(1.5)}
    >>> json.dumps(data, cls=NumpyJSONEncoder)
    '{"array": [1, 2, 3], "value": 1.5}'
    '''
    
    def default(self, obj: Any) -> Any:
        '''Convert NumPy types to JSON-serializable Python types.'''
        if HAS_NUMPY:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.complexfloating):
                return {'real': float(obj.real), 'imag': float(obj.imag)}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
        return super().default(obj)


# ========== Legacy Compatibility ==========

# Backward compatibility with original function name
seperate_file_folder = _separate_file_folder


# ========== Module Test ==========

if __name__ == '__main__':
    # Basic functionality test
    print("DataIOHandler module loaded successfully!")
    print(f"Supported formats: {list(SUPPORTED_FORMATS.keys())}")
    print(f"\nAvailable backends:")
    print(f"  NumPy: {HAS_NUMPY}")
    print(f"  Pandas: {HAS_PANDAS}")
    print(f"  HDF5 (h5py): {HAS_H5PY}")
    print(f"  TIFF (tifffile): {HAS_TIFFFILE}")
    print(f"  Excel (openpyxl): {HAS_OPENPYXL}")
    print(f"  NIfTI (nibabel): {HAS_NIBABEL}")