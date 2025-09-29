'''
Module for input and output of data, e.g. reading files.
Created on 2025-05-20.
Creator: PFM
'''

# Imports
import os
import pickle
import numpy as np
import pandas as pd

#__________________________________________________________________________________
def read_TFAnalyzer_3000(filepath):
    """
    Reads a TF Analyzer 3000 ferroelectric tester data file and extracts
    the overview table and individual hysteresis measurements.

    The file contains an overview table followed by multiple hysteresis
    measurement data blocks. Both overview and measurement data are parsed
    into pandas DataFrames. Measurement metadata is stored alongside its data.

    Parameters
    ----------
    filepath : str
        Path to the TF Analyzer data file.

    Returns
    -------
    overview_df : pd.DataFrame or None
        DataFrame containing the overview data table.
        None if the overview table is not found.
    
    measurements : dict
        Dictionary where each key is a measurement identifier (e.g., 'Measurement_1')
        and each value is a dictionary with metadata keys and a 'data' key containing
        a pandas DataFrame of the hysteresis measurement data.
    """
    overview_df = None
    measurements = {}

    # Read all lines from the file
    with open(filepath, "r") as file:
        lines = file.readlines()

    overview_lines = []
    measurement_blocks = []
    current_block = []
    in_overview = False
    in_measurement = False

    for line in lines:
        # Detect the start of the overview table section
        if "Overview" in line or ("Sweep" in line and "Sample" in line):
            in_overview = True
            continue

        # Collect lines for the overview table
        if in_overview:
            # End of overview section triggered by empty line or comment line
            if line.strip() == "" or line.startswith("#"):
                in_overview = False
                # Convert the collected overview lines to a DataFrame using tab delimiter
                overview_df = pd.read_csv(pd.compat.StringIO("\n".join(overview_lines)),
                                          sep="\t")
                continue
            overview_lines.append(line.rstrip())

        # Detect the start of a measurement data section
        if "Measurement" in line or ("Sweep" in line and "Field" in line):
            # Save any previous measurement block collected
            in_measurement = True
            if current_block:
                measurement_blocks.append(current_block)
                current_block = []
            continue

        # Collect lines for the current measurement block
        if in_measurement:
            # End of measurement block triggered by empty line or comment
            if line.strip() == "" or line.startswith("#"):
                in_measurement = False
                if current_block:
                    measurement_blocks.append(current_block)
                    current_block = []
                continue
            current_block.append(line.rstrip())

    # If file ends while still collecting a measurement block, save it too
    if current_block:
        measurement_blocks.append(current_block)

    # Parse each measurement block into metadata dictionary with DataFrame
    for i, block in enumerate(measurement_blocks):
        meta = {}
        header = []
        data = []

        for line in block:
            # Metadata lines usually start with #, containing key-value pairs
            if line.startswith("#"):
                kv = line[1:].split(":", 1)
                if len(kv) == 2:
                    meta[kv[0].strip()] = kv[1].strip()
            # Header line detected by presence of alphabets
            elif not header and any(c.isalpha() for c in line):
                header = [h.strip() for h in line.split()]
            else:
                # Data line: convert all values to float
                data.append([float(x) for x in line.split()])

        # Create DataFrame for the measurement data
        meas_df = pd.DataFrame(data, columns=header)
        meta["data"] = meas_df
        measurements[f"Measurement_{i + 1}"] = meta

    return overview_df, measurements


#__________________________________________________________________________________
def read_txt(filename, path='', deler=',', skro=0, header=False, kwargs = dict()):
    '''
    Extracts data from a file with arbitrary deliminiter using numpy.
    
    :param filename: Filename to be opened. Can be whole path or just the name.
    :param path: Path to save the object in. If empty, attempt to split up 
    the filename. Fallback is current working directory.
    :param deler: Deliminiter to be used. Default is ','.
    :param skro: Number of rows to skip. Default is 0.
    :param header: Whether header is to be extracted. Default is False
    :param kwargs: Additional arguments to be passed to numpy.loadtxt.
    :return: Data extracted from the file and the header as a tuple. 
    '''
    # Fill path if needed
    path, filename = fill_path(path,filename)

    # Check extract header
    if header:
        
        # Check if header can be extracted
        if not skro:
            raise ValueError('Header cannot be extracted if no rows are skipped!')
        
        # Now extract the header
        header = np.loadtxt()

#__________________________________________________________________________________
def data_extractor(filename, path='', mode='auto'):
    '''
    Extracts data from a file. The file is opened in the given mode.
    The mode is automatically detected if not given. The returning format for the data depends on the mode used. 
    
    :param filename: Filename to be opened. Can be whole path or just the name.
    :param path: Path to save the object in. If empty, attempt to split up the filename. Fallback is current working directory.
    :param mode: Mode to open the file in. Default is 'auto'.
    :return: Data extracted from the file. For "txt" it a tuple with data and headers; numpy.array, list.
    '''


#__________________________________________________________________________________
def seperate_file_folder(dir):
    '''
    Splits a path into file and directory. If the dir is empty, exception is raised.

    :param dir: Directory to be split.
    :return: Tuple of file and directory.
    '''

    if dir == '':
        raise ValueError('Cannot split empty path!')

    # split the path abd return the file and directory
    return os.path.split(dir)

#__________________________________________________________________________________
def fill_path(path, filename, fallback='cwd'):
    '''
    Checks wether path is empty, if yes attempting to split filename. If not succesful, fallback is used.
    
    :param filename: Filename for possible splitting.
    :param path: Path to be checked.
    :param fallback: Fallback to be used if path is empty. Default is 'cwd'.
    :return: Tuple of path and filename.
    '''
    # Check wether path is empty == if yes, try to split up the filename
    if path == '':
        path, filename = seperate_file_folder(filename)
    
        # Fallback is current path
        if path == '':
            match fallback: 
                case 'cwd':
                    path = os.getcwd()
                case _: 
                    raise ValueError('Fallback unknown!')
    
    return path, filename

#__________________________________________________________________________________
def save_Object(obj, filename, path='', format='.pkl'):
    '''
    Saves an Object in the path. Any object can be saved. Target format is pickle.

    :param obj: Object to be saved
    :param filename: Filename to save the object as
    :param path: Path to save the object in. If empty, attempt to split up the filename. Fallback is current working directory.
    :param format: Format to save the object in. Default is ".pkl". 
    :return: None
    '''
    
    # Filling path if necessary
    path, filename = fill_path(path, filename)

    # Check wether path exists and if not, create it!
    if not os.path.exists(path):
        os.makedirs(path)
    
    #Now the file is saved!
    if format == '.pkl':
        filename = filename + ('' if filename[-4:] == '.pkl' else format)
        print('Saving pickle: '+ filename)
        with open(os.path.join(path, filename), 'wb') as f: 
            pickle.dump(obj, f)
    else:
        raise ValueError('Format not supported!')
    print('Done!')

#__________________________________________________________________________________
if __name__ == "__main__":
    print("Module for input and output of data, e.g. reading files.")