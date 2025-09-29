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
import re
from typing import Dict, List, Tuple, Any
import warnings


#__________________________________________________________________________________
def read_TFAnalyzer_3000(file_path):
    # Read in all lines of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Iterate through all the lines process line by line.
    data_dict = dict()
    main_key = '' # main key in the dict which is type of measurement  
    sub_key = '' # sub key of the sub dict
    skips = 0
    for i, line in enumerate(lines): # iterate over all lines
        if skips: # skip lines from past operations 
            skips -= 1
            continue
        if line in ['DynamicHysteresis\n']: # sub iteration
            main_key = line.strip()
            data_dict[main_key] = dict()
            
            # header block
            j = i + 1
            while lines[j] != '\n':
                skips =+ 1
                key, val = lines[j].strip().split(': ', 1)
                try:
                    data_dict[main_key][key] = float(val)
                except:
                    data_dict[main_key][key] = val
                finally:
                    j += 1
            
            # table block iterations
            while lines[j] not in ['DynamicHysteresisResult',]:
                skips += 1
                if lines[j] == '\n': # new table
                    sub_key = lines[j+1].strip()
                    data_dict[main_key][sub_key] = dict()
                    j += 2
                    skips += 1
                elif 'Time [s]' in lines[j]: # begin of data table
                    header_line_index = j
                    while lines[j] != '\n':
                        j += 1
                        if j >= len(lines):
                            break
                    data_dict[main_key][sub_key]['table'] = pd.read_csv(file_path, 
                                                                        header=header_line_index, # header start 
                                                                        delimiter='\t', 
                                                                        usecols=range(9), # if not specified, an empty col will be added 
                                                                        nrows=j-header_line_index-1, #skip last empty row
                                                                        skip_blank_lines=False) # otherwise carriage return is not counted in row index
                else: # header entries added
                    key, val = lines[j].strip().split(': ', 1)
                    try:
                        data_dict[main_key][sub_key][key] = float(val)
                    except:
                        data_dict[main_key][sub_key][key] = val
                    finally:
                        j += 1
                
                if j >= len(lines):
                            break
    return data_dict
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