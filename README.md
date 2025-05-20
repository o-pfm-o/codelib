# PFM
Repository with useful code snippets that can be used across multiple projects. 

## Python
For python modules import use the following snippet: **import PFM.PYTHON as pfm**. Submodules can be accessed by using the .submodule addition.

The following main modules are available in the PYTHON: 
- **__init_.py**: Main pfm module containing functions that include and use function from other modules. Not for cross imports. 
- **visa.py**: Module for visa interfacing with pyvisa. 
- **display**: Module for plotting. 
- **evaluate**: Module for calculating and data evaluation. 
- **process**: Module for reading and writing data. 