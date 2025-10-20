"""
mplgrapher.py - Enhanced matplotlib plotting utilities

A comprehensive matplotlib plotting module with PEP8 compliance, extensive
docstrings, and flexible subplot capabilities. This module provides a wide
range of plotting functions for 2D, 3D, and specialized visualizations.

Author: Enhanced from original display.py
Version: 2.1
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from matplotlib import figure
from matplotlib import axes

# Type aliases to avoid SubFigure/Figure union issues
FigureType = Any  # Can be Figure or SubFigure
AxesType = Any    # Can be Axes or AxesSubplot or Axes3D


class FlexibleSubplot:
    """
    A class to handle flexible subplot arrangements with support for
    adding pre-rendered plots and custom layouts.
    """
    
    def __init__(self, nrows: int = 1, ncols: int = 1, 
                 figsize: Optional[Tuple[float, float]] = None,
                 **kwargs: Any):
        """
        Initialize a flexible subplot container.
        
        Parameters
        ----------
        nrows : int, default=1
            Number of subplot rows
        ncols : int, default=1
            Number of subplot columns
        figsize : tuple, optional
            Figure size in inches (width, height)
        **kwargs : dict
            Additional keyword arguments for plt.subplots()
        """
        self.figure, self.axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                              **kwargs)
        self.nrows = nrows
        self.ncols = ncols
        self.subplot_functions: List[Any] = []
        
    def add_plot(self, row: int, col: int, plot_func: Any, 
                 *args: Any, **kwargs: Any) -> AxesType:
        """
        Add a plot to a specific subplot position.
        
        Parameters
        ----------
        row : int
            Row index (0-based)
        col : int
            Column index (0-based)
        plot_func : callable
            Plotting function to execute
        *args : tuple
            Positional arguments for the plotting function
        **kwargs : dict
            Keyword arguments for the plotting function
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes object for the subplot
        """
        if self.nrows == 1 and self.ncols == 1:
            ax = self.axes
        elif self.nrows == 1 or self.ncols == 1:
            ax = self.axes[max(row, col)]  # type: ignore
        else:
            ax = self.axes[row, col]  # type: ignore
            
        # Execute the plotting function on the specified axes
        if callable(plot_func):
            plot_func(*args, ax=ax, **kwargs)
            
        return ax  # type: ignore
    
    def show(self) -> None:
        """Display the subplot arrangement."""
        plt.tight_layout()
        plt.show()
        
    def save(self, filename: str, **kwargs: Any) -> None:
        """
        Save the subplot arrangement to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        **kwargs : dict
            Additional keyword arguments for savefig()
        """
        self.figure.savefig(filename, **kwargs)


def graph_hist(data: Union[List[float], np.ndarray], bins: Union[int, str] = 'auto',
               ax: Optional[AxesType] = None, title: str = "Histogram",
               xlabel: str = "Values", ylabel: str = "Frequency",
               color: str = 'blue', alpha: float = 0.7, 
               **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a histogram plot from data.
    
    Parameters
    ----------
    data : array-like
        Input data for histogram
    bins : int or str, default='auto'
        Number of bins or binning strategy
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure
    title : str, default="Histogram"
        Plot title
    xlabel : str, default="Values"
        X-axis label
    ylabel : str, default="Frequency"
        Y-axis label
    color : str, default='blue'
        Histogram color
    alpha : float, default=0.7
        Transparency level
    **kwargs : dict
        Additional keyword arguments for hist()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, 1000)
    >>> fig, ax = graph_hist(data, bins=30, color='red')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Create histogram
    ax.hist(data, bins=bins, color=color, alpha=alpha, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def graph_none(ax: Optional[AxesType] = None, title: str = "Empty Plot",
               figsize: Tuple[float, float] = (8, 6)) -> Tuple[FigureType, AxesType]:
    """
    Create an empty plot/graph.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Empty Plot"
        Plot title
    figsize : tuple, default=(8, 6)
        Figure size in inches
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> fig, ax = graph_none(title="Placeholder Plot")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def graph_textbox(text: str, ax: Optional[AxesType] = None, 
                  title: str = "Text Display",
                  fontsize: int = 12, ha: str = 'center', 
                  va: str = 'center', 
                  bbox_props: Optional[Dict[str, Any]] = None,
                  figsize: Tuple[float, float] = (8, 6)) -> Tuple[FigureType, AxesType]:
    """
    Create a plot with text content instead of data.
    
    Parameters
    ----------
    text : str
        Text content to display
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Text Display"
        Plot title
    fontsize : int, default=12
        Font size for text
    ha : str, default='center'
        Horizontal alignment
    va : str, default='center'
        Vertical alignment
    bbox_props : dict, optional
        Text box properties
    figsize : tuple, default=(8, 6)
        Figure size in inches
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> fig, ax = graph_textbox("Sample\\nMultiline\\nText", 
    ...                        fontsize=14, bbox_props={'boxstyle': 'round'})
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Default text box properties
    if bbox_props is None:
        bbox_props = {'boxstyle': 'round,pad=0.5', 'facecolor': 'lightgray', 
                      'alpha': 0.8}
    
    # Add text to the center of the plot
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=fontsize,
            ha=ha, va=va, bbox=bbox_props)
    
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax  # type: ignore


def graph_2d(x_data: Union[List[float], np.ndarray], 
             y_data: Union[List[float], np.ndarray, List[np.ndarray]],
             ax: Optional[AxesType] = None, title: str = "2D Plot",
             xlabel: str = "X", ylabel: str = "Y", 
             plot_type: str = 'line', **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a 2D plot of one or multiple datasets.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_data : array-like or list of array-like
        Y-axis data (can be multiple datasets)
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="2D Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    plot_type : str, default='line'
        Type of plot ('line', 'scatter', 'both')
    **kwargs : dict
        Additional keyword arguments for plotting functions
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> fig, ax = graph_2d(x, y, title="Sine Wave", plot_type='line')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Handle multiple datasets
    if isinstance(y_data, list) and len(y_data) > 0 and isinstance(y_data[0], (list, np.ndarray)):
        for i, y in enumerate(y_data):
            if plot_type == 'line':
                ax.plot(x_data, y, label=f'Dataset {i+1}', **kwargs)
            elif plot_type == 'scatter':
                ax.scatter(x_data, y, label=f'Dataset {i+1}', **kwargs)
            elif plot_type == 'both':
                ax.plot(x_data, y, 'o-', label=f'Dataset {i+1}', **kwargs)
        ax.legend()
    else:
        # Single dataset
        if plot_type == 'line':
            ax.plot(x_data, y_data, **kwargs)
        elif plot_type == 'scatter':
            ax.scatter(x_data, y_data, **kwargs)
        elif plot_type == 'both':
            ax.plot(x_data, y_data, 'o-', **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def flex_subplot(subplot_spec: Union[Tuple[int, int], Dict[str, Any]], 
                 figsize: Optional[Tuple[float, float]] = None,
                 **kwargs: Any) -> FlexibleSubplot:
    """
    Create a flexible subplot arrangement that can accommodate pre-rendered plots.
    
    Parameters
    ----------
    subplot_spec : tuple or dict
        Subplot specification. Tuple (nrows, ncols) or dict with layout info
    figsize : tuple, optional
        Figure size in inches (width, height)
    **kwargs : dict
        Additional keyword arguments for subplot creation
        
    Returns
    -------
    FlexibleSubplot
        Flexible subplot container
        
    Examples
    --------
    >>> # Create a 2x2 subplot arrangement
    >>> subplots = flex_subplot((2, 2), figsize=(12, 10))
    >>> # Add different plots to each position
    >>> subplots.add_plot(0, 0, line_2d, x, y1, title="Line Plot")
    >>> subplots.add_plot(0, 1, scatter_2d, x, y2, title="Scatter Plot")
    >>> subplots.show()
    """
    if isinstance(subplot_spec, tuple):
        nrows, ncols = subplot_spec
        return FlexibleSubplot(nrows, ncols, figsize=figsize, **kwargs)
    elif isinstance(subplot_spec, dict):
        # Handle more complex specifications
        nrows = subplot_spec.get('nrows', 1)
        ncols = subplot_spec.get('ncols', 1)
        return FlexibleSubplot(nrows, ncols, figsize=figsize, **kwargs)
    else:
        raise ValueError("subplot_spec must be tuple (nrows, ncols) or dict")


def save_pyplot_figure(fig: FigureType, filename: str, dpi: int = 300,
                       bbox_inches: str = 'tight', 
                       **kwargs: Any) -> None:
    """
    Save a matplotlib figure to file with common settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filename : str
        Output filename (should include extension)
    dpi : int, default=300
        Resolution in dots per inch
    bbox_inches : str, default='tight'
        Bbox setting for figure bounds
    **kwargs : dict
        Additional keyword arguments for savefig()
        
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_pyplot_figure(fig, "my_plot.png", dpi=150)
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved as: {filename}")


# New plotting functions below this line

def line_2d(x_data: Union[List[float], np.ndarray], y_data: Union[List[float], np.ndarray],
            ax: Optional[AxesType] = None, title: str = "Line Plot",
            xlabel: str = "X", ylabel: str = "Y", 
            color: Optional[str] = None, linewidth: float = 2.0,
            linestyle: str = '-', marker: Optional[str] = None,
            **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a 2D line plot.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_data : array-like
        Y-axis data
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Line Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    color : str, optional
        Line color
    linewidth : float, default=2.0
        Line width
    linestyle : str, default='-'
        Line style
    marker : str, optional
        Marker style
    **kwargs : dict
        Additional keyword arguments for plot()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x)
    >>> fig, ax = line_2d(x, y, color='red', linewidth=3)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.plot(x_data, y_data, color=color, linewidth=linewidth,
            linestyle=linestyle, marker=marker, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def scatter_2d(x_data: Union[List[float], np.ndarray], y_data: Union[List[float], np.ndarray],
               ax: Optional[AxesType] = None, title: str = "Scatter Plot",
               xlabel: str = "X", ylabel: str = "Y",
               s: Union[float, np.ndarray] = 50, 
               c: Union[str, np.ndarray] = 'blue',
               alpha: float = 0.7, **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a 2D scatter plot.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_data : array-like
        Y-axis data
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Scatter Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    s : float or array-like, default=50
        Marker size(s)
    c : str or array-like, default='blue'
        Marker color(s)
    alpha : float, default=0.7
        Transparency level
    **kwargs : dict
        Additional keyword arguments for scatter()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = np.random.randn(100)
    >>> colors = np.random.rand(100)
    >>> fig, ax = scatter_2d(x, y, c=colors, s=60, alpha=0.8)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    scatter = ax.scatter(x_data, y_data, s=s, c=c, alpha=alpha, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar if colors are array-like
    if hasattr(c, '__len__') and not isinstance(c, str):
        plt.colorbar(scatter, ax=ax)
    
    return fig, ax  # type: ignore


def barplot(x_data: Union[List[Any], np.ndarray], y_data: Union[List[float], np.ndarray],
            ax: Optional[AxesType] = None, title: str = "Bar Plot",
            xlabel: str = "Categories", ylabel: str = "Values",
            color: Union[str, List[str]] = 'blue', width: float = 0.8,
            **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a simple bar plot.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data (categories or positions)
    y_data : array-like
        Y-axis data (values)
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Bar Plot"
        Plot title
    xlabel : str, default="Categories"
        X-axis label
    ylabel : str, default="Values"
        Y-axis label
    color : str or list, default='blue'
        Bar color(s)
    width : float, default=0.8
        Bar width
    **kwargs : dict
        Additional keyword arguments for bar()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> categories = ['A', 'B', 'C', 'D']
    >>> values = [3, 7, 2, 5]
    >>> fig, ax = barplot(categories, values, color='green')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.bar(x_data, y_data, color=color, width=width, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig, ax  # type: ignore


def errorband_2d(x_data: Union[List[float], np.ndarray], 
                 y_center: Union[List[float], np.ndarray],
                 y_error: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
                 ax: Optional[AxesType] = None, title: str = "Error Band Plot",
                 xlabel: str = "X", ylabel: str = "Y",
                 color: str = 'blue', alpha: float = 0.3,
                 linewidth: float = 2.0, **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create an error band plot using fill_between.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_center : array-like
        Central Y values
    y_error : array-like or tuple
        Error values. If tuple, (lower_error, upper_error)
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Error Band Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    color : str, default='blue'
        Color for line and fill
    alpha : float, default=0.3
        Transparency for error band
    linewidth : float, default=2.0
        Line width for center line
    **kwargs : dict
        Additional keyword arguments for fill_between()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = np.sin(x)
    >>> error = 0.1 * np.random.rand(50)
    >>> fig, ax = errorband_2d(x, y, error, color='red', alpha=0.4)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    y_center = np.array(y_center)
    
    # Handle different error formats
    if isinstance(y_error, tuple) and len(y_error) == 2:
        y_lower, y_upper = y_error
        y1 = y_center - y_lower
        y2 = y_center + y_upper
    else:
        y_error = np.array(y_error)
        y1 = y_center - y_error
        y2 = y_center + y_error
    
    # Plot center line
    ax.plot(x_data, y_center, color=color, linewidth=linewidth)
    
    # Fill error band
    ax.fill_between(x_data, y1, y2, color=color, alpha=alpha, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def histogram(data: Union[List[float], np.ndarray], bins: Union[int, str] = 'auto',
              ax: Optional[AxesType] = None, title: str = "Histogram",
              xlabel: str = "Values", ylabel: str = "Frequency",
              color: str = 'skyblue', alpha: float = 0.7,
              density: bool = False, **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a standard histogram plot.
    
    Parameters
    ----------
    data : array-like
        Input data
    bins : int or str, default='auto'
        Number of bins or binning strategy
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Histogram"
        Plot title
    xlabel : str, default="Values"
        X-axis label
    ylabel : str, default="Frequency"
        Y-axis label
    color : str, default='skyblue'
        Histogram color
    alpha : float, default=0.7
        Transparency level
    density : bool, default=False
        Whether to normalize to form probability density
    **kwargs : dict
        Additional keyword arguments for hist()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> data = np.random.normal(100, 15, 1000)
    >>> fig, ax = histogram(data, bins=30, color='green', density=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.hist(data, bins=bins, color=color, alpha=alpha, 
            density=density, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def errorbar_2d(x_data: Union[List[float], np.ndarray], y_data: Union[List[float], np.ndarray],
                xerr: Optional[Union[List[float], np.ndarray]] = None,
                yerr: Optional[Union[List[float], np.ndarray]] = None,
                ax: Optional[AxesType] = None, title: str = "Error Bar Plot",
                xlabel: str = "X", ylabel: str = "Y",
                fmt: str = 'o', capsize: float = 5,
                **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a 2D error bar plot.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_data : array-like
        Y-axis data
    xerr : array-like, optional
        X-axis error values
    yerr : array-like, optional
        Y-axis error values
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Error Bar Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    fmt : str, default='o'
        Format string for markers
    capsize : float, default=5
        Error bar cap size
    **kwargs : dict
        Additional keyword arguments for errorbar()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 3, 5, 4, 6]
    >>> y_err = [0.2, 0.3, 0.1, 0.4, 0.2]
    >>> fig, ax = errorbar_2d(x, y, yerr=y_err, fmt='s', capsize=8)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.errorbar(x_data, y_data, xerr=xerr, yerr=yerr, 
                fmt=fmt, capsize=capsize, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def imageplot(image_data: np.ndarray, ax: Optional[AxesType] = None,
              title: str = "Image Plot", extent: Optional[Tuple[float, float, float, float]] = None,
              cmap: str = 'viridis', aspect: Any = 'equal',
              origin: Any = 'lower', **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Display data as an image using imshow.
    
    Parameters
    ----------
    image_data : array-like
        2D array representing image data
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Image Plot"
        Plot title
    extent : tuple, optional
        (left, right, bottom, top) extent of the image
    cmap : str, default='viridis'
        Colormap name
    aspect : str, default='equal'
        Aspect ratio ('equal' or 'auto')
    origin : str, default='lower'
        Origin position ('lower' or 'upper')
    **kwargs : dict
        Additional keyword arguments for imshow()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> # Create sample 2D data
    >>> x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    >>> z = np.sin(x**2 + y**2)
    >>> fig, ax = imageplot(z, cmap='plasma', title="Sine Wave Pattern")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    im = ax.imshow(image_data, extent=extent, cmap=cmap,  # type: ignore
                   aspect=aspect, origin=origin, **kwargs)
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    return fig, ax  # type: ignore


def colormesh(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
              ax: Optional[AxesType] = None, title: str = "Color Mesh Plot",
              xlabel: str = "X", ylabel: str = "Y",
              cmap: str = 'viridis', **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a pseudocolor plot using pcolormesh.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate data
    y_data : array-like
        Y-coordinate data
    z_data : array-like
        Z-value data (colors)
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Color Mesh Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    cmap : str, default='viridis'
        Colormap name
    **kwargs : dict
        Additional keyword arguments for pcolormesh()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(-3, 3, 20)
    >>> y = np.linspace(-2, 2, 15)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = np.sin(X) * np.cos(Y)
    >>> fig, ax = colormesh(X, Y, Z, cmap='coolwarm')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    mesh = ax.pcolormesh(x_data, y_data, z_data, cmap=cmap, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(mesh, ax=ax)
    
    return fig, ax  # type: ignore


def contourplot(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                levels: Optional[Union[int, List[float]]] = None,
                ax: Optional[AxesType] = None, title: str = "Contour Plot",
                xlabel: str = "X", ylabel: str = "Y",
                cmap: str = 'viridis', **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a contour plot.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate data
    y_data : array-like
        Y-coordinate data
    z_data : array-like
        Z-value data
    levels : int or list, optional
        Number of levels or specific level values
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Contour Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    cmap : str, default='viridis'
        Colormap name
    **kwargs : dict
        Additional keyword arguments for contour()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.linspace(-3, 3, 50)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = X**2 + Y**2
    >>> fig, ax = contourplot(X, Y, Z, levels=15)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    if levels is None:
        levels = 10
    
    cs = ax.contour(x_data, y_data, z_data, levels=levels, cmap=cmap, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.clabel(cs, inline=True, fontsize=8)
    
    return fig, ax  # type: ignore


def contourfill(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                levels: Optional[Union[int, List[float]]] = None,
                ax: Optional[AxesType] = None, title: str = "Filled Contour Plot",
                xlabel: str = "X", ylabel: str = "Y",
                cmap: str = 'viridis', **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a filled contour plot.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate data
    y_data : array-like
        Y-coordinate data
    z_data : array-like
        Z-value data
    levels : int or list, optional
        Number of levels or specific level values
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Filled Contour Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    cmap : str, default='viridis'
        Colormap name
    **kwargs : dict
        Additional keyword arguments for contourf()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(-2, 2, 30)
    >>> y = np.linspace(-2, 2, 30)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = np.exp(-(X**2 + Y**2))
    >>> fig, ax = contourfill(X, Y, Z, levels=20, cmap='plasma')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    if levels is None:
        levels = 10
    
    cs = ax.contourf(x_data, y_data, z_data, levels=levels, cmap=cmap, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(cs, ax=ax)
    
    return fig, ax  # type: ignore


def quiver_2d(x_data: np.ndarray, y_data: np.ndarray,
              u_data: np.ndarray, v_data: np.ndarray,
              ax: Optional[AxesType] = None, title: str = "Vector Field Plot",
              xlabel: str = "X", ylabel: str = "Y",
              color: str = 'black', scale: Optional[float] = None,
              **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a 2D vector field plot using quiver.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate data
    y_data : array-like
        Y-coordinate data
    u_data : array-like
        X-component of vectors
    v_data : array-like
        Y-component of vectors
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Vector Field Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    color : str, default='black'
        Arrow color
    scale : float, optional
        Scale factor for arrows
    **kwargs : dict
        Additional keyword arguments for quiver()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(-2, 2, 10)
    >>> y = np.linspace(-2, 2, 10)
    >>> X, Y = np.meshgrid(x, y)
    >>> U = -Y  # Velocity field
    >>> V = X
    >>> fig, ax = quiver_2d(X, Y, U, V, color='blue')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.quiver(x_data, y_data, u_data, v_data, color=color, 
              scale=scale, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig, ax  # type: ignore


def streamplot(x_data: np.ndarray, y_data: np.ndarray,
               u_data: np.ndarray, v_data: np.ndarray,
               ax: Optional[AxesType] = None, title: str = "Stream Plot",
               xlabel: str = "X", ylabel: str = "Y",
               color: Optional[Union[str, np.ndarray]] = None,
               density: Union[float, Tuple[float, float]] = 1, **kwargs: Any) -> Tuple[FigureType, AxesType]:
    """
    Create a streamline plot.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate data
    y_data : array-like
        Y-coordinate data
    u_data : array-like
        X-component of velocity field
    v_data : array-like
        Y-component of velocity field
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Stream Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    color : str or array-like, optional
        Streamline color(s)
    density : float or tuple, default=1
        Streamline density
    **kwargs : dict
        Additional keyword arguments for streamplot()
        
    Returns
    -------
    tuple
        Figure and Axes objects
        
    Examples
    --------
    >>> x = np.linspace(-3, 3, 100)
    >>> y = np.linspace(-3, 3, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> U = -Y
    >>> V = X
    >>> fig, ax = streamplot(X, Y, U, V, density=2, color='red')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.streamplot(x_data, y_data, u_data, v_data, color=color, 
                  density=density, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig, ax  # type: ignore


# Twin axis and inset functions

def twinx_2d(x_data1: Union[List[float], np.ndarray], y_data1: Union[List[float], np.ndarray],
             x_data2: Union[List[float], np.ndarray], y_data2: Union[List[float], np.ndarray],
             ax: Optional[AxesType] = None, title: str = "Twin X-Axis Plot",
             xlabel: str = "X", ylabel1: str = "Y1", ylabel2: str = "Y2",
             plot_type1: str = 'line', plot_type2: str = 'line',
             color1: str = 'blue', color2: str = 'red',
             label1: Optional[str] = None, label2: Optional[str] = None,
             **kwargs: Any) -> Tuple[FigureType, AxesType, AxesType]:
    """
    Create a 2D plot with twin x-axis (two y-axes sharing the same x-axis).
    
    Parameters
    ----------
    x_data1 : array-like
        X-axis data for first plot
    y_data1 : array-like
        Y-axis data for first plot (left y-axis)
    x_data2 : array-like
        X-axis data for second plot
    y_data2 : array-like
        Y-axis data for second plot (right y-axis)
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Twin X-Axis Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel1 : str, default="Y1"
        Left y-axis label
    ylabel2 : str, default="Y2"
        Right y-axis label
    plot_type1 : str, default='line'
        Type of first plot ('line', 'scatter', 'bar')
    plot_type2 : str, default='line'
        Type of second plot ('line', 'scatter', 'bar')
    color1 : str, default='blue'
        Color for first plot and left y-axis
    color2 : str, default='red'
        Color for second plot and right y-axis
    label1 : str, optional
        Label for first dataset
    label2 : str, optional
        Label for second dataset
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    tuple
        Figure, first Axes, and twin Axes objects
        
    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y1 = np.sin(x)
    >>> y2 = np.exp(x / 10)
    >>> fig, ax1, ax2 = twinx_2d(x, y1, x, y2, ylabel1="Sin", ylabel2="Exp",
    ...                          color1='blue', color2='red')
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    else:
        ax1 = ax
        fig = ax1.figure
    
    # Create twin axes sharing x-axis
    ax2 = ax1.twinx()  # type: ignore
    
    # Plot first dataset on left y-axis
    if plot_type1 == 'line':
        line1 = ax1.plot(x_data1, y_data1, color=color1, label=label1)
    elif plot_type1 == 'scatter':
        line1 = ax1.scatter(x_data1, y_data1, color=color1, label=label1)
    elif plot_type1 == 'bar':
        line1 = ax1.bar(x_data1, y_data1, color=color1, label=label1, alpha=0.7)
    
    # Plot second dataset on right y-axis
    if plot_type2 == 'line':
        line2 = ax2.plot(x_data2, y_data2, color=color2, label=label2)
    elif plot_type2 == 'scatter':
        line2 = ax2.scatter(x_data2, y_data2, color=color2, label=label2)
    elif plot_type2 == 'bar':
        line2 = ax2.bar(x_data2, y_data2, color=color2, label=label2, alpha=0.7)
    
    # Set labels
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax2.set_ylabel(ylabel2, color=color2)
    ax1.set_title(title)
    
    # Color the y-axis ticks and spines
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.spines['left'].set_color(color1)
    ax2.spines['right'].set_color(color2)
    
    # Add grid on left axis
    ax1.grid(True, alpha=0.3)
    
    # Create combined legend if labels provided
    if label1 or label2:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    return fig, ax1, ax2  # type: ignore


def twiny_2d(x_data1: Union[List[float], np.ndarray], y_data1: Union[List[float], np.ndarray],
             x_data2: Union[List[float], np.ndarray], y_data2: Union[List[float], np.ndarray],
             ax: Optional[AxesType] = None, title: str = "Twin Y-Axis Plot",
             xlabel1: str = "X1", xlabel2: str = "X2", ylabel: str = "Y",
             plot_type1: str = 'line', plot_type2: str = 'line',
             color1: str = 'blue', color2: str = 'red',
             label1: Optional[str] = None, label2: Optional[str] = None,
             **kwargs: Any) -> Tuple[FigureType, AxesType, AxesType]:
    """
    Create a 2D plot with twin y-axis (two x-axes sharing the same y-axis).
    
    Parameters
    ----------
    x_data1 : array-like
        X-axis data for first plot (bottom x-axis)
    y_data1 : array-like
        Y-axis data for first plot
    x_data2 : array-like
        X-axis data for second plot (top x-axis)
    y_data2 : array-like
        Y-axis data for second plot
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Twin Y-Axis Plot"
        Plot title
    xlabel1 : str, default="X1"
        Bottom x-axis label
    xlabel2 : str, default="X2"
        Top x-axis label
    ylabel : str, default="Y"
        Y-axis label
    plot_type1 : str, default='line'
        Type of first plot ('line', 'scatter', 'bar')
    plot_type2 : str, default='line'
        Type of second plot ('line', 'scatter', 'bar')
    color1 : str, default='blue'
        Color for first plot and bottom x-axis
    color2 : str, default='red'
        Color for second plot and top x-axis
    label1 : str, optional
        Label for first dataset
    label2 : str, optional
        Label for second dataset
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    tuple
        Figure, first Axes, and twin Axes objects
        
    Examples
    --------
    >>> x1 = np.linspace(0, 10, 100)
    >>> x2 = np.linspace(0, 2*np.pi, 100)
    >>> y = np.linspace(0, 10, 100)
    >>> fig, ax1, ax2 = twiny_2d(x1, y, x2, y, xlabel1="Linear", xlabel2="Radians",
    ...                          color1='blue', color2='red')
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    else:
        ax1 = ax
        fig = ax1.figure
    
    # Create twin axes sharing y-axis
    ax2 = ax1.twiny()  # type: ignore
    
    # Plot first dataset on bottom x-axis
    if plot_type1 == 'line':
        line1 = ax1.plot(x_data1, y_data1, color=color1, label=label1)
    elif plot_type1 == 'scatter':
        line1 = ax1.scatter(x_data1, y_data1, color=color1, label=label1)
    elif plot_type1 == 'bar':
        line1 = ax1.barh(y_data1, x_data1, color=color1, label=label1, alpha=0.7)
    
    # Plot second dataset on top x-axis
    if plot_type2 == 'line':
        line2 = ax2.plot(x_data2, y_data2, color=color2, label=label2)
    elif plot_type2 == 'scatter':
        line2 = ax2.scatter(x_data2, y_data2, color=color2, label=label2)
    elif plot_type2 == 'bar':
        line2 = ax2.barh(y_data2, x_data2, color=color2, label=label2, alpha=0.7)
    
    # Set labels
    ax1.set_xlabel(xlabel1, color=color1)
    ax2.set_xlabel(xlabel2, color=color2)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    
    # Color the x-axis ticks and spines
    ax1.tick_params(axis='x', labelcolor=color1)
    ax2.tick_params(axis='x', labelcolor=color2)
    ax1.spines['bottom'].set_color(color1)
    ax2.spines['top'].set_color(color2)
    
    # Add grid on bottom axis
    ax1.grid(True, alpha=0.3)
    
    # Create combined legend if labels provided
    if label1 or label2:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    return fig, ax1, ax2  # type: ignore


def inset_2d(x_outer: Union[List[float], np.ndarray], y_outer: Union[List[float], np.ndarray],
             x_inner: Union[List[float], np.ndarray], y_inner: Union[List[float], np.ndarray],
             ax: Optional[AxesType] = None, title: str = "Plot with Inset",
             xlabel_outer: str = "X", ylabel_outer: str = "Y",
             xlabel_inner: str = "X", ylabel_inner: str = "Y",
             plot_type_outer: str = 'line', plot_type_inner: str = 'line',
             inset_bounds: Tuple[float, float, float, float] = (0.6, 0.6, 0.35, 0.35),
             inset_loc: str = 'upper right',
             color_outer: str = 'blue', color_inner: str = 'red',
             **kwargs: Any) -> Tuple[FigureType, AxesType, AxesType]:
    """
    Create a 2D plot with an inset (smaller plot inside main plot).
    
    Parameters
    ----------
    x_outer : array-like
        X-axis data for outer (main) plot
    y_outer : array-like
        Y-axis data for outer (main) plot
    x_inner : array-like
        X-axis data for inner (inset) plot
    y_inner : array-like
        Y-axis data for inner (inset) plot
    ax : matplotlib.axes.Axes, optional
        Axes object to use. If None, creates new figure
    title : str, default="Plot with Inset"
        Main plot title
    xlabel_outer : str, default="X"
        Outer plot x-axis label
    ylabel_outer : str, default="Y"
        Outer plot y-axis label
    xlabel_inner : str, default="X"
        Inner plot x-axis label
    ylabel_inner : str, default="Y"
        Inner plot y-axis label
    plot_type_outer : str, default='line'
        Type of outer plot ('line', 'scatter', 'bar')
    plot_type_inner : str, default='line'
        Type of inner plot ('line', 'scatter', 'bar')
    inset_bounds : tuple, default=(0.6, 0.6, 0.35, 0.35)
        Inset position and size [x0, y0, width, height] in axes coordinates
    inset_loc : str, default='upper right'
        Location string for inset ('upper right', 'upper left', 'lower left', 
        'lower right', 'center', etc.)
    color_outer : str, default='blue'
        Color for outer plot
    color_inner : str, default='red'
        Color for inner plot
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    tuple
        Figure, outer Axes, and inset Axes objects
        
    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> x_zoom = np.linspace(2, 4, 50)
    >>> y_zoom = np.sin(x_zoom)
    >>> fig, ax_main, ax_inset = inset_2d(x, y, x_zoom, y_zoom,
    ...                                   inset_bounds=(0.5, 0.5, 0.4, 0.4))
    """
    if ax is None:
        fig, ax_outer = plt.subplots(figsize=(10, 8))
    else:
        ax_outer = ax
        fig = ax_outer.figure
    
    # Plot outer (main) plot
    if plot_type_outer == 'line':
        ax_outer.plot(x_outer, y_outer, color=color_outer, linewidth=2)
    elif plot_type_outer == 'scatter':
        ax_outer.scatter(x_outer, y_outer, color=color_outer)
    elif plot_type_outer == 'bar':
        ax_outer.bar(x_outer, y_outer, color=color_outer, alpha=0.7)
    
    ax_outer.set_xlabel(xlabel_outer)
    ax_outer.set_ylabel(ylabel_outer)
    ax_outer.set_title(title)
    ax_outer.grid(True, alpha=0.3)
    
    # Create inset axes
    # Using inset_axes from mpl_toolkits.axes_grid1.inset_locator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Convert bounds tuple to width and height percentages
    x0, y0, width, height = inset_bounds
    
    ax_inner = inset_axes(ax_outer, 
                         width=f"{width*100}%", 
                         height=f"{height*100}%",
                         bbox_to_anchor=(x0, y0, width, height),
                         bbox_transform=ax_outer.transAxes,
                         loc='lower left',
                         borderpad=0)
    
    # Plot inner (inset) plot
    if plot_type_inner == 'line':
        ax_inner.plot(x_inner, y_inner, color=color_inner, linewidth=2)
    elif plot_type_inner == 'scatter':
        ax_inner.scatter(x_inner, y_inner, color=color_inner, s=20)
    elif plot_type_inner == 'bar':
        ax_inner.bar(x_inner, y_inner, color=color_inner, alpha=0.7)
    
    # Set labels for inset (optional, can be small)
    ax_inner.set_xlabel(xlabel_inner, fontsize=8)
    ax_inner.set_ylabel(ylabel_inner, fontsize=8)
    ax_inner.tick_params(labelsize=8)
    ax_inner.grid(True, alpha=0.3)
    
    return fig, ax_outer, ax_inner  # type: ignore


# 3D plotting functions

def errorband_3d(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                 z_error: np.ndarray, ax: Optional[Any] = None,
                 title: str = "3D Error Band Plot",
                 xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
                 color: str = 'blue', alpha: float = 0.3,
                 **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a 3D error band plot using fill_between.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_data : array-like
        Y-axis data (should be constant for proper fill_between3d)
    z_data : array-like
        Z-axis center values
    z_error : array-like
        Z-axis error values
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="3D Error Band Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    color : str, default='blue'
        Color for line and fill
    alpha : float, default=0.3
        Transparency for error band
    **kwargs : dict
        Additional keyword arguments
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = np.zeros_like(x)  # Constant y for proper 3D fill
    >>> z = np.sin(x)
    >>> z_err = 0.1 * np.ones_like(z)
    >>> fig, ax = errorband_3d(x, y, z, z_err)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    # Plot center line
    ax.plot(x_data, y_data, z_data, color=color, linewidth=2)
    
    # Create error bands
    z_lower = z_data - z_error
    z_upper = z_data + z_error
    
    # Fill between using 3D capability
    for i in range(len(x_data) - 1):
        x_seg = [x_data[i], x_data[i+1], x_data[i+1], x_data[i]]
        y_seg = [y_data[i], y_data[i+1], y_data[i+1], y_data[i]]
        z_seg = [z_lower[i], z_lower[i+1], z_upper[i+1], z_upper[i]]
        
        ax.plot_trisurf(x_seg, y_seg, z_seg, color=color, alpha=alpha)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    return fig, ax  # type: ignore


def line_3d(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
            ax: Optional[Any] = None, title: str = "3D Line Plot",
            xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
            color: Optional[str] = None, linewidth: float = 2.0,
            **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a 3D line plot.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_data : array-like
        Y-axis data
    z_data : array-like
        Z-axis data
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="3D Line Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    color : str, optional
        Line color
    linewidth : float, default=2.0
        Line width
    **kwargs : dict
        Additional keyword arguments for plot3D()
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> t = np.linspace(0, 4*np.pi, 100)
    >>> x = np.cos(t)
    >>> y = np.sin(t)
    >>> z = t
    >>> fig, ax = line_3d(x, y, z, color='red', title="3D Spiral")
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    ax.plot3D(x_data, y_data, z_data, color=color, linewidth=linewidth, **kwargs)  # type: ignore
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    return fig, ax  # type: ignore


def quiver_3d(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
              u_data: np.ndarray, v_data: np.ndarray, w_data: np.ndarray,
              ax: Optional[Any] = None, title: str = "3D Vector Field",
              xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
              color: str = 'black', **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a 3D vector field plot.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate data
    y_data : array-like
        Y-coordinate data
    z_data : array-like
        Z-coordinate data
    u_data : array-like
        X-component of vectors
    v_data : array-like
        Y-component of vectors
    w_data : array-like
        Z-component of vectors
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="3D Vector Field"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    color : str, default='black'
        Arrow color
    **kwargs : dict
        Additional keyword arguments for quiver()
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> # Create a simple 3D vector field
    >>> x, y, z = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3))
    >>> u = x
    >>> v = y  
    >>> w = z
    >>> fig, ax = quiver_3d(x, y, z, u, v, w, color='blue')
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    ax.quiver(x_data, y_data, z_data, u_data, v_data, w_data, 
              color=color, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    return fig, ax  # type: ignore


def scatter_3d(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
               ax: Optional[Any] = None, title: str = "3D Scatter Plot",
               xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
               c: Optional[Union[str, np.ndarray]] = None,
               s: Union[float, np.ndarray] = 50, alpha: float = 0.8,
               **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a 3D scatter plot.
    
    Parameters
    ----------
    x_data : array-like
        X-axis data
    y_data : array-like
        Y-axis data
    z_data : array-like
        Z-axis data
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="3D Scatter Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    c : str or array-like, optional
        Marker colors
    s : float or array-like, default=50
        Marker sizes
    alpha : float, default=0.8
        Transparency level
    **kwargs : dict
        Additional keyword arguments for scatter()
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> # Random 3D scatter data
    >>> x = np.random.randn(100)
    >>> y = np.random.randn(100)
    >>> z = np.random.randn(100)
    >>> colors = np.random.rand(100)
    >>> fig, ax = scatter_3d(x, y, z, c=colors, s=60)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    scatter = ax.scatter(x_data, y_data, z_data, c=c, s=s, alpha=alpha, **kwargs)  # type: ignore
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    # Add colorbar if colors are array-like
    if c is not None and hasattr(c, '__len__') and not isinstance(c, str):
        plt.colorbar(scatter, ax=ax, shrink=0.5)
    
    return fig, ax  # type: ignore


def surface(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
            ax: Optional[Any] = None, title: str = "3D Surface Plot",
            xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
            cmap: str = 'viridis', alpha: float = 0.8,
            **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a 3D surface plot.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate meshgrid data
    y_data : array-like
        Y-coordinate meshgrid data
    z_data : array-like
        Z-value data
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="3D Surface Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    cmap : str, default='viridis'
        Colormap name
    alpha : float, default=0.8
        Surface transparency
    **kwargs : dict
        Additional keyword arguments for plot_surface()
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> # Create sample surface data
    >>> x = np.linspace(-2, 2, 30)
    >>> y = np.linspace(-2, 2, 30)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = np.sin(np.sqrt(X**2 + Y**2))
    >>> fig, ax = surface(X, Y, Z, cmap='plasma')
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    surf = ax.plot_surface(x_data, y_data, z_data, cmap=cmap, alpha=alpha, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5)
    
    return fig, ax  # type: ignore


def trisurfplot(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                ax: Optional[Any] = None, title: str = "Triangulated Surface",
                xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
                cmap: str = 'viridis', alpha: float = 0.8,
                **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a triangulated surface plot.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate data (1D)
    y_data : array-like
        Y-coordinate data (1D)
    z_data : array-like
        Z-value data (1D)
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="Triangulated Surface"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    cmap : str, default='viridis'
        Colormap name
    alpha : float, default=0.8
        Surface transparency
    **kwargs : dict
        Additional keyword arguments for plot_trisurf()
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> # Random triangulated surface
    >>> n = 200
    >>> x = np.random.rand(n) * 4 - 2
    >>> y = np.random.rand(n) * 4 - 2
    >>> z = x**2 + y**2
    >>> fig, ax = trisurfplot(x, y, z, cmap='coolwarm')
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    surf = ax.plot_trisurf(x_data, y_data, z_data, cmap=cmap, alpha=alpha, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5)
    
    return fig, ax  # type: ignore


def voxelplot(voxel_data: np.ndarray, ax: Optional[Any] = None,
              title: str = "Voxel Plot", xlabel: str = "X", 
              ylabel: str = "Y", zlabel: str = "Z",
              facecolors: Optional[np.ndarray] = None,
              alpha: float = 0.8, **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a 3D voxel plot.
    
    Parameters
    ----------
    voxel_data : array-like
        3D boolean array indicating filled voxels
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="Voxel Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    facecolors : array-like, optional
        Colors for each voxel face
    alpha : float, default=0.8
        Voxel transparency
    **kwargs : dict
        Additional keyword arguments for voxels()
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> # Create a simple cube pattern
    >>> voxels = np.zeros((10, 10, 10), dtype=bool)
    >>> voxels[2:8, 2:8, 2:8] = True  # Hollow cube
    >>> voxels[4:6, 4:6, 4:6] = False
    >>> fig, ax = voxelplot(voxels, alpha=0.7)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    ax.voxels(voxel_data, facecolors=facecolors, alpha=alpha, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    return fig, ax  # type: ignore


def wireframe(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
              ax: Optional[Any] = None, title: str = "Wireframe Plot",
              xlabel: str = "X", ylabel: str = "Y", zlabel: str = "Z",
              color: str = 'blue', linewidth: float = 1.0,
              **kwargs: Any) -> Tuple[FigureType, Any]:
    """
    Create a 3D wireframe plot.
    
    Parameters
    ----------
    x_data : array-like
        X-coordinate meshgrid data
    y_data : array-like
        Y-coordinate meshgrid data
    z_data : array-like
        Z-value data
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes object. If None, creates new figure
    title : str, default="Wireframe Plot"
        Plot title
    xlabel : str, default="X"
        X-axis label
    ylabel : str, default="Y"
        Y-axis label
    zlabel : str, default="Z"
        Z-axis label
    color : str, default='blue'
        Wireframe color
    linewidth : float, default=1.0
        Line width
    **kwargs : dict
        Additional keyword arguments for plot_wireframe()
        
    Returns
    -------
    tuple
        Figure and 3D Axes objects
        
    Examples
    --------
    >>> # Create wireframe of a saddle surface
    >>> x = np.linspace(-3, 3, 25)
    >>> y = np.linspace(-3, 3, 25)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = X**2 - Y**2
    >>> fig, ax = wireframe(X, Y, Z, color='red', linewidth=1.5)
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure  # type: ignore
    
    ax.plot_wireframe(x_data, y_data, z_data, color=color, 
                      linewidth=linewidth, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    
    return fig, ax  # type: ignore


# Utility functions for the module

def set_style(style: str = 'default') -> None:
    """
    Set matplotlib style for all plots.
    
    Parameters
    ----------
    style : str, default='default'
        Style name (e.g., 'seaborn', 'ggplot', 'dark_background')
        
    Examples
    --------
    >>> set_style('seaborn-v0_8')  # Use seaborn style
    >>> set_style('dark_background')  # Use dark theme
    """
    plt.style.use(style)


def close_all_figures() -> None:
    """Close all open matplotlib figures to free memory."""
    plt.close('all')


# Module metadata
__version__ = "2.1"
__author__ = "Enhanced matplotlib plotting module"
__all__ = [
    # Original functions (enhanced)
    'graph_hist', 'graph_none', 'graph_textbox', 'graph_2d', 
    'flex_subplot', 'save_pyplot_figure',
    
    # New 2D plotting functions
    'line_2d', 'scatter_2d', 'barplot', 'errorband_2d', 'histogram',
    'errorbar_2d', 'imageplot', 'colormesh', 'contourplot', 
    'contourfill', 'quiver_2d', 'streamplot',
    
    # Twin axis and inset functions
    'twinx_2d', 'twiny_2d', 'inset_2d',
    
    # New 3D plotting functions
    'errorband_3d', 'line_3d', 'quiver_3d', 'scatter_3d', 
    'surface', 'trisurfplot', 'voxelplot', 'wireframe',
    
    # Utility functions
    'set_style', 'close_all_figures',
    
    # Classes
    'FlexibleSubplot'
]
