"""
Elementary Transcendental Functions Module
==========================================

This module contains all elementary transcendental functions including:
- Exponential functions
- Logarithmic functions  
- Power functions
- Trigonometric functions (all variants)
- Inverse trigonometric functions
- Hyperbolic functions
- Inverse hyperbolic functions
- Gudermannian function

Total Functions: 41
"""

import numpy as np

# =============================================================================
# EXPONENTIAL FUNCTIONS (3 functions)
# =============================================================================

def exponential_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential function: f(x) = a * exp(b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Exponential rate
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * np.exp(b * x) + c


def exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential decay: f(x) = a * exp(-b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Initial amplitude
    b : float
        Decay rate (positive)
    c : float
        Asymptotic value
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * np.exp(-b * x) + c


def exponential_growth(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential growth: f(x) = a * exp(b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Initial amplitude
    b : float
        Growth rate (positive)
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * np.exp(b * x) + c


# =============================================================================
# LOGARITHMIC FUNCTIONS (3 functions)
# =============================================================================

def natural_logarithm(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Natural logarithm: f(x) = a * ln(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Scale parameter
    b : float
        Input scaling
    c : float
        Horizontal shift
    d : float
        Vertical offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(arg > 0, arg, 1e-10)
    return a * np.log(arg) + d


def common_logarithm(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Common logarithm: f(x) = a * log₁₀(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Scale parameter
    b : float
        Input scaling
    c : float
        Horizontal shift
    d : float
        Vertical offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(arg > 0, arg, 1e-10)
    return a * np.log10(arg) + d


def binary_logarithm(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Binary logarithm: f(x) = a * log₂(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Scale parameter
    b : float
        Input scaling
    c : float
        Horizontal shift
    d : float
        Vertical offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(arg > 0, arg, 1e-10)
    return a * np.log2(arg) + d


# =============================================================================
# POWER FUNCTIONS (2 functions)
# =============================================================================

def power_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Power function: f(x) = a * x^b + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (must be positive for non-integer b)
    a : float
        Amplitude parameter
    b : float
        Power exponent
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    return a * np.power(x_safe, b) + c


def allometric_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Allometric function: f(x) = a * x^b
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Scaling coefficient
    b : float
        Allometric exponent
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    return a * np.power(x_safe, b)


# =============================================================================
# TRIGONOMETRIC FUNCTIONS (16 functions)
# =============================================================================

def sine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Sine function: f(x) = amp * sin(2π*freq*x + phase) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (typically time)
    amp : float
        Amplitude
    freq : float
        Frequency (Hz or cycles per unit x)
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return amp * np.sin(2 * np.pi * freq * x + phase) + offset


def cosine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Cosine function: f(x) = amp * cos(2π*freq*x + phase) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (typically time)
    amp : float
        Amplitude
    freq : float
        Frequency (Hz or cycles per unit x)
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return amp * np.cos(2 * np.pi * freq * x + phase) + offset


def tangent_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Tangent function: f(x) = amp * tan(2π*freq*x + phase) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    result = amp * np.tan(arg) + offset
    # Handle infinities by capping values
    result = np.clip(result, -1e6, 1e6)
    return result


def cotangent_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Cotangent function: f(x) = amp * cot(2π*freq*x + phase) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    result = amp / np.tan(arg + 1e-10) + offset
    result = np.clip(result, -1e6, 1e6)
    return result


def secant_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Secant function: f(x) = amp * sec(2π*freq*x + phase) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    cos_val = np.cos(arg)
    cos_val = np.where(np.abs(cos_val) < 1e-10, 1e-10, cos_val)
    result = amp / cos_val + offset
    result = np.clip(result, -1e6, 1e6)
    return result


def cosecant_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Cosecant function: f(x) = amp * csc(2π*freq*x + phase) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    sin_val = np.sin(arg)
    sin_val = np.where(np.abs(sin_val) < 1e-10, 1e-10, sin_val)
    result = amp / sin_val + offset
    result = np.clip(result, -1e6, 1e6)
    return result


def exsecant_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Exsecant function: f(x) = amp * (sec(2π*freq*x + phase) - 1) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    cos_val = np.cos(arg)
    cos_val = np.where(np.abs(cos_val) < 1e-10, 1e-10, cos_val)
    result = amp * (1/cos_val - 1) + offset
    result = np.clip(result, -1e6, 1e6)
    return result


def excosecant_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Excosecant function: f(x) = amp * (csc(2π*freq*x + phase) - 1) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    sin_val = np.sin(arg)
    sin_val = np.where(np.abs(sin_val) < 1e-10, 1e-10, sin_val)
    result = amp * (1/sin_val - 1) + offset
    result = np.clip(result, -1e6, 1e6)
    return result


def versine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Versine function: f(x) = amp * (1 - cos(2π*freq*x + phase)) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    return amp * (1 - np.cos(arg)) + offset


def coversine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Coversine function: f(x) = amp * (1 - sin(2π*freq*x + phase)) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    return amp * (1 - np.sin(arg)) + offset


def vercosine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Vercosine function: f(x) = amp * (1 + cos(2π*freq*x + phase)) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    return amp * (1 + np.cos(arg)) + offset


def covercosine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Covercosine function: f(x) = amp * (1 + sin(2π*freq*x + phase)) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    return amp * (1 + np.sin(arg)) + offset


def haversine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Haversine function: f(x) = amp * sin²((2π*freq*x + phase)/2) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = (2 * np.pi * freq * x + phase) / 2
    return amp * np.sin(arg)**2 + offset


def hacoversine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Hacoversine function: f(x) = amp * cos²((2π*freq*x + phase)/2) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = (2 * np.pi * freq * x + phase) / 2
    return amp * np.cos(arg)**2 + offset


def havercosine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Havercosine function: f(x) = amp * (1 + cos(2π*freq*x + phase))/2 + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    return amp * (1 + np.cos(arg)) / 2 + offset


def hacovercosine_function(x: np.ndarray, amp: float, freq: float, phase: float, offset: float) -> np.ndarray:
    """
    Hacovercosine function: f(x) = amp * (1 + sin(2π*freq*x + phase))/2 + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset (radians)
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * freq * x + phase
    return amp * (1 + np.sin(arg)) / 2 + offset


# =============================================================================
# INVERSE TRIGONOMETRIC FUNCTIONS (6 functions)
# =============================================================================

def arcsin_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Arcsine function: f(x) = a * arcsin(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.clip(arg, -1 + 1e-10, 1 - 1e-10)
    return a * np.arcsin(arg) + d


def arccos_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Arccosine function: f(x) = a * arccos(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.clip(arg, -1 + 1e-10, 1 - 1e-10)
    return a * np.arccos(arg) + d


def arctan_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Arctangent function: f(x) = a * arctan(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    return a * np.arctan(arg) + d


def arccot_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Arccotangent function: f(x) = a * arccot(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    return a * (np.pi/2 - np.arctan(arg)) + d


def arcsec_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Arcsecant function: f(x) = a * arcsec(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(np.abs(arg) >= 1, arg, np.sign(arg) * 1.001)
    return a * np.arccos(1/arg) + d


def arccsc_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Arccosecant function: f(x) = a * arccsc(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(np.abs(arg) >= 1, arg, np.sign(arg) * 1.001)
    return a * np.arcsin(1/arg) + d


# =============================================================================
# HYPERBOLIC FUNCTIONS (6 functions)
# =============================================================================

def hyperbolic_sinh(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Hyperbolic sine: f(x) = a * sinh(b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * np.sinh(b * x) + c


def hyperbolic_cosh(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Hyperbolic cosine: f(x) = a * cosh(b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * np.cosh(b * x) + c


def hyperbolic_tanh(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Hyperbolic tangent: f(x) = a * tanh(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Slope parameter
    c : float
        Center position
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * np.tanh(b * (x - c)) + d


def hyperbolic_coth(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Hyperbolic cotangent: f(x) = a * coth(b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * x
    return a / np.tanh(arg + 1e-10) + c


def hyperbolic_sech(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Hyperbolic secant: f(x) = a * sech(b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a / np.cosh(b * x) + c


def hyperbolic_csch(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Hyperbolic cosecant: f(x) = a * csch(b*x) + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * x
    return a / np.sinh(arg + 1e-10) + c


# =============================================================================
# INVERSE HYPERBOLIC FUNCTIONS (6 functions)
# =============================================================================

def arcsinh_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Inverse hyperbolic sine: f(x) = a * arcsinh(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    return a * np.arcsinh(arg) + d


def arccosh_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Inverse hyperbolic cosine: f(x) = a * arccosh(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(arg >= 1, arg, 1 + 1e-10)
    return a * np.arccosh(arg) + d


def arctanh_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Inverse hyperbolic tangent: f(x) = a * arctanh(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.clip(arg, -1 + 1e-10, 1 - 1e-10)
    return a * np.arctanh(arg) + d


def arccoth_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Inverse hyperbolic cotangent: f(x) = a * arccoth(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(np.abs(arg) > 1, arg, np.sign(arg) * 1.001)
    return a * np.arctanh(1/arg) + d


def arcsech_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Inverse hyperbolic secant: f(x) = a * arcsech(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.clip(arg, 1e-10, 1)
    return a * np.arccosh(1/arg) + d


def arccsch_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Inverse hyperbolic cosecant: f(x) = a * arccsch(b*(x - c)) + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Center parameter
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    arg = np.where(arg != 0, arg, 1e-10)
    return a * np.arcsinh(1/arg) + d


# =============================================================================
# GUDERMANNIAN FUNCTION (1 function)
# =============================================================================

def gudermannian_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Gudermannian function: f(x) = a * gd(b*x) + c = a * arctan(tanh(b*x/2)) + c
    
    The Gudermannian function relates circular and hyperbolic functions.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * x / 2
    return a * np.arctan(np.tanh(arg)) + c


# Function registry for elementary transcendental functions
TRANSCENDENTAL_FUNCTIONS = {
    # Exponential functions
    'exponential': (exponential_function, ['a', 'b', 'c']),
    'exp_decay': (exponential_decay, ['a', 'b', 'c']),
    'exp_growth': (exponential_growth, ['a', 'b', 'c']),
    
    # Logarithmic functions
    'natural_log': (natural_logarithm, ['a', 'b', 'c', 'd']),
    'common_log': (common_logarithm, ['a', 'b', 'c', 'd']),
    'binary_log': (binary_logarithm, ['a', 'b', 'c', 'd']),
    
    # Power functions
    'power': (power_function, ['a', 'b', 'c']),
    'allometric': (allometric_function, ['a', 'b']),
    
    # Trigonometric functions
    'sine': (sine_function, ['amp', 'freq', 'phase', 'offset']),
    'cosine': (cosine_function, ['amp', 'freq', 'phase', 'offset']),
    'tangent': (tangent_function, ['amp', 'freq', 'phase', 'offset']),
    'cotangent': (cotangent_function, ['amp', 'freq', 'phase', 'offset']),
    'secant': (secant_function, ['amp', 'freq', 'phase', 'offset']),
    'cosecant': (cosecant_function, ['amp', 'freq', 'phase', 'offset']),
    'exsecant': (exsecant_function, ['amp', 'freq', 'phase', 'offset']),
    'excosecant': (excosecant_function, ['amp', 'freq', 'phase', 'offset']),
    'versine': (versine_function, ['amp', 'freq', 'phase', 'offset']),
    'coversine': (coversine_function, ['amp', 'freq', 'phase', 'offset']),
    'vercosine': (vercosine_function, ['amp', 'freq', 'phase', 'offset']),
    'covercosine': (covercosine_function, ['amp', 'freq', 'phase', 'offset']),
    'haversine': (haversine_function, ['amp', 'freq', 'phase', 'offset']),
    'hacoversine': (hacoversine_function, ['amp', 'freq', 'phase', 'offset']),
    'havercosine': (havercosine_function, ['amp', 'freq', 'phase', 'offset']),
    'hacovercosine': (hacovercosine_function, ['amp', 'freq', 'phase', 'offset']),
    
    # Inverse trigonometric functions
    'arcsin': (arcsin_function, ['a', 'b', 'c', 'd']),
    'arccos': (arccos_function, ['a', 'b', 'c', 'd']),
    'arctan': (arctan_function, ['a', 'b', 'c', 'd']),
    'arccot': (arccot_function, ['a', 'b', 'c', 'd']),
    'arcsec': (arcsec_function, ['a', 'b', 'c', 'd']),
    'arccsc': (arccsc_function, ['a', 'b', 'c', 'd']),
    
    # Hyperbolic functions
    'sinh': (hyperbolic_sinh, ['a', 'b', 'c']),
    'cosh': (hyperbolic_cosh, ['a', 'b', 'c']),
    'tanh': (hyperbolic_tanh, ['a', 'b', 'c', 'd']),
    'coth': (hyperbolic_coth, ['a', 'b', 'c']),
    'sech': (hyperbolic_sech, ['a', 'b', 'c']),
    'csch': (hyperbolic_csch, ['a', 'b', 'c']),
    
    # Inverse hyperbolic functions
    'arcsinh': (arcsinh_function, ['a', 'b', 'c', 'd']),
    'arccosh': (arccosh_function, ['a', 'b', 'c', 'd']),
    'arctanh': (arctanh_function, ['a', 'b', 'c', 'd']),
    'arccoth': (arccoth_function, ['a', 'b', 'c', 'd']),
    'arcsech': (arcsech_function, ['a', 'b', 'c', 'd']),
    'arccsch': (arccsch_function, ['a', 'b', 'c', 'd']),
    
    # Gudermannian function
    'gudermannian': (gudermannian_function, ['a', 'b', 'c']),
}

print(f"Elementary Transcendental Functions Module loaded: {len(TRANSCENDENTAL_FUNCTIONS)} functions")