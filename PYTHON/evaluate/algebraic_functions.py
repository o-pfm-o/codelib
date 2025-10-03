"""
Algebraic Functions Module
=========================

This module contains all algebraic functions including polynomials, 
rational functions, and root functions.

Total Functions: 11
"""

import numpy as np

# =============================================================================
# ALGEBRAIC FUNCTIONS (11 functions)
# =============================================================================

def constant_function(x: np.ndarray, a: float) -> np.ndarray:
    """
    Constant function: f(x) = a
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Constant value
        
    Returns
    -------
    np.ndarray
        Function values (all equal to a)
    """
    return np.full_like(x, a)


def polynomial_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Linear function: f(x) = a*x + b
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Slope parameter
    b : float
        Intercept parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * x + b


def polynomial_quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Quadratic function: f(x) = a*x² + b*x + c
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Quadratic coefficient
    b : float
        Linear coefficient
    c : float
        Constant term
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * x**2 + b * x + c


def polynomial_cubic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Cubic function: f(x) = a*x³ + b*x² + c*x + d
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a, b, c, d : float
        Polynomial coefficients
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * x**3 + b * x**2 + c * x + d


def polynomial_quartic(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """
    Quartic function: f(x) = a*x⁴ + b*x³ + c*x² + d*x + e
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a, b, c, d, e : float
        Polynomial coefficients
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def polynomial_quintic(x: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """
    Quintic function: f(x) = a*x⁵ + b*x⁴ + c*x³ + d*x² + e*x + f
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a, b, c, d, e, f : float
        Polynomial coefficients
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


def rational_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Rational function: f(x) = (a*x + b) / (c*x + d)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a, b : float
        Numerator coefficients
    c, d : float
        Denominator coefficients
        
    Returns
    -------
    np.ndarray
        Function values
    """
    denominator = c * x + d
    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    return (a * x + b) / denominator


def square_root_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Square root function: f(x) = a * √(b*x + c)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Shift parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * x + c
    # Ensure non-negative argument
    arg = np.where(arg >= 0, arg, 0)
    return a * np.sqrt(arg)


def cube_root_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Cube root function: f(x) = a * ∛(b*x + c)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Shift parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * x + c
    # Cube root is defined for all real numbers
    return a * np.cbrt(arg)


def nth_root_function(x: np.ndarray, a: float, b: float, c: float, n: float) -> np.ndarray:
    """
    N-th root function: f(x) = a * (b*x + c)^(1/n)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Shift parameter
    n : float
        Root index
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * x + c
    n = max(abs(n), 1e-10)
    
    if int(n) % 2 == 1:  # Odd root
        return a * np.sign(arg) * np.power(np.abs(arg), 1.0/n)
    else:  # Even root
        arg = np.where(arg >= 0, arg, 0)
        return a * np.power(arg, 1.0/n)


# Function registry for algebraic functions
ALGEBRAIC_FUNCTIONS = {
    'constant': (constant_function, ['a']),
    'linear': (polynomial_linear, ['a', 'b']),
    'quadratic': (polynomial_quadratic, ['a', 'b', 'c']),
    'cubic': (polynomial_cubic, ['a', 'b', 'c', 'd']),
    'quartic': (polynomial_quartic, ['a', 'b', 'c', 'd', 'e']),
    'quintic': (polynomial_quintic, ['a', 'b', 'c', 'd', 'e', 'f']),
    'rational': (rational_function, ['a', 'b', 'c', 'd']),
    'square_root': (square_root_function, ['a', 'b', 'c']),
    'cube_root': (cube_root_function, ['a', 'b', 'c']),
    'nth_root': (nth_root_function, ['a', 'b', 'c', 'n']),
}

print(f"Algebraic Functions Module loaded: {len(ALGEBRAIC_FUNCTIONS)} functions")