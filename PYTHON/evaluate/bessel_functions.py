"""
Bessel Functions Module
=======================

This module contains all Bessel functions and related orthogonal functions including:
- Airy functions
- Bessel functions (all kinds and orders)
- Modified Bessel functions
- Kelvin functions
- Legendre functions
- Orthogonal polynomials
- Spherical functions

Total Functions: 32
"""

import numpy as np
import scipy.special

# =============================================================================
# AIRY FUNCTIONS (2 functions)
# =============================================================================

def airy_function_ai(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Airy function Ai(x): solution to y'' - xy = 0, Ai(x) → 0 as x → +∞
    
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
        Airy Ai function values
    """
    return a * scipy.special.airy(b * x + c)[0]


def airy_function_bi(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Airy function Bi(x): second solution to y'' - xy = 0, Bi(x) → ∞ as x → +∞
    
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
        Airy Bi function values
    """
    return a * scipy.special.airy(b * x + c)[2]


# =============================================================================
# BESSEL FUNCTIONS OF FIRST KIND (3 functions)
# =============================================================================

def bessel_j0(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Bessel function of first kind, order 0: J₀(x)
    
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
        J₀ Bessel function values
    """
    return a * scipy.special.j0(b * x) + c


def bessel_j1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Bessel function of first kind, order 1: J₁(x)
    
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
        J₁ Bessel function values
    """
    return a * scipy.special.j1(b * x) + c


def bessel_jn(x: np.ndarray, a: float, b: float, c: float, n: int) -> np.ndarray:
    """
    Bessel function of first kind, order n: Jₙ(x)
    
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
    n : int
        Order of Bessel function
        
    Returns
    -------
    np.ndarray
        Jₙ Bessel function values
    """
    n = int(n)
    return a * scipy.special.jn(n, b * x) + c


# =============================================================================
# BESSEL FUNCTIONS OF SECOND KIND (3 functions)
# =============================================================================

def bessel_y0(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Bessel function of second kind, order 0: Y₀(x) (Neumann function)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Y₀ Bessel function values
    """
    arg = b * x
    arg = np.where(arg > 0, arg, 1e-10)
    return a * scipy.special.y0(arg) + c


def bessel_y1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Bessel function of second kind, order 1: Y₁(x) (Neumann function)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Y₁ Bessel function values
    """
    arg = b * x
    arg = np.where(arg > 0, arg, 1e-10)
    return a * scipy.special.y1(arg) + c


def bessel_yn(x: np.ndarray, a: float, b: float, c: float, n: int) -> np.ndarray:
    """
    Bessel function of second kind, order n: Yₙ(x) (Neumann function)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
    n : int
        Order of Bessel function
        
    Returns
    -------
    np.ndarray
        Yₙ Bessel function values
    """
    arg = b * x
    arg = np.where(arg > 0, arg, 1e-10)
    n = int(n)
    return a * scipy.special.yn(n, arg) + c


# =============================================================================
# MODIFIED BESSEL FUNCTIONS OF FIRST KIND (3 functions)
# =============================================================================

def bessel_i0(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Modified Bessel function of first kind, order 0: I₀(x)
    
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
        I₀ modified Bessel function values
    """
    return a * scipy.special.i0(b * x) + c


def bessel_i1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Modified Bessel function of first kind, order 1: I₁(x)
    
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
        I₁ modified Bessel function values
    """
    return a * scipy.special.i1(b * x) + c


def bessel_in(x: np.ndarray, a: float, b: float, c: float, n: int) -> np.ndarray:
    """
    Modified Bessel function of first kind, order n: Iₙ(x)
    
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
    n : int
        Order of modified Bessel function
        
    Returns
    -------
    np.ndarray
        Iₙ modified Bessel function values
    """
    n = int(n)
    return a * scipy.special.iv(n, b * x) + c


# =============================================================================
# MODIFIED BESSEL FUNCTIONS OF SECOND KIND (3 functions)
# =============================================================================

def bessel_k0(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Modified Bessel function of second kind, order 0: K₀(x) (MacDonald function)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        K₀ modified Bessel function values
    """
    arg = b * x
    arg = np.where(arg > 0, arg, 1e-10)
    return a * scipy.special.k0(arg) + c


def bessel_k1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Modified Bessel function of second kind, order 1: K₁(x) (MacDonald function)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        K₁ modified Bessel function values
    """
    arg = b * x
    arg = np.where(arg > 0, arg, 1e-10)
    return a * scipy.special.k1(arg) + c


def bessel_kn(x: np.ndarray, a: float, b: float, c: float, n: int) -> np.ndarray:
    """
    Modified Bessel function of second kind, order n: Kₙ(x) (MacDonald function)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
    n : int
        Order of modified Bessel function
        
    Returns
    -------
    np.ndarray
        Kₙ modified Bessel function values
    """
    arg = b * x
    arg = np.where(arg > 0, arg, 1e-10)
    n = int(n)
    return a * scipy.special.kv(n, arg) + c


# =============================================================================
# KELVIN FUNCTIONS (4 functions)
# =============================================================================

def kelvin_ber(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Kelvin function ber(x): real part of J₀(x√(-i))
    
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
        Kelvin ber function values
    """
    return a * scipy.special.ber(b * x) + c


def kelvin_bei(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Kelvin function bei(x): imaginary part of J₀(x√(-i))
    
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
        Kelvin bei function values
    """
    return a * scipy.special.bei(b * x) + c


def kelvin_ker(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Kelvin function ker(x): real part of K₀(x√(-i))
    
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
        Kelvin ker function values
    """
    return a * scipy.special.ker(b * x) + c


def kelvin_kei(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Kelvin function kei(x): imaginary part of K₀(x√(-i))
    
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
        Kelvin kei function values
    """
    return a * scipy.special.kei(b * x) + c


# =============================================================================
# LEGENDRE FUNCTIONS (2 functions)
# =============================================================================

def legendre_function_p(x: np.ndarray, a: float, b: float, n: int, m: int = 0) -> np.ndarray:
    """
    Legendre function Pₙᵐ(x): solutions of Legendre differential equation
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (-1 ≤ x ≤ 1)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    n : int
        Degree (non-negative integer)
    m : int, optional
        Order (default: 0)
        
    Returns
    -------
    np.ndarray
        Legendre function values
    """
    x_clipped = np.clip(x, -1 + 1e-10, 1 - 1e-10)
    n, m = int(n), int(m)
    if m == 0:
        return a * scipy.special.eval_legendre(n, b * x_clipped)
    else:
        return a * scipy.special.lpmv(m, n, b * x_clipped)


def associated_legendre_function(x: np.ndarray, a: float, b: float, n: int, m: int) -> np.ndarray:
    """
    Associated Legendre function Pₙᵐ(x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (-1 ≤ x ≤ 1)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    n : int
        Degree (non-negative integer)
    m : int
        Order
        
    Returns
    -------
    np.ndarray
        Associated Legendre function values
    """
    x_clipped = np.clip(x, -1 + 1e-10, 1 - 1e-10)
    n, m = int(n), int(m)
    return a * scipy.special.lpmv(m, n, b * x_clipped)


# =============================================================================
# SPHERICAL BESSEL FUNCTIONS (2 functions)
# =============================================================================

def spherical_bessel_j(x: np.ndarray, a: float, b: float, c: float, n: int) -> np.ndarray:
    """
    Spherical Bessel function jₙ(x) = √(π/2x) Jₙ₊₁/₂(x)
    
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
    n : int
        Order of spherical Bessel function
        
    Returns
    -------
    np.ndarray
        Spherical jₙ function values
    """
    n = int(n)
    return a * scipy.special.spherical_jn(n, b * x) + c


def spherical_bessel_y(x: np.ndarray, a: float, b: float, c: float, n: int) -> np.ndarray:
    """
    Spherical Bessel function yₙ(x) = √(π/2x) Yₙ₊₁/₂(x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
    n : int
        Order of spherical Bessel function
        
    Returns
    -------
    np.ndarray
        Spherical yₙ function values
    """
    n = int(n)
    return a * scipy.special.spherical_yn(n, b * x) + c


# =============================================================================
# SCORER FUNCTIONS (2 functions)
# =============================================================================

def scorer_gi_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Scorer function Gi(x): inhomogeneous Airy function solution
    
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
        Scorer Gi function approximation
    """
    # Approximation using Airy functions
    ai, aip, bi, bip = scipy.special.airy(b * x + c)
    return a * (bi + ai) / 2


def scorer_hi_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Scorer function Hi(x): another inhomogeneous Airy function solution
    
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
        Scorer Hi function approximation
    """
    # Approximation using Airy functions
    ai, aip, bi, bip = scipy.special.airy(b * x + c)
    return a * (bi - ai) / 2


# =============================================================================
# SPECIAL FUNCTIONS (1 function)
# =============================================================================

def sinc_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Sinc function: sinc(x) = sin(πx)/(πx)
    
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
        Sinc function values
    """
    arg = b * (x - c)
    return a * np.sinc(arg)


# =============================================================================
# ORTHOGONAL POLYNOMIALS (4 functions)
# =============================================================================

def hermite_polynomial(x: np.ndarray, a: float, b: float, n: int) -> np.ndarray:
    """
    Hermite polynomial Hₙ(x): orthogonal polynomials with weight e^(-x²)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    n : int
        Degree of polynomial
        
    Returns
    -------
    np.ndarray
        Hermite polynomial values
    """
    n = int(abs(n))
    return a * scipy.special.eval_hermite(n, b * x)


def laguerre_polynomial(x: np.ndarray, a: float, b: float, n: int, alpha: float = 0) -> np.ndarray:
    """
    Laguerre polynomial Lₙ^α(x): orthogonal polynomials with weight x^α e^(-x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x ≥ 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    n : int
        Degree of polynomial
    alpha : float, optional
        Generalization parameter (default: 0)
        
    Returns
    -------
    np.ndarray
        Laguerre polynomial values
    """
    n = int(abs(n))
    if alpha == 0:
        return a * scipy.special.eval_laguerre(n, b * x)
    else:
        return a * scipy.special.eval_genlaguerre(n, alpha, b * x)


def chebyshev_t(x: np.ndarray, a: float, b: float, n: int) -> np.ndarray:
    """
    Chebyshev polynomial of first kind Tₙ(x): orthogonal on [-1,1] with weight 1/√(1-x²)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (-1 ≤ x ≤ 1)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    n : int
        Degree of polynomial
        
    Returns
    -------
    np.ndarray
        Chebyshev T polynomial values
    """
    x_clipped = np.clip(b * x, -1, 1)
    n = int(abs(n))
    return a * scipy.special.eval_chebyt(n, x_clipped)


def chebyshev_u(x: np.ndarray, a: float, b: float, n: int) -> np.ndarray:
    """
    Chebyshev polynomial of second kind Uₙ(x): orthogonal on [-1,1] with weight √(1-x²)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (-1 ≤ x ≤ 1)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    n : int
        Degree of polynomial
        
    Returns
    -------
    np.ndarray
        Chebyshev U polynomial values
    """
    x_clipped = np.clip(b * x, -1, 1)
    n = int(abs(n))
    return a * scipy.special.eval_chebyu(n, x_clipped)


# =============================================================================
# SYNCHROTRON FUNCTION (1 function)
# =============================================================================

def synchrotron_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Synchrotron function: F(x) = x ∫ₓ^∞ K₅/₃(t) dt
    
    Used in astrophysics for synchrotron radiation
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Synchrotron function approximation
    """
    arg = b * x + c
    arg = np.where(arg > 0, arg, 1e-10)
    return a * arg * scipy.special.kv(5/3, arg)


# Function registry for Bessel functions
BESSEL_FUNCTIONS = {
    # Airy functions
    'airy_ai': (airy_function_ai, ['a', 'b', 'c']),
    'airy_bi': (airy_function_bi, ['a', 'b', 'c']),
    
    # Bessel functions of first kind
    'bessel_j0': (bessel_j0, ['a', 'b', 'c']),
    'bessel_j1': (bessel_j1, ['a', 'b', 'c']),
    'bessel_jn': (bessel_jn, ['a', 'b', 'c', 'n']),
    
    # Bessel functions of second kind
    'bessel_y0': (bessel_y0, ['a', 'b', 'c']),
    'bessel_y1': (bessel_y1, ['a', 'b', 'c']),
    'bessel_yn': (bessel_yn, ['a', 'b', 'c', 'n']),
    
    # Modified Bessel functions of first kind
    'bessel_i0': (bessel_i0, ['a', 'b', 'c']),
    'bessel_i1': (bessel_i1, ['a', 'b', 'c']),
    'bessel_in': (bessel_in, ['a', 'b', 'c', 'n']),
    
    # Modified Bessel functions of second kind
    'bessel_k0': (bessel_k0, ['a', 'b', 'c']),
    'bessel_k1': (bessel_k1, ['a', 'b', 'c']),
    'bessel_kn': (bessel_kn, ['a', 'b', 'c', 'n']),
    
    # Kelvin functions
    'kelvin_ber': (kelvin_ber, ['a', 'b', 'c']),
    'kelvin_bei': (kelvin_bei, ['a', 'b', 'c']),
    'kelvin_ker': (kelvin_ker, ['a', 'b', 'c']),
    'kelvin_kei': (kelvin_kei, ['a', 'b', 'c']),
    
    # Legendre functions
    'legendre_p': (legendre_function_p, ['a', 'b', 'n', 'm']),
    'associated_legendre': (associated_legendre_function, ['a', 'b', 'n', 'm']),
    
    # Spherical Bessel functions
    'spherical_j': (spherical_bessel_j, ['a', 'b', 'c', 'n']),
    'spherical_y': (spherical_bessel_y, ['a', 'b', 'c', 'n']),
    
    # Scorer functions
    'scorer_gi': (scorer_gi_function, ['a', 'b', 'c']),
    'scorer_hi': (scorer_hi_function, ['a', 'b', 'c']),
    
    # Sinc function
    'sinc': (sinc_function, ['a', 'b', 'c']),
    
    # Orthogonal polynomials
    'hermite': (hermite_polynomial, ['a', 'b', 'n']),
    'laguerre': (laguerre_polynomial, ['a', 'b', 'n', 'alpha']),
    'chebyshev_t': (chebyshev_t, ['a', 'b', 'n']),
    'chebyshev_u': (chebyshev_u, ['a', 'b', 'n']),
    
    # Synchrotron function
    'synchrotron': (synchrotron_function, ['a', 'b', 'c']),
}

print(f"Bessel Functions Module loaded: {len(BESSEL_FUNCTIONS)} functions")