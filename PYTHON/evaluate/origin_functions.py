"""
Origin Functions Module
=======================

This module contains functions commonly used in Origin software including:
- Origin Basic Functions (26 functions)
- Exponential Functions (5 functions) 
- Growth/Sigmoidal Functions (4 functions)
- Peak Functions (4 functions)
- Chromatography Functions (5 functions)
- Specialized Application Functions (8 functions)

Total Functions: 52
"""

import numpy as np
import scipy.special
import scipy.stats

# =============================================================================
# ORIGIN BASIC FUNCTIONS (26 functions)
# =============================================================================

def allometric1_freundlich(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Classical Freundlich model: y = a * x^b
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Scale coefficient
    b : float
        Allometric exponent
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    return a * np.power(x_safe, b)


def beta_origin(x: np.ndarray, y0: float, A: float, xc: float, w1: float, w2: float, w3: float) -> np.ndarray:
    """
    Beta function: y = y0 + A * [(x-xc)^(w2-1) * (w1-(x-xc))^(w3-1)] / [w1^(w2+w3-2)]
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    xc : float
        Center position
    w1 : float
        Width parameter 1
    w2 : float
        Shape parameter 2
    w3 : float
        Shape parameter 3
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_shifted = x - xc
    x_shifted = np.clip(x_shifted, 0, w1)  # Ensure valid range
    
    numerator = np.power(x_shifted, w2 - 1) * np.power(w1 - x_shifted, w3 - 1)
    denominator = np.power(w1, w2 + w3 - 2)
    
    return y0 + A * numerator / (denominator + 1e-10)


def boltzmann_sigmoidal(x: np.ndarray, A1: float, A2: float, x0: float, dx: float) -> np.ndarray:
    """
    Boltzmann sigmoidal curve: y = A2 + (A1-A2) / (1 + exp((x-x0)/dx))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    A1 : float
        Initial asymptote
    A2 : float
        Final asymptote
    x0 : float
        Center (inflection point)
    dx : float
        Time constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    exp_arg = (x - x0) / (dx + 1e-10)
    exp_arg = np.clip(exp_arg, -500, 500)  # Prevent overflow
    return A2 + (A1 - A2) / (1 + np.exp(exp_arg))


def dhyperbl_double_hyperbola(x: np.ndarray, P1: float, P2: float, P3: float, P4: float, P5: float) -> np.ndarray:
    """
    Double rectangular hyperbola: y = P1*x/(P2+x) + P3*x/(P4+x) + P5
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    P1, P2, P3, P4, P5 : float
        Function parameters
        
    Returns
    -------
    np.ndarray
        Function values
    """
    term1 = P1 * x / (P2 + x + 1e-10)
    term2 = P3 * x / (P4 + x + 1e-10)
    return term1 + term2 + P5


def exp_assoc(x: np.ndarray, y0: float, A1: float, t1: float, A2: float, t2: float) -> np.ndarray:
    """
    Exponential associate: y = y0 + A1*(1-exp(-x/t1)) + A2*(1-exp(-x/t2))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1 : float
        Amplitude 1
    t1 : float
        Time constant 1
    A2 : float
        Amplitude 2
    t2 : float
        Time constant 2
        
    Returns
    -------
    np.ndarray
        Function values
    """
    term1 = A1 * (1 - np.exp(-x / (abs(t1) + 1e-10)))
    term2 = A2 * (1 - np.exp(-x / (abs(t2) + 1e-10)))
    return y0 + term1 + term2


def exp_dec1(x: np.ndarray, y0: float, A: float, t: float) -> np.ndarray:
    """
    Exponential decay with time constant: y = y0 + A*exp(-x/t)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    t : float
        Time constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return y0 + A * np.exp(-x / (abs(t) + 1e-10))


def exp_dec2(x: np.ndarray, y0: float, A1: float, t1: float, A2: float, t2: float) -> np.ndarray:
    """
    Two-phase exponential decay: y = y0 + A1*exp(-x/t1) + A2*exp(-x/t2)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1 : float
        Amplitude 1
    t1 : float
        Time constant 1
    A2 : float
        Amplitude 2
    t2 : float
        Time constant 2
        
    Returns
    -------
    np.ndarray
        Function values
    """
    term1 = A1 * np.exp(-x / (abs(t1) + 1e-10))
    term2 = A2 * np.exp(-x / (abs(t2) + 1e-10))
    return y0 + term1 + term2


def exp_dec3(x: np.ndarray, y0: float, A1: float, t1: float, A2: float, t2: float, A3: float, t3: float) -> np.ndarray:
    """
    Three-phase exponential decay: y = y0 + A1*exp(-x/t1) + A2*exp(-x/t2) + A3*exp(-x/t3)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1, A2, A3 : float
        Amplitudes
    t1, t2, t3 : float
        Time constants
        
    Returns
    -------
    np.ndarray
        Function values
    """
    term1 = A1 * np.exp(-x / (abs(t1) + 1e-10))
    term2 = A2 * np.exp(-x / (abs(t2) + 1e-10))
    term3 = A3 * np.exp(-x / (abs(t3) + 1e-10))
    return y0 + term1 + term2 + term3


def exp_decay1_offset(x: np.ndarray, y0: float, A1: float, x0: float, t1: float) -> np.ndarray:
    """
    Exponential decay with offset: y = y0 + A1*exp(-(x-x0)/t1)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1 : float
        Amplitude
    x0 : float
        Time offset
    t1 : float
        Time constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return y0 + A1 * np.exp(-(x - x0) / (abs(t1) + 1e-10))


def exp_decay2_offset(x: np.ndarray, y0: float, A1: float, x0: float, t1: float, A2: float, t2: float) -> np.ndarray:
    """
    Two-phase exponential decay with offset: y = y0 + A1*exp(-(x-x0)/t1) + A2*exp(-(x-x0)/t2)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1, A2 : float
        Amplitudes
    x0 : float
        Time offset
    t1, t2 : float
        Time constants
        
    Returns
    -------
    np.ndarray
        Function values
    """
    term1 = A1 * np.exp(-(x - x0) / (abs(t1) + 1e-10))
    term2 = A2 * np.exp(-(x - x0) / (abs(t2) + 1e-10))
    return y0 + term1 + term2


def exp_decay3_offset(x: np.ndarray, y0: float, A1: float, x0: float, t1: float, A2: float, t2: float, A3: float, t3: float) -> np.ndarray:
    """
    Three-phase exponential decay with offset: y = y0 + A1*exp(-(x-x0)/t1) + A2*exp(-(x-x0)/t2) + A3*exp(-(x-x0)/t3)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1, A2, A3 : float
        Amplitudes
    x0 : float
        Time offset
    t1, t2, t3 : float
        Time constants
        
    Returns
    -------
    np.ndarray
        Function values
    """
    term1 = A1 * np.exp(-(x - x0) / (abs(t1) + 1e-10))
    term2 = A2 * np.exp(-(x - x0) / (abs(t2) + 1e-10))
    term3 = A3 * np.exp(-(x - x0) / (abs(t3) + 1e-10))
    return y0 + term1 + term2 + term3


def exp_grow1_offset(x: np.ndarray, y0: float, A1: float, x0: float, t1: float) -> np.ndarray:
    """
    Exponential growth with offset: y = y0 + A1*exp((x-x0)/t1)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1 : float
        Amplitude
    x0 : float
        Time offset
    t1 : float
        Time constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return y0 + A1 * np.exp((x - x0) / (abs(t1) + 1e-10))


def exp_grow2_offset(x: np.ndarray, y0: float, A1: float, x0: float, t1: float, A2: float, t2: float) -> np.ndarray:
    """
    Two-phase exponential growth with offset: y = y0 + A1*exp((x-x0)/t1) + A2*exp((x-x0)/t2)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A1, A2 : float
        Amplitudes
    x0 : float
        Time offset
    t1, t2 : float
        Time constants
        
    Returns
    -------
    np.ndarray
        Function values
    """
    term1 = A1 * np.exp((x - x0) / (abs(t1) + 1e-10))
    term2 = A2 * np.exp((x - x0) / (abs(t2) + 1e-10))
    return y0 + term1 + term2


def gauss_area(x: np.ndarray, y0: float, A: float, xc: float, w: float) -> np.ndarray:
    """
    Area version of Gaussian: y = y0 + A / (w*sqrt(π/2)) * exp(-2*(x-xc)²/w²)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Area under curve
    xc : float
        Center position
    w : float
        Width parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    norm_factor = 1 / (w_safe * np.sqrt(np.pi / 2))
    exponent = -2 * (x - xc)**2 / w_safe**2
    return y0 + A * norm_factor * np.exp(exponent)


def gauss_amp(x: np.ndarray, y0: float, A: float, xc: float, w: float) -> np.ndarray:
    """
    Amplitude version of Gaussian: y = y0 + A * exp(-(x-xc)²/w²)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude (peak height)
    xc : float
        Center position
    w : float
        Width parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    exponent = -(x - xc)**2 / w_safe**2
    return y0 + A * np.exp(exponent)


def hyperbl_michaelis_menten(x: np.ndarray, P1: float, P2: float) -> np.ndarray:
    """
    Hyperbola function (Michaelis-Menten): y = P1*x / (P2+x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    P1 : float
        Maximum value (Vmax)
    P2 : float
        Half-saturation constant (Km)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return P1 * x / (P2 + x + 1e-10)


def logistic_dose_response(x: np.ndarray, A1: float, A2: float, x0: float, p: float) -> np.ndarray:
    """
    Logistic dose response: y = A2 + (A1-A2) / (1 + (x/x0)^p)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (dose)
    A1 : float
        Lower asymptote
    A2 : float
        Upper asymptote
    x0 : float
        EC50 (half-maximum effective concentration)
    p : float
        Hill coefficient (slope parameter)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    x0_safe = abs(x0) + 1e-10
    ratio = x_safe / x0_safe
    return A2 + (A1 - A2) / (1 + np.power(ratio, p))


def log_normal(x: np.ndarray, y0: float, A: float, xc: float, w: float) -> np.ndarray:
    """
    Log-Normal function: y = y0 + A / (x*w*sqrt(2π)) * exp(-[ln(x/xc)]²/(2w²))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 0)
    y0 : float
        Baseline offset
    A : float
        Area under curve
    xc : float
        Median
    w : float
        Shape parameter (log standard deviation)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.where(x > 0, x, 1e-10)
    xc_safe = abs(xc) + 1e-10
    w_safe = abs(w) + 1e-10
    
    norm_factor = A / (x_safe * w_safe * np.sqrt(2 * np.pi))
    ln_ratio = np.log(x_safe / xc_safe)
    exponent = -(ln_ratio**2) / (2 * w_safe**2)
    
    return y0 + norm_factor * np.exp(exponent)


def lorentz_peak(x: np.ndarray, y0: float, A: float, xc: float, w: float) -> np.ndarray:
    """
    Lorentzian peak: y = y0 + 2A/π * w / (4(x-xc)² + w²)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Area under peak
    xc : float
        Center position
    w : float
        Full width at half maximum (FWHM)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    numerator = 2 * A * w_safe / np.pi
    denominator = 4 * (x - xc)**2 + w_safe**2
    return y0 + numerator / denominator


def pulse_function(x: np.ndarray, y0: float, A: float, x0: float, t1: float, t2: float, P: float) -> np.ndarray:
    """
    Pulse function: y = y0 + A * (1-exp(-(x-x0)/t1)) * (1-exp(-(x-x0)^P/t2))^(1/P)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    x0 : float
        Time offset
    t1 : float
        Rise time constant
    t2 : float
        Decay time constant
    P : float
        Power parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_shifted = x - x0
    t1_safe = abs(t1) + 1e-10
    t2_safe = abs(t2) + 1e-10
    P_safe = abs(P) + 1e-10
    
    rise_term = 1 - np.exp(-x_shifted / t1_safe)
    decay_term = np.power(1 - np.exp(-np.power(x_shifted, P_safe) / t2_safe), 1/P_safe)
    
    return y0 + A * rise_term * decay_term


def rational0(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Rational function type 0: y = (1+b*x) / (a+c*x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Constant term in denominator
    b : float
        Linear coefficient in numerator
    c : float
        Linear coefficient in denominator
        
    Returns
    -------
    np.ndarray
        Function values
    """
    numerator = 1 + b * x
    denominator = a + c * x
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    return numerator / denominator


def sine_origin(x: np.ndarray, A: float, xc: float, w: float) -> np.ndarray:
    """
    Sine function: y = A * sin(π(x-xc)/w)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    A : float
        Amplitude
    xc : float
        Phase shift (center)
    w : float
        Period parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    return A * np.sin(np.pi * (x - xc) / w_safe)


def voigt_profile(x: np.ndarray, y0: float, A: float, xc: float, wG: float, wL: float) -> np.ndarray:
    """
    Voigt profile: convolution of Gaussian and Lorentzian
    Approximated using Pseudo-Voigt for computational efficiency
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    xc : float
        Center position
    wG : float
        Gaussian width
    wL : float
        Lorentzian width
        
    Returns
    -------
    np.ndarray
        Function values (Pseudo-Voigt approximation)
    """
    # Pseudo-Voigt approximation
    wG_safe = abs(wG) + 1e-10
    wL_safe = abs(wL) + 1e-10
    
    # Calculate mixing parameter
    eta = 1.36603 * (wL_safe / wG_safe) - 0.47719 * (wL_safe / wG_safe)**2 + 0.11116 * (wL_safe / wG_safe)**3
    eta = np.clip(eta, 0, 1)
    
    # Gaussian component
    gaussian = np.exp(-(x - xc)**2 / (2 * wG_safe**2))
    
    # Lorentzian component  
    lorentzian = 1 / (1 + (x - xc)**2 / wL_safe**2)
    
    return y0 + A * (eta * lorentzian + (1 - eta) * gaussian)


# =============================================================================
# EXPONENTIAL FUNCTIONS (5 functions)
# =============================================================================

def asymptotic1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Asymptotic regression model: y = a - b*exp(-c*x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Asymptotic value
    b : float
        Range parameter
    c : float
        Rate parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a - b * np.exp(-c * x)


def box_lucas1(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Box Lucas model: y = a * (1 - exp(-b*x))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Asymptotic maximum
    b : float
        Rate parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * (1 - np.exp(-b * x))


def chapman_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Chapman model: y = a * (1 - exp(-b*(x-c)))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Asymptotic maximum
    b : float
        Rate parameter
    c : float
        Lag parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * (1 - np.exp(-b * (x - c)))


def exponential_growth(x: np.ndarray, y0: float, A: float, R0: float) -> np.ndarray:
    """
    Exponential growth: y = y0 + A * exp(R0*x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Initial value
    A : float
        Pre-exponential factor
    R0 : float
        Growth rate
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return y0 + A * np.exp(R0 * x)


def monomolecular_growth(x: np.ndarray, A: float, k: float, xc: float) -> np.ndarray:
    """
    Monomolecular growth model: y = A * (1 - exp(-k*(x-xc)))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    A : float
        Maximum value
    k : float
        Growth rate constant
    xc : float
        Lag time
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return A * (1 - np.exp(-k * (x - xc)))


# =============================================================================
# GROWTH/SIGMOIDAL FUNCTIONS (4 functions)
# =============================================================================

def hill_function(x: np.ndarray, Vmax: float, k: float, n: float) -> np.ndarray:
    """
    Hill function: y = Vmax * x^n / (k^n + x^n)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    Vmax : float
        Maximum response
    k : float
        Half-saturation constant
    n : float
        Hill coefficient (cooperativity)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    k_safe = abs(k) + 1e-10
    numerator = Vmax * np.power(x_safe, n)
    denominator = np.power(k_safe, n) + np.power(x_safe, n)
    return numerator / denominator


def gompertz_growth(x: np.ndarray, a: float, k: float, xc: float) -> np.ndarray:
    """
    Gompertz growth model: y = a * exp(-exp(-k*(x-xc)))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Upper asymptote
    k : float
        Growth rate
    xc : float
        Inflection point
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return a * np.exp(-np.exp(-k * (x - xc)))


def sigmoidal_logistic1(x: np.ndarray, a: float, k: float, xc: float) -> np.ndarray:
    """
    Sigmoidal logistic function type 1: y = a / (1 + exp(-k*(x-xc)))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Upper asymptote
    k : float
        Slope parameter
    xc : float
        Inflection point
        
    Returns
    -------
    np.ndarray
        Function values
    """
    exp_arg = -k * (x - xc)
    exp_arg = np.clip(exp_arg, -500, 500)  # Prevent overflow
    return a / (1 + np.exp(exp_arg))


def sigmoidal_logistic2(x: np.ndarray, y0: float, a: float, x0: float, Wmax: float) -> np.ndarray:
    """
    Sigmoidal logistic function type 2: y = (y0 + a*exp(Wmax*(x-x0)/4)) / (1 + exp(Wmax*(x-x0)/4))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Lower asymptote
    a : float
        Range parameter
    x0 : float
        Inflection point
    Wmax : float
        Maximum growth rate
        
    Returns
    -------
    np.ndarray
        Function values
    """
    exp_arg = Wmax * (x - x0) / 4
    exp_arg = np.clip(exp_arg, -500, 500)
    exp_term = np.exp(exp_arg)
    return (y0 + a * exp_term) / (1 + exp_term)


# =============================================================================
# PEAK FUNCTIONS (4 functions)
# =============================================================================

def extreme_value(x: np.ndarray, y0: float, A: float, xc: float, w: float) -> np.ndarray:
    """
    Extreme value function: y = y0 + A * exp(-exp(-(x-xc)/w) - (x-xc)/w)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    xc : float
        Mode position
    w : float
        Width parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    z = -(x - xc) / w_safe
    return y0 + A * np.exp(-np.exp(z) + z)


def pearson_vii(x: np.ndarray, y0: float, A: float, xc: float, w: float, m: float) -> np.ndarray:
    """
    Pearson VII peak function: y = y0 + A * [1 + 4*(2^(1/m)-1)*(x-xc)²/w²]^(-m)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    xc : float
        Center position
    w : float
        Full width at half maximum
    m : float
        Shape parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    m_safe = abs(m) + 1e-10
    
    factor = 4 * (np.power(2, 1/m_safe) - 1)
    term = 1 + factor * (x - xc)**2 / w_safe**2
    return y0 + A * np.power(term, -m_safe)


def pseudo_voigt1(x: np.ndarray, y0: float, A: float, xc: float, w: float, mu: float) -> np.ndarray:
    """
    Pseudo-Voigt function (single width): y = y0 + A * [μ*L(x) + (1-μ)*G(x)]
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    xc : float
        Center position
    w : float
        Width parameter
    mu : float
        Mixing parameter (0=Gaussian, 1=Lorentzian)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    mu = np.clip(mu, 0, 1)
    
    # Gaussian component
    gaussian = np.exp(-0.5 * ((x - xc) / w_safe)**2)
    
    # Lorentzian component
    lorentzian = 1 / (1 + ((x - xc) / w_safe)**2)
    
    return y0 + A * (mu * lorentzian + (1 - mu) * gaussian)


def pseudo_voigt2(x: np.ndarray, y0: float, A: float, xc: float, wG: float, wL: float, mu: float) -> np.ndarray:
    """
    Pseudo-Voigt function (separate widths): y = y0 + A * [μ*L(x,wL) + (1-μ)*G(x,wG)]
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude
    xc : float
        Center position
    wG : float
        Gaussian width
    wL : float
        Lorentzian width
    mu : float
        Mixing parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    wG_safe = abs(wG) + 1e-10
    wL_safe = abs(wL) + 1e-10
    mu = np.clip(mu, 0, 1)
    
    # Gaussian component
    gaussian = np.exp(-0.5 * ((x - xc) / wG_safe)**2)
    
    # Lorentzian component
    lorentzian = 1 / (1 + ((x - xc) / wL_safe)**2)
    
    return y0 + A * (mu * lorentzian + (1 - mu) * gaussian)


# =============================================================================
# CHROMATOGRAPHY FUNCTIONS (5 functions)
# =============================================================================

def chesler_cram_peak(x: np.ndarray, y0: float, H: float, tr: float, W: float, B1: float, B2: float) -> np.ndarray:
    """
    Chesler-Cram peak function: Complex chromatographic peak model
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (time)
    y0 : float
        Baseline offset
    H : float
        Peak height
    tr : float
        Retention time
    W : float
        Peak width
    B1 : float
        Asymmetry parameter 1
    B2 : float
        Asymmetry parameter 2
        
    Returns
    -------
    np.ndarray
        Function values
    """
    # Simplified Chesler-Cram approximation
    W_safe = abs(W) + 1e-10
    t_norm = (x - tr) / W_safe
    
    # Asymmetric Gaussian-like shape
    if B1 != 0 or B2 != 0:
        asym_factor = 1 + B1 * t_norm + B2 * t_norm**2
        gaussian = np.exp(-0.5 * t_norm**2 * asym_factor)
    else:
        gaussian = np.exp(-0.5 * t_norm**2)
    
    return y0 + H * gaussian


def edgeworth_cramer_peak(x: np.ndarray, y0: float, A: float, xc: float, w: float, a3: float, a4: float) -> np.ndarray:
    """
    Edgeworth-Cramer peak function: y = y0 + A/(w*sqrt(2π)) * exp(-z²/2) * [1 + a3*H3(z)/3! + a4*H4(z)/4!]
    where z = (x-xc)/w and H_n are Hermite polynomials
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Area under peak
    xc : float
        Center position
    w : float
        Width parameter
    a3 : float
        Skewness parameter
    a4 : float
        Kurtosis parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    z = (x - xc) / w_safe
    
    # Hermite polynomials H3(z) and H4(z)
    H3 = z**3 - 3*z
    H4 = z**4 - 6*z**2 + 3
    
    # Base Gaussian
    gaussian = np.exp(-0.5 * z**2) / (w_safe * np.sqrt(2 * np.pi))
    
    # Edgeworth-Cramer correction
    correction = 1 + a3 * H3 / 6 + a4 * H4 / 24
    
    return y0 + A * gaussian * correction


def exponentially_modified_gaussian(x: np.ndarray, y0: float, A: float, xc: float, w: float, t0: float) -> np.ndarray:
    """
    Exponentially modified Gaussian: y = y0 + A/(2*t0) * exp(w²/(2*t0²) - (x-xc)/t0) * erfc((w²/t0 - (x-xc))/w/sqrt(2))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Area under peak
    xc : float
        Gaussian center
    w : float
        Gaussian width
    t0 : float
        Exponential decay time
        
    Returns
    -------
    np.ndarray
        Function values
    """
    t0_safe = abs(t0) + 1e-10
    w_safe = abs(w) + 1e-10
    
    exp_term = np.exp(w_safe**2 / (2 * t0_safe**2) - (x - xc) / t0_safe)
    erfc_arg = (w_safe**2 / t0_safe - (x - xc)) / (w_safe * np.sqrt(2))
    erfc_term = scipy.special.erfc(erfc_arg)
    
    return y0 + A / (2 * t0_safe) * exp_term * erfc_term


def gram_charlier_peak(x: np.ndarray, y0: float, A: float, xc: float, w: float, a3: float, a4: float, a5: float) -> np.ndarray:
    """
    Gram-Charlier peak function: y = y0 + A/(w*sqrt(2π)) * exp(-z²/2) * sum(ai*Hi(z)/i!) for i=3,4,5
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Area under peak
    xc : float
        Center position
    w : float
        Width parameter
    a3, a4, a5 : float
        Hermite polynomial coefficients
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    z = (x - xc) / w_safe
    
    # Hermite polynomials
    H3 = z**3 - 3*z
    H4 = z**4 - 6*z**2 + 3
    H5 = z**5 - 10*z**3 + 15*z
    
    # Base Gaussian
    gaussian = np.exp(-0.5 * z**2) / (w_safe * np.sqrt(2 * np.pi))
    
    # Gram-Charlier series
    series = 1 + a3 * H3 / 6 + a4 * H4 / 24 + a5 * H5 / 120
    
    return y0 + A * gaussian * series


def giddings_peak(x: np.ndarray, y0: float, A: float, xc: float, w: float, c: float) -> np.ndarray:
    """
    Giddings peak function: y = y0 + A/w * exp(-(x-xc)²/w² + c*(x-xc))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y0 : float
        Baseline offset
    A : float
        Amplitude parameter
    xc : float
        Center position
    w : float
        Width parameter
    c : float
        Asymmetry parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    w_safe = abs(w) + 1e-10
    x_diff = x - xc
    exponent = -x_diff**2 / w_safe**2 + c * x_diff
    return y0 + A / w_safe * np.exp(exponent)


# =============================================================================
# SPECIALIZED APPLICATION FUNCTIONS (8 functions)
# =============================================================================

def biphasic_dose_response(x: np.ndarray, y0: float, A1: float, EC50_1: float, n1: float, A2: float, EC50_2: float, n2: float) -> np.ndarray:
    """
    Biphasic dose response: y = y0 + A1*x^n1/(EC50_1^n1 + x^n1) + A2*x^n2/(EC50_2^n2 + x^n2)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (dose)
    y0 : float
        Baseline response
    A1, A2 : float
        Maximum responses for each phase
    EC50_1, EC50_2 : float
        Half-maximum concentrations
    n1, n2 : float
        Hill coefficients
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    
    # First phase
    term1 = A1 * np.power(x_safe, n1) / (np.power(abs(EC50_1) + 1e-10, n1) + np.power(x_safe, n1))
    
    # Second phase
    term2 = A2 * np.power(x_safe, n2) / (np.power(abs(EC50_2) + 1e-10, n2) + np.power(x_safe, n2))
    
    return y0 + term1 + term2


def dose_response(x: np.ndarray, y0: float, A: float, EC50: float, n: float) -> np.ndarray:
    """
    Dose response function: y = y0 + A*x^n/(EC50^n + x^n)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (dose)
    y0 : float
        Baseline response
    A : float
        Maximum response
    EC50 : float
        Half-maximum effective concentration
    n : float
        Hill coefficient
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    EC50_safe = abs(EC50) + 1e-10
    return y0 + A * np.power(x_safe, n) / (np.power(EC50_safe, n) + np.power(x_safe, n))


def one_site_binding(x: np.ndarray, Bmax: float, Kd: float) -> np.ndarray:
    """
    One-site binding: y = Bmax * x / (Kd + x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (ligand concentration)
    Bmax : float
        Maximum binding capacity
    Kd : float
        Dissociation constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return Bmax * x / (abs(Kd) + x + 1e-10)


def two_site_binding(x: np.ndarray, Bmax1: float, Kd1: float, Bmax2: float, Kd2: float) -> np.ndarray:
    """
    Two-site binding: y = Bmax1*x/(Kd1+x) + Bmax2*x/(Kd2+x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (ligand concentration)
    Bmax1, Bmax2 : float
        Maximum binding capacities
    Kd1, Kd2 : float
        Dissociation constants
        
    Returns
    -------
    np.ndarray
        Function values
    """
    site1 = Bmax1 * x / (abs(Kd1) + x + 1e-10)
    site2 = Bmax2 * x / (abs(Kd2) + x + 1e-10)
    return site1 + site2


def michaelis_menten_kinetics(x: np.ndarray, Vmax: float, Km: float) -> np.ndarray:
    """
    Michaelis-Menten enzyme kinetics: y = Vmax * x / (Km + x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (substrate concentration)
    Vmax : float
        Maximum reaction velocity
    Km : float
        Michaelis constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    return Vmax * x / (abs(Km) + x + 1e-10)


def competitive_inhibition(x: np.ndarray, Vmax: float, Km: float, I: float, Ki: float) -> np.ndarray:
    """
    Competitive inhibition: y = Vmax * x / (Km*(1 + I/Ki) + x)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (substrate concentration)
    Vmax : float
        Maximum reaction velocity
    Km : float
        Michaelis constant
    I : float
        Inhibitor concentration
    Ki : float
        Inhibition constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    apparent_Km = Km * (1 + I / (abs(Ki) + 1e-10))
    return Vmax * x / (apparent_Km + x + 1e-10)


def noncompetitive_inhibition(x: np.ndarray, Vmax: float, Km: float, I: float, Ki: float) -> np.ndarray:
    """
    Non-competitive inhibition: y = Vmax * x / ((1 + I/Ki) * (Km + x))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (substrate concentration)
    Vmax : float
        Maximum reaction velocity
    Km : float
        Michaelis constant
    I : float
        Inhibitor concentration
    Ki : float
        Inhibition constant
        
    Returns
    -------
    np.ndarray
        Function values
    """
    inhibition_factor = 1 + I / (abs(Ki) + 1e-10)
    return Vmax * x / (inhibition_factor * (Km + x + 1e-10))


def allosteric_sigmoidal(x: np.ndarray, Vmax: float, K: float, n: float) -> np.ndarray:
    """
    Allosteric sigmoidal kinetics: y = Vmax * x^n / (K^n + x^n)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (substrate concentration)
    Vmax : float
        Maximum reaction velocity
    K : float
        Half-saturation constant
    n : float
        Hill coefficient (cooperativity)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    x_safe = np.abs(x) + 1e-10
    K_safe = abs(K) + 1e-10
    return Vmax * np.power(x_safe, n) / (np.power(K_safe, n) + np.power(x_safe, n))


# Function registry for Origin functions
ORIGIN_FUNCTIONS = {
    # Origin Basic Functions (26)
    'allometric1_freundlich': (allometric1_freundlich, ['a', 'b']),
    'beta_origin': (beta_origin, ['y0', 'A', 'xc', 'w1', 'w2', 'w3']),
    'boltzmann': (boltzmann_sigmoidal, ['A1', 'A2', 'x0', 'dx']),
    'dhyperbl': (dhyperbl_double_hyperbola, ['P1', 'P2', 'P3', 'P4', 'P5']),
    'exp_assoc': (exp_assoc, ['y0', 'A1', 't1', 'A2', 't2']),
    'exp_dec1': (exp_dec1, ['y0', 'A', 't']),
    'exp_dec2': (exp_dec2, ['y0', 'A1', 't1', 'A2', 't2']),
    'exp_dec3': (exp_dec3, ['y0', 'A1', 't1', 'A2', 't2', 'A3', 't3']),
    'exp_decay1': (exp_decay1_offset, ['y0', 'A1', 'x0', 't1']),
    'exp_decay2': (exp_decay2_offset, ['y0', 'A1', 'x0', 't1', 'A2', 't2']),
    'exp_decay3': (exp_decay3_offset, ['y0', 'A1', 'x0', 't1', 'A2', 't2', 'A3', 't3']),
    'exp_grow1': (exp_grow1_offset, ['y0', 'A1', 'x0', 't1']),
    'exp_grow2': (exp_grow2_offset, ['y0', 'A1', 'x0', 't1', 'A2', 't2']),
    'gauss_area': (gauss_area, ['y0', 'A', 'xc', 'w']),
    'gauss_amp': (gauss_amp, ['y0', 'A', 'xc', 'w']),
    'hyperbl': (hyperbl_michaelis_menten, ['P1', 'P2']),
    'logistic_dose': (logistic_dose_response, ['A1', 'A2', 'x0', 'p']),
    'log_normal': (log_normal, ['y0', 'A', 'xc', 'w']),
    'lorentz': (lorentz_peak, ['y0', 'A', 'xc', 'w']),
    'pulse': (pulse_function, ['y0', 'A', 'x0', 't1', 't2', 'P']),
    'rational0': (rational0, ['a', 'b', 'c']),
    'sine_origin': (sine_origin, ['A', 'xc', 'w']),
    'voigt': (voigt_profile, ['y0', 'A', 'xc', 'wG', 'wL']),
    
    # Exponential Functions (5)
    'asymptotic1': (asymptotic1, ['a', 'b', 'c']),
    'box_lucas1': (box_lucas1, ['a', 'b']),
    'chapman': (chapman_model, ['a', 'b', 'c']),
    'exponential_origin': (exponential_growth, ['y0', 'A', 'R0']),
    'monomolecular': (monomolecular_growth, ['A', 'k', 'xc']),
    
    # Growth/Sigmoidal Functions (4)
    'hill': (hill_function, ['Vmax', 'k', 'n']),
    'gompertz': (gompertz_growth, ['a', 'k', 'xc']),
    'slogistic1': (sigmoidal_logistic1, ['a', 'k', 'xc']),
    'slogistic2': (sigmoidal_logistic2, ['y0', 'a', 'x0', 'Wmax']),
    
    # Peak Functions (4)
    'extreme': (extreme_value, ['y0', 'A', 'xc', 'w']),
    'pearson_vii': (pearson_vii, ['y0', 'A', 'xc', 'w', 'm']),
    'pseudo_voigt1': (pseudo_voigt1, ['y0', 'A', 'xc', 'w', 'mu']),
    'pseudo_voigt2': (pseudo_voigt2, ['y0', 'A', 'xc', 'wG', 'wL', 'mu']),
    
    # Chromatography Functions (5)
    'chesler_cram': (chesler_cram_peak, ['y0', 'H', 'tr', 'W', 'B1', 'B2']),
    'edgeworth_cramer': (edgeworth_cramer_peak, ['y0', 'A', 'xc', 'w', 'a3', 'a4']),
    'exp_modified_gaussian': (exponentially_modified_gaussian, ['y0', 'A', 'xc', 'w', 't0']),
    'gram_charlier': (gram_charlier_peak, ['y0', 'A', 'xc', 'w', 'a3', 'a4', 'a5']),
    'giddings': (giddings_peak, ['y0', 'A', 'xc', 'w', 'c']),
    
    # Specialized Application Functions (8)
    'biphasic_dose_response': (biphasic_dose_response, ['y0', 'A1', 'EC50_1', 'n1', 'A2', 'EC50_2', 'n2']),
    'dose_response': (dose_response, ['y0', 'A', 'EC50', 'n']),
    'one_site_binding': (one_site_binding, ['Bmax', 'Kd']),
    'two_site_binding': (two_site_binding, ['Bmax1', 'Kd1', 'Bmax2', 'Kd2']),
    'michaelis_menten': (michaelis_menten_kinetics, ['Vmax', 'Km']),
    'competitive_inhibition': (competitive_inhibition, ['Vmax', 'Km', 'I', 'Ki']),
    'noncompetitive_inhibition': (noncompetitive_inhibition, ['Vmax', 'Km', 'I', 'Ki']),
    'allosteric_sigmoidal': (allosteric_sigmoidal, ['Vmax', 'K', 'n']),
}

print(f"Origin Functions Module loaded: {len(ORIGIN_FUNCTIONS)} functions")