"""
Zeta Functions Module
====================

This module contains Riemann zeta and related L-functions including:
- Riemann zeta function
- Dirichlet functions
- Hurwitz zeta function
- Polylogarithms
- Fermi-Dirac integrals
- Related transcendental functions

Total Functions: 16
"""

import numpy as np
import scipy.special

# =============================================================================
# RIEMANN ZETA AND RELATED FUNCTIONS (16 functions)
# =============================================================================

def riemann_zeta_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Riemann zeta function: ζ(s) = Σₙ₌₁^∞ n^(-s) for Re(s) > 1
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable s
    a : float
        Amplitude parameter
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Riemann zeta function values
    """
    arg = b * x
    arg = np.where(arg > 1, arg, 1 + 1e-10)
    return a * scipy.special.zeta(arg, 1)


def riemann_xi_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Riemann Xi function: ξ(s) = ½s(s-1)π^(-s/2)Γ(s/2)ζ(s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable s
    a : float
        Amplitude parameter
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Riemann Xi function values
    """
    s = b * x
    s = np.where(s > 1, s, 1 + 1e-10)
    zeta_s = scipy.special.zeta(s, 1)
    gamma_s2 = scipy.special.gamma(s/2)
    xi = 0.5 * s * (s - 1) * np.pi**(-s/2) * gamma_s2 * zeta_s
    return a * xi


def dirichlet_eta_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Dirichlet eta function: η(s) = Σₙ₌₁^∞ (-1)^(n-1) n^(-s) = (1-2^(1-s))ζ(s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable s
    a : float
        Amplitude parameter
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Dirichlet eta function values
    """
    s = b * x
    factor = 1 - 2**(1 - s)
    zeta_s = scipy.special.zeta(s, 1)
    return a * factor * zeta_s


def dirichlet_beta_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Dirichlet beta function: β(s) = Σₙ₌₀^∞ (-1)ⁿ (2n+1)^(-s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable s
    a : float
        Amplitude parameter
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Dirichlet beta function approximation
    """
    s = b * x
    # Approximation using series
    result = np.zeros_like(s)
    for n in range(20):  # First 20 terms
        result += (-1)**n / (2*n + 1)**s
    return a * result


def dirichlet_l_function(x: np.ndarray, a: float, b: float, chi: int = 1) -> np.ndarray:
    """
    Dirichlet L-function: L(s,χ) = Σₙ₌₁^∞ χ(n) n^(-s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable s
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    chi : int, optional
        Character parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Dirichlet L-function approximation
    """
    s = b * x
    s = np.where(s > 1, s, 1 + 1e-10)
    return a * scipy.special.zeta(s, 1) * (1 + chi * 0.1)  # Simplified


def hurwitz_zeta_function(x: np.ndarray, a: float, b: float, q: float = 1) -> np.ndarray:
    """
    Hurwitz zeta function: ζ(s,q) = Σₙ₌₀^∞ (n+q)^(-s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable s
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    q : float, optional
        Hurwitz parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Hurwitz zeta function values
    """
    s = b * x
    s = np.where(s > 1, s, 1 + 1e-10)
    q = max(q, 1e-10)
    return a * scipy.special.zeta(s, q)


def legendre_chi_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Legendre chi function: χ(x) = Σₙ₌₀^∞ x^(2n+1)/(2n+1) for |x| < 1
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (|x| < 1)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Legendre chi function values
    """
    x_clipped = np.clip(b * x, -0.99, 0.99)
    result = np.zeros_like(x_clipped)
    for n in range(20):  # First 20 terms
        result += x_clipped**(2*n + 1) / (2*n + 1)
    return a * result


def lerch_transcendent(x: np.ndarray, a: float, lambda_param: float, s: float) -> np.ndarray:
    """
    Lerch transcendent: Φ(z,s,a) = Σₙ₌₀^∞ z^n (n+a)^(-s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (used as shifting parameter)
    a : float
        Amplitude parameter
    lambda_param : float
        Parameter z (|z| < 1 for convergence)
    s : float
        Complex parameter s
        
    Returns
    -------
    np.ndarray
        Lerch transcendent approximation
    """
    z = np.clip(lambda_param, -0.99, 0.99)
    result = np.zeros_like(x)
    for n in range(20):  # First 20 terms
        result += z**n / (n + a)**s
    return a * result


def polylogarithm(x: np.ndarray, a: float, s: float) -> np.ndarray:
    """
    Polylogarithm: Li_s(z) = Σₙ₌₁^∞ z^n n^(-s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable z (|z| < 1 for convergence)
    a : float
        Amplitude parameter
    s : float
        Complex parameter s
        
    Returns
    -------
    np.ndarray
        Polylogarithm values
    """
    x_clipped = np.clip(x, -0.99, 0.99)
    result = np.zeros_like(x_clipped)
    for n in range(1, 21):  # First 20 terms
        result += x_clipped**n / n**s
    return a * result


def incomplete_polylogarithm(x: np.ndarray, a: float, s: float, b: float) -> np.ndarray:
    """
    Incomplete polylogarithm: Li_s^{(b)}(z) = Σₙ₌₁^b z^n n^(-s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable z
    a : float
        Amplitude parameter
    s : float
        Complex parameter s
    b : float
        Upper limit of sum
        
    Returns
    -------
    np.ndarray
        Incomplete polylogarithm values
    """
    x_clipped = np.clip(x, -0.99, 0.99)
    b_int = max(1, int(b))
    result = np.zeros_like(x_clipped)
    for n in range(1, b_int + 1):
        result += x_clipped**n / n**s
    return a * result


def clausen_function(x: np.ndarray, a: float, n: int = 2) -> np.ndarray:
    """
    Clausen function: Cl_n(θ) = Σₖ₌₁^∞ sin(kθ)/k^n (for even n) or cos(kθ)/k^n (for odd n)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable θ
    a : float
        Amplitude parameter
    n : int, optional
        Order of Clausen function (default: 2)
        
    Returns
    -------
    np.ndarray
        Clausen function values
    """
    n = max(1, int(n))
    result = np.zeros_like(x)
    for k in range(1, 21):  # First 20 terms
        if n % 2 == 0:  # Even n
            result += np.sin(k * x) / k**n
        else:  # Odd n
            result += np.cos(k * x) / k**n
    return a * result


def complete_fermi_dirac_integral(x: np.ndarray, a: float, j: float) -> np.ndarray:
    """
    Complete Fermi-Dirac integral: F_j(η) = (1/Γ(j+1)) ∫₀^∞ t^j/(e^(t-η)+1) dt
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable η (chemical potential)
    a : float
        Amplitude parameter
    j : float
        Order parameter
        
    Returns
    -------
    np.ndarray
        Fermi-Dirac integral approximation
    """
    eta = x
    z = 1 / (1 + np.exp(-eta))
    return a * polylogarithm(z, 1, j + 1)


def dilogarithm(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Dilogarithm: Li₂(z) = -∫₀^z ln(1-t)/t dt
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable z
    a : float
        Amplitude parameter
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Dilogarithm values
    """
    return a * polylogarithm(b * x, 1, 2)


def incomplete_fermi_dirac_integral(x: np.ndarray, a: float, j: float, upper_limit: float) -> np.ndarray:
    """
    Incomplete Fermi-Dirac integral: F_j^{(b)}(η) with finite upper integration limit
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable η
    a : float
        Amplitude parameter
    j : float
        Order parameter
    upper_limit : float
        Upper integration limit
        
    Returns
    -------
    np.ndarray
        Incomplete Fermi-Dirac integral approximation
    """
    result = complete_fermi_dirac_integral(x, 1, j)
    factor = 1 - np.exp(-upper_limit)
    return a * result * factor


def kummer_function(x: np.ndarray, a: float, alpha: float, beta: float, z: float) -> np.ndarray:
    """
    Kummer's function M(α,β,z): confluent hypergeometric function ₁F₁(α;β;z)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (scaling factor)
    a : float
        Amplitude parameter
    alpha : float
        First parameter of confluent hypergeometric function
    beta : float
        Second parameter of confluent hypergeometric function
    z : float
        Argument scaling
        
    Returns
    -------
    np.ndarray
        Kummer function values
    """
    return a * scipy.special.hyp1f1(alpha, beta, z * x)


def riesz_function(x: np.ndarray, a: float, s: float) -> np.ndarray:
    """
    Riesz function: related to Riemann zeta function, R_s(x) = x^s ζ(s)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    s : float
        Parameter s
        
    Returns
    -------
    np.ndarray
        Riesz function approximation
    """
    s_val = s
    zeta_val = scipy.special.zeta(s_val, 1)
    return a * x**s_val * zeta_val


# Function registry for zeta functions
ZETA_FUNCTIONS = {
    'riemann_zeta': (riemann_zeta_function, ['a', 'b']),
    'riemann_xi': (riemann_xi_function, ['a', 'b']),
    'dirichlet_eta': (dirichlet_eta_function, ['a', 'b']),
    'dirichlet_beta': (dirichlet_beta_function, ['a', 'b']),
    'dirichlet_l': (dirichlet_l_function, ['a', 'b', 'chi']),
    'hurwitz_zeta': (hurwitz_zeta_function, ['a', 'b', 'q']),
    'legendre_chi': (legendre_chi_function, ['a', 'b']),
    'lerch_transcendent': (lerch_transcendent, ['a', 'lambda_param', 's']),
    'polylogarithm': (polylogarithm, ['a', 's']),
    'incomplete_polylogarithm': (incomplete_polylogarithm, ['a', 's', 'b']),
    'clausen': (clausen_function, ['a', 'n']),
    'complete_fermi_dirac': (complete_fermi_dirac_integral, ['a', 'j']),
    'dilogarithm': (dilogarithm, ['a', 'b']),
    'incomplete_fermi_dirac': (incomplete_fermi_dirac_integral, ['a', 'j', 'upper_limit']),
    'kummer': (kummer_function, ['a', 'alpha', 'beta', 'z']),
    'riesz': (riesz_function, ['a', 's']),
}

print(f"Zeta Functions Module loaded: {len(ZETA_FUNCTIONS)} functions")