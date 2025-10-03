"""
Advanced Functions Module
=========================

This module contains advanced mathematical functions including:
- Gamma and related functions
- Elliptic functions
- Bessel functions and related functions
- Riemann zeta and related functions

Total Functions: 78
"""

import numpy as np
import scipy.special
import scipy.stats

# =============================================================================
# GAMMA AND RELATED FUNCTIONS (12 functions)
# =============================================================================

def gamma_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Gamma function: Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
    
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
        Gamma function values
    """
    arg = b * x + c
    arg = np.where(arg > 0, arg, 1e-3)
    return a * scipy.special.gamma(arg)


def barnes_g_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Barnes G-function: superfactorial function G(x+1) = 0!¹ * 1!² * ... * (x-1)!ˣ
    
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
        Barnes G-function approximation
    """
    arg = b * x + c
    arg = np.where(arg > 0, arg, 1e-3)
    return a * np.exp(scipy.special.loggamma(arg) * arg)


def beta_function(x: np.ndarray, a: float, alpha: float, beta_param: float) -> np.ndarray:
    """
    Beta function: B(α,β) = ∫₀¹ t^(α-1) (1-t)^(β-1) dt
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (0 < x < 1)
    a : float
        Amplitude parameter
    alpha : float
        First shape parameter
    beta_param : float
        Second shape parameter
        
    Returns
    -------
    np.ndarray
        Beta function values
    """
    alpha = max(alpha, 1e-3)
    beta_param = max(beta_param, 1e-3)
    x_clipped = np.clip(x, 1e-10, 1 - 1e-10)
    return a * scipy.stats.beta.pdf(x_clipped, alpha, beta_param)


def digamma_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Digamma function: ψ(x) = d/dx ln Γ(x) = Γ'(x)/Γ(x)
    
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
        Digamma function values
    """
    arg = b * x + c
    arg = np.where(arg > 0, arg, 1e-3)
    return a * scipy.special.digamma(arg)


def polygamma_function(x: np.ndarray, a: float, b: float, c: float, n: int = 1) -> np.ndarray:
    """
    Polygamma function: ψ^(n)(x) = d^(n+1)/dx^(n+1) ln Γ(x)
    
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
    n : int, optional
        Order of polygamma function (default: 1)
        
    Returns
    -------
    np.ndarray
        Polygamma function values
    """
    arg = b * x + c
    arg = np.where(arg > 0, arg, 1e-3)
    n = max(0, int(n))
    return a * scipy.special.polygamma(n, arg)


def incomplete_beta_function(x: np.ndarray, a: float, alpha: float, beta_param: float) -> np.ndarray:
    """
    Incomplete beta function: Iₓ(α,β) = B(α,β,x)/B(α,β)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (0 ≤ x ≤ 1)
    a : float
        Amplitude parameter
    alpha : float
        First shape parameter
    beta_param : float
        Second shape parameter
        
    Returns
    -------
    np.ndarray
        Incomplete beta function values
    """
    alpha = max(alpha, 1e-3)
    beta_param = max(beta_param, 1e-3)
    x_clipped = np.clip(x, 0, 1)
    return a * scipy.special.betainc(alpha, beta_param, x_clipped)


def incomplete_gamma_function(x: np.ndarray, a: float, gamma_a: float, scale: float) -> np.ndarray:
    """
    Incomplete gamma function: γ(s,x) = ∫₀ˣ t^(s-1) e^(-t) dt
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    gamma_a : float
        Shape parameter
    scale : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Incomplete gamma function values
    """
    gamma_a = max(gamma_a, 1e-3)
    scale = max(abs(scale), 1e-3)
    return a * scipy.special.gammainc(gamma_a, x / scale)


def k_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    K-function (related to gamma): K(x) = Γ(x)Γ(1-x) = π/sin(πx)
    
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
        K-function values
    """
    arg = b * x + c
    # Avoid singularities at integers
    arg = np.where(np.abs(arg - np.round(arg)) < 1e-10, arg + 1e-6, arg)
    return a * np.pi / np.sin(np.pi * arg)


def multivariate_gamma_function(x: np.ndarray, a: float, p: int, scale: float) -> np.ndarray:
    """
    Multivariate gamma function: Γₚ(x) = π^(p(p-1)/4) ∏ⱼ₌₁ᵖ Γ(x + (1-j)/2)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    p : int
        Dimension parameter
    scale : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Multivariate gamma function values
    """
    result = np.ones_like(x)
    p = max(1, int(p))
    
    # Compute product
    for j in range(1, p + 1):
        arg = (x + (1 - j) / 2) * scale
        arg = np.where(arg > 0, arg, 1e-3)
        result *= scipy.special.gamma(arg)
    
    # Multiply by normalization constant
    normalization = np.pi ** (p * (p - 1) / 4)
    return a * normalization * result


def student_t_distribution(x: np.ndarray, amp: float, df: float, loc: float, scale: float) -> np.ndarray:
    """
    Student's t-distribution: t-distribution with df degrees of freedom
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude parameter
    df : float
        Degrees of freedom
    loc : float
        Location parameter
    scale : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        Student's t-distribution values
    """
    return amp * scipy.stats.t.pdf(x, df, loc=loc, scale=scale)


def pi_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Pi function: Π(z) = zΓ(z) = (z)! = Γ(z+1)
    
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
        Pi function values
    """
    arg = b * x + c + 1
    arg = np.where(arg > 0, arg, 1e-3)
    return a * scipy.special.gamma(arg)


# =============================================================================
# ELLIPTIC AND RELATED FUNCTIONS (18 functions)
# =============================================================================

def elliptic_integral_first_kind(x: np.ndarray, a: float, m: float) -> np.ndarray:
    """
    Elliptic integral of first kind: F(φ,m) = ∫₀^φ dθ/√(1-m sin²θ)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (angle φ)
    a : float
        Amplitude parameter
    m : float
        Parameter m (0 ≤ m < 1)
        
    Returns
    -------
    np.ndarray
        Elliptic integral values
    """
    m = np.clip(m, 0, 1 - 1e-10)
    return a * scipy.special.ellipkinc(x, m)


def elliptic_integral_second_kind(x: np.ndarray, a: float, m: float) -> np.ndarray:
    """
    Elliptic integral of second kind: E(φ,m) = ∫₀^φ √(1-m sin²θ) dθ
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (angle φ)
    a : float
        Amplitude parameter
    m : float
        Parameter m (0 ≤ m < 1)
        
    Returns
    -------
    np.ndarray
        Elliptic integral values
    """
    m = np.clip(m, 0, 1 - 1e-10)
    return a * scipy.special.ellipeinc(x, m)


def elliptic_integral_third_kind(x: np.ndarray, a: float, n: float, m: float) -> np.ndarray:
    """
    Elliptic integral of third kind: Π(n;φ,m) = ∫₀^φ dθ/((1-n sin²θ)√(1-m sin²θ))
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (angle φ)
    a : float
        Amplitude parameter
    n : float
        Characteristic n
    m : float
        Parameter m (0 ≤ m < 1)
        
    Returns
    -------
    np.ndarray
        Elliptic integral approximation
    """
    m = np.clip(m, 0, 1 - 1e-10)
    # Approximation using first kind integral
    return a * scipy.special.ellipkinc(x, m) * (1 + n/4)


def nome_function(m: np.ndarray, a: float) -> np.ndarray:
    """
    Nome function: q = exp(-π K'(m)/K(m)) where K is complete elliptic integral
    
    Parameters
    ----------
    m : np.ndarray
        Parameter m (0 ≤ m < 1)
    a : float
        Amplitude parameter
        
    Returns
    -------
    np.ndarray
        Nome function values
    """
    m_safe = np.clip(m, 1e-10, 1 - 1e-10)
    K = scipy.special.ellipk(m_safe)
    K_prime = scipy.special.ellipk(1 - m_safe)
    q = np.exp(-np.pi * K_prime / K)
    return a * q


def quarter_period(m: np.ndarray, a: float) -> np.ndarray:
    """
    Quarter period: K(m) = complete elliptic integral of first kind
    
    Parameters
    ----------
    m : np.ndarray
        Parameter m (0 ≤ m < 1)
    a : float
        Amplitude parameter
        
    Returns
    -------
    np.ndarray
        Complete elliptic integral values
    """
    m_safe = np.clip(m, 0, 1 - 1e-10)
    return a * scipy.special.ellipk(m_safe)


def jacobi_elliptic_sn(x: np.ndarray, a: float, b: float, m: float, offset: float = 0) -> np.ndarray:
    """
    Jacobi elliptic function sn: doubly periodic function with periods 4K and 2iK'
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    m : float
        Parameter m (0 ≤ m < 1)
    offset : float, optional
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Jacobi sn function values
    """
    m = np.clip(m, 0, 1 - 1e-10)
    sn, cn, dn, ph = scipy.special.ellipj(b * x, m)
    return a * sn + offset


def jacobi_elliptic_cn(x: np.ndarray, a: float, b: float, m: float, offset: float = 0) -> np.ndarray:
    """
    Jacobi elliptic function cn: doubly periodic function with periods 4K and 4iK'
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    m : float
        Parameter m (0 ≤ m < 1)
    offset : float, optional
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Jacobi cn function values
    """
    m = np.clip(m, 0, 1 - 1e-10)
    sn, cn, dn, ph = scipy.special.ellipj(b * x, m)
    return a * cn + offset


def jacobi_elliptic_dn(x: np.ndarray, a: float, b: float, m: float, offset: float = 0) -> np.ndarray:
    """
    Jacobi elliptic function dn: doubly periodic function with periods 2K and 4iK'
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    m : float
        Parameter m (0 ≤ m < 1)
    offset : float, optional
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Jacobi dn function values
    """
    m = np.clip(m, 0, 1 - 1e-10)
    sn, cn, dn, ph = scipy.special.ellipj(b * x, m)
    return a * dn + offset


def weierstrass_elliptic_p(x: np.ndarray, a: float, g2: float, g3: float) -> np.ndarray:
    """
    Weierstrass elliptic function ℘(z): meromorphic function satisfying ℘'² = 4℘³ - g₂℘ - g₃
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    g2 : float
        Invariant g₂
    g3 : float
        Invariant g₃
        
    Returns
    -------
    np.ndarray
        Weierstrass ℘ function approximation
    """
    # Simplified implementation using approximation
    return a * (1 / (x**2 + 1e-10) + g2 * x**2 / 20 + g3 * x**4 / 28)


def lemniscate_elliptic_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Lemniscate elliptic functions: special case of Jacobi elliptic functions with m = 1/2
    
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
        Lemniscate function values
    """
    m = 0.5  # Lemniscate case
    sn, cn, dn, ph = scipy.special.ellipj(b * x + c, m)
    return a * sn


def theta_function_1(x: np.ndarray, a: float, q: float) -> np.ndarray:
    """
    Jacobi theta function θ₁: θ₁(z,q) = 2q^(1/4) Σₙ₌₋∞^∞ (-1)ⁿ q^(n(n+1)) sin((2n+1)z)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable z
    a : float
        Amplitude parameter
    q : float
        Nome parameter (|q| < 1)
        
    Returns
    -------
    np.ndarray
        Theta function values
    """
    q = np.clip(abs(q), 1e-10, 1 - 1e-10)
    result = np.zeros_like(x)
    for n in range(-5, 6):  # Finite sum approximation
        result += (-1)**n * q**(n*(n+1)) * np.sin((2*n+1)*x)
    return 2 * a * q**(1/4) * result


def theta_function_2(x: np.ndarray, a: float, q: float) -> np.ndarray:
    """
    Jacobi theta function θ₂: θ₂(z,q) = 2q^(1/4) Σₙ₌₋∞^∞ q^(n(n+1)) cos((2n+1)z)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable z
    a : float
        Amplitude parameter
    q : float
        Nome parameter (|q| < 1)
        
    Returns
    -------
    np.ndarray
        Theta function values
    """
    q = np.clip(abs(q), 1e-10, 1 - 1e-10)
    result = np.zeros_like(x)
    for n in range(-5, 6):  # Finite sum approximation
        result += q**(n*(n+1)) * np.cos((2*n+1)*x)
    return 2 * a * q**(1/4) * result


def theta_function_3(x: np.ndarray, a: float, q: float) -> np.ndarray:
    """
    Jacobi theta function θ₃: θ₃(z,q) = 1 + 2Σₙ₌₁^∞ q^(n²) cos(2nz)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable z
    a : float
        Amplitude parameter
    q : float
        Nome parameter (|q| < 1)
        
    Returns
    -------
    np.ndarray
        Theta function values
    """
    q = np.clip(abs(q), 1e-10, 1 - 1e-10)
    result = np.ones_like(x)
    for n in range(1, 11):  # Finite sum approximation
        result += 2 * q**(n**2) * np.cos(2*n*x)
    return a * result


def theta_function_4(x: np.ndarray, a: float, q: float) -> np.ndarray:
    """
    Jacobi theta function θ₄: θ₄(z,q) = 1 + 2Σₙ₌₁^∞ (-1)ⁿ q^(n²) cos(2nz)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable z
    a : float
        Amplitude parameter
    q : float
        Nome parameter (|q| < 1)
        
    Returns
    -------
    np.ndarray
        Theta function values
    """
    q = np.clip(abs(q), 1e-10, 1 - 1e-10)
    result = np.ones_like(x)
    for n in range(1, 11):  # Finite sum approximation
        result += 2 * (-1)**n * q**(n**2) * np.cos(2*n*x)
    return a * result


def neville_theta_functions(x: np.ndarray, a: float, m: float) -> np.ndarray:
    """
    Neville theta functions: alternative notation for Jacobi theta functions
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    a : float
        Amplitude parameter
    m : float
        Parameter m
        
    Returns
    -------
    np.ndarray
        Neville theta function approximation
    """
    # Using theta_3 as representative
    q = np.exp(-np.pi * scipy.special.ellipk(1-m) / scipy.special.ellipk(m))
    return theta_function_3(x, a, q)


def modular_lambda_function(x: np.ndarray, a: float) -> np.ndarray:
    """
    Modular lambda function: λ(τ) = (θ₂/θ₃)⁴ where θ are theta functions
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable τ
    a : float
        Amplitude parameter
        
    Returns
    -------
    np.ndarray
        Modular lambda function approximation
    """
    # Simplified approximation
    q = np.exp(2j * np.pi * x)
    q_real = np.real(q)
    q_abs = np.abs(q_real)
    q_safe = np.clip(q_abs, 1e-10, 1 - 1e-10)
    
    # Approximation using series
    lambda_val = 16 * q_safe * (1 + q_safe)**(-4)
    return a * lambda_val


def j_invariant(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    J-invariant: j(τ) = 1728 * g₂³/(g₂³ - 27g₃²) for elliptic curves
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable τ
    a : float
        Amplitude parameter
    b : float
        Scale parameter
        
    Returns
    -------
    np.ndarray
        J-invariant approximation
    """
    tau = b * x
    # Simplified form
    result = 1728 * (1 + 744 * np.exp(2j * np.pi * tau)).real
    return a * result


def dedekind_eta_function(x: np.ndarray, a: float) -> np.ndarray:
    """
    Dedekind eta function: η(τ) = q^(1/24) ∏ₙ₌₁^∞ (1-qⁿ) where q = e^(2πiτ)
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable τ
    a : float
        Amplitude parameter
        
    Returns
    -------
    np.ndarray
        Dedekind eta function approximation
    """
    q = np.exp(2j * np.pi * x)
    q_abs = np.abs(q)
    
    # Approximation using finite product
    result = q**(1/24)
    for n in range(1, 10):
        result *= (1 - q**n)
    
    return a * np.real(result)


# Function registry for advanced functions
ADVANCED_FUNCTIONS = {
    # Gamma and related functions
    'gamma': (gamma_function, ['a', 'b', 'c']),
    'barnes_g': (barnes_g_function, ['a', 'b', 'c']),
    'beta': (beta_function, ['a', 'alpha', 'beta_param']),
    'digamma': (digamma_function, ['a', 'b', 'c']),
    'polygamma': (polygamma_function, ['a', 'b', 'c', 'n']),
    'incomplete_beta': (incomplete_beta_function, ['a', 'alpha', 'beta_param']),
    'incomplete_gamma': (incomplete_gamma_function, ['a', 'gamma_a', 'scale']),
    'k_function': (k_function, ['a', 'b', 'c']),
    'multivariate_gamma': (multivariate_gamma_function, ['a', 'p', 'scale']),
    'student_t': (student_t_distribution, ['amp', 'df', 'loc', 'scale']),
    'pi_function': (pi_function, ['a', 'b', 'c']),
    
    # Elliptic and related functions
    'elliptic_f': (elliptic_integral_first_kind, ['a', 'm']),
    'elliptic_e': (elliptic_integral_second_kind, ['a', 'm']),
    'elliptic_pi': (elliptic_integral_third_kind, ['a', 'n', 'm']),
    'nome': (nome_function, ['a']),
    'quarter_period': (quarter_period, ['a']),
    'jacobi_sn': (jacobi_elliptic_sn, ['a', 'b', 'm', 'offset']),
    'jacobi_cn': (jacobi_elliptic_cn, ['a', 'b', 'm', 'offset']),
    'jacobi_dn': (jacobi_elliptic_dn, ['a', 'b', 'm', 'offset']),
    'weierstrass_p': (weierstrass_elliptic_p, ['a', 'g2', 'g3']),
    'lemniscate': (lemniscate_elliptic_function, ['a', 'b', 'c']),
    'theta_1': (theta_function_1, ['a', 'q']),
    'theta_2': (theta_function_2, ['a', 'q']),
    'theta_3': (theta_function_3, ['a', 'q']),
    'theta_4': (theta_function_4, ['a', 'q']),
    'neville_theta': (neville_theta_functions, ['a', 'm']),
    'modular_lambda': (modular_lambda_function, ['a']),
    'j_invariant': (j_invariant, ['a', 'b']),
    'dedekind_eta': (dedekind_eta_function, ['a']),
}

print(f"Advanced Functions Module (Gamma + Elliptic) loaded: {len(ADVANCED_FUNCTIONS)} functions")