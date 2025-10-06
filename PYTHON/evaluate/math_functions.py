"""
Mathematical functions module
_____________________________________________________________________________

Advanced Functions
=========================

Contains advanced mathematical functions including:
- Gamma and related functions
- Elliptic functions
- Bessel functions and related functions
- Riemann zeta and related functions

Total Functions: 78
"""

import numpy as np
import scipy.special
import scipy.stats
import scipy.signal

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


"""
Algebraic Functions
=========================

Contains all algebraic functions including polynomials, 
rational functions, and root functions.

Total Functions: 11
"""

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

"""
Bessel Functions 
=======================

Contains all Bessel functions and related orthogonal functions including:
- Airy functions
- Bessel functions (all kinds and orders)
- Modified Bessel functions
- Kelvin functions
- Legendre functions
- Orthogonal polynomials
- Spherical functions

Total Functions: 32
"""

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

"""
Origin Functions
=======================

Contains functions commonly used in Origin software including:
- Origin Basic Functions (26 functions)
- Exponential Functions (5 functions) 
- Growth/Sigmoidal Functions (4 functions)
- Peak Functions (4 functions)
- Chromatography Functions (5 functions)
- Specialized Application Functions (8 functions)

Total Functions: 52
"""

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


def exponential_origin(x: np.ndarray, y0: float, A: float, R0: float) -> np.ndarray:
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
    'exponential_origin': (exponential_origin, ['y0', 'A', 'R0']),
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

"""
Special Functions
========================

Contains special mathematical functions including:
- Piecewise functions
- Arithmetic functions  
- Antiderivatives of elementary functions
- Error functions and integrals

Total Functions: 30
"""

# =============================================================================
# PIECEWISE SPECIAL FUNCTIONS (11 functions)
# =============================================================================

def indicator_function(x: np.ndarray, x_min: float, x_max: float, amp: float, offset: float = 0) -> np.ndarray:
    """
    Indicator function: f(x) = amp if x_min ≤ x ≤ x_max, else offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    x_min : float
        Lower bound of indicator interval
    x_max : float
        Upper bound of indicator interval
    amp : float
        Amplitude inside interval
    offset : float, optional
        Value outside interval (default: 0)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    indicator = np.where((x >= x_min) & (x <= x_max), 1, 0)
    return amp * indicator + offset


def step_function(x: np.ndarray, x0: float, amp: float, offset: float = 0) -> np.ndarray:
    """
    Step function: f(x) = amp * (x ≥ x0) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    x0 : float
        Step position
    amp : float
        Step amplitude
    offset : float, optional
        Baseline value (default: 0)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    step = np.where(x >= x0, 1, 0)
    return amp * step + offset


def heaviside_step_function(x: np.ndarray, x0: float, amp: float, offset: float = 0) -> np.ndarray:
    """
    Heaviside step function: f(x) = amp * H(x - x0) + offset
    
    The Heaviside function is 0 for x < x0, 0.5 for x = x0, and 1 for x > x0
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    x0 : float
        Step position
    amp : float
        Step amplitude
    offset : float, optional
        Baseline value (default: 0)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    step = np.where(x < x0, 0, np.where(x == x0, 0.5, 1))
    return amp * step + offset


def sawtooth_wave(x: np.ndarray, amp: float, period: float, phase: float, offset: float) -> np.ndarray:
    """
    Sawtooth wave: f(x) = amp * sawtooth(2π*(x + phase)/period) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    period : float
        Period of the wave
    phase : float
        Phase shift
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * (x + phase) / period
    return amp * scipy.signal.sawtooth(arg) + offset


def square_wave(x: np.ndarray, amp: float, period: float, phase: float, offset: float, duty: float = 0.5) -> np.ndarray:
    """
    Square wave: f(x) = amp * square(2π*(x + phase)/period, duty) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    period : float
        Period of the wave
    phase : float
        Phase shift
    offset : float
        DC offset
    duty : float, optional
        Duty cycle (0 < duty < 1, default: 0.5)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * (x + phase) / period
    duty = np.clip(duty, 0.01, 0.99)
    return amp * scipy.signal.square(arg, duty=duty) + offset


def triangle_wave(x: np.ndarray, amp: float, period: float, phase: float, offset: float) -> np.ndarray:
    """
    Triangle wave: f(x) = amp * triangle(2π*(x + phase)/period) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    amp : float
        Amplitude
    period : float
        Period of the wave
    phase : float
        Phase shift
    offset : float
        DC offset
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = 2 * np.pi * (x + phase) / period
    return amp * scipy.signal.sawtooth(arg, width=0.5) + offset # type: ignore


def rectangular_function(x: np.ndarray, x0: float, width: float, amp: float, offset: float = 0) -> np.ndarray:
    """
    Rectangular function: f(x) = amp * rect((x - x0) / width) + offset
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    x0 : float
        Center position
    width : float
        Width of rectangle
    amp : float
        Amplitude
    offset : float, optional
        Baseline value (default: 0)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    half_width = width / 2
    rect = np.where(np.abs(x - x0) <= half_width, 1, 0)
    return amp * rect + offset


def floor_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Floor function: f(x) = a * floor(b*(x - c))
    
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
    arg = b * (x - c)
    return a * np.floor(arg)


def ceiling_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Ceiling function: f(x) = a * ceil(b*(x - c))
    
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
    arg = b * (x - c)
    return a * np.ceil(arg)


def sign_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Sign function: f(x) = a * sign(b*(x - c))
    
    Returns -1, 0, or +1 depending on sign of argument
    
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
    arg = b * (x - c)
    return a * np.sign(arg)


def absolute_value_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Absolute value function: f(x) = a * |b*(x - c)| + d
    
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
    d : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Function values
    """
    arg = b * (x - c)
    return a * np.abs(arg) + d


# =============================================================================
# ARITHMETIC FUNCTIONS (9 functions)
# =============================================================================

def sigma_function(n: np.ndarray, k: float = 1, amp: float = 1) -> np.ndarray:
    """
    Sigma function (divisor function): σ_k(n) = sum of k-th powers of divisors of n
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    k : float, optional
        Power exponent (default: 1)
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    result = np.zeros_like(n)
    n_int = np.round(n).astype(int)
    
    for i, val in enumerate(n_int):
        if val <= 0:
            result[i] = 0
        else:
            divisors = [d for d in range(1, int(val) + 1) if val % d == 0]
            result[i] = sum(d**k for d in divisors)
    
    return amp * result


def euler_totient_function(n: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Euler's totient function: φ(n) = number of integers ≤ n coprime to n
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Function values
    """
    result = np.zeros_like(n)
    n_int = np.round(n).astype(int)
    
    for i, val in enumerate(n_int):
        if val <= 0:
            result[i] = 0
        elif val == 1:
            result[i] = 1
        else:
            count = 0
            for k in range(1, int(val) + 1):
                if np.gcd(k, int(val)) == 1:
                    count += 1
            result[i] = count
    
    return amp * result


def prime_counting_function(x: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Prime counting function: π(x) ≈ x/ln(x) (asymptotic approximation)
    
    Parameters
    ----------
    x : np.ndarray
        Positive real numbers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Approximate number of primes ≤ x
    """
    x_safe = np.where(x > 1, x, 2)
    return amp * x_safe / np.log(x_safe)


def partition_function(n: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Partition function: p(n) ≈ exp(π√(2n/3))/(4n√3) (Hardy-Ramanujan approximation)
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Approximate number of partitions
    """
    n_safe = np.where(n > 0, n, 1)
    return amp * np.exp(np.pi * np.sqrt(2 * n_safe / 3)) / (4 * n_safe * np.sqrt(3))


def mobius_function(n: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Möbius μ function: μ(n) = (-1)^k if n is product of k distinct primes, 0 if square factor
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Möbius function values
    """
    result = np.zeros_like(n)
    n_int = np.round(n).astype(int)
    
    for i, val in enumerate(n_int):
        if val <= 0:
            result[i] = 0
        elif val == 1:
            result[i] = 1
        else:
            factors = 0
            temp = val
            for p in range(2, int(np.sqrt(val)) + 1):
                if temp % (p * p) == 0:  # Square factor
                    result[i] = 0
                    break
                if temp % p == 0:
                    factors += 1
                    temp //= p
            else:
                if temp > 1:
                    factors += 1
                result[i] = (-1) ** factors
    
    return amp * result


def liouville_function(n: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Liouville function: λ(n) = (-1)^Ω(n) where Ω(n) is number of prime factors with multiplicity
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Liouville function values
    """
    result = np.zeros_like(n)
    n_int = np.round(n).astype(int)
    
    for i, val in enumerate(n_int):
        if val <= 0:
            result[i] = 0
        else:
            omega = 0  # Count prime factors with multiplicity
            temp = val
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    omega += 1
                    temp //= d
                d += 1
            if temp > 1:
                omega += 1
            result[i] = (-1) ** omega
    
    return amp * result


def von_mangoldt_function(n: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Von Mangoldt function: Λ(n) = log p if n = p^k for prime p, 0 otherwise
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Von Mangoldt function values
    """
    result = np.zeros_like(n)
    n_int = np.round(n).astype(int)
    
    for i, val in enumerate(n_int):
        if val <= 1:
            result[i] = 0
        else:
            # Check if val is a prime power
            for p in range(2, int(val) + 1):
                power = 1
                while p ** power < val:
                    power += 1
                if p ** power == val:
                    # Check if p is prime
                    is_prime = True
                    for d in range(2, int(np.sqrt(p)) + 1):
                        if p % d == 0:
                            is_prime = False
                            break
                    if is_prime:
                        result[i] = np.log(p)
                        break
    
    return amp * result


def carmichael_function(n: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Carmichael function: λ(n) ≈ n/log(log(n+1)+1) (approximation for fitting)
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Carmichael function approximation
    """
    n_safe = np.where(n > 0, n, 1)
    return amp * n_safe / np.log(np.log(n_safe + 1) + 1)


def radical_function(n: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    Radical function: rad(n) = product of distinct prime factors of n
    
    Parameters
    ----------
    n : np.ndarray
        Positive integers
    amp : float, optional
        Amplitude parameter (default: 1)
        
    Returns
    -------
    np.ndarray
        Radical function values
    """
    result = np.zeros_like(n)
    n_int = np.round(n).astype(int)
    
    for i, val in enumerate(n_int):
        if val <= 0:
            result[i] = 0
        elif val == 1:
            result[i] = 1
        else:
            radical = 1
            temp = val
            for p in range(2, int(val) + 1):
                if temp % p == 0:
                    radical *= p
                    while temp % p == 0:
                        temp //= p
                if temp == 1:
                    break
            result[i] = radical
    
    return amp * result


# =============================================================================
# ANTIDERIVATIVES OF ELEMENTARY FUNCTIONS (10 functions)
# =============================================================================

def logarithmic_integral(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Logarithmic integral: li(x) = ∫(dt/ln(t)) from 2 to x
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (x > 2)
    a : float
        Amplitude parameter
    b : float
        Scale parameter
    c : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Logarithmic integral values
    """
    x_safe = np.where(x > 2, x, 2.001)
    return a * scipy.special.expi(np.log(x_safe * b)) + c


def exponential_integral(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential integral: Ei(x) = ∫(e^t/t)dt
    
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
        Exponential integral values
    """
    return a * scipy.special.expi(b * x) + c


def sine_integral(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Sine integral: Si(x) = ∫(sin(t)/t)dt from 0 to x
    
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
        Sine integral values
    """
    return a * scipy.special.shichi(b * x)[0] + c


def cosine_integral(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Cosine integral: Ci(x) = -∫(cos(t)/t)dt from x to ∞
    
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
        Cosine integral values
    """
    x_safe = np.where(x > 0, x, 1e-10)
    return a * scipy.special.shichi(b * x_safe)[1] + c


def inverse_tangent_integral(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Inverse tangent integral: Ti(x) = ∫(arctan(t)/t)dt (series approximation)
    
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
        Inverse tangent integral approximation
    """
    x_scaled = b * x
    result = np.zeros_like(x_scaled)
    for n in range(1, 10):  # First 10 terms
        result += (-1)**(n-1) * x_scaled**(2*n-1) / ((2*n-1)**2)
    return a * result + c


def error_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Error function: erf(x) = (2/√π) ∫e^(-t²)dt from 0 to x
    
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
        Error function values
    """
    return a * scipy.special.erf(b * (x - c)) + d


def fresnel_integral_s(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Fresnel integral S: S(x) = ∫sin(πt²/2)dt from 0 to x
    
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
        Fresnel S integral values
    """
    fresnel_s, _ = scipy.special.fresnel(b * x)
    return a * fresnel_s + c


def fresnel_integral_c(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Fresnel integral C: C(x) = ∫cos(πt²/2)dt from 0 to x
    
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
        Fresnel C integral values
    """
    _, fresnel_c = scipy.special.fresnel(b * x)
    return a * fresnel_c + c


def dawson_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Dawson function: D(x) = e^(-x²) ∫e^(t²)dt from 0 to x
    
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
        Dawson function values
    """
    return a * scipy.special.dawsn(b * x) + c


def faddeeva_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Faddeeva function: w(z) = e^(-z²)erfc(-iz)
    
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
        Faddeeva function values (real part)
    """
    return a * np.real(scipy.special.wofz(b * x)) + c


# Function registry for special functions
SPECIAL_FUNCTIONS = {
    # Piecewise functions
    'indicator': (indicator_function, ['x_min', 'x_max', 'amp', 'offset']),
    'step': (step_function, ['x0', 'amp', 'offset']),
    'heaviside': (heaviside_step_function, ['x0', 'amp', 'offset']),
    'sawtooth': (sawtooth_wave, ['amp', 'period', 'phase', 'offset']),
    'square_wave': (square_wave, ['amp', 'period', 'phase', 'offset', 'duty']),
    'triangle_wave': (triangle_wave, ['amp', 'period', 'phase', 'offset']),
    'rectangular': (rectangular_function, ['x0', 'width', 'amp', 'offset']),
    'floor': (floor_function, ['a', 'b', 'c']),
    'ceiling': (ceiling_function, ['a', 'b', 'c']),
    'sign': (sign_function, ['a', 'b', 'c']),
    'absolute': (absolute_value_function, ['a', 'b', 'c', 'd']),
    
    # Arithmetic functions
    'sigma': (sigma_function, ['k', 'amp']),
    'euler_totient': (euler_totient_function, ['amp']),
    'prime_counting': (prime_counting_function, ['amp']),
    'partition': (partition_function, ['amp']),
    'mobius': (mobius_function, ['amp']),
    'liouville': (liouville_function, ['amp']),
    'von_mangoldt': (von_mangoldt_function, ['amp']),
    'carmichael': (carmichael_function, ['amp']),
    'radical': (radical_function, ['amp']),
    
    # Antiderivatives
    'logarithmic_integral': (logarithmic_integral, ['a', 'b', 'c']),
    'exponential_integral': (exponential_integral, ['a', 'b', 'c']),
    'sine_integral': (sine_integral, ['a', 'b', 'c']),
    'cosine_integral': (cosine_integral, ['a', 'b', 'c']),
    'inverse_tangent_integral': (inverse_tangent_integral, ['a', 'b', 'c']),
    'error_function': (error_function, ['a', 'b', 'c', 'd']),
    'fresnel_s': (fresnel_integral_s, ['a', 'b', 'c']),
    'fresnel_c': (fresnel_integral_c, ['a', 'b', 'c']),
    'dawson': (dawson_function, ['a', 'b', 'c']),
    'faddeeva': (faddeeva_function, ['a', 'b', 'c']),
}

"""
Elementary Transcendental Functions
==========================================

Contains all elementary transcendental functions including:
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

"""
Zeta Functions
====================

Contains Riemann zeta and related L-functions including:
- Riemann zeta function
- Dirichlet functions
- Hurwitz zeta function
- Polylogarithms
- Fermi-Dirac integrals
- Related transcendental functions

Total Functions: 16
"""

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