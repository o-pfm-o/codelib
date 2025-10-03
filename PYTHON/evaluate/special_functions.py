"""
Special Functions Module
========================

This module contains special mathematical functions including:
- Piecewise functions
- Arithmetic functions  
- Antiderivatives of elementary functions
- Error functions and integrals

Total Functions: 30
"""

import numpy as np
import scipy.special
import scipy.signal

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

print(f"Special Functions Module loaded: {len(SPECIAL_FUNCTIONS)} functions")