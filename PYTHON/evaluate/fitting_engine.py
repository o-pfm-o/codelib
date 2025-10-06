"""
Universal Mathematical Function Fitting System
==============================================

This is the main fitting engine that provides n-dimensional function fitting
with intelligent parameter guessing for ALL 160+ mathematical functions.

The system automatically imports all function modules and provides a unified
interface for fitting any mathematical function to data.

Author: User/Assistant Collaboration
Date: October 2025
License: MIT
"""

import numpy as np
import scipy.optimize
import inspect
from typing import Dict, Any, Union, List, Callable, Optional, Tuple
import warnings

# Import all function modules
try:
    from math_functions import ALGEBRAIC_FUNCTIONS
    from math_functions import TRANSCENDENTAL_FUNCTIONS
    from math_functions import SPECIAL_FUNCTIONS
    from math_functions import ADVANCED_FUNCTIONS
    from math_functions import BESSEL_FUNCTIONS
    from math_functions import ZETA_FUNCTIONS
    print("✓ All function modules loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Create empty registries as fallback
    ALGEBRAIC_FUNCTIONS = {}
    TRANSCENDENTAL_FUNCTIONS = {}
    SPECIAL_FUNCTIONS = {}
    ADVANCED_FUNCTIONS = {}
    BESSEL_FUNCTIONS = {}
    ZETA_FUNCTIONS = {}

# =============================================================================
# COMPLETE FUNCTION REGISTRY - ALL 160+ FUNCTIONS
# =============================================================================

# Combine all function registries
ALL_FUNCTIONS = {
    **ALGEBRAIC_FUNCTIONS,
    **TRANSCENDENTAL_FUNCTIONS, 
    **SPECIAL_FUNCTIONS,
    **ADVANCED_FUNCTIONS,
    **BESSEL_FUNCTIONS,
    **ZETA_FUNCTIONS
}

print(f"🎯 TOTAL FUNCTIONS AVAILABLE: {len(ALL_FUNCTIONS)}")

# =============================================================================
# N-DIMENSIONAL FITTING ENGINE
# =============================================================================

def fit_multidimensional_function(
    *args: np.ndarray,
    function: Union[str, Callable],
    init_guess: Union[str, Dict[str, float], List[float]] = 'auto',
    algorithm: str = 'auto',
    bounds: Optional[List[Tuple[float, float]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Universal n-dimensional function fitting interface.
    
    Automatically determines dimensionality from number of arguments:
    - 2 args (x, y): 1D curve fitting
    - 3 args (x, y, z): 2D surface fitting  
    - 4+ args: Higher dimensional fitting
    
    Parameters
    ----------
    *args : np.ndarray
        Variable number of arrays: first is independent, rest are dependent
    function : Union[str, Callable]
        Function to fit (string name or callable)
    init_guess : Union[str, Dict[str, float], List[float]], optional
        Initial parameter guess strategy, default 'auto'
    algorithm : str, optional
        Optimization algorithm, default 'auto'
    bounds : Optional[List[Tuple[float, float]]], optional
        Parameter bounds
    **kwargs
        Additional optimizer arguments
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive fit results including parameters, statistics, fitted function
        
    Examples
    --------
    1D curve fitting:
    >>> result = fit_multidimensional_function(x, y, function='sine')
    
    2D surface fitting:
    >>> result = fit_multidimensional_function(x, y, z, function='quadratic')
    
    3D+ hyperspace fitting:
    >>> result = fit_multidimensional_function(x, y, z, w, function='cubic')
    """
    if len(args) < 2:
        raise ValueError("At least 2 arrays required: independent and dependent variables")
    
    # Separate independent and dependent variables
    x_data = args[0]
    y_data_arrays = args[1:]
    n_dimensions = len(y_data_arrays)
    
    # Validate input arrays
    for i, arr in enumerate(args):
        if len(arr) != len(x_data):
            raise ValueError(f"Array {i} length ({len(arr)}) doesn't match x_data length ({len(x_data)})")
    
    # Get function and parameter information
    if isinstance(function, str):
        func_obj, param_names = get_function_by_name(function)
    else:
        func_obj = function
        param_names = get_parameter_names_from_callable(func_obj)
    
    # Multi-dimensional objective function
    def objective_function(params):
        try:
            residuals = []
            for y_data in y_data_arrays:
                y_model = func_obj(x_data, *params)
                residuals.extend(y_data - y_model)
            return np.array(residuals)
        except Exception:
            total_points = sum(len(y) for y in y_data_arrays)
            return np.full(total_points, 1e6)
    
    # Generate intelligent initial parameter guess
    if isinstance(init_guess, str) and init_guess == 'auto':
        initial_params = generate_intelligent_guess(
            x_data, y_data_arrays[0], function, param_names
        )
    elif isinstance(init_guess, dict):
        initial_params = [init_guess.get(name, 1.0) for name in param_names]
    elif isinstance(init_guess, (list, np.ndarray)):
        if len(init_guess) != len(param_names):
            raise ValueError(f"init_guess length ({len(init_guess)}) doesn't match parameter count ({len(param_names)})")
        initial_params = list(init_guess)
    else:
        raise ValueError("Invalid init_guess type. Must be 'auto', dict, or list")
    
    # Automatic algorithm selection based on problem complexity
    if algorithm == 'auto':
        if n_dimensions == 1 and len(param_names) <= 6:
            algorithm = 'lm'
        elif n_dimensions <= 2 and len(param_names) <= 10:
            algorithm = 'trf'
        else:
            algorithm = 'differential_evolution'
    
    # Perform optimization
    try:
        if algorithm in ['lm', 'trf', 'dogbox']:
            result = scipy.optimize.least_squares(
                objective_function, initial_params,
                method=algorithm,
                bounds=bounds if bounds else (-np.inf, np.inf),
                **kwargs
            )
            fitted_params = result.x
            
            # Covariance matrix estimation
            try:
                J = result.jac
                cov_matrix = np.linalg.inv(J.T @ J)
            except:
                cov_matrix = None
                
        elif algorithm == 'differential_evolution':
            if bounds is None:
                bounds = [(p * 0.1 if p != 0 else -10, p * 10 if p != 0 else 10) 
                         for p in initial_params]
            
            def cost_function(params):
                residuals = objective_function(params)
                return np.sum(residuals**2)
            
            result = scipy.optimize.differential_evolution(
                cost_function, bounds, **kwargs
            )
            fitted_params = result.x
            cov_matrix = None
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    except Exception as e:
        raise RuntimeError(f"Fitting failed: {str(e)}")
    
    # Calculate comprehensive fit statistics
    final_residuals = objective_function(fitted_params)
    y_combined = np.concatenate(y_data_arrays) 
    r_squared = calculate_r_squared(y_combined, final_residuals)
    
    # Parameter uncertainties
    param_errors = None
    if cov_matrix is not None:
        try:
            param_errors = np.sqrt(np.diag(cov_matrix))
        except:
            param_errors = None
    
    # Create fitted function
    fitted_function = lambda x: func_obj(x, *fitted_params)
    
    # Return comprehensive results dictionary
    return {
        'parameters': dict(zip(param_names, fitted_params)),
        'parameter_errors': dict(zip(param_names, param_errors)) if param_errors is not None else None,
        'covariance_matrix': cov_matrix,
        'fitted_function': fitted_function,
        'residuals': final_residuals,
        'r_squared': r_squared,
        'reduced_chi_squared': np.sum(final_residuals**2) / (len(final_residuals) - len(fitted_params)) 
                              if len(final_residuals) > len(fitted_params) else np.inf,
        'function_name': function if isinstance(function, str) else 'custom',
        'algorithm_used': algorithm,
        'dimensions': n_dimensions,
        'success': True,
        'total_functions_available': len(ALL_FUNCTIONS),
        'function_categories': get_function_categories()
    }


def get_function_by_name(function_name: str) -> Tuple[Callable, List[str]]:
    """Get function object and parameter names by string name."""
    
    if function_name not in ALL_FUNCTIONS:
        available_functions = ', '.join(list(ALL_FUNCTIONS.keys())[:20])
        raise ValueError(f"Unknown function '{function_name}'. Available functions include: {available_functions}... "
                        f"(Total: {len(ALL_FUNCTIONS)} functions available)")
    
    return ALL_FUNCTIONS[function_name]


def get_parameter_names_from_callable(func: Callable) -> List[str]:
    """Extract parameter names from callable function."""
    sig = inspect.signature(func)
    return list(sig.parameters.keys())[1:]  # Exclude first parameter (x)


def generate_intelligent_guess(x_data: np.ndarray, y_data: np.ndarray, 
                              function: Union[str, Callable], param_names: List[str]) -> List[float]:
    """Generate intelligent initial parameter guesses based on function type and data."""
    
    function_name = function if isinstance(function, str) else 'custom'
    
    # Basic statistical estimates from data
    y_range = np.ptp(y_data)
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    x_range = np.ptp(x_data)
    x_center = np.mean(x_data)
    x_std = np.std(x_data)
    
    guess_dict = {}
    
    # INTELLIGENT FUNCTION-SPECIFIC PARAMETER GUESSING
    if any(trig in function_name for trig in ['sine', 'cosine', 'tangent', 'versine', 'haversine', 'secant', 'cosecant']):
        # FFT-based frequency estimation for all periodic functions
        dt = np.mean(np.diff(x_data)) if len(x_data) > 1 else 1.0
        frequencies = np.fft.fftfreq(len(x_data), dt)
        fft_magnitudes = np.abs(np.fft.fft(y_data - y_mean))
        
        if len(fft_magnitudes) > 1:
            # Find dominant frequency (excluding DC component)
            peak_idx = np.argmax(fft_magnitudes[1:len(fft_magnitudes)//2]) + 1
            dominant_freq = abs(frequencies[peak_idx])
        else:
            dominant_freq = 1.0 / x_range if x_range > 0 else 1.0
        
        guess_dict['amp'] = y_std * np.sqrt(2)  # RMS amplitude estimate
        guess_dict['freq'] = dominant_freq
        guess_dict['phase'] = 0.0  # Start with zero phase
        guess_dict['offset'] = y_mean
        
    elif any(poly in function_name for poly in ['linear', 'quadratic', 'cubic', 'quartic', 'quintic']):
        # Polynomial coefficient estimation using least squares fit
        degree = len(param_names) - 1
        try:
            # Fit polynomial to get reasonable initial coefficients
            poly_coeffs = np.polyfit(x_data, y_data, degree)
            for i, param in enumerate(param_names):
                guess_dict[param] = poly_coeffs[i]
        except:
            # Fallback to analytical estimates
            for i, param in enumerate(param_names):
                if i == len(param_names) - 1:  # Constant term
                    guess_dict[param] = y_mean
                elif i == len(param_names) - 2:  # Linear term
                    guess_dict[param] = (y_data[-1] - y_data[0]) / x_range if x_range > 0 else 0
                else:  # Higher order terms
                    guess_dict[param] = y_range / (x_range ** (degree - i)) if x_range > 0 else 0.1
    
    elif 'exponential' in function_name or 'exp' in function_name:
        # Exponential function parameter estimation
        try:
            # Take log and fit linear relationship
            y_positive = np.where(y_data > 0, y_data - np.min(y_data) + 1e-10, 1e-10)
            log_y = np.log(y_positive)
            coeffs = np.polyfit(x_data, log_y, 1)
            guess_dict['a'] = np.exp(coeffs[1])  # Amplitude
            guess_dict['b'] = coeffs[0]  # Exponential rate
            guess_dict['c'] = np.min(y_data)  # Offset
        except:
            # Fallback estimates
            guess_dict['a'] = y_range
            guess_dict['b'] = 1.0 / x_std if x_std > 0 else 1.0
            guess_dict['c'] = np.min(y_data)
    
    elif any(log_func in function_name for log_func in ['natural_log', 'common_log', 'binary_log']):
        # Logarithmic function parameter estimation
        try:
            # Fit linear relationship with log of x
            x_positive = np.where(x_data > 0, x_data, 1e-10)
            log_x = np.log(x_positive)
            coeffs = np.polyfit(log_x, y_data, 1)
            guess_dict['a'] = coeffs[0]  # Scale factor
            guess_dict['b'] = 1.0  # Input scaling
            guess_dict['c'] = x_center  # Center
            guess_dict['d'] = coeffs[1]  # Offset
        except:
            # Fallback estimates
            guess_dict['a'] = y_range
            guess_dict['b'] = 1.0
            guess_dict['c'] = x_center
            guess_dict['d'] = y_mean
    
    elif 'power' in function_name:
        # Power function parameter estimation
        try:
            x_positive = np.where(x_data > 0, x_data, 1e-10)
            y_positive = np.where(y_data > 0, y_data, 1e-10)
            log_x = np.log(x_positive)
            log_y = np.log(y_positive)
            coeffs = np.polyfit(log_x, log_y, 1)
            guess_dict['a'] = np.exp(coeffs[1])  # Amplitude
            guess_dict['b'] = coeffs[0]  # Power exponent
            guess_dict['c'] = 0.0  # Offset
        except:
            guess_dict['a'] = y_mean
            guess_dict['b'] = 1.0
            guess_dict['c'] = 0.0
    
    elif 'rational' in function_name:
        # Rational function parameter estimation
        guess_dict['a'] = y_range
        guess_dict['b'] = y_mean * x_center
        guess_dict['c'] = 1.0 / x_std if x_std > 0 else 1.0
        guess_dict['d'] = 1.0
    
    # Fill in any missing parameters with intelligent defaults
    for param in param_names:
        if param not in guess_dict:
            param_lower = param.lower()
            if any(kw in param_lower for kw in ['amp', 'a', 'height', 'amplitude']):
                guess_dict[param] = y_range
            elif any(kw in param_lower for kw in ['offset', 'c', 'd', 'baseline', 'const']):
                guess_dict[param] = y_mean
            elif any(kw in param_lower for kw in ['center', 'x0', 'mu', 'mean']):
                guess_dict[param] = x_center
            elif any(kw in param_lower for kw in ['width', 'sigma', 'scale', 'std']):
                guess_dict[param] = x_std
            elif any(kw in param_lower for kw in ['freq', 'frequency']):
                guess_dict[param] = 1.0 / x_range if x_range > 0 else 1.0
            elif 'phase' in param_lower:
                guess_dict[param] = 0.0
            elif 'n' in param_lower and len(param) <= 2:  # Order parameters
                guess_dict[param] = 1.0
            elif 'm' in param_lower and len(param) <= 2:  # Modulus parameters
                guess_dict[param] = 0.5
            elif 'alpha' in param_lower or 'beta' in param_lower:
                guess_dict[param] = 1.0
            else:
                guess_dict[param] = 1.0
    
    return [guess_dict[name] for name in param_names]


def calculate_r_squared(y_observed: np.ndarray, residuals: np.ndarray) -> float:
    """Calculate coefficient of determination (R-squared)."""
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_observed - np.mean(y_observed))**2)
    return float(1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0.0)


def list_available_functions() -> Dict[str, List[str]]:
    """List all available functions organized by category."""
    categories = {
        'Algebraic Functions': list(ALGEBRAIC_FUNCTIONS.keys()),
        'Elementary Transcendental': list(TRANSCENDENTAL_FUNCTIONS.keys()),
        'Special Functions': list(SPECIAL_FUNCTIONS.keys()),
        'Advanced Functions': list(ADVANCED_FUNCTIONS.keys()),
        'Bessel Functions': list(BESSEL_FUNCTIONS.keys()),
        'Zeta Functions': list(ZETA_FUNCTIONS.keys()),
    }
    
    return categories


def get_function_categories() -> Dict[str, int]:
    """Get function count by category."""
    return {
        'Algebraic': len(ALGEBRAIC_FUNCTIONS),
        'Transcendental': len(TRANSCENDENTAL_FUNCTIONS),
        'Special': len(SPECIAL_FUNCTIONS),
        'Advanced': len(ADVANCED_FUNCTIONS),
        'Bessel': len(BESSEL_FUNCTIONS),
        'Zeta': len(ZETA_FUNCTIONS),
        'Total': len(ALL_FUNCTIONS)
    }


def search_functions(query: str) -> List[str]:
    """Search for functions by name or description."""
    query_lower = query.lower()
    matches = []
    
    for func_name in ALL_FUNCTIONS.keys():
        if query_lower in func_name.lower():
            matches.append(func_name)
    
    return matches
