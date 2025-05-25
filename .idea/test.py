"""
# Numerical Root-Finding Methods: Comprehensive Implementation
# ==========================================================
#
# This file contains a comprehensive implementation of various numerical
# root-finding methods with detailed explanations in comments.
#
# The code is structured into three main components:
# 1. Core Functions: Equation processing and evaluation
# 2. Numerical Methods: Implementation of various root-finding algorithms
# 3. Visualization: Code for plotting and displaying tabular results
"""

import numpy as np
import re
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from typing import Tuple, List, Optional, Dict, Union, Any


# ==========================================================
# PART 1: CORE FUNCTIONS - EQUATION PROCESSING AND EVALUATION
# ==========================================================

def preprocess_equation(eqn: str) -> str:
    """
    # Equation Preprocessing
    # ---------------------
    # This function converts user-friendly equation format to Python-compatible format.
    # It handles:
    # - Replacing ^ with ** for exponentiation
    # - Handling mathematical functions (sin, cos, etc.)
    # - Processing constants like pi and e
    # - Performing syntax validation
    #
    # Example: "x^2 + sin(x)" becomes "x**2 + math.sin(x)"

    Args:
        eqn: The equation as a string

    Returns:
        Processed equation ready for evaluation

    Raises:
        ValueError: If the equation is invalid or contains unsupported functions
    """
    if not eqn:
        raise ValueError("Equation cannot be empty")

    # Check for common mistakes and provide helpful error messages
    if "=" in eqn:
        raise ValueError(
            "Please provide equation in the form f(x) without equals sign. For example, 'x^2 - 4' instead of 'x^2 = 4'")

    # First, handle special cases with negative exponents
    # Replace patterns like e^-x with e^(-x)
    eqn = re.sub(r'(\^|\*\*)(-[a-zA-Z0-9\.]+)', r'\1(\2)', eqn)

    # Check for unsupported functions before replacement
    raw_funcs = re.findall(r'([a-zA-Z]+)\(', eqn)
    supported_raw = {'ln', 'log', 'exp', 'sqrt', 'sin', 'cos', 'tan'}
    for func in raw_funcs:
        if func not in supported_raw and func != 'e':  # 'e' might be part of scientific notation
            raise ValueError(
                f"Unsupported function: {func}(). Supported functions are: ln(), log(), exp(), sqrt(), sin(), cos(), tan()")

    # Standard replacements - but don't add np. prefix
    eqn = eqn.replace('^', '**')
    eqn = eqn.replace("ln(", "log(")  # Use log for natural logarithm
    # Keep other function names as is - we'll provide them in the local_dict

    # Handle implicit multiplication
    eqn = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', eqn)
    eqn = re.sub(r'([a-zA-Z\)])(\d)', r'\1*\2', eqn)
    eqn = re.sub(r'(\))(\()', r'\1*\2', eqn)

    # Replace standalone 'e' with Euler's number, but preserve scientific notation
    eqn = re.sub(r'(?<!\d)e(?![\+\-]?\d)', "e", eqn)  # Just use 'e', we'll provide it in local_dict

    # Replace pi with pi (we'll provide it in local_dict)
    eqn = eqn.replace("pi", "pi")

    # Check for basic syntax errors
    try:
        # Just compile to check syntax, don't execute
        compile(eqn, '<string>', 'eval')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in equation: {str(e)}")

    return eqn


def evaluate_function(x: float, eqn: str) -> Optional[float]:
    """
    # Function Evaluation
    # -----------------
    # This function evaluates the mathematical expression at a specific x value.
    # It uses the preprocessed equation and provides detailed error handling.
    #
    # The function creates a dictionary with all necessary mathematical functions
    # and constants, then evaluates the expression using Python's eval().

    Args:
        x: Point at which to evaluate the function
        eqn: The equation as a string

    Returns:
        Function value or None if evaluation fails

    Raises:
        ValueError: If the equation evaluation fails with a specific error
    """
    try:
        processed_eqn = preprocess_equation(eqn)

        # Create a dictionary with all the functions and constants needed
        local_dict = {
            "x": x,
            "e": np.e,
            "pi": np.pi,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "log": np.log,  # Natural logarithm
            "log10": np.log10,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "math": math  # Include math module for additional functions
        }

        # Evaluate the expression using eval()
        return eval(processed_eqn, {"__builtins__": {}}, local_dict)
    except Exception as e:
        # Provide more detailed error information
        if "division by zero" in str(e).lower():
            raise ValueError(f"Division by zero error at x={x}. The function has a vertical asymptote at this point.")
        elif "domain error" in str(e).lower():
            raise ValueError(
                f"Domain error at x={x}. Check if you're taking logarithm of negative numbers or square root of negative numbers.")
        elif "overflow" in str(e).lower():
            raise ValueError(f"Overflow error at x={x}. The function value is too large to compute.")
        else:
            raise ValueError(f"Error evaluating function at x={x}: {str(e)}")


def evaluate_derivative(x: float, eqn: str, h: float = 1e-6) -> Optional[float]:
    """
    # Numerical Differentiation
    # -----------------------
    # This function calculates the derivative of the function at a specific point
    # using the central difference method.
    #
    # The central difference formula is: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    # This provides a more accurate approximation than the forward or backward
    # difference methods.

    Args:
        x: Point at which to evaluate the derivative
        eqn: Function equation as a string
        h: Step size for numerical differentiation

    Returns:
        Derivative value or None if evaluation fails

    Raises:
        ValueError: If derivative calculation fails
    """
    try:
        # Evaluate at x+h and x-h
        f_plus = evaluate_function(x + h, eqn)
        f_minus = evaluate_function(x - h, eqn)

        # Calculate derivative using central difference
        return (f_plus - f_minus) / (2 * h)
    except Exception as e:
        raise ValueError(f"Error evaluating derivative at x={x}: {str(e)}")


# ==========================================================
# PART 2: NUMERICAL METHODS FOR ROOT FINDING
# ==========================================================

def bisection_method(
        eqn: str,
        x_lower: float,
        x_upper: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5,  # Error tolerance in percentage
        find_multiple_roots: bool = True  # Always find multiple roots
) -> Tuple[List[float], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    # Bisection Method
    # --------------
    # The Bisection method is based on the Intermediate Value Theorem:
    # If f(a) and f(b) have opposite signs, and f is continuous, then there exists
    # at least one root in the interval [a, b].
    #
    # Algorithm steps:
    # 1. Start with interval [a, b] where f(a) and f(b) have opposite signs
    # 2. Calculate midpoint c = (a + b) / 2
    # 3. If f(c) is close enough to zero, c is the root
    # 4. Otherwise, determine which subinterval contains the root:
    #    - If f(a) * f(c) < 0, the root is in [a, c]
    #    - If f(a) * f(c) > 0, the root is in [c, b]
    # 5. Update the interval and repeat
    #
    # Advantages:
    # - Always converges if initial conditions are met
    # - Simple to implement
    # - Reliable
    #
    # Disadvantages:
    # - Slow convergence (linear)
    # - Requires initial interval with sign change

    Args:
        eqn: The function equation as a string
        x_lower: Lower bound (Xl)
        x_upper: Upper bound (Xu)
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage
        find_multiple_roots: If True, attempt to find all roots in the interval

    Returns:
        Tuple of (roots, table_data, points)

    Raises:
        ValueError: If the method fails or inputs are invalid
    """
    if x_lower >= x_upper:
        raise ValueError("Lower bound must be less than upper bound")

    # Check if the function changes sign in the interval
    try:
        f_lower = evaluate_function(x_lower, eqn)
        f_upper = evaluate_function(x_upper, eqn)
    except Exception as e:
        raise ValueError(f"Error evaluating function at interval bounds: {str(e)}")

    points = [(x_lower, f_lower), (x_upper, f_upper)]
    table_data = []
    roots = []

    # Check for sign change
    sign_change = f_lower * f_upper < 0

    # Add initial information to table
    table_data.append({
        "Iteration": "0",
        "Xl": f"{x_lower:.6f}",
        "Xr": "",
        "Xu": f"{x_upper:.6f}",
        "f(Xl)": f"{f_lower:.6f}",
        "f(Xr)": "",
        "|Ea|,%": "",
        "f(Xl).f(Xr)": "<0" if sign_change else ">0",
        "Remark": "Initial interval" + (" (sign change detected)" if sign_change else " (no sign change)")
    })

    # If we're finding multiple roots, we'll divide the interval into subintervals
    if find_multiple_roots:
        # Divide the interval into subintervals to search for multiple roots
        num_subintervals = 10  # Number of subintervals to check
        subinterval_size = (x_upper - x_lower) / num_subintervals

        for i in range(num_subintervals):
            sub_lower = x_lower + i * subinterval_size
            sub_upper = sub_lower + subinterval_size

            try:
                f_sub_lower = evaluate_function(sub_lower, eqn)
                f_sub_upper = evaluate_function(sub_upper, eqn)

                # Check if there's a sign change in this subinterval
                if f_sub_lower * f_sub_upper <= 0 or abs(f_sub_lower) < tolerance or abs(f_sub_upper) < tolerance:
                    # Apply bisection to this subinterval
                    root, sub_table_data, sub_points = _bisection_single_root(
                        eqn, sub_lower, sub_upper, tolerance, max_iterations // num_subintervals,
                        error_tolerance, True  # Always ignore sign change check
                    )

                    if root is not None:
                        # Check if this is a new root (not too close to existing ones)
                        is_new_root = not roots or min(abs(root - r) for r in roots) > tolerance * 10
                        if is_new_root:
                            roots.append(root)

                            # Add subinterval information to table
                            for row in sub_table_data:
                                row["Remark"] = f"Subinterval {i + 1}: " + row["Remark"]
                                table_data.append(row)

                            points.extend(sub_points)
            except Exception:
                # Skip problematic subintervals
                continue

        # If no roots found with subintervals, try the original interval
        if not roots:
            root, single_table_data, single_points = _bisection_single_root(
                eqn, x_lower, x_upper, tolerance, max_iterations,
                error_tolerance, True  # Always ignore sign change check
            )

            if root is not None:
                roots.append(root)
                table_data.extend(single_table_data)
                points.extend(single_points)
    else:
        # Just find a single root in the original interval
        root, single_table_data, single_points = _bisection_single_root(
            eqn, x_lower, x_upper, tolerance, max_iterations,
            error_tolerance, True  # Always ignore sign change check
        )

        if root is not None:
            roots.append(root)
            table_data.extend(single_table_data)
            points.extend(single_points)

    return roots, table_data, points


def _bisection_single_root(
        eqn: str,
        x_lower: float,
        x_upper: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5,  # Error tolerance in percentage
        ignore_sign_change: bool = False  # Ignore sign change check
) -> Tuple[Optional[float], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    # Bisection Method - Single Root Helper
    # -----------------------------------
    # This helper function implements the bisection algorithm for a single root
    # in the given interval. It's used by the main bisection_method function
    # to find individual roots in subintervals.

    Args:
        eqn: The function equation as a string
        x_lower: Lower bound (Xl)
        x_upper: Upper bound (Xu)
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage
        ignore_sign_change: If True, ignore the sign change check

    Returns:
        Tuple of (root, table_data, points)
    """
    try:
        f_lower = evaluate_function(x_lower, eqn)
        f_upper = evaluate_function(x_upper, eqn)
    except Exception as e:
        raise ValueError(f"Error evaluating function at interval bounds: {str(e)}")

    points = [(x_lower, f_lower), (x_upper, f_upper)]
    table_data = []
    x_r_old = None

    # Check for sign change
    sign_change = f_lower * f_upper < 0

    # If no sign change and we're not ignoring it, return None
    if not sign_change and not ignore_sign_change:
        return None, [], points

    # Check if either endpoint is already a root
    if abs(f_lower) < tolerance:
        return x_lower, [], points
    if abs(f_upper) < tolerance:
        return x_upper, [], points

    # Bisection iterations
    for iteration in range(1, max_iterations + 1):
        # Calculate midpoint
        x_r = (x_lower + x_upper) / 2
        try:
            f_r = evaluate_function(x_r, eqn)
        except Exception as e:
            raise ValueError(f"Error at iteration {iteration}: {str(e)}")

        points.append((x_r, f_r))

        # Calculate relative error if possible
        rel_error = ""
        if x_r_old is not None and x_r != 0:
            error = abs((x_r - x_r_old) / x_r) * 100
            rel_error = f"{error:.10f}"

        # Determine which subinterval contains the root
        product = f_lower * f_r
        product_display = "<0" if product < 0 else ">0"

        # FIXED: Proper interval selection based on sign change
        if product < 0:
            remark = "1st subinterval"
            x_upper = x_r
            f_upper = f_r
        else:
            remark = "2nd subinterval"
            x_lower = x_r
            f_lower = f_r

        # Add to table data
        table_data.append({
            "Iteration": str(iteration),
            "Xl": f"{x_lower:.6f}",
            "Xr": f"{x_r:.6f}",
            "Xu": f"{x_upper:.6f}",
            "f(Xl)": f"{f_lower:.6f}",
            "f(Xr)": f"{f_r:.6f}",
            "|Ea|,%": rel_error,
            "f(Xl).f(Xr)": product_display,
            "Remark": remark
        })

        # FIXED: Check termination criteria - use absolute value for function
        if abs(f_r) < tolerance:
            # Function value is close enough to zero
            table_data[-1]["Remark"] += " - Converged (function value)"
            return x_r, table_data, points

        # Check relative error if available
        if x_r_old is not None:
            error = abs((x_r - x_r_old) / x_r) * 100
            if error < error_tolerance:
                table_data[-1]["Remark"] += " - Converged (error tolerance)"
                return x_r, table_data, points

        # Update previous midpoint
        x_r_old = x_r

    # If max iterations reached without convergence
    if iteration == max_iterations:
        table_data[-1]["Remark"] += " - Max iterations reached without convergence"
        # Return the best approximation we have
        return x_r, table_data, points

    return None, table_data, points


def newton_raphson_method(
        eqn: str,
        x_initial: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5,  # Error tolerance in percentage
        find_multiple_roots: bool = True,  # Added parameter to find multiple roots
        search_range: Tuple[float, float] = (-10, 10)  # Search range for multiple roots
) -> Tuple[Union[Optional[float], List[float]], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    # Newton-Raphson Method
    # -------------------
    # The Newton-Raphson method uses derivative information to find roots:
    # - Based on linear approximation of the function at each iteration
    #
    # Algorithm steps:
    # 1. Start with initial guess x₀
    # 2. Calculate f(x₀) and f'(x₀)
    # 3. Compute next approximation: x₁ = x₀ - f(x₀)/f'(x₀)
    # 4. Repeat until convergence
    #
    # Advantages:
    # - Fast convergence (quadratic) when close to root
    # - Requires only one initial guess
    #
    # Disadvantages:
    # - May diverge if initial guess is poor
    # - Requires derivative calculation
    # - Can fail if derivative is zero or very small

    Args:
        eqn: The function equation as a string
        x_initial: Initial guess for the root
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage
        find_multiple_roots: If True, attempt to find all roots in the search range
        search_range: Tuple of (min, max) to search for multiple roots

    Returns:
        Tuple of (roots, table_data, points)

    Raises:
        ValueError: If the method fails or inputs are invalid
    """
    # If we're just finding a single root, use the original implementation
    if not find_multiple_roots:
        root, table_data, points = _newton_raphson_single_root(
            eqn, x_initial, tolerance, max_iterations, error_tolerance
        )
        return root, table_data, points

    # For multiple roots, we'll divide the search range into subintervals
    x_min, x_max = search_range
    num_subintervals = 10  # Number of subintervals to check
    subinterval_size = (x_max - x_min) / num_subintervals

    roots = []
    all_table_data = []
    all_points = []

    # First try with the provided initial guess
    root, table_data, points = _newton_raphson_single_root(
        eqn, x_initial, tolerance, max_iterations // 2, error_tolerance
    )

    if root is not None:
        roots.append(root)
        all_table_data.extend(table_data)
        all_points.extend(points)

    # Then try different starting points across the search range
    for i in range(num_subintervals):
        x_start = x_min + i * subinterval_size + subinterval_size / 2  # Use midpoint of subinterval

        # Skip if too close to already found roots
        if any(abs(x_start - r) < tolerance * 10 for r in roots):
            continue

        try:
            # Only use a few iterations to see if this starting point converges
            root, sub_table_data, sub_points = _newton_raphson_single_root(
                eqn, x_start, tolerance, max_iterations // num_subintervals, error_tolerance
            )

            if root is not None:
                # Check if this is a new root (not too close to existing ones)
                is_new_root = not roots or min(abs(root - r) for r in roots) > tolerance * 10
                if is_new_root:
                    roots.append(root)

                    # Add subinterval information to table
                    for row in sub_table_data:
                        row["Remark"] = f"Starting point {i + 1}: " + row["Remark"]
                        all_table_data.append(row)

                    all_points.extend(sub_points)
        except Exception:
            # Skip problematic starting points
            continue

    # If no roots found, return the original result with single root
    if not roots and not all_table_data:
        root, table_data, points = _newton_raphson_single_root(
            eqn, x_initial, tolerance, max_iterations, error_tolerance
        )
        if root is not None:
            return [root], table_data, points
        return [], table_data, points

    return roots, all_table_data, all_points


def _newton_raphson_single_root(
        eqn: str,
        x_initial: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5  # Error tolerance in percentage
) -> Tuple[Optional[float], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    # Newton-Raphson Method - Single Root Helper
    # ---------------------------------------
    # This helper function implements the Newton-Raphson algorithm for a single root
    # starting from the given initial guess. It's used by the main newton_raphson_method
    # function to find individual roots from different starting points.

    Args:
        eqn: The function equation as a string
        x_initial: Initial guess for the root
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage

    Returns:
        Tuple of (root, table_data, points)
    """
    x_i = x_initial
    points = []
    table_data = []

    # Test if we can evaluate the equation at the initial point
    try:
        f_x = evaluate_function(x_i, eqn)
        f_prime_x = evaluate_derivative(x_i, eqn)
    except Exception as e:
        raise ValueError(f"Cannot initialize Newton-Raphson method: {str(e)}")

    points.append((x_i, f_x))

    for iteration in range(1, max_iterations + 1):
        try:
            f_x = evaluate_function(x_i, eqn)
            f_prime_x = evaluate_derivative(x_i, eqn)
        except Exception as e:
            raise ValueError(f"Error at iteration {iteration}: {str(e)}")

        # FIXED: Check for division by zero or very small derivative
        if abs(f_prime_x) < 1e-10:
            # Add small perturbation to avoid division by zero
            f_prime_x = 1e-10 if f_prime_x >= 0 else -1e-10
            remark = "Derivative near zero - using perturbation"
        else:
            remark = "Continuing"

        # Calculate next approximation
        x_next = x_i - f_x / f_prime_x

        # FIXED: Limit step size to prevent large jumps
        max_step = 1.0
        if abs(x_next - x_i) > max_step:
            x_next = x_i + max_step * (1 if x_next > x_i else -1)
            remark = "Step size limited to prevent divergence"

        # Calculate relative error
        if x_next != 0:
            error = abs((x_next - x_i) / x_next) * 100
            error_str = f"{error:.6f}"
        else:
            error = 0
            error_str = "0.000000"

        # FIXED: Check for convergence using absolute value
        if abs(f_x) < tolerance:
            remark = "Converged (function value)"
        elif error < error_tolerance:
            remark = "Converged (error tolerance)"

        # Check for divergence or oscillation
        if abs(x_next) > 1e6 or (iteration > 3 and abs(x_next - x_i) > 1e3):
            remark = "Possible divergence detected"
            table_data.append({
                "Iteration": str(iteration),
                "Xi": f"{x_i:.6f}",
                "|Ea|,%": error_str,
                "f(Xi)": f"{f_x:.6f}",
                "f'(Xi)": f"{f_prime_x:.6f}",
                "Remark": remark
            })
            return None, table_data, points

        # Add to table data
        table_data.append({
            "Iteration": str(iteration),
            "Xi": f"{x_i:.6f}",
            "|Ea|,%": error_str,
            "f(Xi)": f"{f_x:.6f}",
            "f'(Xi)": f"{f_prime_x:.6f}",
            "Remark": remark
        })

        # FIXED: Termination check with absolute value
        if abs(f_x) < tolerance or error < error_tolerance:
            break

        # Update for next iteration
        x_i = x_next
        points.append((x_i, f_x))

    # If max iterations reached without convergence
    if iteration == max_iterations and abs(f_x) >= tolerance and error >= error_tolerance:
        table_data[-1]["Remark"] = "Max iterations reached without convergence"
        return None, table_