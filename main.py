import numpy as np
import numexpr as ne
import re
from typing import Tuple, List, Optional, Dict, Union, Any


def preprocess_equation(eqn: str) -> str:
    """
    Preprocess the equation string for evaluation.

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
    Evaluate the function at point x.

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
            "sqrt": np.sqrt
        }

        return ne.evaluate(processed_eqn, local_dict=local_dict)
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
    Evaluate the derivative of the function at point x using central difference.

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
        # Preprocess the equation once to avoid doing it twice
        processed_eqn = preprocess_equation(eqn)

        # Create a dictionary with all the functions and constants needed
        local_dict_plus = {
            "x": x + h,
            "e": np.e,
            "pi": np.pi,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "log": np.log,  # Natural logarithm
            "log10": np.log10,
            "exp": np.exp,
            "sqrt": np.sqrt
        }

        local_dict_minus = {
            "x": x - h,
            "e": np.e,
            "pi": np.pi,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "log": np.log,  # Natural logarithm
            "log10": np.log10,
            "exp": np.exp,
            "sqrt": np.sqrt
        }

        # Evaluate at x+h and x-h using the processed equation directly
        f_plus = ne.evaluate(processed_eqn, local_dict=local_dict_plus)
        f_minus = ne.evaluate(processed_eqn, local_dict=local_dict_minus)

        return (f_plus - f_minus) / (2 * h)
    except Exception as e:
        raise ValueError(f"Error evaluating derivative at x={x}: {str(e)}")


def incremental_search(
        eqn: str,
        x_start: float,
        initial_step: float,
        tolerance: float = 0.005,
        max_iterations: int = 200  # Increased to find more roots
) -> Tuple[List[float], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    Incremental search method that handles polynomial and transcendental equations.

    Args:
        eqn: The function equation as a string
        x_start: Initial x value (Xl)
        initial_step: Initial step size (Δx)
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations

    Returns:
        Tuple of (roots, table_data, points)

    Raises:
        ValueError: If the calculation fails
    """
    x = x_start
    step = initial_step
    points = []
    roots = []
    table_data = []

    if initial_step <= 0:
        raise ValueError("Step size must be positive")

    # Test if we can evaluate the equation at the starting point
    try:
        evaluate_function(x_start, eqn)
    except Exception as e:
        raise ValueError(f"Cannot evaluate function at starting point x={x_start}: {str(e)}")

    iteration = 0
    sign_changes_count = 0
    while iteration < max_iterations:
        iteration += 1
        x_next = x + step

        try:
            f_x = evaluate_function(x, eqn)
            f_x_next = evaluate_function(x_next, eqn)
        except Exception as e:
            # Skip this interval and continue
            x = x_next
            continue

        points.extend([(x, f_x), (x_next, f_x_next)])

        # Determine if root has been passed
        product = f_x * f_x_next
        product_display = "<0" if product < 0 else ">0"

        if abs(f_x) < tolerance:
            # Found a root at x
            if not roots or min(abs(x - root) for root in roots) > tolerance:
                roots.append(x)
                remark = "Root found"
                step = initial_step  # Reset step size after finding root
                x = x_next  # Move to next interval
        elif product < 0:
            # Root found in this interval
            sign_changes_count += 1
            remark = f"Sign change #{sign_changes_count} detected - reducing step"

            # Refine the root using bisection within this interval
            a, b = (x, x_next) if x < x_next else (x_next, x)
            root_found = False

            for _ in range(15):  # Increased refinement iterations for better accuracy
                mid = (a + b) / 2
                try:
                    f_mid = evaluate_function(mid, eqn)
                    f_a = evaluate_function(a, eqn)
                except Exception:
                    break

                points.append((mid, f_mid))

                if abs(f_mid) < tolerance:
                    # Only add if this is a new root
                    if not roots or min(abs(mid - root) for root in roots) > tolerance:
                        roots.append(mid)
                        root_found = True
                    break
                elif f_mid * f_a < 0:
                    # Root is in left half
                    b = mid
                else:
                    # Root is in right half
                    a = mid

            # Add the refined root if found by bisection
            if not root_found and b - a < tolerance * 10:
                mid = (a + b) / 2
                if not roots or min(abs(mid - root) for root in roots) > tolerance:
                    roots.append(mid)

            # Reduce step size for next iteration
            step = step / 2 if step > tolerance * 10 else step
        else:
            remark = "No sign change - continuing search"
            x = x_next  # Move to next interval

        # Add to table data
        table_data.append({
            "Iteration": str(iteration),
            "Xl": f"{x:.6f}",
            "Δx": f"{step:.6f}",
            "Xu": f"{x_next:.6f}",
            "f(Xl)": f"{f_x:.6f}",
            "f(Xu)": f"{f_x_next:.6f}",
            "Product": product_display,
            "Remark": remark
        })

        # Early termination if we've found sufficient roots (increased to 10)
        if len(roots) >= 10:
            break

    # If no roots found, add helpful message to table
    if not roots and table_data:
        table_data[-1]["Remark"] = "No roots found in the search range. Try different starting point or step size."

    return roots, table_data, points








# Bisection method
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
    Bisection method for finding roots of equations.
    Modified to find multiple roots and ignore sign change check for the first iteration.

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












# Bisection single root helper function
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
    Helper function for bisection_method that finds a single root in an interval.

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














#Newton-Raphson method
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
    Newton-Raphson method for finding roots of equations.
    Modified to optionally find multiple roots.

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





# Newton-Raphson
def _newton_raphson_single_root(
        eqn: str,
        x_initial: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5  # Error tolerance in percentage
) -> Tuple[Optional[float], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    Helper function for newton_raphson_method that finds a single root.

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
        return None, table_data, points

    # Add the final point for plotting
    if x_i not in [p[0] for p in points]:
        try:
            f_x = evaluate_function(x_i, eqn)
            points.append((x_i, f_x))
        except:
            pass

    return x_i, table_data, points






# FIXED: Secant method
def secant_method(
        eqn: str,
        x0: float,
        x1: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5,  # Error tolerance in percentage
        find_multiple_roots: bool = True,  # Added parameter to find multiple roots
        search_range: Tuple[float, float] = (-10, 10),  # Search range for multiple roots
        multiplicity_threshold: float = 1e-4  # Threshold for detecting duplicate roots
) -> Tuple[Union[Optional[float], List[float]], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    Secant method for finding roots of equations.
    Modified to handle duplicate roots and optionally find multiple roots.

    Args:
        eqn: The function equation as a string
        x0: First initial guess
        x1: Second initial guess
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage
        find_multiple_roots: If True, attempt to find all roots in the search range
        search_range: Tuple of (min, max) to search for multiple roots
        multiplicity_threshold: Threshold for detecting duplicate roots

    Returns:
        Tuple of (roots, table_data, points)

    Raises:
        ValueError: If the method fails or inputs are invalid
    """
    if x0 == x1:
        raise ValueError("Initial guesses must be different")

    # If we're just finding a single root, use the modified implementation
    if not find_multiple_roots:
        root, table_data, points = _secant_single_root(
            eqn, x0, x1, tolerance, max_iterations, error_tolerance, multiplicity_threshold
        )
        return root, table_data, points

    # For multiple roots, we'll try different starting pairs across the search range
    x_min, x_max = search_range
    num_pairs = 15  # Increased number of starting guess pairs to try
    interval_size = (x_max - x_min) / num_pairs

    roots = []
    all_table_data = []
    all_points = []

    # First try with the provided initial guesses
    root, table_data, points = _secant_single_root(
        eqn, x0, x1, tolerance, max_iterations // 2, error_tolerance, multiplicity_threshold
    )

    if root is not None:
        roots.append(root)
        all_table_data.extend(table_data)
        all_points.extend(points)

    # Then try different starting points across the search range
    for i in range(num_pairs):
        start_point = x_min + i * interval_size
        # Use two points close to each other as starting guesses
        x0_new = start_point
        x1_new = start_point + interval_size * 0.5

        # Skip if too close to already found roots
        if any(abs(start_point - r) < tolerance * 20 for r in roots):
            continue

        try:
            # Only use a few iterations to see if this starting point converges
            root, sub_table_data, sub_points = _secant_single_root(
                eqn, x0_new, x1_new, tolerance, max_iterations // num_pairs,
                error_tolerance, multiplicity_threshold
            )

            if root is not None:
                # Check if this is a new root (not too close to existing ones)
                # Use a larger threshold for duplicate roots
                is_new_root = not roots or min(abs(root - r) for r in roots) > tolerance * 20
                if is_new_root:
                    roots.append(root)

                    # Add pair information to table
                    for row in sub_table_data:
                        row["Remark"] = f"Starting pair {i + 1}: " + row["Remark"]
                        all_table_data.append(row)

                    all_points.extend(sub_points)
        except Exception:
            # Skip problematic starting points
            continue

    # If no roots found, return the original result with single root
    if not roots and not all_table_data:
        root, table_data, points = _secant_single_root(
            eqn, x0, x1, tolerance, max_iterations, error_tolerance, multiplicity_threshold
        )
        if root is not None:
            return [root], table_data, points
        return [], table_data, points

    # Sort roots for better presentation
    roots.sort()

    return roots, all_table_data, all_points


# FIXED: Secant single root helper function
def _secant_single_root(
        eqn: str,
        x0: float,
        x1: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5,  # Error tolerance in percentage
        multiplicity_threshold: float = 1e-4  # Threshold for detecting duplicate roots
) -> Tuple[Optional[float], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    Helper function for secant_method that finds a single root.
    Modified to better handle duplicate roots.

    Args:
        eqn: The function equation as a string
        x0: First initial guess
        x1: Second initial guess
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage
        multiplicity_threshold: Threshold for detecting duplicate roots

    Returns:
        Tuple of (root, table_data, points)
    """
    # Evaluate function at initial points
    try:
        f0 = evaluate_function(x0, eqn)
        f1 = evaluate_function(x1, eqn)
    except Exception as e:
        raise ValueError(f"Error evaluating function at initial points: {str(e)}")

    # Initialize variables
    points = [(x0, f0), (x1, f1)]
    table_data = []

    # Check if either initial point is already a root
    if abs(f0) < tolerance:
        table_data.append({
            "Iteration": "0",
            "Xi-1": f"{x0:.6f}",
            "Xi": f"{x1:.6f}",
            "Xi+1": "",
            "|Ea|,%": "",
            "f(Xi-1)": f"{f0:.6f}",
            "f(Xi)": f"{f1:.6f}",
            "f(Xi+1)": "",
            "Remark": "Initial value x0 is already a root"
        })
        return x0, table_data, points

    if abs(f1) < tolerance:
        table_data.append({
            "Iteration": "0",
            "Xi-1": f"{x0:.6f}",
            "Xi": f"{x1:.6f}",
            "Xi+1": "",
            "|Ea|,%": "",
            "f(Xi-1)": f"{f0:.6f}",
            "f(Xi)": f"{f1:.6f}",
            "f(Xi+1)": "",
            "Remark": "Initial value x1 is already a root"
        })
        return x1, table_data, points

    # Add initial points to table
    table_data.append({
        "Iteration": "0",
        "Xi-1": f"{x0:.6f}",
        "Xi": f"{x1:.6f}",
        "Xi+1": "",
        "|Ea|,%": "",
        "f(Xi-1)": f"{f0:.6f}",
        "f(Xi)": f"{f1:.6f}",
        "f(Xi+1)": "",
        "Remark": "Initial values"
    })

    # Add additional evaluation points for better visualization
    # This helps create a smoother curve
    x_range = abs(x1 - x0) * 5  # Extend range for better visualization
    x_min = min(x0, x1) - x_range
    x_max = max(x0, x1) + x_range

    # Add more points for visualization (but don't include in table data)
    num_extra_points = 20
    for x in np.linspace(x_min, x_max, num_extra_points):
        try:
            f_x = evaluate_function(x, eqn)
            # Only add if not too close to existing points
            if all(abs(x - p[0]) > 1e-6 for p in points):
                points.append((x, f_x))
        except Exception:
            continue

    # Variables to track potential duplicate roots
    consecutive_small_changes = 0
    potential_duplicate_root = False

    # Secant method iterations
    for iteration in range(1, max_iterations + 1):
        # FIXED: Check for division by zero or very small denominator
        if abs(f1 - f0) < multiplicity_threshold:
            # For potential duplicate roots, use a modified approach
            # Instead of giving up, we'll use a damped secant method
            potential_duplicate_root = True
            remark = "Very small denominator detected - using perturbation"

            # Use a damping factor to stabilize the iteration
            damping_factor = 0.1
            # Modified secant formula with damping
            x2 = x1 - damping_factor * f1 * (x1 - x0) / (f1 - f0 + multiplicity_threshold)
        else:
            # Standard secant formula
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            remark = "Continuing"

        try:
            f2 = evaluate_function(x2, eqn)
        except Exception as e:
            raise ValueError(f"Error at iteration {iteration}: {str(e)}")

        points.append((x2, f2))

        # FIXED: Calculate relative error
        if x2 != 0:
            error = abs((x2 - x1) / x2) * 100
            error_str = f"{error:.6f}"
        else:
            error = 0
            error_str = "0.000000"

        # Check for very small changes in x - indicator of duplicate root
        if abs(x2 - x1) < tolerance * 10:
            consecutive_small_changes += 1
            if consecutive_small_changes >= 3:
                potential_duplicate_root = True
                remark = "Consecutive small changes - possible duplicate root"
        else:
            consecutive_small_changes = 0

        # FIXED: Check for convergence using absolute value
        if abs(f2) < tolerance:
            if potential_duplicate_root:
                remark = "Converged to duplicate root (function value)"
            else:
                remark = "Converged (function value)"
        elif error < error_tolerance:
            if potential_duplicate_root:
                remark = "Converged to duplicate root (error tolerance)"
            else:
                remark = "Converged (error tolerance)"

        # FIXED: Limit step size to prevent large jumps
        max_step = 1.0
        if abs(x2 - x1) > max_step:
            x2 = x1 + max_step * (1 if x2 > x1 else -1)
            remark = "Step size limited to prevent divergence"
            # Recalculate function value at new x2
            f2 = evaluate_function(x2, eqn)

        # Add to table data
        table_data.append({
            "Iteration": str(iteration),
            "Xi-1": f"{x0:.6f}",
            "Xi": f"{x1:.6f}",
            "Xi+1": f"{x2:.6f}",
            "|Ea|,%": error_str,
            "f(Xi-1)": f"{f0:.6f}",
            "f(Xi)": f"{f1:.6f}",
            "f(Xi+1)": f"{f2:.6f}",
            "Remark": remark
        })

        # FIXED: Check termination criteria with absolute value
        if abs(f2) < tolerance or error < error_tolerance:
            break

        # Update for next iteration
        x0, f0 = x1, f1
        x1, f1 = x2, f2

    # If max iterations reached without convergence
    if iteration == max_iterations and abs(f2) >= tolerance and error >= error_tolerance:
        table_data[-1]["Remark"] = "Max iterations reached without convergence"
        return None, table_data, points

    return x1, table_data, points







#Graphical Method
def graphical_method(
        eqn: str,
        x_lower: float,
        x_upper: float,
        num_points: int = 1000,
        tolerance: float = 1e-5
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """
    Estimate roots graphically by evaluating the function at multiple points.
    Handles polynomial and transcendental equations.

    Args:
        eqn: The function equation as a string
        x_lower: Lower bound of the range
        x_upper: Upper bound of the range
        num_points: Number of points to evaluate
        tolerance: Tolerance for root finding

    Returns:
        Tuple of (roots, points)

    Raises:
        ValueError: If the method fails or inputs are invalid
    """
    if x_lower >= x_upper:
        raise ValueError("Lower bound must be less than upper bound")

    if num_points < 10:
        raise ValueError("Number of points must be at least 10")

    x_vals = np.linspace(x_lower, x_upper, num_points)
    y_vals = []
    roots = []
    points = []
    sign_changes = []
    potential_roots = []  # For roots without sign changes

    # Evaluate function at all points
    for x in x_vals:
        try:
            y = evaluate_function(x, eqn)
            y_vals.append(y)
            points.append((x, y))
        except Exception:
            # Skip points where evaluation fails
            y_vals.append(np.nan)
            continue

    # Clean up NaNs from points
    points = [(x, y) for x, y in points if not np.isnan(y)]

    # Find all sign change intervals
    for i in range(len(y_vals) - 1):
        if np.isnan(y_vals[i]) or np.isnan(y_vals[i + 1]):
            continue

        if y_vals[i] * y_vals[i + 1] <= 0:  # Sign change or zero crossing
            sign_changes.append((x_vals[i], x_vals[i + 1]))

    # Look for potential roots without sign changes (local minima of |f(x)|)
    for i in range(1, len(y_vals) - 1):
        if np.isnan(y_vals[i - 1]) or np.isnan(y_vals[i]) or np.isnan(y_vals[i + 1]):
            continue

        # Check if |f(x)| has a local minimum
        abs_prev = abs(y_vals[i - 1])
        abs_curr = abs(y_vals[i])
        abs_next = abs(y_vals[i + 1])

        if abs_curr < abs_prev and abs_curr < abs_next and abs_curr < tolerance * 10:
            potential_roots.append(x_vals[i])

    # Refine each root using bisection method
    for a, b in sign_changes:
        # Check if endpoints are actual roots
        try:
            f_a = evaluate_function(a, eqn)
            f_b = evaluate_function(b, eqn)

            if abs(f_a) < tolerance:
                # a is already a root
                if not roots or min(abs(a - root) for root in roots) > tolerance:
                    roots.append(a)
                continue

            if abs(f_b) < tolerance:
                # b is already a root
                if not roots or min(abs(b - root) for root in roots) > tolerance:
                    roots.append(b)
                continue
        except Exception:
            continue

        # Initial values
        x_left = a
        x_right = b

        try:
            f_left = evaluate_function(x_left, eqn)
            f_right = evaluate_function(x_right, eqn)

            # Skip if no sign change
            if f_left * f_right > 0:
                continue
        except Exception:
            continue

        # Bisection iterations
        for _ in range(50):  # Max refinements
            x_mid = (x_left + x_right) / 2
            try:
                f_mid = evaluate_function(x_mid, eqn)
            except Exception:
                break

            points.append((x_mid, f_mid))  # Track evaluation points

            if abs(f_mid) < tolerance:
                # Found a root
                if not roots or min(abs(x_mid - root) for root in roots) > tolerance:
                    roots.append(x_mid)
                break
            elif f_mid * f_left < 0:
                # Root is in left half
                x_right = x_mid
            else:
                # Root is in right half
                x_left = x_mid
                f_left = f_mid

    # Refine potential roots (those without sign changes)
    for x_pot in potential_roots:
        # Use a small interval around the potential root
        delta = (x_upper - x_lower) / num_points
        a, b = x_pot - delta, x_pot + delta

        # Use golden section search to find minimum of |f(x)|
        for _ in range(20):  # Limited refinement iterations
            # Golden ratio
            golden_ratio = (np.sqrt(5) - 1) / 2

            # Calculate intermediate points
            c = b - golden_ratio * (b - a)
            d = a + golden_ratio * (b - a)

            try:
                f_c = abs(evaluate_function(c, eqn))
                f_d = abs(evaluate_function(d, eqn))

                points.append((c, f_c * np.sign(evaluate_function(c, eqn))))
                points.append((d, f_d * np.sign(evaluate_function(d, eqn))))

                if f_c < f_d:
                    b = d
                else:
                    a = c

                if abs(b - a) < tolerance:
                    break
            except Exception:
                break

        # Check if we found a root
        x_min = (a + b) / 2
        try:
            f_min = evaluate_function(x_min, eqn)
            if abs(f_min) < tolerance:
                # Only add if this is a new root
                if not roots or min(abs(x_min - root) for root in roots) > tolerance:
                    roots.append(x_min)
        except Exception:
            continue

    return roots, points


def regula_falsi_method(
        eqn: str,
        x_lower: float,
        x_upper: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5,  # Error tolerance in percentage
        allow_no_sign_change: bool = False,  # Parameter to handle no sign change
        find_multiple_roots: bool = True,  # Added parameter to find multiple roots
        num_subintervals: int = 50,  # Increased number of subintervals for better root detection
        root_separation_threshold: float = 0.1  # Minimum distance between distinct roots
) -> Tuple[Union[Optional[float], List[float]], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    Regula Falsi method (False Position Method) for finding roots of equations.
    Enhanced to find multiple roots across the given interval.

    Args:
        eqn: The function equation as a string
        x_lower: Lower bound (XL)
        x_upper: Upper bound (XU)
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage
        allow_no_sign_change: If True, attempt to find roots even without sign change
        find_multiple_roots: If True, attempt to find all roots in the interval
        num_subintervals: Number of subintervals to check for multiple roots
        root_separation_threshold: Minimum distance between distinct roots

    Returns:
        Tuple of (roots, table_data, points)

    Raises:
        ValueError: If the method fails or inputs are invalid
    """
    if x_lower >= x_upper:
        raise ValueError("Lower bound must be less than upper bound")

    # If we're just finding a single root, use the original implementation
    if not find_multiple_roots:
        root, table_data, points = _regula_falsi_single_root(
            eqn, x_lower, x_upper, tolerance, max_iterations, error_tolerance, allow_no_sign_change
        )
        return root, table_data, points

    # For multiple roots, we'll divide the search range into subintervals
    subinterval_size = (x_upper - x_lower) / num_subintervals

    roots = []
    all_table_data = []
    all_points = []

    # First, try to find roots at the endpoints
    try:
        f_lower = evaluate_function(x_lower, eqn)
        f_upper = evaluate_function(x_upper, eqn)
        all_points.extend([(x_lower, f_lower), (x_upper, f_upper)])

        # Check if endpoints are roots
        if abs(f_lower) < tolerance:
            roots.append(x_lower)
        if abs(f_upper) < tolerance:
            roots.append(x_upper)
    except Exception:
        pass

    # Add more evaluation points for better visualization
    extra_points = np.linspace(x_lower, x_upper, 200)
    for x in extra_points:
        try:
            y = evaluate_function(x, eqn)
            all_points.append((x, y))
        except Exception:
            continue

    # Search for roots in each subinterval
    for i in range(num_subintervals):
        sub_lower = x_lower + i * subinterval_size
        sub_upper = sub_lower + subinterval_size

        try:
            f_sub_lower = evaluate_function(sub_lower, eqn)
            f_sub_upper = evaluate_function(sub_upper, eqn)

            # Check if there's a sign change in this subinterval
            sign_change = f_sub_lower * f_sub_upper <= 0

            # Also check if function values are close to zero
            near_zero_lower = abs(f_sub_lower) < tolerance * 10
            near_zero_upper = abs(f_sub_upper) < tolerance * 10

            if sign_change or near_zero_lower or near_zero_upper or allow_no_sign_change:
                # Apply regula falsi to this subinterval
                root, sub_table_data, sub_points = _regula_falsi_single_root(
                    eqn, sub_lower, sub_upper, tolerance, max_iterations // 2,
                    error_tolerance, True  # Allow no sign change for subintervals
                )

                if root is not None:
                    # Check if this is a new root (not too close to existing ones)
                    is_new_root = not roots or min(abs(root - r) for r in roots) > root_separation_threshold
                    if is_new_root:
                        roots.append(root)

                        # Add subinterval information to table
                        for row in sub_table_data:
                            row["Remark"] = f"Subinterval {i + 1}: " + row["Remark"]
                            all_table_data.append(row)

                        all_points.extend(sub_points)
        except Exception:
            # Skip problematic subintervals
            continue

    # If no roots found with subintervals, try the original interval
    if not roots:
        root, single_table_data, single_points = _regula_falsi_single_root(
            eqn, x_lower, x_upper, tolerance, max_iterations,
            error_tolerance, allow_no_sign_change
        )

        if root is not None:
            roots = [root]  # Return as a list for consistent output
            all_table_data = single_table_data
            all_points.extend(single_points)
        else:
            # No roots found
            return [], single_table_data, all_points

    # Sort roots for better presentation
    roots.sort()

    return roots, all_table_data, all_points








#Regula Falsi Method

def _regula_falsi_single_root(
        eqn: str,
        x_lower: float,
        x_upper: float,
        tolerance: float = 0.0001,
        max_iterations: int = 100,
        error_tolerance: float = 0.5,  # Error tolerance in percentage
        allow_no_sign_change: bool = False  # Parameter to handle no sign change
) -> Tuple[Optional[float], List[Dict[str, str]], List[Tuple[float, float]]]:
    """
    Helper function for regula_falsi_method that finds a single root.
    Implements the standard Regula Falsi algorithm with Illinois modification
    for faster convergence.

    Args:
        eqn: The function equation as a string
        x_lower: Lower bound (XL)
        x_upper: Upper bound (XU)
        tolerance: Tolerance for root finding
        max_iterations: Maximum number of iterations
        error_tolerance: Error tolerance in percentage
        allow_no_sign_change: If True, attempt to find roots even without sign change

    Returns:
        Tuple of (root, table_data, points)
    """
    # Check if the function changes sign in the interval
    try:
        f_lower = evaluate_function(x_lower, eqn)
        f_upper = evaluate_function(x_upper, eqn)
    except Exception as e:
        raise ValueError(f"Error evaluating function at interval bounds: {str(e)}")

    # Check for sign change
    sign_change = f_lower * f_upper < 0

    # Check if either endpoint is already a root
    if abs(f_lower) < tolerance:
        return x_lower, [], [(x_lower, f_lower)]
    if abs(f_upper) < tolerance:
        return x_upper, [], [(x_upper, f_upper)]

    # If no sign change and we're not allowing that case, try to find a minimum
    if not sign_change and not allow_no_sign_change:
        # If both function values have the same sign, look for a minimum of |f(x)|
        # This can help find roots where the function touches but doesn't cross the x-axis
        if abs(f_lower) < abs(f_upper):
            # If lower bound is closer to zero, start search from there
            x_min, f_min = x_lower, f_lower
        else:
            # If upper bound is closer to zero, start search from there
            x_min, f_min = x_upper, f_upper

        # Check if we're already close enough to a root
        if abs(f_min) < tolerance:
            return x_min, [], [(x_min, f_min)]

        # Otherwise, return None to indicate no root found
        error_msg = (f"No sign change in interval: f({x_lower}) = {f_lower:.6f}, f({x_upper}) = {f_upper:.6f}\n"
                     f"The Regula Falsi method requires f(a) and f(b) to have opposite signs.")
        raise ValueError(error_msg)

    # Initialize variables
    points = [(x_lower, f_lower), (x_upper, f_upper)]
    table_data = []
    x_r_old = None
    side_counter = 0  # Track which side is being modified repeatedly
    last_modified_side = None  # Track which side was last modified

    # Add initial information to table
    table_data.append({
        "Iteration": "0",
        "XL": f"{x_lower:.6f}",
        "XU": f"{x_upper:.6f}",
        "XR": "",
        "|Ea|,%": "",
        "f(XL)": f"{f_lower:.6f}",
        "f(XU)": f"{f_upper:.6f}",
        "f(XR)": "",
        "f(XL)*f(XR)": "",
        "Remark": "Initial interval" + (" (sign change detected)" if sign_change else " (no sign change)")
    })

    # Regula Falsi iterations
    for iteration in range(1, max_iterations + 1):
        # Calculate false position point using the formula from the algorithm
        # xR = (xU·f(xL) - xL·f(xU)) / (f(xL) - f(xU))
        x_r = (x_upper * f_lower - x_lower * f_upper) / (f_lower - f_upper)

        try:
            f_r = evaluate_function(x_r, eqn)
        except Exception as e:
            raise ValueError(f"Error at iteration {iteration}: {str(e)}")

        points.append((x_r, f_r))

        # Calculate relative error if possible
        rel_error = ""
        if x_r_old is not None and x_r != 0:
            error = abs((x_r - x_r_old) / x_r) * 100
            rel_error = f"{error:.6f}"

        # Check if we've found the root exactly
        if abs(f_r) < tolerance:
            table_data.append({
                "Iteration": str(iteration),
                "XL": f"{x_lower:.6f}",
                "XU": f"{x_upper:.6f}",
                "XR": f"{x_r:.6f}",
                "|Ea|,%": rel_error,
                "f(XL)": f"{f_lower:.6f}",
                "f(XU)": f"{f_upper:.6f}",
                "f(XR)": f"{f_r:.6f}",
                "f(XL)*f(XR)": "≈0",
                "Remark": "Root found (function value within tolerance)"
            })
            return x_r, table_data, points

        # Determine which subinterval contains the root
        product = f_lower * f_r
        product_display = "<0" if product < 0 else ">0"

        # Step 3a: If f(xL)·f(xR) < 0, root is between xL and xR
        if product < 0:
            # Update upper bound
            x_upper_old = x_upper
            x_upper = x_r
            f_upper = f_r
            remark = "Root in 1st subinterval (XL,XR)"

            # Check if we're repeatedly modifying the upper bound
            if last_modified_side == "upper":
                side_counter += 1
                if side_counter >= 2:
                    # Apply Illinois modification to accelerate convergence
                    f_lower = f_lower * 0.5
                    remark += " - Illinois modification applied"
            else:
                side_counter = 0

            last_modified_side = "upper"
        # Step 3b: If f(xL)·f(xR) > 0, root is between xR and xU
        else:
            # Update lower bound
            x_lower_old = x_lower
            x_lower = x_r
            f_lower = f_r
            remark = "Root in 2nd subinterval (XR,XU)"

            # Check if we're repeatedly modifying the lower bound
            if last_modified_side == "lower":
                side_counter += 1
                if side_counter >= 2:
                    # Apply Illinois modification to accelerate convergence
                    f_upper = f_upper * 0.5
                    remark += " - Illinois modification applied"
            else:
                side_counter = 0

            last_modified_side = "lower"

        # Add to table data
        table_data.append({
            "Iteration": str(iteration),
            "XL": f"{x_lower:.6f}",
            "XU": f"{x_upper:.6f}",
            "XR": f"{x_r:.6f}",
            "|Ea|,%": rel_error,
            "f(XL)": f"{f_lower:.6f}",
            "f(XU)": f"{f_upper:.6f}",
            "f(XR)": f"{f_r:.6f}",
            "f(XL)*f(XR)": product_display,
            "Remark": remark
        })

        # Check relative error if available
        if x_r_old is not None and x_r != 0:
            error = abs((x_r - x_r_old) / x_r) * 100
            if error < error_tolerance:
                table_data[-1]["Remark"] += " - Converged (error tolerance met)"
                return x_r, table_data, points

        # Check if interval is becoming too small
        if abs(x_upper - x_lower) < tolerance:
            table_data[-1]["Remark"] += " - Converged (interval size within tolerance)"
            return x_r, table_data, points

        # Update previous false position point
        x_r_old = x_r

    # If max iterations reached without convergence
    if iteration == max_iterations:
        table_data[-1]["Remark"] += " - Max iterations reached without convergence"
        # Return the best approximation we have
        return x_r, table_data, points

    return None, table_data, points