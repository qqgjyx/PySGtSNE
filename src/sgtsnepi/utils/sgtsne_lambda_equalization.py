import numpy as np
from scipy.sparse import csc_matrix
import warnings

from typing import Literal

from scipy.optimize import root_scalar


def sgtsne_lambda_equalization(
    D: csc_matrix,
    lambda_: float,
    max_iter: int = 50,
    tol_binary: float = 1e-5,
    algorithm: Literal[
        "custom_bisection",
        "bisection",
        "brentq",
        "brenth",
        "bisect",
        "ridder",
        "newton",
        "secant",
        "halley",
    ] = "custom_bisection",
) -> csc_matrix:
    """Binary search for the scales of column-wise conditional probabilities.

    Binary search for the scales of column-wise conditional probabilities
    from exp(-D) to exp(-D/σ²)/z equalized by λ.

    Parameters
    ----------
    D : scipy.sparse.csc_matrix
        N x N sparse matrix of "distance square"
        (column-wise conditional, local distances)
    lambda_ : float
        The equalization parameter
    max_iter : int, optional
        Maximum number of iterations for binary search, by default 50
    tol_binary : float, optional
        Tolerance for binary search convergence, by default 1e-5
    algorithm : str, optional
        The root finding algorithm to use, by default "custom_bisection"

    Returns
    -------
    scipy.sparse.csc_matrix
        The column-wise conditional probability matrix

    Notes
    -----
    .. versionadded:: 0.1.0

    Author
    ------
    Xiaobai Sun (MATLAB prototype on May 12, 2019)
    Dimitris Floros (translation to Julia)
    Juntang Wang (translation to Python on Nov 16, 2024)
    """

    #############################################################################
    #                          private helper functions                         #
    #############################################################################

    def colsum(D, j, sigma=1.0):
        """Helper function to compute column sum"""

        # minimum possible value (python float precision)
        D_min = np.finfo(float).tiny

        vals = D.data[D.indptr[j] : D.indptr[j + 1]]
        sum_j = np.sum(np.exp(-vals * sigma))
        return max(sum_j, D_min)

    def colupdate(D, j, sigma):
        """Helper function to update column values"""
        start, end = D.indptr[j], D.indptr[j + 1]
        D.data[start:end] = np.exp(-D.data[start:end] * sigma)

    #############################################################################
    #                 parameter setting & memory pre-allocations                #
    #############################################################################

    n = D.shape[0]
    cond_P = D.copy()

    i_diff = np.zeros(n)
    i_count = np.zeros(n)
    i_tval = np.zeros(n)
    sigma_sq = np.ones(n)

    #############################################################################
    #                       pre-calculate average entropy                       #
    #############################################################################

    for j in range(n):  # loop over all columns of D
        sum_j = colsum(D, j)
        i_tval[j] = sum_j - lambda_  # difference from λ

    #############################################################################
    #                        search for σ²                                      #
    #############################################################################

    if algorithm == "custom_bisection":
        for j in range(n):  # loop over all columns of D
            fval = i_tval[j]
            lb, ub = -1000.0, np.inf  # lower and upper bounds

            iter_count = 0

            while abs(fval) > tol_binary and iter_count < max_iter:
                iter_count += 1

                if fval > 0:  # update lower bound
                    lb = sigma_sq[j]
                    sigma_sq[j] = 2 * lb if np.isinf(ub) else 0.5 * (lb + ub)
                else:  # update upper bound
                    ub = sigma_sq[j]
                    sigma_sq[j] = 0.5 * ub if np.isinf(lb) else 0.5 * (lb + ub)

                # Re-calculate local entropy
                sum_j = colsum(D, j, sigma_sq[j])
                fval = sum_j - lambda_

            # Post-recording
            i_diff[j] = fval
            i_count[j] = iter_count
            colupdate(cond_P, j, sigma_sq[j])

    else:  # Use any scipy.optimize root finding method
        for j in range(n):
            # Define the objective function
            def objective(x):
                return colsum(D, j, x) - lambda_

            try:
                # For methods that require brackets
                if algorithm in ["brentq", "brenth", "bisect", "ridder"]:
                    result = root_scalar(
                        objective,
                        method=algorithm,
                        bracket=[-1000.0, np.inf],
                        xtol=tol_binary,
                        maxiter=max_iter,
                        full_output=True,
                    )
                # For methods that require initial guess
                elif algorithm in ["newton", "secant", "halley"]:
                    result = root_scalar(
                        objective,
                        method=algorithm,
                        x0=1.0,
                        xtol=tol_binary,
                        maxiter=max_iter,
                        full_output=True,
                    )
                else:
                    raise ValueError(f"Unsupported root finding method: {algorithm}")

                sigma_sq[j] = result.root
                i_diff[j] = objective(sigma_sq[j])
                i_count[j] = result.iterations
                colupdate(cond_P, j, sigma_sq[j])

            except (ValueError, RuntimeError) as e:
                # If root finding fails, use the initial value
                msg = (
                    f"Failed for column {j} with {algorithm} method: {str(e)}"
                )
                warnings.warn(msg)
                sigma_sq[j] = 1.0
                i_diff[j] = objective(sigma_sq[j])
                i_count[j] = 0
                colupdate(cond_P, j, sigma_sq[j])

    #############################################################################
    #                      display post-information to user                     #
    #############################################################################

    avg_iter = np.ceil(np.sum(i_count) / n)
    nc_idx = np.sum(np.abs(i_diff) > tol_binary)

    if nc_idx == 0:
        print(f"✅ All {n} elements converged numerically, avg(#iter) = {avg_iter}")
    else:
        warnings.warn(f"There are {nc_idx} non-convergent elements out of {n}")

    n_neg = np.sum(sigma_sq < 0)
    if n_neg > 0:
        warnings.warn(
            f"There are {n_neg} nodes with negative γᵢ; consider decreasing λ"
        )

    return cond_P
