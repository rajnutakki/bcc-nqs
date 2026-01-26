import numpy as np
from typing import Sequence


def vscore(
    E: float | np.ndarray, varE: float | np.ndarray, N: int | np.ndarray
) -> float | np.ndarray:
    """
    Compute the V-score V = NVarE/(E^2) (assuming E_inf = 0)
    """
    return N * varE / (E**2)


def rel_err(
    E: float | np.ndarray, E_ref: float | np.ndarray, abs_numerator=True
) -> float | np.ndarray:
    """
    Compute the relative error between E and E_ref
    """
    if abs_numerator:
        return np.abs(E - E_ref) / np.abs(E_ref)
    else:
        return (E - E_ref) / np.abs(E_ref)


def to_array(data: list):
    """
    Convert data in the list to a numpy array of floats
    """
    return np.array(data, dtype=float)


def get_ievav(results: dict, N: int, complex_energy: bool = True):
    """
    Get iters, energy, vscore, acceptance, energy variance from results dictionary
    """
    if complex_energy:
        e = to_array(results["Energy"]["Mean"]["real"])
    else:
        e = to_array(results["Energy"]["Mean"])
    var = to_array(results["Energy"]["Variance"])
    v_score = vscore(e, var, N)
    acceptance = to_array(results["acceptance"]["value"])
    iters = to_array(results["Energy"]["iters"])
    return iters, e, v_score, acceptance, var


def compute_expectation(data: np.ndarray | Sequence, n_iters: int):
    """
    Compute mean(data[-n_iters:]),std(data[-n_iters:]) converting data to an array if necessary
    """
    data = to_array(data)
    return np.mean(data[-n_iters:]), np.std(data[-n_iters:])
