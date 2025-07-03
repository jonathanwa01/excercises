from collections.abc import Sequence
from typing import Union

import numpy as np
import plotly.graph_objects as go


class Polynomial:
    """Class which represents a complex polynomial."""

    def __init__(self, coeffs: Sequence[complex] | np.ndarray) -> None:
        """
        Initialize polynomial instance.

        Args:
            coeffs (Sequence[complex]): Coefficients of the polynomial a_n, ..., a_0

        """
        # Remove leading zeros to avoid degree inflation
        self.coeffs: np.ndarray = np.trim_zeros(np.array(coeffs, dtype=complex), "f")

    def __call__(self, z: complex) -> complex:
        return complex(np.polyval(self.coeffs, z))

    def __add__(self, other: Union["Polynomial", complex]) -> "Polynomial":
        if isinstance(other, Polynomial):
            len_self = len(self.coeffs)
            len_other = len(other.coeffs)
            if len_self < len_other:
                padded_self = np.pad(self.coeffs, (len_other - len_self, 0), constant_values=0)
                padded_other = other.coeffs
            else:
                padded_self = self.coeffs
                padded_other = np.pad(other.coeffs, (len_self - len_other, 0), constant_values=0)
            return Polynomial(padded_self + padded_other)

        if isinstance(other, (int, float, complex)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[-1] += other
            return Polynomial(new_coeffs)

        return NotImplemented

    def __radd__(self, other: Union["Polynomial", complex]) -> "Polynomial":
        return self + other

    def __sub__(self, other: Union["Polynomial", complex]) -> "Polynomial":
        if isinstance(other, Polynomial):
            len_self = len(self.coeffs)
            len_other = len(other.coeffs)
            if len_self < len_other:
                padded_self = np.pad(self.coeffs, (len_other - len_self, 0), constant_values=0)
                padded_other = other.coeffs
            else:
                padded_self = self.coeffs
                padded_other = np.pad(other.coeffs, (len_self - len_other, 0), constant_values=0)
            return Polynomial(padded_self - padded_other)

        if isinstance(other, (int, float, complex)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[-1] -= other
            return Polynomial(new_coeffs)

        return NotImplemented

    def __rsub__(self, other: Union["Polynomial", complex]) -> "Polynomial":
        return -self + other

    def __neg__(self) -> "Polynomial":
        return Polynomial(-self.coeffs)

    def degree(self) -> int:
        return len(self.coeffs) - 1


def compute_preimages(f: Polynomial, k: int, z: complex) -> np.ndarray:
    """
    Compute all preimages f^(-k)(z).

    Args:
        f (Polynomial): Polynomial of which the preimages should be computed.
        k (int): Degree of preimages.
        z (complex): Initial point.

    Returns:
        np.ndarray: Array of preimages of shape (d, 1)

    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    preimages = np.roots((f - z).coeffs)
    if k == 1:
        return preimages

    # recursively compute preimages and stack into an array of dim (n^k, 1)
    all_preimages = [compute_preimages(f, k - 1, point).reshape(-1, 1) for point in preimages]

    return np.vstack(all_preimages)


if __name__ == "__main__":
    preimages = compute_preimages(Polynomial([1, 0, -1]), 15, 1)
    fig = go.Figure(
        go.Scatter(
            x=preimages.real.flatten().tolist(),
            y=preimages.imag.flatten().tolist(),
            mode="markers",
        ),
    )
    fig.show()
