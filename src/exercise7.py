import random
from collections.abc import Sequence
from typing import Union
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


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

    def derivative(self) -> "Polynomial":
        return Polynomial(np.polyder(self.coeffs))

    def __repr__(self) -> str:  # noqa: C901
        terms = []
        degree = len(self.coeffs) - 1

        def format_coeff(c: complex) -> str:
            # Format complex coeff in a concise way:
            re = c.real
            im = c.imag
            parts = []
            # Handle real part if significant
            if abs(re) > 1e-14:
                parts.append(f"{re:.14g}")
            # Handle imaginary part if significant
            if abs(im) > 1e-14:
                sign = "+" if im >= 0 and parts else ""
                im_part = f"{sign}{im:.14g}j"
                parts.append(im_part)
            # If both parts are near zero, show 0
            if not parts:
                return "0"
            return "".join(parts)

        for i, coeff in enumerate(self.coeffs):
            power = degree - i
            if abs(coeff) < 1e-14:
                continue  # skip zero coefficients

            coeff_str = format_coeff(coeff)

            # For terms with coeff 1 or -1 (real 1 or -1 and zero imag), omit coeff for x terms
            if power > 0 and abs(coeff - 1) < 1e-14 and abs(coeff.imag) < 1e-14:
                coeff_str = ""
            elif power > 0 and abs(coeff + 1) < 1e-14 and abs(coeff.imag) < 1e-14:
                coeff_str = "-"

            if power > 1:
                term = f"{coeff_str}x^{power}"
            elif power == 1:
                term = f"{coeff_str}x"
            else:
                term = coeff_str

            terms.append(term)

        if not terms:
            return "0"

        # Combine terms with proper + / - signs
        result = terms[0]
        for term in terms[1:]:
            # Check if term starts with '-' to use correct sign
            if term.startswith("-"):
                result += " - " + term[1:]
            else:
                result += " + " + term

        return result

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

# Not much faster than above, since all roots are calcualted...
# Also recursion depth is limited in python.
def compute_preimages_randomized_recursively(f: Polynomial, k: int, z: complex) -> np.ndarray:
    """
    Compute the preimages f^(-k)(z) randomized.

    For an initial point z, this method computes the the sequence z_(n+1) = g(z_n),
    where g draws a random preimage of f^(-1)(z_n).

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
    random_preimage = random.choice(preimages)
    if k == 1:
        return random_preimage

    return np.vstack([compute_preimages_randomized_recursively(f, k - 1, random_preimage), random_preimage])

def compute_preimages_randomized(f: Polynomial, k: int, z: complex) -> np.ndarray:
    """
    Compute the preimages f^(-k)(z) randomized and iteratively.

    For an initial point z, this method computes the the sequence z_(n+1) = g(z_n),
    where g draws a random preimage of f^(-1)(z_n).

    Args:
        f (Polynomial): Polynomial of which the preimages should be computed.
        k (int): Degree of preimages.
        z (complex): Initial point.

    Returns:
        np.ndarray: Array of preimages of shape (d, 1)

    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    preimages = []
    curr = z

    for _ in tqdm(range(k), desc="Sampling roots of attractor:"):
        roots = np.roots((f - curr).coeffs)
        curr = random.choice(roots)
        preimages.append(curr)

    return np.array(preimages).reshape(-1,1)

if __name__ == "__main__":
    f = Polynomial([1, 0, -1])
    preimages = compute_preimages_randomized(f, 200000, 1)[10000:]
    fig = go.Figure(
        go.Scatter(
            x=preimages.real.flatten().tolist(),
            y=preimages.imag.flatten().tolist(),
            mode="markers",
        ),
    )
    fig.update_layout(
        title=f"Julia Set of {f}",
    )
    fig.show()
