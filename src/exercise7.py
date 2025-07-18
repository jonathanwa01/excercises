import random
from collections.abc import Sequence
from typing import Union

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from scipy.optimize import root
from tqdm import tqdm


def find_attracting_orbits(c: complex, max_period: int = 5, tol: float = 1e-6) -> tuple[int, list[complex]] | None:
    """
    Find attracting periodic orbits of f_c(z) = z^2 + c.

    Args:
        c (complex): Parameter c.
        max_period (int): Maximum period to check.
        tol (float): Numerical tolerance.

    Returns:
        ptional[Tuple[int, list[complex]]]: (period, periodic_orbit) if found, else None.

    """

    def f(z: complex, n: int) -> complex:
        z = complex(z)
        for _ in range(n):
            z = z**2 + c
        return z

    for period in range(1, max_period + 1):
        # Try multiple initial guesses
        guesses = np.exp(2j * np.pi * np.linspace(0, 1, 10))

        for guess in guesses:
            # calculate f^period(z) - z = 0
            sol = root(
                lambda z, period=period: np.array(
                    [
                        (f(z[0] + 1j * z[1], period) - (z[0] + 1j * z[1])).real,
                        (f(z[0] + 1j * z[1], period) - (z[0] + 1j * z[1])).imag,
                    ],
                ),
                [guess.real, guess.imag],
                tol=tol,
            )  # type: ignore # noqa: PGH003
            if sol.success:
                z_sol: complex = sol.x[0] + 1j * sol.x[1]

                # Compute (f^p)'(z_sol)
                dz: complex = complex(1, 0)
                z = z_sol
                for _ in range(period):
                    dz = dz * 2 * z
                    z = z**2 + c

                if abs(dz) < 1:  # attracting fix-point if |dz| = |(f^p)'(z_sol)| < 1
                    orbit = [z_sol]
                    z_next = z_sol
                    for _ in range(period - 1):
                        z_next = z_next**2 + c
                        orbit.append(z_next)

                    return period, orbit

    # No attracting orbit found
    return None

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

    for _ in tqdm(range(k), desc="Sampling points of attractor:"):
        roots = np.roots((f - curr).coeffs)
        curr = random.choice(roots)
        preimages.append(curr)

    return np.array(preimages).reshape(-1, 1)


def ex1() -> None:  # noqa: D103
    f = Polynomial([1, 0, -0.3 + 0.3j])
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


# Initialize the Dash app
app = dash.Dash(__name__)


# Layout with sliders for real and imaginary parts
app.layout = html.Div(
    [
        html.H2("Interactive Plot of Julia set z^2 + c"),
        html.Div(
            [
                html.Label("Real part of c:"),
                dcc.Slider(
                    id="real-slider",
                    min=-4,
                    max=4,
                    step=0.01,
                    value=-0.12,
                    marks={i: str(i) for i in range(-4, 5)},
                ),
            ],
            style={"margin-bottom": "30px"},
        ),
        html.Div(
            [
                html.Label("Imaginary part of c:"),
                dcc.Slider(
                    id="imag-slider",
                    min=-4,
                    max=4,
                    step=0.01,
                    value=0.74,
                    marks={i: str(i) for i in range(-4, 5)},
                ),
            ],
            style={"margin-bottom": "30px"},
        ),
        html.Div(
            [
                html.Label("Show periodic points:"),
                dcc.Checklist(
                    id="show-periodic",
                    options=["show"],
                    value=[],
                    inline=True,
                ),
            ],
            style={"margin-bottom": "30px"},
        ),
        html.Div(id="z-display", style={"margin-bottom": "20px", "fontWeight": "bold"}),
        dcc.Graph(id="complex-plot"),
    ],
)


# Callback to update plot
@app.callback(
    Output("complex-plot", "figure"),
    Output("z-display", "children"),
    Input("real-slider", "value"),
    Input("imag-slider", "value"),
    Input("show-periodic", "value"),
)
def update_plot(real: float, imag: float, show_periodic: list[str]) -> tuple[go.Figure, str]:
    """
    Callbacl function for updating plot.

    Args:
        real (float): Real part of c
        imag (float): Imaginary part of c
        show_periodic (list[str]): List of selected checklist values (e.g., ["show"] if enabled, [] if disabled).
            If non-empty, the function will plot the attracting periodic orbit points on top of the Julia set.
            Typically controlled by a Dash dcc.Checklist component.

    Returns:
        tuple[go.Figure, str]: Figure and error message

    """
    try:
        # Safely parse the complex number
        z = complex(real, imag)
        f: Polynomial = Polynomial([1, 0, z])

        preimages = compute_preimages_randomized(f, 20000, 1)[10000:]

        fig = go.Figure(
            go.Scatter(
                x=preimages.real.flatten().tolist(),
                y=preimages.imag.flatten().tolist(),
                mode="markers",
                marker={"size": 1, "color": "blue"},
                name="Julia Set",
            ),
        )
        if show_periodic:
            result = find_attracting_orbits(z)
            if result:
                period, orbit = result
                fig.add_trace(
                    go.Scatter(
                        x=[pt.real for pt in orbit],
                        y=[pt.imag for pt in orbit],
                        mode="markers+text",
                        marker={"size": 8, "color": "red", "symbol": "circle"},
                        text=[f"P{period}"] * len(orbit),
                        textposition="top center",
                        name=f"Period-{period} orbit",
                    ),
                )
        fig.update_layout(
            title=f"Julia Set of {f}",
            showlegend=True,
        )
        return fig, ""

    except Exception as e:  # noqa: BLE001
        # Handle invalid complex number format
        empty_fig = go.Figure()
        return empty_fig, f"Invalid complex number: {e!s}"


if __name__ == "__main__":
    app.run(debug=True)
    # ex1()
