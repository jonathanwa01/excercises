from collections.abc import Iterable
import logging
from typing import Callable
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from exercise5 import approximate_attractor_randomized

# logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def box_count(points: np.ndarray, eps: float) -> int:
    """
    Approximates number of boxes needed for covering the points with sets of diam eps.

    Args:
        points (np.ndarray): ndarray of shape (sample_size, d), providing the
            points row by row
        eps (float): Diameter of sets to cover the points with.

    """
    boxes = set()
    # scales the boxis to unit length
    scaled = np.floor(points / eps)

    for coord in scaled:
        boxes.add(tuple(coord))
    return len(boxes)


def estimate_box_dimension(points: np.ndarray, min_exp: int = 3, max_exp: int = 10) -> float:
    """
    Estimate the box-counting dimension of a point cloud.

    It calculates a log-log linear regression on box counts.
    This function overlays grids of side length ε = 2^-k for k in the range
    [min_exp, max_exp], counts how many boxes contain at least one point,
    and fits a linear model to log(N(ε)) vs. log(1/ε). The slope of the line
    approximates the box-counting (Minkowski) dimension.

    Args:
        points: A NumPy array of shape (n_points, d) containing the sampled
            point cloud, where each row is a d-dimensional point.
        min_exp: The minimum exponent k to use for ε = 2^-k.
        max_exp: The maximum exponent k to use for ε = 2^-k.

    Returns:
        A float representing the estimated box-counting dimension.

    """
    epsilons = [2**-k for k in range(min_exp, max_exp + 1)]
    counts = [box_count(points, eps) for eps in tqdm(epsilons, desc="Computing box counts")]

    log_eps = np.log(1 / np.array(epsilons))
    log_N = np.log(counts)

    # Since N(ε) ≈ ε^(-d), use least-squares linear regression for following equation:
    # logN(ε) ≈ -d⋅logε+C
    d, _ = np.polyfit(log_eps, log_N, 1)
    return d


def loglog_box_covering_samples(points: np.ndarray,
                             iteration_range: Iterable[float] = list(range(3, 10))) -> list[tuple[float, float]]:
    """
    Computes log-log data points for box-counting analysis of a point set.

    For each scale δₖ = 2^(-k), this function calculates the number of boxes Nₖ 
    of size δₖ required to cover the given set of points, and returns the 
    corresponding log-log pairs (log₂(δₖ), log₂(Nₖ)).

    Args:
        points (np.ndarray): Array of points representing the set to analyze.
        iteration_range (Iterable[int]): Iterable of integer exponents k to define 
                                         the box size as δₖ = 2^(-k). Default is range(1, 10).

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple (log₂(1/δₖ), log₂(Nₖ)).
    """

    epsilons = [2**-k for k in iteration_range]
    counts = [box_count(points, eps) for eps in tqdm(epsilons, desc="Computing box counts")]

    log_eps: np.ndarray = np.log(1 / np.array(epsilons))
    log_N: np.ndarray = np.log(counts)

    return log_eps, log_N


if __name__ == "__main__":
    number_of_samples = int(1e6)
    number_of_iterations = 20

    iterated_function_system: dict[str, list[Callable[..., tuple[float, ...]]]] = {
        # "ex_functions": [
        #     lambda x, y: (0.8 * x + 0.1, 0.8 * y + 0.04),
        #     lambda x, y: (0.6 * x + 0.19, 0.6 * y + 0.5),
        #     lambda x, y: (0.446 * (x - y) + 0.266, 0.466 * (x + y) + 0.067),
        #     lambda x, y: (0.446 * (x + y) + 0.456, 0.446 * (x - y) + 0.434),
        # ],
        "Sierpinski Triangle": [
            lambda x, y: (0.5 * (x - 3), 0.5 * y),
            lambda x, y: (0.5 * (x + 3), 0.5 * y),
            lambda x, y: (0.5 * x, 0.5 * (y + 3)),
        ],
        "Black Spleenwort Fern": [
            lambda _, y: (0, 0.16 * y),
            lambda x, y: (0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6),
            lambda x, y: (0.2 * x - 0.26 * y, 0.24 * x + 0.22 * y + 1.6),
            lambda x, y: (-0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44),
        ],
    }

    for name, func in iterated_function_system.items():
        logger.info(f"Calculating attractor for {name}.")
        high_res_samples = approximate_attractor_randomized(
            func,
            [(1.0, 1.0)],
            number_of_samples=number_of_samples,
            number_of_iterations=number_of_iterations,
        )
        points = np.array(high_res_samples)
        logger.info(f"Calculating box covering samples for {name}.")
        x_vals, y_vals = loglog_box_covering_samples(points, list(range(1,20)))
        # calculate the slope (boxplot dim)
        d, C = np.polyfit(x_vals, y_vals, 1)

        x_fit = np.array(x_vals)
        y_fit = d * x_fit + C

        # Create the figure
        fig = go.Figure()

        # (log-log plot)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+lines',
            name='log-log data'
        ))

        # Dimension line
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f'Fit: y = {d:.4f}x + {C:.4f}',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="log(δₖ)",
            yaxis_title="log(Nₖ)",
            title=f"Log Log Plot box-couting dimension for {name}"
        )
        fig.add_annotation(
            text=f"Slope = {d}",
            xref="paper", yref="paper",
            x=0.05, y=0.95,  # Position in figure coordinates (0 to 1)
            showarrow=False,
            font=dict(size=12, color="black"),
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="lightgray",
            opacity=0.8
        )
        fig.show()
