import logging

import numpy as np
from numba import njit, prange  # type: ignore noqa: PGH003
from vispy import app, color, scene
from vispy.scene.cameras import PanZoomCamera

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@njit
def is_close(a: complex, b: complex, tol: float = 1e-6) -> bool:
    """
    Check if two complex numbers are close within a tolerance.

    Args:
        a (complex): First number.
        b (complex): Second number.
        tol (float): Tolerance.

    Returns:
        bool: True if |a - b| < tol, else False.

    """
    return abs(a - b) < tol


@njit
def find_critical_orbit_period(c: complex, max_iter: int = 1000, max_period: int = 6) -> int:
    """
    Find the period of the attracting orbit (if any) by following the critical orbit z=0.

    This is done by the following. Start with a critical point (z=0 is the only one).
    It gets attracted to the attracting orbit. Then apply z**2 + c repeatedly until the
    sequence either diverges to infinity or closes to an orbit.

    Args:
        c (complex): Parameter c.
        max_iter (int): Maximum number of iterations.
        max_period (int): Maximum period to check.

    Returns:
        int: Detected period (1, 2, ..., max_period), or 0 if none.

    """
    z = 0.0 + 0.0j
    orbit_buffer: np.ndarray = np.empty(max_period, dtype=np.complex128)

    for i in range(max_iter):
        z = z * z + c
        if np.isinf(z.real) or np.isinf(z.imag) or np.isnan(z.real) or np.isnan(z.imag):
            return 0  # no attracting cycle, since it escaped to infinity
        # store the first max_period points
        if i < max_period:
            orbit_buffer[i] = z
        else:
            # check for cicles (up to numerical tolerance)
            for p in range(1, max_period + 1):
                if is_close(z, orbit_buffer[i % p]):
                    return p  # found period p, since circle closed
    return 0  # no cycle detected


@njit(parallel=True)
def compute_period_grid_parallel(  # noqa: PLR0913
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_period: int,
) -> np.ndarray:
    """
    Compute the attracting period for each point c on a complex grid.

    Args:
        xmin (float): Minimum real part.
        xmax (float): Maximum real part.
        ymin (float): Minimum imaginary part.
        ymax (float): Maximum imaginary part.
        width (int): Grid width.
        height (int): Grid height.
        max_period (int): Maximum period to check.

    Returns:
        np.ndarray: 2D array of detected periods, shape (height, width).

    """
    period_grid = np.zeros((height, width), dtype=np.uint8)
    for i in prange(height):  # like range, but for parallelization using numda
        for j in range(width):
            x = xmin + (xmax - xmin) * j / (width - 1)
            y = ymin + (ymax - ymin) * i / (height - 1)
            c = x + 1j * y
            period = find_critical_orbit_period(c, max_iter=1000, max_period=max_period)
            period_grid[i, j] = period
    return period_grid


def normalize_coloring(arr: np.ndarray, max_value: int) -> np.ndarray:
    """
    Normalize an array to [0,1] range.

    Args:
        arr (np.ndarray): Input array.
        max_value (int): Maximum value.

    Returns:
        np.ndarray: Normalized array.

    """
    return arr / max_value if max_value != 0 else arr


def period_to_colormap(period_array: np.ndarray, max_period: int, cmap_name: str = "inferno") -> np.ndarray:
    """
    Map period values to RGB colors using a colormap.

    Args:
        period_array (np.ndarray): 2D array of period values.
        max_period (int): Maximum period value.
        cmap_name (str): VisPy colormap name.

    Returns:
        np.ndarray: 3D uint8 RGB image, shape (height, width, 3).

    """
    cmap = color.get_colormap(cmap_name)
    normalized = normalize_coloring(period_array, max_period)
    rgba = cmap.map(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def setup_scene(rgb_image: np.ndarray, canvas_size: tuple[int, int]) -> scene.SceneCanvas:
    """
    Set up the VisPy canvas to display the RGB image.

    Args:
        rgb_image (np.ndarray): Image array (height, width, 3).
        canvas_size (tuple[int, int]): Canvas size in pixels.

    Returns:
        scene.SceneCanvas: The created VisPy canvas.

    """
    canvas = scene.SceneCanvas(title="Mandelbrot Period Coloring", keys="interactive", size=canvas_size, show=True)
    view = canvas.central_widget.add_view()
    scene.visuals.Image(rgb_image, parent=view.scene, method="subdivide")

    camera = PanZoomCamera(aspect=1)
    view.camera = camera
    view.camera.flip = (0, 1, 0)
    view.camera.set_range()
    return canvas


def main() -> None:  # noqa: D103
    width, height = 600, 600
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_period = 50

    logger.info("Computing period grid...")
    period_array = compute_period_grid_parallel(xmin, xmax, ymin, ymax, width, height, max_period)

    logger.info("Mapping periods to colormap...")
    rgb_image = period_to_colormap(period_array, max_period)
    rgb_image = rgb_image.reshape((height, width, 3))

    logger.info("Setting up VisPy scene...")
    setup_scene(rgb_image, (width, height))

    app.run()


if __name__ == "__main__":
    main()
