import logging

import numpy as np
from scipy.optimize import root
from vispy import app, color, scene
from vispy.scene.cameras import PanZoomCamera
from vispy.scene.visuals import Image
from numba import njit, prange

# logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@njit
def is_outside_mandelbrot(c: complex, N: int, r: float) -> int:
    """
    Check whether a complex number c escapes the radius r within N iterations.

    Args:
        c (complex): The complex number to be checked.
        r (float): The escape radius. If the magnitude of the iterated value exceeds r,
                   the point is considered to have escaped.
        N (int): The maximum number of iterations to check.

    Returns:
        int: The iteration count at which the escape occurred.
            If the point does not escape, the second element will be N.

    """
    z = complex(0, 0)
    for k in range(N):
        z = z**2 + c

        if abs(z) > r:
            return k
    return N

@njit(parallel=True)
def compute_mandelbrot_grid(  # noqa: PLR0913
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    r: float,
    N: int,
) -> np.ndarray:
    """
    Compute the Mandelbrot escape times for a grid of complex numbers.

    Each point in the complex plane is tested using the Mandelbrot iteration,
    and the number of iterations it takes to escape a given radius is recorded.

    Args:
        x_min (float): Minimum real value of the complex plane.
        x_max (float): Maximum real value of the complex plane.
        y_min (float): Minimum imaginary value of the complex plane.
        y_max (float): Maximum imaginary value of the complex plane.
        width (int): Number of pixels along the horizontal axis.
        height (int): Number of pixels along the vertical axis.
        r (float): Escape radius. If the magnitude of z exceeds this, the point is considered to have escaped.
        N (int): Maximum number of iterations to test for each point.

    Returns:
        np.ndarray: A 2D array of shape (height, width) where each value is the number of iterations
                    before escape, or N if the point did not escape.

    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    # iteration matrix of shape (height, width) with numbers of iterations before espace
    iterations = np.zeros((height, width), dtype=np.uint16)

    for i in prange(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            iterations[i, j] = is_outside_mandelbrot(c, N, r)

    return iterations


def normalize_coloring(iteration_arr: np.ndarray, max_iter: int) -> np.ndarray:
    """
    Normalize to a float in [0, 1].

    Args:
        iteration_arr (np.ndarray): Array of number of iterations to escape
        max_iter (int): Number of total iterations

    Returns:
        np.ndarray: Normalized array.

    """
    return iteration_arr / max_iter


def iterations_to_colormap(iter_array: np.ndarray, max_iter: int, cmap_name: str = "inferno") -> np.ndarray:
    """
    Map Mandelbrot iteration counts to RGB color values using a colormap.

    This function converts a 2D array of iteration counts (escape times) into
    a corresponding 3D RGB image array by:
    - Normalizing the values to the [0, 1] range
    - Applying a colormap of vispy (e.g., 'inferno') to generate RGBA values
    - Converting the result to 8-bit RGB format for display

    Args:
        iter_array (np.ndarray): A 2D array of integers representing the number of iterations taken to escape.
        max_iter (int): The maximum number of iterations used during the Mandelbrot computation.
            Used for normalizing iteration values.
        cmap_name (str, optional): The name of the VisPy colormap to use (default is 'inferno').

    Returns:
        np.ndarray: A 3D uint8 array of shape (height, width, 3), containing RGB color values
            in the range [0, 255], suitable for image rendering.

    """
    cmap = color.get_colormap(cmap_name)
    normalized = normalize_coloring(iter_array, max_iter)
    rgba = cmap.map(normalized)
    # scale to [0, 255] since vispy color map creates colors in [0,1]
    return (rgba[..., :3] * 255).astype(np.uint8)


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


# https://vispy.org/gallery/scene/image.html


def setup_scene(
    rgb_image: np.ndarray,
    canvas_size: tuple[int, int],
) -> tuple[scene.SceneCanvas, Image, scene.cameras.PanZoomCamera]:
    """
    Set up the VisPy canvas and scene for displaying the Mandelbrot image.

    Args:
        rgb_image (np.ndarray): RGB image array of shape (H, W, 3)
        canvas_size (tuple): Tuple (width, height) specifying canvas size

    Returns:
        tuple: (canvas, image visual, camera) for further interaction

    """
    canvas = scene.SceneCanvas(title="Mandelbrot Set", keys="interactive", size=canvas_size, show=True)
    view = canvas.central_widget.add_view()

    image = Image(rgb_image, parent=view.scene)

    camera = PanZoomCamera(aspect=1)
    view.camera = camera
    view.camera.flip = (0, 1, 0)
    view.camera.set_range()
    return canvas, image, camera


def main() -> None:  # noqa: D103
    width, height = 800, 800
    xmin, xmax = -2.0, 2.0
    ymin, ymax = -2.0, 2.0
    radius = 2.0
    max_iter = 100

    logger.info("Computing Mandelbrot set...")
    escape_array = compute_mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, radius, max_iter)

    logger.info("Generating RGB image...")
    rgb_image = iterations_to_colormap(escape_array, max_iter)
    rgb_image = rgb_image.reshape((height, width, 3))

    logger.info("Setting up VisPy canvas...")
    _ = setup_scene(rgb_image, (width, height))

    app.run()


if __name__ == "__main__":
    main()
