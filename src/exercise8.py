import logging

import numpy as np
from numba import njit
from vispy import app, color, scene
from vispy.scene.cameras import PanZoomCamera
from vispy.util.event import Event

# logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# fastly compiles with
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
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y

    # iteration matrix of shape (height, width) with numbers of iterations before espace
    iterations = np.zeros_like(C, dtype=np.uint16)

    for i in range(height):
        for j in range(width):
            iterations[i, j] = is_outside_mandelbrot(C[i, j], N, r)

    return iterations


def normalize_coloring(iteration_arr: np.ndarray, max_iter: int) -> np.ndarray:
    """
    Normalize to a float in [0, 1].

    Args:
        iteration_arr (np.ndarray): Array of number of iterations to escape
        max_iter (int): Number of total iterations

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


# https://vispy.org/gallery/scene/image.html


def setup_scene(
    rgb_image: np.ndarray,
    canvas_size: tuple[int, int],
) -> tuple[scene.SceneCanvas, scene.visuals.Image, scene.widgets.ViewBox]:
    """
    Set up the VisPy canvas and scene for displaying the Mandelbrot image.

    Args:
        rgb_image (np.ndarray): RGB image array of shape (H, W, 3)
        canvas_size (tuple): Tuple (width, height) specifying canvas size
        on_zoom (Callable): On zoom callback

    Returns:
        tuple: (canvas, image visual, view) for further interaction

    """
    canvas = scene.SceneCanvas(title="Mandelbrot Set", keys="interactive", size=canvas_size, show=True)
    view = canvas.central_widget.add_view()

    image = scene.visuals.Image(rgb_image, parent=view.scene, method="subdivide")

    camera = PanZoomCamera(aspect=1)
    view.camera = camera
    view.camera.flip = (0, 1, 0)
    view.camera.set_range()
    return canvas, image, view


def main() -> None:  # noqa: D103
    # Initial image dimensions
    width, height = 800, 800

    # Initial zoom parameters
    center_x, center_y = -0.743643135, 0.131825963  # Interesting deep-zoom point
    zoom_factor = 0.95
    frame = {"count": 0}
    x_width = 4.0
    y_height = 4.0

    max_iter_start = 100
    radius = 2.0

    # Compute initial frame
    x_min = center_x - x_width / 2
    x_max = center_x + x_width / 2
    y_min = center_y - y_height / 2
    y_max = center_y + y_height / 2

    mandelbrot_grid = compute_mandelbrot_grid(x_min, x_max, y_min, y_max, width, height, radius, max_iter_start)
    rgb_image = iterations_to_colormap(mandelbrot_grid, max_iter_start)
    rgb_image = rgb_image.reshape((height, width, 3))

    # Setup scene
    canvas, image, view = setup_scene(rgb_image, (width, height))

    def update_frame(_event: Event) -> None:
        # Zoom in
        frame["count"] += 1
        nonlocal x_width, y_height, image

        x_width *= zoom_factor
        y_height *= zoom_factor
        x_min = center_x - x_width / 2
        x_max = center_x + x_width / 2
        y_min = center_y - y_height / 2
        y_max = center_y + y_height / 2

        # Increase iteration depth with zoom
        max_iter = min(1000, max_iter_start + frame["count"] * 2)

        logger.info(
            "Frame %d: x=[%.6f, %.6f], y=[%.6f, %.6f], iter=%d",
            frame["count"],
            x_min,
            x_max,
            y_min,
            y_max,
            max_iter,
        )

        mandelbrot_grid = compute_mandelbrot_grid(x_min, x_max, y_min, y_max, width, height, radius, max_iter)

        rgb_image = iterations_to_colormap(mandelbrot_grid, max_iter)
        rgb_image = rgb_image.reshape((height, width, 3))

        logger.info("Shape: %s", rgb_image.shape)

        # Remove previous visual and add new one
        nonlocal image
        image.parent = None
        # ascontiguousarray for performance
        image = scene.visuals.Image(
            np.ascontiguousarray(rgb_image),
            parent=view.scene,
            method="subdivide",
        )

    # Start zoom timer
    _timer = app.Timer(interval=0.1, connect=update_frame, start=True)

    app.run()


if __name__ == "__main__":
    main()
