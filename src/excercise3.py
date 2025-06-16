import logging
import ddg
import bpy
from ddg.geometry import (
    Quadric,
    Pencil,
    Point,
)
from ddg.blender import props
from ddg.blender.collection import collection
from ddg.blender.animation import clear_animation_data
import numpy as np
from numpy.linalg import svd

ddg.blender.scene.clear(deep=True)
cam = ddg.blender.camera.camera(location=(10, 10, 10))
ddg.blender.camera.look_at_point(cam, (2.5, 2.5, 0))
ddg.blender.light.light(location=(0, 1, 8))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

base_points = [
    Point(np.array([1.0, 1.0, 1.0])),
    Point(np.array([-1.0, 1.0, 1.0])),
    Point(np.array([1.0, -1.0, 1.0])),
    Point(np.array([-1.0, -1.0, 1.0])),
]

# add basepoints to the scene
for basepoint in base_points:
    ddg.blender.convert(basepoint, f"Basepoint{id(basepoint)}")


def _pencil_from_base_point_matrix(base_points: np.ndarray) -> Pencil:
    """
    Calculates the pencil through the given basepoints.
    Points must have dimension 3 (homogeneus coordinates)

    Args:
        base_points (np.ndarray): Homogeneous coordinates of basepoints
         row by row of shape (4,3)
    """
    if base_points.shape != (4, 3):
        raise ValueError(
            "Missmatching number of points. The base_points"
            "matrix should be of shape (4,3), where the rows are given by the"
            "homogeneous coordinates of the poits.")

    A: list = []
    for x, y, z in base_points:
        A.append([
            x**2,           # a
            2 * x * y,      # b
            2 * x * z,      # c
            y**2,           # d
            2 * y * z,      # e
            z**2            # f
        ])

    # calculate the kernal of A using singular value decomposition
    M: np.ndarray = np.array(A)
    U, S, Vt = svd(M)

    # calculate the rank of A which coincides with the rank of S in the svd
    # due to numerical floating-point precision, we allow a small tollerance
    # to be considered as 0
    tol = 1e-10
    rank = np.sum(S > tol)

    if rank < 4:
        raise ValueError(
            "Pencil can not be uniquely determined, since basepoints"
            " do not lie in general position"
        )

    # kernal of shape (2,6)
    kernel: np.ndarray = Vt[-2:, :]

    def vec_to_matrix(vec):
        a, b, c, d, e, f = vec
        return np.array([
            [a, b, c],
            [b, d, e],
            [c, e, f],
        ])

    # create symmetric values and create Quadrics
    return Pencil(vec_to_matrix(kernel[0, :]), vec_to_matrix(kernel[1, :]))


def pencil_from_base_points(points: list[Point]) -> Pencil:
    """
    Calculates the pencil through the given basepoints.
    Points must have dimension 3 (homogeneus coordinates) and len(points)
    must be 4.

    Args:
        base_points (np.ndarray): Array of shape (4,3)
    """
    if len(points) != 4:
        raise ValueError(f"len(points) must be 4, but it is {len(points)}")
    return _pencil_from_base_point_matrix(
        np.array([point.point for point in points])
    )


def _get_quadric_of_pencil(t: tuple[float, float], pencil: Pencil) -> Quadric:
    """
    Calculates quadric of pencil with parameter t.
    We identify [1, a] <-> a and [0, 1] <-> inf.

    Args:
        t (tuple[float, float]): Parameter t in RP1
        pencil (Pencil): Pencil
    """
    return Quadric(pencil.matrix(t))


pencil: Pencil = pencil_from_base_points(base_points)
col = collection(f"Pencil_{id(pencil)}")


def _update_pencil(t: float,
                   pencil: Pencil,
                   ) -> None:
    """
    Callback function for parametrization of the pencil.

    Args:
        t (float): Parameter to change quadrics of the pencil
    """
    ddg.blender.collection.clear(
        [col],
        deep=True
    )
    logger.info(f"Update basepoints with parameter t = {t}")
    quadric: Quadric = _get_quadric_of_pencil((1.0, t), pencil)
    ddg.blender.convert(
        quadric,
        "Quadric",
        collection=col,
    )


props.add_props_with_callback(
    lambda t: _update_pencil(t, pencil),
    ("t"),
    1.0,
)

scene = bpy.context.scene
clear_animation_data(scene)
scene.render.resolution_x = 1000
scene.render.resolution_y = 1000


FPS = 100


ddg.blender.animation.set_keyframe(scene, 0, "t", -6.0)
ddg.blender.animation.set_keyframe(scene, FPS, "t", 0)
ddg.blender.animation.set_keyframe(scene, 2*FPS, "t", 6.0)

# render annimation video
ddg.blender.render.setup_eevee_renderer()
ddg.blender.render.set_film_transparency()
ddg.blender.render.set_world_background()

# ddg.blender.render.set_render_output_images("pencil", time=True)
# ddg.blender.render.render_animation(start=0, end=2*FPS, camera=cam)
