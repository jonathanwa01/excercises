import bpy
import numpy as np
import ddg
from ddg.blender import props
from ddg.geometry import (
    Pencil,
    Point,
    Quadric,
    Subspace,
    QuadricIntersection,
)
from ddg.blender.collection import collection
from ddg.geometry._quadrics import touching_cone, polarize
import logging


def _setup_freestyle() -> None:
    fs_settings = ddg.blender.freestyle.settings()
    fs_settings.as_render_pass = False
    fs_settings.use_smoothness = False
    ddg.blender.freestyle.lineset(fs_settings, "Freestyle Line")


def _update_pencil(t: float,
                   pencil: Pencil,
                   basepoint: Point,
                   col: bpy.types.Collection,
                   ) -> None:
    """
    Callback function for parametrization of the pencil.

    Args:
        t (float): Parameter to change quadrics of the pencil
    """
    ddg.blender.collection.clear(
        [col],
        deep=True,
    )
    ddg.blender.object.clear_empty_objects()
    bounding_box = [2, 2, 2]

    logger.info(f"Update Quadric with parameter t = {t} for collection {col}")
    quadric: Quadric = pencil.quadric(t)
    ddg.blender.convert(
        quadric,
        "Quadric",
        collection=col,
        bounding_box=bounding_box,
    )

    # calculate touching cone
    tangent_cone: Quadric | Subspace = touching_cone(basepoint, quadric)
    ddg.blender.convert(
        tangent_cone,
        "Tangent_Cone",
        collection=col,
    )
    # Polar plane
    polar_plane: Subspace = polarize(basepoint, quadric)

    ddg.blender.convert(
        polar_plane,
        "Polar_plane",
        collection=col,
        bounding_box=bounding_box,
    )

    intersection: (QuadricIntersection
                   | Subspace
                   | Quadric) = ddg.geometry.intersect(polar_plane, quadric)
    ddg.blender.convert(
        intersection,
        "Intersection",
        collection=col,
    )


if __name__ == "__main__":
    # Blender setup
    ddg.blender.scene.clear(deep=True)
    cam = ddg.blender.camera.camera(location=(10, 10, 10))
    ddg.blender.camera.look_at_point(cam, (2.5, 2.5, 0))
    ddg.blender.light.light(location=(0, 1, 8))

    # setup freestyle
    _setup_freestyle()

    # logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    Q1 = Quadric(np.diag((1, 1, -1, -1)))
    Q2 = Quadric(np.diag((2, 1, 1, -1)))

    basepoint = Point(np.array([1.0, -2.0, 2.0, 1.0]))
    ddg.blender.convert(basepoint, f"Point_{id(basepoint)}")

    pencil = Pencil(Q1, Q2)
    col = collection("Pencil")
    _update_pencil(1.0, pencil, basepoint, col)

    props.add_props_with_callback(
        lambda t: _update_pencil(t, pencil, basepoint, col),
        ("t"),
        1.0,
    )
