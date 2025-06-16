import logging
from typing import Callable
import ddg
from ddg.geometry import (
    Quadric,
    Subspace,
    quadric_normalization,
    subspace_from_affine_points
)
from ddg.blender.props import add_props_with_callback
from ddg.blender.collection import collection
import numpy as np


def lines_through_one_sheeted_hyperboloid(
        Q: Quadric) -> Callable[[float], tuple[Subspace, Subspace]]:
    """
    Calculates the lines in given quadric.

    Args:
        Q (Quadric): One sheeted hyperboloid.
    """

    # normalize quadric:
    # F is given st. F^tMF = normalized
    # np.diag((1,1,-1,-1)) == F.T @ Q.matrix @ F
    sgn, F = quadric_normalization(Q)

    if sgn.plus != 2:
        raise ValueError("Quadric does not contain lines.")

    def line_parametrization(phi: float) -> tuple[Subspace, Subspace]:
        """
        Calculates the parametrization of the lines.

        It does so by transforming the parametrization of the standard
        one-sheeted hyberboloid.

        Args:
            phi (float): phi in [0, 2*pi)

        Returns:
            tuple[Subspace, Subspace]: Lines contained in the one-sheeted
                hyperboloid parameterized by phi
        """
        line_1: Subspace = subspace_from_affine_points(
            np.array([
                np.cos(phi) - np.sin(phi),
                np.sin(phi) + np.cos(phi),
                1,
            ]),
            np.array([
                np.cos(phi),
                np.sin(phi),
                0,
            ])
        )

        line_2: Subspace = subspace_from_affine_points(
            np.array([
                np.cos(phi) - np.sin(phi),
                np.sin(phi) + np.cos(phi),
                -1,
            ]),
            np.array([
                np.cos(phi),
                np.sin(phi),
                0,
            ])
        )

        line_1 = line_1.transform(F)
        line_2 = line_2.transform(F)
        return line_1, line_2

    return line_parametrization


lines_collection = collection("Lines")


def _update_line(t: float, q: Quadric) -> None:
    """
    Callback function for parametrization of the lines.

    Args:
        t (float): Parameter to change the line
    """
    ddg.blender.collection.clear(
        [lines_collection],
        deep=True
    )
    line_1, line_2 = lines_through_one_sheeted_hyperboloid(q)(t)
    ddg.blender.convert(
        line_1,
        "Line_1",
        collection=lines_collection,
    )
    ddg.blender.convert(
        line_2,
        "Line_2",
        collection=lines_collection,
    )


if __name__ == "__main__":
    # Blender setup
    ddg.blender.scene.clear(deep=True)
    cam = ddg.blender.camera.camera(location=(10, 10, 10))
    ddg.blender.camera.look_at_point(cam, (2.5, 2.5, 0))
    ddg.blender.light.light(location=(0, 1, 8))

    # logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    q = Quadric(np.diag((2, 5, -1, -1)))
    ddg.blender.convert(q, "Quadric")
    add_props_with_callback(
        lambda t: _update_line(t, q),
        ("t"),
        1.0,
    )
