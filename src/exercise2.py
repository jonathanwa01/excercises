import ddg
from ddg.arrays import from_discrete_curve
from ddg.nets import DiscreteCurve, DiscreteDomain
from ddg.geometry import (
    subspace_from_affine_points_and_directions,
    subspace_from_affine_points,
    intersect,
    Subspace,
)
import numpy as np

ddg.blender.scene.clear(deep=True)
cam = ddg.blender.camera.camera(location=(10, 10, 10))
ddg.blender.camera.look_at_point(cam, (2.5, 2.5, 0))
ddg.blender.light.light(location=(0, 1, 8))


# curve: DiscreteCurve = DiscreteCurve(lambda t: (t, t**2), (-10, 10, False))
# ddg.blender.convert(curve, "Curve")

# circle
# Parameters
radius = 1.0
n_points = 100  # Number of points to sample
# Define the parameter domain
parameters = np.linspace(0, 2 * np.pi, n_points, endpoint=True)
# Create a list of points on the circle
points = [(radius * np.cos(theta), radius * np.sin(theta))
          for theta in parameters]
# Create the domain for the discrete curve
domain = DiscreteDomain([[0, n_points - 1]])

# Create the DiscreteCurve
curve = DiscreteCurve(lambda x: points[x], domain)
ddg.blender.convert(curve, "Circle")


def tangents(curve: DiscreteCurve) -> DiscreteCurve:
    points: np.ndarray = from_discrete_curve(curve).points
    tangent_directions = points[1:] - points[:-1]
    return DiscreteCurve(
        lambda x: subspace_from_affine_points_and_directions(
                    points[x], tangent_directions[x]
                ),
        [0, len(tangent_directions) - 1, False]
    )


for tangent in tangents(curve):
    ddg.blender.convert(tangent, f"Tangents{id(tangent)}")
    pass


def normals(curve: DiscreteCurve) -> DiscreteCurve:
    points: np.ndarray = from_discrete_curve(curve).points
    return DiscreteCurve(
        lambda x: ddg.geometry.euclidean(2).perpendicular_bisector(
            subspace_from_affine_points(points[x]),
            subspace_from_affine_points(points[x+1]),
        ),
        [0, len(points) - 2, False]
    )


for normal in normals(curve):
    ddg.blender.convert(normal, f"Normal{id(normal)}")
    pass


def envelope(curves: list[Subspace]) -> DiscreteCurve:
    """
    Calculates the envelope of a given one parameter family of curves.

    Args:
        curves (DiscreteCurve): One parameter family of discrete curves

    Returns
        DiscreteCurve: The envelope
    """
    return DiscreteCurve(
        lambda n: intersect(curves[n], curves[n+1]),
        [0, len(curves) - 2, False]
    )


# normal = normals(curve).evaluate()
# print("NORMAL")
# print(normal)
# curvee = envelope(normal)

# ddg.blender.convert(curvee, "Envelope")
