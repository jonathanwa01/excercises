import ddg
from ddg.arrays import from_discrete_curve
from ddg.geometry._subspaces import Point
from ddg.nets import DiscreteCurve, DiscreteDomain
from ddg.geometry import (
    subspace_from_affine_points_and_directions,
    subspace_from_affine_points,
    intersect,
    Subspace,
)
import numpy as np

from ddg.nets._domain import DiscreteInterval

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
#domain = DiscreteDomain([[0, n_points - 1]])

# Create the DiscreteCurve
#curve = DiscreteCurve(lambda x: points[x], domain)
#ddg.blender.convert(curve, "Circle")


def tangents(curve: DiscreteCurve) -> DiscreteCurve:
    """
    Computes the discrete tangent lines of a given discrete curve.

    Parameters:
        curve (DiscreteCurve): The input discrete curve from which tangents are computed.

    Returns:
        DiscreteCurve: A discrete curve where each point is a tangent line (as an affine subspace) 
                       to the original curve at a corresponding segment.
    """
    points: np.ndarray = from_discrete_curve(curve).points
    tangent_directions = points[1:] - points[:-1]
    return DiscreteCurve(
        lambda x: subspace_from_affine_points_and_directions(
                    points[x], tangent_directions[x]
                ),
        [0, len(tangent_directions) - 1, False]
    )


# for tangent in tangents(curve):
#     # ddg.blender.convert(tangent, f"Tangents{id(tangent)}")
#     pass


def normals(curve: DiscreteCurve) -> DiscreteCurve:
    points: np.ndarray = from_discrete_curve(curve).points
    return DiscreteCurve(
        lambda x: ddg.geometry.euclidean(2).perpendicular_bisector(
            subspace_from_affine_points(points[x]),
            subspace_from_affine_points(points[x+1]),
        ),
        [0, len(points) - 2, False]
    )


# for normal in normals(curve):
#     # ddg.blender.convert(normal, f"Normal{id(normal)}")
#     pass


def envelope(curve: ddg.nets.DiscreteCurve) -> ddg.nets.DiscreteCurve:
    """
    Calculates the envelope of a given one parameter family of curves.

    Args:
        curves (DiscreteCurve): One parameter family of discrete curves

    Returns
        DiscreteCurve: The envelope
    """
    domain: DiscreteInterval = curve.domain
    return DiscreteCurve(
        lambda n: ddg.geometry.intersect(curve(n), curve(n+1)),
        DiscreteInterval([domain.interval[0], domain.interval[1]-1]))


def orthogonal_trajectory(curve: DiscreteCurve, point: Point) -> DiscreteCurve:
    """
    Constructs an orthogonal trajectory to a given discrete curve, starting from a specified point.

    The resulting trajectory is built by reflecting the initial point across the angle bisectors of 
    consecutive segments of the input curve, ensuring orthogonal progression relative to the curve's shape.

    Args:
        curve (DiscreteCurve): The input discrete curve along which the orthogonal trajectory is constructed.
        point (Point): The initial point from which the orthogonal trajectory begins.

    Returns:
        DiscreteCurve: A new discrete curve representing the orthogonal trajectory.
    
    """
    domain: DiscreteInterval = curve.domain
    points = [point]
    for n, in DiscreteInterval([domain.interval[0], domain.interval[1]-1]).traverser:
        bisector = ddg.geometry.euclidean(2).angle_bisector_orientation_preserving(curve(n), curve(n+1))
        points.append(ddg.geometry.euclidean(2).reflect_in_hyperplane(points[-1], bisector))
    return DiscreteCurve(lambda n: points[n - domain.interval[0]], [domain.interval[0], domain.interval[1]])


def evolute(curve: DiscreteCurve) -> DiscreteCurve:
    """
    Computes the evolute of a discrete curve.

    The evolute is defined as the envelope of the normals of the input curve,
    representing the locus of centers of curvature (i.e., the centers of osculating circles).

    Args:
        curve (DiscreteCurve): The input discrete curve for which the evolute is computed.

    Returns:
        DiscreteCurve: A new discrete curve representing the evolute of the input curve.
    """
    return envelope(normals(curve))

def midpoint(p1: Point, p2: Point) -> Point:
    return ddg.geometry.subspace_from_affine_points(0.5*(p1.affine_point + p2.affine_point))

def involute(curve: DiscreteCurve, point: Point=None) -> DiscreteCurve:
    """
    Computes the involute of a given discrete curve.

    The involute is constructed as the orthogonal trajectory to the tangents of the input curve, 
    starting from a given initial point. If no point is provided, the default starting point is the 
    midpoint of the first segment of the curve.

    Args:
        curve (DiscreteCurve): The input discrete curve for which the involute is computed.
        point (Point, optional): The initial point of the involute. If None, the midpoint 
                                              of the first segment of the curve is used.

    Returns:
        DiscreteCurve: A new discrete curve representing the involute of the input curve.

    """
    if point is None:
        domain: DiscreteInterval = curve.domain
        start = domain.interval[0]
        point = midpoint(curve(start), curve(start+1))
    return orthogonal_trajectory(tangents(curve), point)


def involutes(curve: ddg.nets.DiscreteCurve) -> ddg.nets.DiscreteCurve:
    domain: DiscreteInterval = curve.domain
    def fct(n):
        shifted_curve = DiscreteCurve(curve.fct, DiscreteInterval([domain.interval[0]+n, domain.interval[1]]))
        return involute(shifted_curve)
    return DiscreteCurve(fct, DiscreteInterval([domain.interval[0], domain.interval[1]-1]))

# curvee = envelope(normal)

#ddg.blender.convert(curvee, "Envelope")



# material
red = ddg.blender.material.material(color=(122, 11, 15))
red_light = ddg.blender.material.material(color=(252, 73, 3))
blue = ddg.blender.material.material(color=(3, 69, 252))
blue_light = ddg.blender.material.material(color=(3, 194, 252))
green = ddg.blender.material.material(color=(3, 252, 119))


def ellipse(a=2, b=1):
    def fct(u):
        return ddg.geometry.Point([a*np.cos(u), b*np.sin(u), 1])
    domain = [0,2*np.pi, True]
    return ddg.nets.SmoothCurve(fct, domain)


def affine_curve(curve: ddg.nets.DiscreteCurve):
    def fct(n):
        return curve.fct(n).affine_point
    return ddg.nets.DiscreteCurve(fct, curve.domain)

# sample curve
N = 20
dcurve = ddg.nets.sample_smooth_net(ellipse(), [N, 't'])


# curve
ddg.blender.convert(
    affine_curve(dcurve),
    "curve",
    material=blue_light,
)

# tangents
for k,t in enumerate(tangents(dcurve)):
    ddg.blender.convert(
        t,
        f"tangent{k}",
        material=blue,
        curve_radius=0.005,
        collection = ddg.blender.collection.collection("tangents")
    )

# normals
for k,n in enumerate(normals(dcurve)):
    ddg.blender.convert(
        n,
        f"normal{k}",
        material=red,
        curve_radius=0.005,
        collection = ddg.blender.collection.collection("normals")
    )

# evolute
ddg.blender.convert(
    affine_curve(evolute(dcurve)),
    "evolute",
    material=red_light,
)

# involute
ddg.blender.convert(
    affine_curve(involute(dcurve)),
    "involute",
    material=green,
)

# involutes
for k,i in enumerate(involutes(dcurve)):
    ddg.blender.convert(
        affine_curve(i),
        f"involute{k}",
        material=green,
        collection = ddg.blender.collection.collection("involutes")
    )