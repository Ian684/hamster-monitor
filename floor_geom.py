"""Cage-floor geometry shared by tracker.py (distance) and web.py (checking what the user typed)."""
import math


def floor_shape(sides_cm):
    """Real positions (cm) of the 4 floor points from the lengths of the sides [1→2, 2→3, 3→4, 4→1].

    The 4 points are the ones clicked in 框選籠子範圍 › 籠底四角 (1 top-left, 2 top-right, 3 bottom-right,
    4 bottom-left as seen in the picture): the cage floor's corners, or 4 markers laid on the floor.
    1→2 and 4→3 are taken as parallel: equal lengths = rectangle, different = trapezoid (a cage narrower at
    one end). Point 1 is at (0, 0), point 2 at (top, 0); points 3 and 4 lie on the line y = depth.
    Returns None if the lengths can't form such a shape.
    """
    try:
        top, right, bottom, left = (float(v) for v in sides_cm)
    except (TypeError, ValueError):
        return None
    if min(top, right, bottom, left) <= 0:
        return None
    d = bottom - top
    if abs(d) < 1e-6:
        x = 0.0   # rectangle (or parallelogram): treat as a rectangle, depth from the average side
        depth = (left + right) / 2
    else:
        x = (right ** 2 - left ** 2 - d ** 2) / (2 * d)   # where point 4 sits horizontally relative to point 1
        if left ** 2 - x ** 2 <= 0:
            return None
        depth = math.sqrt(left ** 2 - x ** 2)
    return [(0.0, 0.0), (top, 0.0), (x + bottom, depth), (x, depth)]
