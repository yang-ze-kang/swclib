import math
from anytree import NodeMixin, RenderTree, iterators
from typing import List
from dataclasses import dataclass, field
import numpy as np

from swclib.data.euclidean_point import EuclideanPoint3D

_3D = "3d"
_2D = "2d"


# not used old code
def compute_platform_area(r1, r2, h):
    return (r1 + r2) * h * math.pi


# not used old code
def compute_two_node_area(tn1, tn2, remain_dist):
    """Returns the surface area formed by two nodes"""
    r1 = tn1.radius()
    r2 = tn2.radius()
    d = tn1.distance(tn2)
    print(remain_dist)

    if remain_dist >= d:
        h = d
    else:
        h = remain_dist
        a = remain_dist / d
        r2 = r1 * (1 - a) + r2 * a

    area = compute_platform_area(r1, r2, h)
    return area


# not used old code
def compute_surface_area(tn, range_radius):
    area = 0

    # backtrace
    currentDist = 0
    parent = tn.parent
    while parent and currentDist < range_radius:
        remainDist = range_radius - currentDist
        area += compute_two_node_area(tn, parent, remainDist)
        currentDist += tn.distance(parent)
        tn = parent
        parent = tn.parent

    # forwardtrace
    currentDist = 0
    childList = tn.children
    while len(childList) == 1 and currentDist < range_radius:
        child = childList[0]
        remainDist = range_radius - currentDist
        area += compute_two_node_area(tn, child, remainDist)
        currentDist += tn.distance(child)
        tn = child
        childList = tn.children

    return area


def get_lca(u, v):
    tmp_set = set()
    tmp_u = u
    tmp_v = v
    while u.get_id() != -1:
        tmp_set.add(u.get_id())
        u = u.parent

    while v.get_id() != -1:
        if v.get_id() in tmp_set:
            return v.get_id()
        v = v.parent
    return None


def nodes2coords(nodes: List["SwcNode"]):
    return np.array([node.coord for node in nodes])

class SwcNode(NodeMixin):

    def __init__(self, nid=-1, ntype=0, coord: EuclideanPoint3D=None, radius=1.0, parent=None):
        self.nid = nid
        self.ntype = ntype
        self.coord = coord
        self.radius = radius
        self.parent = parent
        self.root_length = 0.0
        if not isinstance(self.coord, EuclideanPoint3D):
            self.coord = EuclideanPoint3D(self.coord)

    def __getitem__(self, key):
        return self.coord[key]
    
    def __setitem__(self, key, value):
        self.coord[key] = value

    def is_virtual(self):
        """Returns True iff the node is virtual."""
        return self.nid < 0

    def is_regular(self):
        """Returns True iff the node is NOT virtual."""
        return self.nid >= 0

    def distance(self, tn=None, mode=_3D):
        """Returns the distance to another node.
        It returns 0 if either of the nodes is not regular.

        Args:
          tn : the target node for distance measurement
        """
        # make sure itself is a regular node
        if not self.is_regular():
            return 0.0

        # make sure tn is a valid swc node
        if isinstance(tn, SwcNode) and tn.is_regular():
            if mode == _2D:
                return self.coord.distance_to_point_2d(tn.coord)
            return self.coord.distance(tn.coord)

        # euc node is also acceptable
        if isinstance(tn, EuclideanPoint3D):
            if mode == _2D:
                return self.coord.distance_to_point_2d(tn)
            return self.coord.distance(tn)

        return 0.0

    def parent_distance(self):
        """Returns the distance to it parent."""
        return self.distance(self.parent)


    def is_isolated(self):
        if (self.parent is None or self.parent.nid == -1) and len(
            self.children
        ) == 0:
            return True
        return False

    def to_swc_str(self, pid=None, scale=(1.0, 1.0, 1.0)):
        if pid is not None:
            return f"{self.nid} {self.ntype} {self.coord[0]*scale[0]:.13e} {self.coord[1]*scale[1]:.13e} {self.coord[2]*scale[2]:.13e} {self.radius} {pid}\n"
        return f"{self.nid} {self.ntype} {self.coord[0]*scale[0]:.13e} {self.coord[1]*scale[1]:.13e} {self.coord[2]*scale[2]:.13e} {self.radius} {self.parent.nid}\n"


    def __str__(self):
        return "%d (%d): %s, %g" % (
            self.nid,
            self.ntype,
            str(self.coord),
            self.radius,
        )
