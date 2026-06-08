from anytree import NodeMixin
from typing import List
import numpy as np
from collections import deque
import copy
import queue

from swclib.data.euclidean_point import EuclideanPoint3D

_3D = "3d"
_2D = "2d"


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

    def is_regular(self):
        """Returns True iff the node is NOT virtual."""
        return self.nid >= 0
    
    def is_in_roi(self, roi):
        """Returns True iff the node is in the given roi."""
        (xmin, ymin, zmin), (xmax, ymax, zmax) = roi
        return (xmin <= self.coord[0] <= xmax and
                ymin <= self.coord[1] <= ymax and
                zmin <= self.coord[2] <= zmax)
    
    def is_root(self):
        """Returns True iff the node is root."""
        return self.parent is None or self.parent.nid == -1

    def distance(self, tn=None, mode=_3D):
        """Returns the distance to another node.
        It returns 0 if either of the nodes is not regular.

        Args:
          tn : the target node for distance measurement
        """
        # make sure itself is a regular node
        if not self.is_regular():
            return None

        # make sure tn is a valid swc node
        if isinstance(tn, SwcNode) and tn.is_regular():
            if mode == _2D:
                return self.coord.distance_to_point_2d(tn.coord)
            return self.coord.distance(tn.coord)
        
        if isinstance(tn, list) and len(tn) == 3:
            tn = EuclideanPoint3D(tn)

        # euc node is also acceptable
        if isinstance(tn, EuclideanPoint3D):
            if mode == _2D:
                return self.coord.distance_to_point_2d(tn)
            return self.coord.distance(tn)

        return None

    def parent_distance(self):
        """Returns the distance to it parent."""
        return self.distance(self.parent)

    def to_swc_str(self, pid=None, scale=(1.0, 1.0, 1.0)):
        if pid is not None:
            return f"{self.nid} {self.ntype} {self.coord[0]*scale[0]:.13e} {self.coord[1]*scale[1]:.13e} {self.coord[2]*scale[2]:.13e} {self.radius} {pid}\n"
        if self.is_root():
            return f"{self.nid} {self.ntype} {self.coord[0]*scale[0]:.13e} {self.coord[1]*scale[1]:.13e} {self.coord[2]*scale[2]:.13e} {self.radius} -1\n"
        return f"{self.nid} {self.ntype} {self.coord[0]*scale[0]:.13e} {self.coord[1]*scale[1]:.13e} {self.coord[2]*scale[2]:.13e} {self.radius} {self.parent.nid}\n"


    def __str__(self):
        return "%d (%d): %s, %g" % (
            self.nid,
            self.ntype,
            str(self.coord),
            self.radius,
        )
    
    ## --- Below are subtree properties with self as root --- ##
    def get_subtree_node_list(self):
        node_list = []
        q = queue.LifoQueue()
        q.put(self)
        while not q.empty():
            cur = q.get()
            node_list.append(cur)
            for child in cur.children:
                q.put(child)
        return node_list
    
    def get_subtree_length(self, force_update=False):
        node_list = self.get_subtree_node_list()
        result = 0
        for tn in node_list:
            if tn.is_root():
                continue
            result += tn.parent_distance()
        return result
    
    def get_subtree_leafs(self,roi=None):
        leafs = []
        q = queue.LifoQueue()
        if roi is not None and not self.is_in_roi(roi):
            return leafs
        q.put(self)
        while not q.empty():
            cur = q.get()
            if len(cur.children) == 0 and not cur.is_root():
                leafs.append(cur)
            f = True
            for child in cur.children:
                if roi is None or child.is_in_roi(roi):
                    f = False
                    q.put(child)
            if f and len(cur.children) > 0:
                leafs.append(cur)
        return leafs
    
    def get_subtree_fibers(self, roi=None, with_root=False):
        from swclib.data.swc_fiber import SwcFiber
        fibers = []
        leafs = self.get_subtree_leafs(roi)
        for node in leafs:
            fiber = SwcFiber()
            while node != self:
                fiber.append(node)
                node = node.parent
            if with_root and len(fiber) > 0:
                fiber.append(self)
            fiber.reverse()
            if len(fiber) > 1 and fiber not in fibers:
                fibers.append(fiber)
        return fibers
    
    def remove_subtree_fiber(self, leaf_node, with_root=False):
        node = leaf_node
        while node.parent != self and len(node.parent.children) <= 1:
            node = node.parent
        node.parent = None
        if with_root and node.parent == self:
            self = None

    
    def get_rerooted_tree(self, nid_start=1, ntype=None, return_old_nodes=False, return_old2new=False):
        # --- 1) Collect all nodes in the original connected component (the whole tree) ---
        # We can reach all nodes by going to the original root then iterating descendants,
        # but we can also build adjacency on the fly by BFS from new_root_old.
        # Here we BFS using "undirected" neighbors: parent + children.
        q = deque([self])
        parent_old = {self: None}
        old_nodes = []  # BFS order in old graph
        while q:
            node = q.popleft()
            if node in old_nodes:
                continue
            old_nodes.append(node)

            if not node.is_root():
                if node.parent not in parent_old:
                    q.append(node.parent)
                    parent_old[node.parent] = node
            
            for c in node.children:
                if c not in parent_old:
                    q.append(c)
                    parent_old[c] = node

        # --- 2) Create new nodes (one-to-one mapping), without linking parents yet ---
        old2new = {}
        for old in old_nodes:
            new = SwcNode(
                nid=nid_start,
                ntype=old.ntype if ntype is None else ntype,
                coord=copy.deepcopy(old.coord),
                radius=old.radius,
            )
            old2new[old] = new
            nid_start += 1        

        # Link new nodes by the parent map
        for child_old, p_old in parent_old.items():
            child_new = old2new[child_old]
            if p_old is not None:
                child_new.parent = old2new[p_old]
        
        if return_old_nodes:
            return old2new[self], set(old_nodes)
        
        if return_old2new:
            return old2new[self], old2new
        
        return old2new[self]
    
    def get_rerooted_subtree_fibers(self, roi=None, with_root=False):
        from swclib.data.swc_fiber import SwcFiber
        fibers = []
        if not self.is_in_roi(roi):
            return fibers
        fiber = SwcFiber()
        stack = [[self, False]]
        visited = set()
        visited.add(self.nid)
        while len(stack) > 0:
            cur, is_exit = stack.pop()
            if not is_exit:
                fiber.append(cur)
                stack.append([cur, True])
                f = True
                for child in cur.children:
                    if child.is_in_roi(roi) and child.nid not in visited:
                        f = False   
                        visited.add(child.nid)
                        stack.append([child, False])
                if not cur.is_root() and cur.parent.nid not in visited and cur.parent.is_in_roi(roi):
                    f = False
                    visited.add(cur.parent.nid)
                    stack.append([cur.parent, False])
                if f:
                    new_fiber = fiber.copy()
                    if not with_root:
                        new_fiber.nodes = new_fiber.nodes[1:]
                    fibers.append(new_fiber)
            else:
                fiber.pop()
        return fibers
    
    def get_fiber_by_leaf(self, roi=None):
        from swclib.data.swc_fiber import SwcFiber
        fiber = SwcFiber()
        node = self
        while node.parent != None and not node.is_root():
            if node.is_in_roi(roi):
                fiber.append(node)
            else:
                break
            node = node.parent
        fiber.reverse()
        return fiber
