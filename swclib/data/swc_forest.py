import math
import os
import bisect
import queue
import numpy as np
from anytree import iterators, PreOrderIter
import miniball
from typing import TYPE_CHECKING
from scipy.spatial import KDTree
import copy
from collections import deque
import heapq

from swclib.data.swc_node import SwcNode, nodes2coords
from swclib.data.swc_soma import SwcSoma
from swclib.data.swc_fiber import SwcFiber
from swclib.data.euclidean_point import EuclideanPoint3D
from swclib.utils.points import *


class SwcForest:
    """A class for representing one or more SWC trees."""

    def __init__(self, swc=None):
        self.roots = []  # list of root nodes
        self._size = None
        self._total_length = None
        self._name = None
        self.scale = (1.0, 1.0, 1.0)

        self.id_set = set()  # all nodes has unique id.
        self.depth_array = None
        self.LOG_NODE_NUM = None
        self.lca_parent = None
        self.node_list = None
        self.id_node_dict = None

        from swclib.data.swc import Swc

        if isinstance(swc, (str, os.PathLike)):
            self.load(swc)
        elif isinstance(swc, Swc):
            self.load_from_swc(swc)

    def __contains__(self, nid):
        return nid in self.id_set

    def size(self):
        self._size = len(self.id_set)
        return self._size

    def name(self):
        return self._name

    def clear(self):
        self.roots = []
        self.id_set = set()

    def get_edge_num(self):
        edges = 0
        for node in self.get_node_list():
            if not node.is_root():
                edges += 1
        return edges

    def get_preorder_nodes(self):
        preorder_nodes = []
        stack = [root for root in reversed(self.roots) if root is not None]

        while stack:
            node = stack.pop()
            preorder_nodes.append(node)
            stack.extend(reversed(node.children))

        return preorder_nodes

    def get_node_by_nid(self, nid):
        for node in self.get_node_list():
            if node.nid == nid:
                return node
        return None

    def load_list(self, lines):
        self.clear()
        nodeDict = dict()
        for line in lines:
            if not line.strip().startswith("#"):
                if line.strip() == "":
                    continue
                try:
                    data = list(map(float, line.split()))
                    assert len(data) == 7
                except:
                    raise Exception("[Error: SwcForest.load_list] Invalid swc format: {}".format(line))
                if len(data) == 7:
                    nid = int(data[0])
                    ntype = int(data[1])
                    radius = data[5]
                    parentId = data[6]
                    if nid in self.id_set:
                        raise Exception("[Error: SwcForest.load]Same id {}".format(nid))
                    self.id_set.add(nid)
                    tn = SwcNode(
                        nid=nid,
                        ntype=ntype,
                        radius=radius,
                        coord=EuclideanPoint3D(data[2:5]),
                    )
                    nodeDict[nid] = (tn, parentId)
        for _, value in nodeDict.items():
            tn = value[0]
            parentId = value[1]
            if parentId == -1:
                self.roots.append(tn)
            else:
                if parentId not in nodeDict.keys():
                    raise Exception(
                        "[Error: SwcForest.load]Unknown parent id {}".format(parentId)
                    )
                parentNode = nodeDict.get(parentId)
                if parentNode:
                    tn.parent = parentNode[0]

        for node in self.get_node_list():
            if node.is_root():
                continue
            if node.parent.nid != -1:
                node.root_length = node.parent.root_length + node.parent_distance()

    def load(self, path):
        self.clear()
        self._name = os.path.basename(path)
        with open(path, "r") as fp:
            lines = fp.readlines()
            self.load_list(lines)

    def load_from_swc(self, swc):
        """
        Build SwcForest from an in-memory Swc object.
        """
        self.clear()
        self._name = swc.file_name

        nodeDict = {}

        # 1) Create all nodes
        for nid, rec in swc.nodes.items():
            # Flexible field mapping
            _nid = int(rec.get("id", nid))
            _type = int(rec.get("type", rec.get("ntype", 0)))
            _x = float(rec.get("x"))
            _y = float(rec.get("y"))
            _z = float(rec.get("z"))
            _r = float(rec.get("radius", rec.get("radius", 0.0)))
            _pid = int(rec.get("parent", rec["parent"]))

            if _nid in self.id_set:
                raise Exception(f"[Error: SwcForest.load_from_swc] Same id {_nid}")

            self.id_set.add(_nid)
            tn = SwcNode(
                nid=_nid,
                ntype=_type,
                radius=_r,
                coord=EuclideanPoint3D([_x, _y, _z]),
            )
            nodeDict[_nid] = (tn, _pid)

        # 2) Link parents
        for _nid, (tn, parentId) in nodeDict.items():
            if parentId == -1:
                self.roots.append(tn)
            else:
                tn.parent = nodeDict[parentId][0]

        # 3) Precompute root_length
        for node in self.get_node_list(update=True):
            if node.is_root():
                continue
            if node.parent.nid != -1:
                node.root_length = node.parent.root_length + node.parent_distance()

        return self

    def rescale(self, scale):
        self.scale = scale
        for node in self.get_node_list():
            for i in range(3):
                node.coord[i] = node.coord[i] * scale[i]

    def relocation(self, offset):
        """
        offset: (x_offset, y_offset, z_offset)
        """
        for node in self.get_node_list():
            for i in range(3):
                node.coord[i] = node.coord[i] + offset[i]

    def length(self, force_update=True):
        if self._total_length is not None and force_update == False:
            return self._total_length
        node_list = self.get_node_list()
        result = 0
        for tn in node_list:
            if tn.is_root():
                continue
            result += tn.parent_distance()
        self._total_length = result
        return result

    def next_id(self):
        if len(self.id_set) == 0:
            return 1
        return max(self.id_set) + 1

    # use this function if son is a new node
    def add_child(self, node, son_node):
        # swc_node is a module in model, while node and son_node are objects.
        assert isinstance(son_node, SwcNode) and isinstance(node, SwcNode)
        nid = self.next_id()

        son_node.nid = nid
        son_node.parent = node

        self.id_set.add(nid)

        self.length(force_update=True)

    # use this function if son use to be a part of this tree
    def link_child(self, pa, son):
        if not isinstance(pa, SwcNode) or not isinstance(son, SwcNode):
            return False
        son.parent = pa
        for node in PreOrderIter(son):
            if node.nid in self.id_set:
                raise Exception(
                    "[Error: SwcForest.link_child]Node id {} already exists".format(
                        node.nid
                    )
                )
            self.id_set.add(node.nid)

        self.length(force_update=True)
        return True

    def add_tree(self, root):
        assert isinstance(root, SwcNode)
        self.roots.append(root)
        # Use iterative traversal to avoid recursion depth limits on very deep trees.
        stack = [root]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.nid in self.id_set:
                raise Exception(
                    "[Error: SwcForest.link_child]Node id {} already exists".format(
                        node.nid
                    )
                )
            self.id_set.add(node.nid)
            children = list(node.children)
            children.reverse()
            stack.extend(children)

        self.length(force_update=True)

    def remove_node(self, node):
        assert isinstance(node, SwcNode)
        pa = node.parent
        if pa is not None:
            children = list(pa.children)
            children.remove(node)
            pa.children = tuple(children)
        else:
            assert node in self.roots
            self.roots.remove(node)

        for son in node.children:
            son.parent = None
            self.roots.append(son)
        self.id_set.remove(node.nid)
        self.length(force_update=True)

    def remove_tree(self, root):
        assert isinstance(root, SwcNode)
        assert root in self.roots
        self.roots.remove(root)

    def get_node_list(self, update=True, roi=None):
        if self.node_list is None or update:
            self.node_list = []
            q = queue.LifoQueue()
            for root in self.roots:
                q.put(root)
            while not q.empty():
                cur = q.get()
                if roi is None or (roi is not None and cur.is_in_roi(roi)):
                    self.node_list.append(cur)
                for child in cur.children:
                    q.put(child)
        return self.node_list

    def to_str_list(self):
        swc_node_list = self.get_node_list()
        swc_str = []
        for node in swc_node_list:
            swc_str.append(node.to_swc_str())
        return "".join(swc_str)

    def save_to_file(self, path):
        self.get_node_list(update=True)
        swc_str = self.to_str_list()
        with open(path, "w") as fp:
            fp.write(swc_str)

    def get_copy(self):
        new_tree = SwcForest()
        swc_str = self.to_str_list().split("\n")
        new_tree.load_list(swc_str)
        return new_tree

    def get_id_node_dict(self):
        if self.id_node_dict is not None:
            return self.id_node_dict
        self.id_node_dict = {}
        for node in self.get_node_list():
            self.id_node_dict[node.nid] = node
        return self.id_node_dict

    def get_branch_nodes(self):
        """
        get branch nodes (link to more than 3 nodes) of swc tree
        """
        swc_list = self.get_node_list()
        branch_list = []
        for node in swc_list:
            if node.is_root():
                continue
            if len(node.children) > 1:
                branch_list.append(node)
        return branch_list

    def get_leaf_nodes(self, with_isolated_root=False):
        swc_list = self.get_node_list()
        leaf_list = []
        for node in swc_list:
            if with_isolated_root and node.is_root() and len(node.children) == 1:
                leaf_list.append(node)
            if len(node.children) == 0:
                leaf_list.append(node)
        return leaf_list

    def get_nearest_node(self, coord, topk=1):
        if topk <= 0:
            return []

        nodes = self.get_node_list()
        candidates = [(node.distance(coord), node) for node in nodes]

        nearest = heapq.nsmallest(topk, candidates, key=lambda x: x[0])

        if topk == 1:
            dist, node = nearest[0]
            return node, dist

        return [(node, dist) for dist, node in nearest]

    def get_somas(self):
        """The soma labeled as 1. The area near the soma is annotated with straight lines. It is assumed that the soma is a sphere."""
        somas = []
        for child in self.roots:
            if child.ntype == 1:
                anno_center = child
                anno_fibers = []
                for cchild in child.children:
                    cur_node = cchild
                    fiber = SwcFiber()
                    fiber.append(anno_center)
                    while len(cur_node.children) > 0:
                        fiber.append(cur_node)
                        if len(cur_node.children) > 1:  # out soma and branch
                            for c in cur_node.children:
                                new_fiber = fiber.copy()
                                new_fiber.append(c)
                                anno_fibers.append(new_fiber)
                            cur_node = None
                            break
                        next_node = cur_node.children[0]
                        angle = cal_tree_point_angle(
                            anno_center[:], cur_node[:], next_node[:]
                        )
                        cur_node = next_node
                        if angle > 5:
                            break
                    if cur_node is not None:
                        fiber.append(cur_node)
                        anno_fibers.append(fiber)
                C, r2 = miniball.get_bounding_ball(
                    np.array([cs[-1][:] for cs in anno_fibers])
                )
                somas.append(
                    SwcSoma(
                        center=SwcNode(ntype=1, coord=C, radius=np.sqrt(r2)),
                        annoed_center=anno_center,
                        anno_fibers=anno_fibers,
                        scale=self.scale,
                    )
                )
        return somas

    def refine_somas(self, out_path=None, step_size=2.0):
        somas = self.get_somas()
        for soma in somas:
            anno_center = soma.annoed_center
            anno_center.coord = soma.center.coord
            for i, fiber in enumerate(soma.anno_fibers):
                for node in fiber[1:-1]:
                    if node is not None and node.nid in self:
                        self.remove_node(node)
                sampled_points = sample_points_from_point_pair(
                    anno_center.coord[:], fiber[-1].coord[:], step_size
                )
                cur_node = anno_center
                for sp in sampled_points[1:-1]:
                    new_node = SwcNode(
                        ntype=fiber[-1].ntype,
                        coord=EuclideanPoint3D(sp),
                        radius=fiber[-1].radius,
                    )
                    self.add_child(cur_node, new_node)
                    cur_node = new_node
                self.add_child(cur_node, fiber[-1])
        self.get_node_list(update=True)
        if out_path is not None:
            self.save_to_file(out_path)
        return self

    def get_roots(self, return_coords=False):
        roots = self.roots
        roots = nodes2coords(roots) if return_coords else roots
        return roots

    ## -- get fibers related functions -- ##
    def get_fibers(self, only_from_soma=False, min_length=0.0):
        fibers = []
        leaf_nodes = self.get_leaf_nodes()
        for node in leaf_nodes:
            fiber = SwcFiber()
            while node is not None:
                fiber.append(node)
                node = node.parent
            fiber.reverse()
            if len(fiber) > 1 and fiber not in fibers:
                if fiber.length < min_length:
                    continue
                if only_from_soma and fiber[0].ntype == 1:
                    fibers.append(fiber)
                elif not only_from_soma:
                    fibers.append(fiber)
        return fibers

    def get_fibers_by_roi(self, roi):
        """Get segmented fibers that all nodes in the region of interest (ROI)."""
        all_fibers = self.get_fibers()
        fibers = []
        for fiber in all_fibers:
            new_fiber = None
            for node in fiber:
                if node.is_in_roi(roi):
                    if new_fiber is None:
                        new_fiber = SwcFiber()
                    new_fiber.append(node)
                else:
                    if new_fiber is not None:
                        fibers.append(new_fiber)
                        new_fiber = None
            if new_fiber is not None and len(new_fiber) > 1 and new_fiber not in fibers:
                fibers.append(new_fiber)
        return fibers

    def get_fiber_by_leaf(self, leaf_node, roi=None):
        if roi is not None:
            if not leaf_node.is_in_roi(roi):
                return None
        fiber = SwcFiber()
        node = leaf_node
        while node is not None:
            if not node.is_in_roi(roi):
                break
            fiber.append(node)
            node = node.parent
        fiber.reverse()
        return fiber

    def align_roots(self, root_coords, align_roots_thredhold=20.0):
        # find new roots
        root_min_dist = {}
        nodes, node_coords = self.get_node_list(), nodes2coords(self.get_node_list())
        kdTree = KDTree(node_coords)
        for rc in root_coords:
            dist, idx = kdTree.query(rc)
            if dist <= align_roots_thredhold:
                node = nodes[idx]
                prev = root_min_dist.get(node)
                if prev is None or dist < prev:
                    root_min_dist[node] = dist
        new_roots = sorted(root_min_dist.keys(), key=lambda node: (root_min_dist[node], node.nid))

        # build new tree with new roots
        swc = SwcForest()
        vis = set()
        for root in new_roots:
            if root in vis:
                continue
            new_root, visited = root.get_rerooted_tree(
                nid_start=swc.next_id(), return_old_nodes=True
            )
            swc.add_tree(new_root)
            vis |= visited

        # add remaining components (if any) as separate trees
        for node in self.roots:
            if node not in vis:
                new_root, visited = node.get_rerooted_tree(
                    nid_start=swc.next_id(), return_old_nodes=True
                )
                swc.add_tree(new_root)
                vis |= visited
        return swc

    def get_components(self):
        components = []
        for child in self.roots:
            component = []
            for node in PreOrderIter(child):
                component.append(node)
            components.append(component)
        return components

    def get_depth_array(self, node_num):
        self.depth_array = [-1] * (node_num + 10)
        node_list = self.get_node_list()
        for node in node_list:
            self.depth_array[node.nid] = node.depth

    # initialize LCA data structure in swc_tree
    def get_lca_preprocess(self, node_num=-1):
        node_list = self.get_node_list()
        if node_num == -1:
            node_num = self.size()
        self.get_depth_array(node_num)
        self.LOG_NODE_NUM = math.ceil(math.log(node_num, 2)) + 1
        self.lca_parent = np.zeros(shape=(node_num + 10, self.LOG_NODE_NUM), dtype=int)
        tree_node_list = self.get_node_list()

        for node in tree_node_list:
            if node.is_root():
                self.lca_parent[node.nid][0] = -1
            else:
                self.lca_parent[node.nid][0] = node.parent.nid
        valid_nids = [node.nid for node in node_list]
        for k in range(self.LOG_NODE_NUM - 1):
            for v in valid_nids:
                if self.lca_parent[v][k] < 0:
                    self.lca_parent[v][k + 1] = -1
                else:
                    self.lca_parent[v][k + 1] = self.lca_parent[
                        int(self.lca_parent[v][k])
                    ][k]
        return True

    # input node id of two swc_node, calculate LCA
    def get_lca(self, u, v):
        lca_parent = self.lca_parent
        LOG_NODE_NUM = self.LOG_NODE_NUM
        depth_array = self.depth_array

        if depth_array[u] > depth_array[v]:
            u, v = v, u
        for k in range(LOG_NODE_NUM):
            if depth_array[v] - depth_array[u] >> k & 1:
                v = lca_parent[v][k]
        if u == v:
            return u
        for k in range(LOG_NODE_NUM - 1, -1, -1):
            if lca_parent[u][k] != lca_parent[v][k]:
                u = lca_parent[u][k]
                v = lca_parent[v][k]
                if u < 1 or v < 1:
                    return -1
        ans = lca_parent[u][0]
        return ans if ans >= 1 else -1
