import math
import os
import bisect
import queue
import numpy as np
from anytree import iterators, PreOrderIter
import miniball

from swclib.data.swc_node import SwcNode
from swclib.data.swc_soma import SwcSoma
from swclib.data.swc_fiber import SwcFiber
from swclib.data.euclidean_point import EuclideanPoint3D
from swclib.utils.points import *


def Make_Virtual():
    return SwcNode(nid=-1, coord=EuclideanPoint3D([0, 0, 0]))


def get_nearby_swc_node_list(gold_node, test_kdtree, test_pos_node_dict, threshold):
    """
    find all nodes in "test_swc_list" which are close enough to "gold_node"
    sort them by distance

    :param gold_node: swc_node
    :param test_swc_list:
    :param threshold:
    :return:
    """
    if gold_node.is_virtual():
        return
    tmp_list = []
    # find the closest pos for gold node
    target_pos_list = test_kdtree.search_knn(list(gold_node.get_center_as_tuple()), k=5)
    for pos in target_pos_list:
        target_node = test_pos_node_dict[tuple(pos[0].data)]
        if target_node.is_virtual() or gold_node.is_virtual():
            continue
        # only if gold and test nodes are very close(dis < 0.03), they can be considered as the same pos
        if gold_node.distance(target_node) < threshold:
            tmp_list.append(tuple([target_node, target_node.distance(gold_node)]))

    tmp_list.sort(key=lambda x: x[1])
    res_list = []
    for tu in tmp_list:
        res_list.append(tu[0])
    return res_list


class SwcTree:
    """A class for representing one or more SWC trees.
    For simplicity, we always assume that the root is a virtual node.
    """

    def __init__(self, path=None):
        self.root = Make_Virtual()
        self._size = None
        self._total_length = None
        self._name = None
        self.scale = (1.0, 1.0, 1.0)

        self.id_set = set()
        self.depth_array = None
        self.LOG_NODE_NUM = None
        self.lca_parent = None
        self.node_list = None
        self.id_node_dict = None

        if path is not None:
            self.load(path)

    def __contains__(self, nid):
        return nid in self.id_set

    def size(self):
        self._size = len(self.id_set)
        return self._size

    def name(self):
        return self._name

    def clear(self):
        self.root = Make_Virtual()

    # warning: slow, don't use in loop
    def node_from_id(self, nid):
        niter = iterators.PreOrderIter(self.root)
        for tn in niter:
            if tn.nid == nid:
                return tn
        return None

    def load_list(self, lines):
        self.clear()
        nodeDict = dict()
        for line in lines:
            if not line.strip().startswith("#"):
                data = list(map(float, line.split()))
                if len(data) == 7:
                    nid = int(data[0])
                    ntype = int(data[1])
                    radius = data[5]
                    parentId = data[6]
                    if nid in self.id_set:
                        raise Exception("[Error: SwcTree.load]Same id {}".format(nid))
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
                tn.parent = self.root
            else:
                if parentId not in nodeDict.keys():
                    raise Exception(
                        "[Error: SwcTree.load]Unknown parent id {}".format(parentId)
                    )
                parentNode = nodeDict.get(parentId)
                if parentNode:
                    tn.parent = parentNode[0]

        for node in self.get_node_list():
            if node.parent is None:
                continue
            if node.parent.nid != -1:
                node.root_length = node.parent.root_length + node.parent_distance()

    def load(self, path):
        self.clear()
        self._name = os.path.basename(path)
        with open(path, "r") as fp:
            lines = fp.readlines()
            self.load_list(lines)

    def rescale(self, scale):
        self.scale = scale
        for node in self.get_node_list():
            for i in range(3):
                node.coord[i] = node.coord[i] * scale[i]

    def length(self, force_update=False):
        if self._total_length is not None and force_update == False:
            return self._total_length
        node_list = self.get_node_list()
        result = 0
        for tn in node_list:
            if tn.is_virtual() or tn.parent.is_virtual():
                continue
            result += tn.parent_distance()
        return result

    def get_depth_array(self, node_num):
        self.depth_array = [0] * (node_num + 10)
        node_list = self.get_node_list()
        for node in node_list:
            self.depth_array[node.nid] = node.depth - 1

    # initialize LCA data structure in swc_tree
    def get_lca_preprocess(self, node_num=-1):
        if node_num == -1:
            node_num = self.size()
        self.get_depth_array(node_num)
        self.LOG_NODE_NUM = math.ceil(math.log(node_num, 2)) + 1
        self.lca_parent = np.zeros(shape=(node_num + 10, self.LOG_NODE_NUM), dtype=int)
        tree_node_list = self.get_node_list()

        for node in tree_node_list:
            if node.is_virtual():
                continue
            self.lca_parent[node.nid][0] = node.parent.nid
        for k in range(self.LOG_NODE_NUM - 1):
            for v in range(1, node_num + 1):
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
        return lca_parent[u][0]

    def align_roots(self, gold_tree, matches, DEBUG=False):
        offset = EuclideanPoint3D()
        stack = queue.LifoQueue()
        swc_test_list = self.get_node_list()

        for root in gold_tree.root.children:
            gold_anchor = np.array(root._pos)
            if root in matches.keys():
                test_anchor = np.array(matches[root]._pos)
            else:
                nearby_nodes = get_nearby_swc_node_list(
                    gold_node=root,
                    test_swc_list=swc_test_list,
                    threshold=root.radius() / 2,
                )
                if len(nearby_nodes) == 0:
                    continue
                test_anchor = nearby_nodes[0]._pos

            offset._pos = (test_anchor - gold_anchor).tolist()
            if DEBUG:
                print(
                    "off_set:x = {}, y = {}, z = {}".format(
                        offset._pos[0], offset._pos[1], offset._pos[2]
                    )
                )

            stack.put(root)
            while not stack.empty():
                node = stack.get()
                if node.is_virtual():
                    continue

                node._pos[0] += offset._pos[0]
                node._pos[1] += offset._pos[1]
                node._pos[2] += offset._pos[2]

                for son in node.children:
                    stack.put(son)

    def change_root(self, new_root_id):
        stack = queue.LifoQueue()
        swc_list = self.get_node_list()
        list_size = max(self.id_set)
        vis_list = np.zeros(shape=(list_size + 10))
        pa_list = [None] * (list_size + 10)

        for node in swc_list:
            pa_list[node.nid] = node.parent
        new_root = self.node_from_id(new_root_id)

        stack.put(new_root)
        pa_list[new_root_id] = self.root
        while not stack.empty():
            cur_node = stack.get()
            vis_list[cur_node.nid] = True
            for son in cur_node.children:
                if not vis_list[son.nid]:
                    stack.put(son)
                    pa_list[son.nid] = cur_node
            if (
                cur_node.parent is not None
                and cur_node.parent.nid != -1
                and not vis_list[cur_node.parent.nid]
            ):
                stack.put(cur_node.parent)
                pa_list[cur_node.parent.nid] = cur_node

        for i in range(1, len(swc_list)):
            swc_list[i].parent = None
        for i in range(1, len(swc_list)):
            swc_list[i].parent = pa_list[swc_list[i].nid]

    def type_clear(self, col=0, rt_color=2):
        node_list = self.get_node_list()
        for node in node_list:
            node._type = col
        for root in self.root.children:
            root._type = rt_color

    def radius_limit(self, x):
        node_list = self.get_node_list()
        for node in node_list:
            node._radius /= x

    def next_id(self):
        return max(self.id_set) + 1

    # use this function if son is a new node
    def add_child(self, node, son_node):
        # swc_node is a module in model, while node and son_node are objects.
        if not isinstance(son_node, SwcNode) or not isinstance(node, SwcNode):
            return False
        nid = self.next_id()

        son_node.nid = nid
        son_node.parent = node

        self.id_set.add(nid)
        return True

    # use this function if son use to be a part of this tree
    def link_child(self, pa, son):
        if not isinstance(pa, SwcNode) or not isinstance(son, SwcNode):
            return False
        son.parent = pa
        return True

    def remove_node(self, node):
        if not isinstance(node, SwcNode):
            return False
        pa = node.parent
        children = list(pa.children)
        children.remove(node)
        pa.children = tuple(children)

        for son in node.children:
            son.parent = self.root
        self.id_set.remove(node.nid)
        return True

    def unlink_child(self, node):
        if not isinstance(node, SwcNode):
            return False
        pa = node.parent
        children = list(pa.children)
        children.remove(node)
        pa.children = tuple(children)
        node.parent = self.root
        return True

    def get_node_list(self, update=False):
        if self.node_list is None or update:
            self.node_list = []
            q = queue.LifoQueue()
            q.put(self.root)
            while not q.empty():
                cur = q.get()
                self.node_list.append(cur)
                for child in cur.children:
                    q.put(child)
        return self.node_list

    def sort_node_list(self, key="id"):
        """
        index:
            default: order by pre order
            id: order by id
            compress: re-order from 1 to the size of swc tree
        """
        if key == "default":
            self.get_node_list(update=True)
        if key == "id":
            self.node_list.sort(key=lambda node: node.nid)
        if key == "compress":
            self.node_list.sort(key=lambda node: node.nid)
            id_list = []
            for id in self.id_set:
                id_list.append(id)
            id_list.sort()
            for node in self.node_list:
                if node.is_virtual():
                    continue
                new_id = bisect.bisect_left(id_list, node.nid) + 1
                self.id_set.remove(node.nid)
                self.id_set.add(new_id)
                node.nid = new_id

    def to_str_list(self):
        swc_node_list = self.get_node_list()
        swc_str = []
        for node in swc_node_list:
            if node.is_virtual():
                continue
            swc_str.append(node.to_swc_str())
        return "".join(swc_str)

    def save_to_file(self, path):
        self.get_node_list(update=True)
        swc_str = self.to_str_list()
        with open(path, "w") as fp:
            fp.write(swc_str)

    def get_copy(self):
        new_tree = SwcTree()
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

    def get_branch_swc_list(self):
        """
        get branch nodes (link to more than 3 nodes) of swc tree
        """
        swc_list = self.get_node_list()
        branch_list = []
        for node in swc_list:
            if node.is_virtual():
                continue
            if len(node.children) > 1:
                branch_list.append(node)
        return branch_list

    def get_leaf_swc_list(self):
        swc_list = self.get_node_list()
        leaf_list = []
        for node in swc_list:
            if node.is_virtual():
                continue
            if node.parent.is_virtual():
                if len(node.children) == 1:
                    leaf_list.append(node)
            if len(node.children) == 0:
                leaf_list.append(node)
        return leaf_list

    def set_node_type_by_topo(self, root_id=1):
        """
        root_id decide other nodes' id
        branch = root_id + 1
        leaf = root_id + 3
        normal node = root_id + 2
        """
        swc_list = self.get_node_list()
        for node in swc_list:
            if node.is_virtual():
                continue
            if node.parent.nid == -1:
                node.ntype = root_id
            elif len(node.children) > 1:
                node.ntype = root_id + 1
            elif len(node.children) == 1:
                node.ntype = root_id + 2
            else:
                node.ntype = root_id + 3

    def break_branches(self):
        breaked_tree = self.get_copy()
        branch_list = breaked_tree.get_branch_swc_list()
        for branch in branch_list:
            if branch.is_virtual():
                continue
            for branch_son in branch.children:
                breaked_tree.unlink_child(branch_son)
        breaked_tree.get_node_list(update=True)
        return breaked_tree

    def get_somas(self):
        """The soma labeled as 1. The area near the soma is annotated with straight lines. It is assumed that the soma is a sphere."""
        somas = []
        for child in self.root.children:
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
                C, r2 = miniball.get_bounding_ball(np.array([cs[-1][:] for cs in anno_fibers]))
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

    def get_fibers(self, only_from_soma=False):
        fibers = []
        leaf_nodes = self.get_leaf_swc_list()
        for node in leaf_nodes:
            fiber = SwcFiber()
            while node != self.root:
                fiber.append(node)
                node = node.parent
            fiber.reverse()
            if len(fiber) > 1 and fiber not in fibers:
                if only_from_soma and fiber[0].ntype == 1:
                    fibers.append(fiber)
                elif not only_from_soma:
                    fibers.append(fiber)
        return fibers
    
    def get_components(self):
        """
        获取连通分量
        Return: list of node list.
        """
        components = []
        for child in self.root.children():
            component = []
            for node in PreOrderIter(child):
                component.append(node)
            components.append(component)
        return components
