import numpy as np
from scipy.ndimage import distance_transform_edt
import tifffile as tiff
import tqdm
import glob
import os
import copy
from numpy import linalg as LA
from scipy.spatial import distance_matrix
import cv2 as cv


from swclib.data.swc import Swc
from swclib.geometry.Obj3D import Point3D, Sphere, Cone


def swc_to_mask_line(swc: Swc, shape=(256, 256, 256), scale=(1.0, 1.0, 1.0), max_radius=5):
    mask = np.zeros(shape[::-1], dtype=np.uint8)
    def draw_line(p1, p2, r):
        p1 = np.array(p1)
        p2 = np.array(p2)
        diff = p2 - p1
        length = np.linalg.norm(diff)
        if length == 0:
            mask[int(p1[0]), int(p1[1]), int(p1[2])] = 1
            return
        steps = int(length) + 1
        for i in range(steps + 1):
            pos = p1 + diff * i / steps
            z, y, x = np.round(pos).astype(int)
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                mask[z, y, x] = 1

    for nid, node in swc.nodes.items():
        x, y, z, r = node["x"], node["y"], node["z"], node["radius"]
        parent_id = node["parent"]
        if parent_id == -1:
            continue
        parent = swc.nodes[parent_id]
        draw_line((z, y, x), (parent["z"], parent["y"], parent["x"]), r)
    mask_dt = distance_transform_edt(1 - mask, sampling=scale[::-1])
    mask = (mask_dt <= max_radius).astype(np.uint8)
    return mask


def setMarkWithSphere(mark, sphere, mark_shape, use_bbox=False):
    bbox = list(sphere.calBBox()) # xmin,ymin,zmin,xmax,ymax,zmax
    for i in range(3):
        j = i+3
        if (bbox[i]<0):
            bbox[i] = 0
        if (bbox[j]>mark_shape[i]):
            bbox[j] = mark_shape[i]
    (xmin,ymin,zmin,xmax,ymax,zmax) = tuple(bbox)
    (x_idxs,y_idxs,z_idxs)=np.where(mark[xmin:xmax,ymin:ymax,zmin:zmax]==0)
    # points=img_idxs[:3, xmin+x_idxs, ymin+y_idxs, zmin+z_idxs] # 3*M
    # points=points.T # M*3
    if not use_bbox:
        xs = np.asarray(xmin+x_idxs).reshape((len(x_idxs),1))
        ys = np.asarray(ymin+y_idxs).reshape((len(y_idxs),1))
        zs = np.asarray(zmin+z_idxs).reshape((len(z_idxs),1))
        points=np.hstack((xs,ys,zs))

        sphere_c_mat = np.array([sphere.center_point.toList()]) # 1*3
        # 计算所有点到所有球心的距离
        dis_mat = distance_matrix(points,sphere_c_mat) # M*1

        # 判断距离是否小于半径
        res_idxs = np.where(dis_mat<=sphere.radius)[0]
        # value_list = randIntList(lower,upper,len(res_idxs))
        for pos in res_idxs:
            mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 255
        # mark[xmin+x_idxs[res_idxs], ymin+y_idxs[res_idxs], zmin+z_idxs[res_idxs]] = 255
    else:
        # value_list = randIntList(lower,upper,len(res_idxs))
        for (px,py,pz) in zip(x_idxs,y_idxs,z_idxs):
            mark[xmin+x_idxs[px], ymin+y_idxs[py], zmin+z_idxs[pz]] = 255
        # mark[xmin+x_idxs, ymin+y_idxs, zmin+z_idxs] = 255



def setMarkWithCone(mark, cone, mark_shape, use_bbox=False):
    bbox = list(cone.calBBox()) # xmin,ymin,zmin,xmax,ymax,zmax
    for i in range(3):
        j = i+3
        if (bbox[i]<0):
            bbox[i] = 0
        if (bbox[j]>mark_shape[i]):
            bbox[j] = mark_shape[i]
    (xmin,ymin,zmin,xmax,ymax,zmax) = tuple(bbox)

    (x_idxs,y_idxs,z_idxs)=np.where(mark[xmin:xmax,ymin:ymax,zmin:zmax]==0)
    if not use_bbox:
        xs = np.asarray(xmin+x_idxs).reshape((len(x_idxs),1))
        ys = np.asarray(ymin+y_idxs).reshape((len(y_idxs),1))
        zs = np.asarray(zmin+z_idxs).reshape((len(z_idxs),1))
        ns = np.ones((len(z_idxs),1))
        points=np.hstack((xs,ys,zs,ns))

        # 每个圆锥的还原矩阵
        r_min=cone.up_radius
        r_max=cone.bottom_radius
        height=cone.height
        cone_revert_mat = cone.revertMat().T # 4*4

        # 每个椎体还原后坐标
        revert_coor_mat = np.matmul(points, cone_revert_mat) # M*4
        revert_radius_list = LA.norm(revert_coor_mat[:,:2], axis=1) # M

        # Local Indexs
        M = points.shape[0]
        l_idx = np.arange(M) # M (1-dim)
        l_mark = np.ones((M,), dtype=bool)

        # 过滤高度在外部的点
        res_idxs = np.logical_or(revert_coor_mat[l_idx[l_mark],2]<0, revert_coor_mat[l_idx[l_mark],2]>height)
        l_mark[l_idx[l_mark][res_idxs]]=False

        # 过滤半径在外部的点
        res_idxs = revert_radius_list[l_idx[l_mark]]>r_max
        l_mark[l_idx[l_mark][res_idxs]]=False

        # 过滤半径在内部的点
        res_idxs = revert_radius_list[l_idx[l_mark]]<=r_min
        # value_list = randIntList(lower,upper,len(l_idx[l_mark][res_idxs]))
        for pos in l_idx[l_mark][res_idxs]:

            mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 255
        # mark[xmin+x_idxs[l_idx[l_mark][res_idxs]], ymin+y_idxs[l_idx[l_mark][res_idxs]], zmin+z_idxs[l_idx[l_mark][res_idxs]]] = 255
        l_mark[l_idx[l_mark][res_idxs]]=False

        # 计算剩余
        if r_max>r_min:
            res_idxs = ((r_max-revert_radius_list[l_idx[l_mark]])*height/(r_max-r_min)) >= revert_coor_mat[l_idx[l_mark],2]
            # value_list = randIntList(lower,upper,len(l_idx[l_mark][res_idxs]))
            for pos in l_idx[l_mark][res_idxs]:

                mark[xmin+x_idxs[pos], ymin+y_idxs[pos], zmin+z_idxs[pos]] = 255
            # mark[xmin+x_idxs[l_idx[l_mark][res_idxs]], ymin+y_idxs[l_idx[l_mark][res_idxs]], zmin+z_idxs[l_idx[l_mark][res_idxs]]] = 255
            l_mark[l_idx[l_mark][res_idxs]]=False
    else:
        # value_list = randIntList(lower,upper,len(x_idxs))
        for (px,py,pz) in zip(x_idxs,y_idxs,z_idxs):

            mark[xmin+x_idxs[px], ymin+y_idxs[py], zmin+z_idxs[pz]] = 255
        # mark[xmin+x_idxs, ymin+y_idxs, zmin+z_idxs] = 255


def swc_to_mask_sphere_cone(swc: Swc, shape=(256, 256, 256), foreground_value=255, *, r_scale=1.0):
    if isinstance(swc, str) or isinstance(swc, os.PathLike):
        swc = Swc(swc)
    mask = np.zeros(shape, dtype=np.uint8)
    for nid in swc.nodes:
        node = swc.nodes[nid]
        parent_id = node['parent']
        x,y,z,r = node['x'], node['y'], node['z'], node['radius']*r_scale
        setMarkWithSphere(mask, Sphere(Point3D(z,y,x), r), shape)
        if parent_id != -1:
            parent_node = swc.nodes[parent_id]
            px,py,pz,pr = parent_node['x'], parent_node['y'], parent_node['z'], parent_node['radius']*r_scale
            setMarkWithCone(mask, Cone(Point3D(pz,py,px), pr, Point3D(z,y,x), r), shape)
    mask = (mask > 0).astype(np.uint8) * foreground_value
    return mask


class Swc2Mask:

    def __init__(
        self, shape=(300, 300, 300), scale=(1.0, 1.0, 1.0), radius=1, method="line"
    ):
        self.shape = shape
        self.scale = scale
        self.radius = radius
        self.method = method

    def run(self, swc, out_file=None):
        if isinstance(swc, str) or isinstance(swc, os.PathLike):
            swc = Swc(swc)
        if self.method == "line":
            mask = swc_to_mask_line(
                swc, shape=self.shape, scale=self.scale, max_radius=self.radius
            )
        elif self.method == "sphere_cone":
            mask = swc_to_mask_sphere_cone(
                swc, shape=self.shape, foreground_value=255
            )
        else:
            raise NotImplementedError
        if out_file is not None:
            tiff.imwrite(out_file, mask.astype(np.uint8))
        return mask
