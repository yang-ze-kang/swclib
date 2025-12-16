from swclib.data.swc_node import SwcNode


class Swc(object):

    def __init__(self, file_name=None):
        self.nodes = {}
        self.edges = []
        self.bound_box = [0, 0, 0, 0, 0, 0]  # x0,y0,z0,x1,y1,z1
        if file_name:
            self.open(file_name)

    def open(self, file_name):
        with open(file_name) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                id, type, x, y, z, r, pid = map(float, line.split())
                self.nodes[id] = {
                    "id": id,
                    "type": type,
                    "x": x,
                    "y": y,
                    "z": z,
                    "radius": r,
                    "parent": pid,
                }
                self.edges.append((id, pid))
                if x < self.bound_box[0]:
                    self.bound_box[0] = x
                if x > self.bound_box[3]:
                    self.bound_box[3] = x
                if y < self.bound_box[1]:
                    self.bound_box[1] = y
                if y > self.bound_box[4]:
                    self.bound_box[4] = y
                if z < self.bound_box[2]:
                    self.bound_box[2] = z
                if z > self.bound_box[5]:
                    self.bound_box[5] = z
